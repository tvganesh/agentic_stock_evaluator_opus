"""The LLM gateway: the only code in the system allowed to call a model.

Every model call in phases 3 and 5 goes through :meth:`LLMGateway.call`, which enforces:

1. **A slot or no call.** The caller must present a :class:`SlotTicket` from the ledger; the
   gateway redeems it (once) and takes the model, ``max_in``, ``max_out`` and effort from the
   compiled plan -- the caller cannot choose a model or a limit.
2. **Phase and seal.** The call runs inside ``PhaseMachine.model_window()``, which is refused
   outside CLAIM/VETO and opens exactly one reachable host (the model provider).
3. **Measured input.** ``count_tokens`` runs before dispatch; a prompt above ``max_in`` raises
   :class:`PromptOverBudget` (an assembler bug, never silently truncated).
4. **Bounded output.** ``max_tokens = max_out``, which also caps adaptive thinking.
5. **No tools, typed output only.** The request surface has no ``tools`` parameter at all;
   output is constrained by structured outputs to a Pydantic schema (claims or refutations).
6. **Settlement and audit.** Actual usage is settled against the slot's cost bound; every call
   is audited with its ticket, model, prompt hash, token counts, cost and stop reason, and the
   parsed output is appended to the run transcript for replay.

The model provider client is behind the :class:`ModelClient` protocol so tests and offline
runs can substitute a deterministic implementation without touching governance code.

Refusal fallbacks: the server-side ``fallbacks`` option is deliberately not enabled, because
it could re-route a call to a model the approved plan did not price. A refused call simply
yields no claims for that slot.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from ..snapshot.hashing import canonical_json, content_hash
from . import policy
from .audit import AuditLog
from .errors import PromptOverBudget
from .seal import PhaseMachine
from .spend import ADAPTIVE_THINKING_MODELS, SlotLedger, SlotTicket, TokenUsage

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ModelRequest:
    """Everything a model client needs for one call, fully determined by the plan row and prompt."""

    model: str
    max_tokens: int
    system: str
    user: str
    output_type: type[BaseModel]
    thinking: dict[str, Any] | None
    effort: str | None


@dataclass
class ModelResult:
    """What a model client returns: parsed output (or ``None``) plus usage for settlement."""

    parsed: BaseModel | None
    stop_reason: str
    input_tokens: int
    output_tokens: int
    request_id: str | None = None
    detail: str | None = None
    usage_estimated: bool = False
    thinking: str | None = None
    """The model's extended-thinking text, when it produced any.

    Billed as output and charged against ``max_tokens``, so on a thinking model it is most of what
    a call costs: one veto on 18 Sep 2026 spent 15,733 output tokens to emit 136 tokens of JSON.
    Discarding it meant paying for reasoning and keeping only the conclusion.

    It is recorded for review, never for publication. The dossier is assembled from adjudicated
    claims alone, because a claim carries a falsifier the machine re-checks against the snapshot
    while this is unverified prose. Putting it in the report would place a sentence in front of a
    reader that nothing checked."""


class ModelClient(Protocol):
    """Minimal model surface the gateway depends on. Note: no tools, no URLs, no model choice.

    ``permitted_hosts`` is optional: a client that reaches the network declares which hosts its
    model window must open (the Anthropic endpoint, or the loopback set for a locally served
    model), so the gateway never has to know about modes. A client without it gets the default.
    """

    label: str
    permitted_hosts: frozenset[str] | None

    def count_input_tokens(self, request: ModelRequest) -> int:
        """Return the exact input token count for ``request``."""

    def complete(self, request: ModelRequest) -> ModelResult:
        """Run ``request`` and return parsed output and usage."""


class AnthropicModelClient:
    """:class:`ModelClient` backed by the Anthropic Messages API with structured outputs."""

    label = "anthropic"
    permitted_hosts = policy.MODEL_PROVIDER_HOSTS

    def __init__(self, timeout_s: float = 600.0) -> None:
        """Create an SDK client pinned to the policy's provider URL (credentials resolved by the SDK).

        The 10-minute timeout covers a veto call generating up to 16,000 tokens with adaptive thinking.
        """
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(base_url=policy.MODEL_PROVIDER_BASE_URL, timeout=timeout_s)

    @staticmethod
    def _base_kwargs(request: ModelRequest) -> dict[str, Any]:
        """Request fields shared by ``count_tokens`` and ``parse``."""
        kwargs: dict[str, Any] = {
            "model": request.model,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
            "output_format": request.output_type,
        }
        if request.thinking:
            kwargs["thinking"] = request.thinking
        return kwargs

    def count_input_tokens(self, request: ModelRequest) -> int:
        """Count input tokens including the output schema, via ``messages.count_tokens``."""
        return self._client.messages.count_tokens(**self._base_kwargs(request)).input_tokens

    def complete(self, request: ModelRequest) -> ModelResult:
        """Call ``messages.parse``; map refusals, truncation and schema failures to ``parsed=None``."""
        anthropic = self._anthropic
        kwargs = self._base_kwargs(request)
        if request.effort:
            kwargs["output_config"] = {"effort": request.effort}
        try:
            response = self._client.messages.parse(max_tokens=request.max_tokens, **kwargs)
        except anthropic.APIStatusError as exc:  # HTTP error responses are not billed
            return ModelResult(None, "api_error", 0, 0, detail=f"HTTP {exc.status_code}")
        except anthropic.APIConnectionError as exc:  # may have been billed: settle at the bound
            return ModelResult(None, "connection_error", 0, 0, detail=type(exc).__name__, usage_estimated=True)
        except ValueError as exc:  # output did not validate against the schema after generation
            return ModelResult(None, "schema_error", 0, 0, detail=str(exc)[:200], usage_estimated=True)

        usage = response.usage
        input_tokens = (usage.input_tokens or 0) + (usage.cache_creation_input_tokens or 0) + (
            usage.cache_read_input_tokens or 0
        )
        parsed = None
        if response.stop_reason not in ("refusal", "max_tokens"):
            try:
                parsed = response.parsed_output
            except ValueError:
                parsed = None
        detail = None
        if response.stop_reason == "refusal" and getattr(response, "stop_details", None):
            detail = str(getattr(response.stop_details, "category", None))
        return ModelResult(parsed, str(response.stop_reason), input_tokens, usage.output_tokens or 0,
                           request_id=getattr(response, "_request_id", None), detail=detail,
                           thinking=_thinking_text(response))


DEGRADED_STOP_REASONS: frozenset[str] = frozenset({
    "context_truncated", "api_error", "connection_error", "schema_error", "malformed_response", "max_tokens",
})
"""Stop reasons where the model produced no usable answer *because something went wrong*.

A call that returns nothing is ambiguous: a model can legitimately find nothing to say, and an
empty batch is a real finding. These reasons are the other case -- the prompt was cut, the server
errored, the output did not validate -- and they must not be reported as a considered silence.

This distinction has already cost a diagnosis. Two local runs published dossiers in which the veto
recorded no refutations; that read as an auditor with nothing to object to, and it took a canary
probe to establish the auditor had never received the claims at all. Only the gateway can tell the
two apart, because only the gateway sees ``stop_reason``."""


def _thinking_text(response: Any) -> str | None:
    """Join the extended-thinking blocks of a response, or ``None`` if it did no visible thinking.

    Thinking arrives as its own content blocks alongside the answer. Redacted blocks carry opaque
    encrypted data rather than readable text, so they are noted by count rather than decoded.

    Read defensively through ``getattr``: this runs inside a paid call, and a change to the SDK's
    block shape should cost the transcript its reasoning, never cost the run its result.
    """
    blocks = getattr(response, "content", None) or []
    parts, redacted = [], 0
    for block in blocks:
        kind = getattr(block, "type", None)
        if kind == "thinking":
            text = getattr(block, "thinking", None)
            if text:
                parts.append(str(text))
        elif kind == "redacted_thinking":
            redacted += 1
    if redacted:
        parts.append(f"[{redacted} redacted thinking block(s): encrypted by the provider, not readable here]")
    return "\n\n".join(parts) if parts else None


class LLMGateway:
    """Enforces slots, phases, token limits, typed output and audit for every model call."""

    def __init__(
        self,
        *,
        ledger: SlotLedger,
        audit: AuditLog,
        client: ModelClient,
        phases: PhaseMachine,
        transcript_path: Path | None = None,
    ) -> None:
        """Wire the gateway to one run's ledger, audit log, phase machine and model client."""
        self._ledger = ledger
        self._audit = audit
        self._client = client
        self._phases = phases
        self._transcript_path = transcript_path
        self._transcript_lock = threading.Lock()
        self._degraded: list[str] = []
        self._degraded_lock = threading.Lock()

    def drain_degraded(self) -> list[str]:
        """Take the degraded-call messages recorded so far, so a run can surface them as notes.

        Draining rather than reading keeps a note from being reported twice when the orchestrator
        collects after each phase.
        """
        with self._degraded_lock:
            messages, self._degraded = self._degraded, []
        return messages

    @property
    def client_label(self) -> str:
        """Which model client is in use (``anthropic`` or an offline stand-in); shown on the dossier."""
        return self._client.label

    def call(
        self,
        ticket: SlotTicket,
        *,
        system: str,
        user: str,
        output_type: type[T],
        purpose: dict[str, Any],
    ) -> T | None:
        """Make one governed model call and return validated output, or ``None`` if there is none."""
        row = self._ledger.redeem(ticket)
        adaptive = row.model in ADAPTIVE_THINKING_MODELS
        request = ModelRequest(
            model=row.model,
            max_tokens=row.max_out,
            system=system,
            user=user,
            output_type=output_type,
            thinking={"type": "adaptive"} if adaptive else None,
            effort=row.effort if adaptive else None,
        )
        prompt_hash = content_hash(canonical_json({"system": system, "user": user, "schema": output_type.__name__}))

        with self._phases.model_window(getattr(self._client, "permitted_hosts", None)):
            counted = self._client.count_input_tokens(request)
            if counted > row.max_in:
                self._ledger.settle(ticket, TokenUsage(0, 0))
                self._audit.record("model.prompt_over_budget", {
                    "ticket": ticket.ticket_id, "counted_input_tokens": counted, "max_in": row.max_in, **purpose})
                raise PromptOverBudget(f"{ticket.ticket_id}: prompt has {counted} tokens > max_in {row.max_in}")
            result = self._client.complete(request)

        usage = (TokenUsage(counted, row.max_out) if result.usage_estimated
                 else TokenUsage(result.input_tokens, result.output_tokens))
        cost = self._ledger.settle(ticket, usage)
        parsed = result.parsed if isinstance(result.parsed, output_type) else None
        self._audit.record("model.call", {
            "ticket": ticket.ticket_id,
            "slot_class": ticket.slot_class.value,
            "client": self._client.label,
            "model": row.model,
            "prompt_hash": prompt_hash,
            "counted_input_tokens": counted,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "usage_estimated": result.usage_estimated,
            "cost_microusd": cost,
            "stop_reason": result.stop_reason,
            "detail": result.detail,
            "request_id": result.request_id,
            "has_output": parsed is not None,
            **purpose,
        })
        if result.stop_reason in DEGRADED_STOP_REASONS:
            # Recorded, not raised: one bad slot costs its claims, it does not end the run. But it must
            # reach the operator, or a broken call is indistinguishable from a model with nothing to say.
            agent = purpose.get("agent", ticket.slot_class.value)
            detail = f" ({result.detail})" if result.detail else ""
            with self._degraded_lock:
                self._degraded.append(f"{agent}: no usable answer, {result.stop_reason}{detail}")
        self._append_transcript(ticket, row.model, prompt_hash, result, parsed)
        return parsed

    def _append_transcript(
        self, ticket: SlotTicket, model: str, prompt_hash: str, result: ModelResult, parsed: BaseModel | None
    ) -> None:
        """Append the call's parsed output to the run transcript (for replay and review)."""
        if self._transcript_path is None:
            return
        line = json.dumps({
            "ticket": ticket.ticket_id,
            "model": model,
            "prompt_hash": prompt_hash,
            "stop_reason": result.stop_reason,
            "output": parsed.model_dump(mode="json") if parsed is not None else None,
            # Recorded for review, never published. Kept even when the call produced no output: a
            # call truncated at max_tokens spent its whole budget here, and this is the only record
            # of what it was doing. The key is named to make its status unmistakable to a reader.
            "unverified_thinking": result.thinking,
        }, sort_keys=True)
        with self._transcript_lock, self._transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
