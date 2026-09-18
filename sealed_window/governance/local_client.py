"""OpenAI-compatible model client for a model served on this machine (Ollama and similar).

Why this exists
---------------
Running the claim and veto agents against a locally served model makes the agentic layer free
to exercise. The question the operator deferred -- *do model-written falsifiable claims beat the
deterministic screen?* -- needs many runs to answer, and every Anthropic run costs real money.
A local model turns that measurement into electricity and patience.

It is the same governed surface as :class:`~.llm_gateway.AnthropicModelClient`. The gateway
still redeems a slot ticket, still counts input before dispatch, still confines the call to a
model window, and still settles and audits the result. Only the transport and the price differ.

Transport
---------
The standard library's ``urllib``, not ``httpx``: policy forbids a general HTTP client in the
SEALED process role (``SEALED_ROLE_FORBIDDEN_MODULES``), and that rule exists so the sealed
process cannot be talked into fetching something. Requests go to
``policy.LOCAL_MODEL_BASE_URL`` on the loopback interface, and :attr:`permitted_hosts` tells the
gateway to open the loopback set instead of the Anthropic endpoint -- so during a local model
call nothing off this machine is reachable, and an injected instruction has nowhere to send to.
The base URL is re-checked against the policy's loopback set on construction: a non-loopback
host fails closed rather than quietly widening the seal.

Schema enforcement
------------------
The server's OpenAI-compatible route honours ``response_format={"type": "json_schema", ...}``
(verified against Ollama 0.34.0), so the model is *constrained* to the Pydantic schema by the
sampler rather than asked for JSON and hoped over. Output that still fails validation comes back
as ``parsed=None`` with a ``schema_error`` stop reason -- the same shape the Anthropic client
returns for a refusal -- and the run loses that slot's claims and carries on.

Token counting is an estimate
-----------------------------
The server exposes no count-tokens endpoint, and a true count would require a full prefill
(minutes for a 24,000-token analyst prompt). :meth:`count_input_tokens` therefore estimates from
prompt length. This is honest because a local model costs nothing: ``max_in`` here is a
prompt-assembly sanity check, not a cost control, and the plan it belongs to commits $0.00.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pydantic import ValidationError

from . import policy
from .errors import SealViolation
from .llm_gateway import ModelRequest, ModelResult

CHARS_PER_TOKEN = 3.5
"""Rough characters-per-token for the estimate; matches the offline stand-in's assumption."""

DEFAULT_TIMEOUT_S = 3600.0
"""One hour. A veto slot may generate 16,000 tokens, and CPU-only inference on an Intel Mac
runs at single-digit tokens per second, so a correct call can legitimately take over an hour."""

MIN_EVALUATED_PROMPT_RATIO = 0.7
"""Fail closed if the server evaluated materially fewer prompt tokens than we sent.

A locally served model silently discards whatever exceeds its context window -- Ollama drops
tokens from the *front*, so the system prompt and the beginning of the evidence go first -- and
then answers confidently about what is left. Measured on qwen3:8b with the default context: a
6,290-token veto prompt was evaluated as 2,050 tokens, the auditor never saw a single claim, and
it returned an empty refutation list that looked like a considered finding.

That is the failure this system refuses everywhere else. :class:`PromptOverBudget` exists so an
oversize prompt is rejected rather than shortened, on the reasoning that a silently trimmed prompt
produces an answer about incomplete evidence. The same rule has to hold when the *server* does the
trimming, so a short evaluation count ends the call with no output instead of a plausible one.

The ratio is generous because :meth:`count_input_tokens` only estimates: English prose runs near
4 characters per token against the 3.5 assumed here, so the estimate reads about 15% high, and
JSON full of digits tokenizes denser still (which errs safe, raising the server's count). A real
truncation is nothing like that marginal -- the measured case evaluated 33% of the prompt."""

REASONING_EFFORT = "none"
"""Suppress hybrid-reasoning preamble on models that have one (qwen3 and kin).

Not a style preference -- a correctness and cost control. Left on, qwen3 emits a ``<think>``
block *before* the JSON, those tokens count against ``max_tokens``, and a long enough preamble
truncates the answer: the slot returns ``max_tokens`` with no parsed output and the run loses
those claims. Measured on qwen3:8b, the same trivial answer took 56.4s/258 tokens with reasoning
and 3.2s/16 tokens without -- a 17x difference that would turn a short run into hours.

Why this spelling: of the four documented ways to disable it, only ``reasoning_effort`` works on
the OpenAI-compatible route. ``chat_template_kwargs={"enable_thinking": False}``, an in-prompt
``/no_think`` directive and a top-level ``think: false`` were all measured against Ollama 0.34.0
and left reasoning running. Ollama's *native* ``/api/chat`` does honour ``think: false``, but
moving there would tie this client to one vendor; this keeps it portable to any OpenAI-compatible
server. Models without a reasoning mode ignore the field -- verified against llama3.2."""


class LocalOpenAIModelClient:
    """:class:`~.llm_gateway.ModelClient` backed by an OpenAI-compatible server on loopback."""

    label = "local"
    permitted_hosts = policy.LOCAL_MODEL_HOSTS

    def __init__(self, base_url: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        """Pin the client to a loopback base URL, refusing any host the policy does not allow."""
        self._base_url = (base_url or policy.LOCAL_MODEL_BASE_URL).rstrip("/")
        host = urllib.parse.urlsplit(self._base_url).hostname or ""
        if host not in policy.LOCAL_MODEL_HOSTS:
            raise SealViolation(
                f"local model base URL {self._base_url!r} resolves to host {host!r}, which is not in "
                f"the permitted loopback set {sorted(policy.LOCAL_MODEL_HOSTS)}"
            )
        self._timeout_s = timeout_s

    def count_input_tokens(self, request: ModelRequest) -> int:
        """Estimate input tokens from prompt length; the local server offers no exact counter."""
        return math.ceil(len(request.system + request.user) / CHARS_PER_TOKEN)

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        """Build the chat-completions body, constraining output to the request's Pydantic schema."""
        return {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "max_tokens": request.max_tokens,
            "reasoning_effort": REASONING_EFFORT,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_type.__name__,
                    "schema": request.output_type.model_json_schema(),
                },
            },
        }

    def complete(self, request: ModelRequest) -> ModelResult:
        """Call the local server and return parsed output plus usage, mapping every failure to a result.

        Failures never raise into the gateway: a transport error, a non-200 status, malformed JSON
        or a schema mismatch all yield ``parsed=None`` with a stop reason, so one bad slot costs the
        run its claims rather than the whole run.
        """
        body = json.dumps(self._payload(request)).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self._timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # the server answered, but with an error status
            return ModelResult(None, "api_error", 0, 0, detail=f"HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:  # never reached the model
            return ModelResult(None, "connection_error", 0, 0, detail=type(exc).__name__)

        try:
            payload = json.loads(raw)
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            return ModelResult(None, "malformed_response", 0, 0, detail=f"{type(exc).__name__}: {exc}"[:200])

        usage = payload.get("usage") or {}
        estimated = self.count_input_tokens(request)
        input_tokens = int(usage.get("prompt_tokens") or estimated)
        output_tokens = int(usage.get("completion_tokens") or 0)

        # The server reports what it actually evaluated, so a count far below what we sent means the
        # prompt was cut to fit the context window. Refuse the answer: it is about evidence the model
        # never read. Raising the server's context (num_ctx / OLLAMA_CONTEXT_LENGTH) is the operator fix.
        if usage.get("prompt_tokens") and input_tokens < estimated * MIN_EVALUATED_PROMPT_RATIO:
            return ModelResult(
                None, "context_truncated", input_tokens, output_tokens,
                detail=(f"server evaluated {input_tokens} prompt tokens of ~{estimated} sent; "
                        "raise the model server's context window"),
            )
        finish = str(choice.get("finish_reason") or "stop")
        stop_reason = {"stop": "end_turn", "length": "max_tokens"}.get(finish, finish)

        # Read before the truncation check, not after: a call cut off at max_tokens spent its whole
        # budget reasoning, so that is exactly the case where this is the only record of what it did.
        # Suppressed by REASONING_EFFORT today, so normally empty; captured for symmetry with the
        # Anthropic client so the transcript means the same thing whichever model produced it.
        message = choice.get("message") or {}
        thinking = message.get("reasoning_content") or message.get("reasoning") or None

        if stop_reason == "max_tokens":  # truncated JSON is not worth parsing
            return ModelResult(None, stop_reason, input_tokens, output_tokens,
                               detail=f"output truncated at max_tokens={request.max_tokens}",
                               thinking=thinking)

        try:
            parsed = request.output_type.model_validate_json(content)
        except ValidationError as exc:
            return ModelResult(None, "schema_error", input_tokens, output_tokens, detail=str(exc)[:200],
                               thinking=thinking)
        return ModelResult(parsed, stop_reason, input_tokens, output_tokens, thinking=thinking)
