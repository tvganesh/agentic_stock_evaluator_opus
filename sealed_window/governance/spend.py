"""Prepaid spend: the plan compiler and the slot ledger (pillar 3 of ARCHITECTURE_OPUS.md).

The whole cost of a run is compiled and committed *before the first model call*, from
counts known after the deterministic screen. It is a plan with no headroom, not a meter.

Compiler
--------
:func:`compile_plan` turns (snapshot hash, screen-config hash, candidate count) into a
:class:`SpendPlan`: for each slot class, a model, a number of calls, ``max_in`` and
``max_out`` token limits, and the committed cost. The committed total *is* the hard stop.
The plan hash (excluding the compile timestamp) is what the operator approves.

Why committed == hard stop is true, not aspirational: each call's input is measured with
``count_tokens`` before dispatch and refused above ``max_in``; its output is capped by
``max_tokens = max_out`` (which also bounds adaptive thinking tokens). So every call's
actual cost is at most ``max_in x input price + max_out x output price``, and there are
exactly ``calls`` of them. Prompt caching is deliberately off in v1 because cache writes
bill above the base input price and would break that bound unless planned for.

Ledger
------
:class:`SlotLedger` issues one-use :class:`SlotTicket` objects. A ticket is consumed when
issued (the counter only goes down; there are no refunds), must be redeemed exactly once
by the LLM gateway, and is settled against actual usage. No code path builds a model
request without a redeemed ticket. When a slot class is exhausted the ledger raises
:class:`NoSlotAvailable` and the orchestrator moves on -- the run finishes with whatever
survived rather than stopping dead or borrowing budget.

Money is tracked in integer micro-dollars (µ$). Because list prices are whole dollars per
million tokens, µ$ per token equals $ per MTok and all arithmetic stays exact.
"""

from __future__ import annotations

import itertools
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict

from ..snapshot.hashing import hash_object
from .audit import AuditLog
from .errors import InvalidSlotTicket, NoSlotAvailable, PlanNotApproved, SpendViolation

PLAN_VERSION = "spend-plan-v1"
PRICING_VERSION = "anthropic-list-2026-06"

PRICING_USD_PER_MTOK: dict[str, tuple[int, int]] = {
    "claude-opus-5": (5, 25),
    "claude-sonnet-5": (2, 10),
    "claude-haiku-4-5": (1, 5),
}
"""(input, output) list price in whole USD per million tokens == µ$ per token."""

ADAPTIVE_THINKING_MODELS = frozenset({"claude-opus-5", "claude-sonnet-5"})
"""Models that accept ``thinking={"type": "adaptive"}`` and ``output_config.effort``."""


class SlotClass(str, Enum):
    """Kinds of model call a plan can contain."""

    DEEP_FUNDAMENTAL = "deep.fundamental"
    DEEP_TECHNICAL = "deep.technical"
    DEEP_NEWS = "deep.news"
    PROBE = "probe"
    VETO = "veto"


FREE_MODEL_PREFIXES: tuple[str, ...] = ("qwen", "llama", "mistral", "gemma", "phi", "deepseek")
"""Models served locally, priced at zero because the electricity is not billed per token.

A plan run against one of these commits $0.00, which is true and keeps the approval step honest
rather than quoting Anthropic prices for calls Anthropic never receives."""


def price_of(model: str) -> tuple[int, int]:
    """(input, output) price in µ$ per token; zero for locally served models, else from the table."""
    if model in PRICING_USD_PER_MTOK:
        return PRICING_USD_PER_MTOK[model]
    if model.lower().startswith(FREE_MODEL_PREFIXES):
        return (0, 0)
    raise SpendViolation(f"no pricing for model {model!r}; refusing to plan a run whose cost is unknown")


@dataclass(frozen=True)
class SlotClassSpec:
    """Static definition of a slot class: model, token limits, effort, and what one call covers."""

    slot_class: SlotClass
    model: str
    max_in: int
    max_out: int
    effort: str | None
    per: str  # "candidate" | "veto_batch"


PROBES_PER_CANDIDATE = 2
"""Tool calls prepaid per candidate, shared by its three analysts.

The pool enforced at runtime (``agents.tools.ProbePool``) is built from this same constant, so the
number of probes paid for and the number allowed can never drift apart."""

DEFAULT_SLOT_SPECS: tuple[SlotClassSpec, ...] = (
    # Input allowances leave room for a probe: a tool call resends the conversation so far, so the
    # turn after a probe carries the opening slice, the request, and the returned table.
    SlotClassSpec(SlotClass.DEEP_FUNDAMENTAL, "claude-sonnet-5", 24_000, 4_000, "medium", "candidate"),
    SlotClassSpec(SlotClass.DEEP_TECHNICAL, "claude-sonnet-5", 20_000, 3_000, "medium", "candidate"),
    SlotClassSpec(SlotClass.DEEP_NEWS, "claude-haiku-4-5", 20_000, 1_500, None, "candidate"),
    # A probe turn re-reads the conversation and one returned table, then answers briefly: the
    # judgement already happened on the analyst's model, so the cheap model carries the round trip.
    SlotClassSpec(SlotClass.PROBE, "claude-haiku-4-5", 12_000, 1_500, None, "probe"),
    # 24,000 output, not 16,000: on an adaptive-thinking model the budget covers the reasoning as well
    # as the answer, and the reasoning dominates. Measured 18 Sep 2026 -- one veto spent 15,733 output
    # tokens to emit 136 tokens of JSON (0.9%), and another consumed all 16,000 thinking and was cut off
    # before writing any JSON at all, losing every refutation for its batch after the call was paid for.
    # Thinking length tracks how hard the judgement is, not how many claims are shown, so smaller
    # batches do not prevent it: a 31-claim batch was lost while a 37-claim one survived.
    SlotClassSpec(SlotClass.VETO, "claude-sonnet-5", 40_000, 24_000, "high", "veto_batch"),
)
"""Model tiering from the architecture: Sonnet for deep analysis and veto, Haiku for news.

Limits calibrated on the first live Claude run (15 Sep 2026): claim calls used 740-1,311 output
tokens against 3,000-4,000; the veto used 3,996 of 8,000 auditing just 2 instruments, so its
limit was doubled for full batches of 4, where truncation would silently lose every refutation."""

VETO_BATCH_SIZE = 3
"""Candidates whose surviving claims are audited together in one veto call.

Sized by ``max_out``, not ``max_in``. The auditor writes a refutation per claim it attacks, so a
batch that fits the prompt comfortably can still run out of room to answer -- and a veto truncated
mid-JSON yields nothing at all, losing every refutation for those candidates after the call has
already been paid for.

Measured on the run of 18 Sep 2026 (15 candidates, Sonnet, 16,000 output tokens per veto slot):
batches of 46 and 38 claims finished at 14,948 and 9,324 output tokens, while batches of 50 and 43
hit the ceiling and were discarded -- two of four batches lost, 93 claims unaudited. At roughly 12
surviving claims per candidate, three candidates keeps a batch near 35 claims and well inside the
limit. Lowering the batch costs one extra veto call per run; raising ``max_out`` instead would
raise the committed total of every run, including the ones that never needed the room."""


MODEL_MODES = ("anthropic", "local", "offline")
"""The model modes a run may use. Defined here, beside :func:`specs_for_mode`, because the CLI
builds its parser from this tuple in either process role and may not import the orchestrator."""

DEFAULT_LOCAL_MODEL = "qwen3:8b"
"""Local model assumed when none is named; any model the server hosts may be passed instead."""


def specs_for_mode(mode: str, local_model: str = DEFAULT_LOCAL_MODEL) -> tuple[SlotClassSpec, ...]:
    """Slot specs for a run: the Claude tiering, or the same shape against one locally served model.

    A local plan names the model it will actually call and commits $0.00, so the operator approves a
    figure that matches what the run does rather than Anthropic prices for calls never sent.
    """
    if mode in ("anthropic", "offline"):
        # The offline stand-in simulates these models, so it is costed as if it were them.
        return DEFAULT_SLOT_SPECS
    if mode == "local":
        return tuple(
            SlotClassSpec(spec.slot_class, local_model, spec.max_in, spec.max_out, None, spec.per)
            for spec in DEFAULT_SLOT_SPECS
        )
    raise SpendViolation(f"unknown model mode {mode!r}; cannot choose slot specs")


def call_bound_microusd(model: str, max_in: int, max_out: int) -> int:
    """Worst-case cost in µ$ of one call with the given limits."""
    input_price, output_price = price_of(model)
    return max_in * input_price + max_out * output_price


class PlanRow(BaseModel):
    """One slot class in a compiled plan."""

    model_config = ConfigDict(frozen=True)

    slot_class: SlotClass
    model: str
    calls: int
    max_in: int
    max_out: int
    effort: str | None
    committed_microusd: int

    @property
    def per_call_bound_microusd(self) -> int:
        """Worst-case cost of a single call in this slot class."""
        return call_bound_microusd(self.model, self.max_in, self.max_out)


class SpendPlan(BaseModel):
    """A compiled, frozen spend plan. Its hash is what the operator approves."""

    model_config = ConfigDict(frozen=True)

    plan_version: str
    pricing_version: str
    snapshot_hash: str
    screen_config_hash: str
    candidate_count: int
    rows: list[PlanRow]
    committed_total_microusd: int
    hard_stop_microusd: int
    compiled_at: str

    @property
    def plan_hash(self) -> str:
        """Hash of everything except the compile timestamp; third element of the reproducibility triple."""
        return hash_object(self.model_dump(mode="json", exclude={"compiled_at"}))

    def row(self, slot_class: SlotClass) -> PlanRow:
        """Return the row for ``slot_class``; raises ``KeyError`` if the plan has none."""
        for row in self.rows:
            if row.slot_class is slot_class:
                return row
        raise KeyError(slot_class)

    def as_table(self) -> str:
        """Render the plan like the architecture document's example, for the CLI and approval prompt."""
        lines = [
            f"plan {self.plan_hash[:12]}...  snapshot {self.snapshot_hash[:12]}...  "
            f"screen cfg {self.screen_config_hash[:12]}...",
            f"compiled {self.compiled_at}   candidates {self.candidate_count}",
            "",
            f"{'slot class':<18}{'model':<19}{'calls':>6}{'max_in':>9}{'max_out':>9}{'committed':>12}",
            "-" * 73,
        ]
        for r in self.rows:
            lines.append(
                f"{r.slot_class.value:<18}{r.model:<19}{r.calls:>6}{r.max_in:>9,}{r.max_out:>9,}"
                f"{format_usd(r.committed_microusd):>12}"
            )
        lines += [
            "-" * 73,
            f"{'committed total':>52}{format_usd(self.committed_total_microusd):>21}",
            f"{'hard stop':>52}{format_usd(self.hard_stop_microusd):>21}",
        ]
        return "\n".join(lines)


def format_usd(microusd: int) -> str:
    """Format µ$ as dollars, rounding *up* to the cent so displayed commitments never understate."""
    return f"${math.ceil(microusd / 10_000) / 100:,.2f}"


def compile_plan(
    *,
    snapshot_hash: str,
    screen_config_hash: str,
    candidate_count: int,
    specs: tuple[SlotClassSpec, ...] = DEFAULT_SLOT_SPECS,
    veto_batch_size: int = VETO_BATCH_SIZE,
    veto_candidate_count: int | None = None,
    ceiling_microusd: int | None = None,
) -> SpendPlan:
    """Compile the spend plan for a screened run.

    ``veto_candidate_count`` is how many candidates the auditor will read, which need not be all of
    them: analysts are cheap and the veto is not, so a run may claim across a wide field and audit
    only the front of it. Defaults to every candidate. The committed total must describe what the
    run can actually spend, so a narrower veto has to lower the number the operator approves rather
    than leaving slots reserved for calls that will never be made.

    ``ceiling_microusd`` is an operator limit checked at compile time: a plan above it is
    refused (narrow the screen), never silently trimmed at runtime.
    """
    if candidate_count < 0:
        raise ValueError("candidate_count must be >= 0")
    vetoed = candidate_count if veto_candidate_count is None else min(veto_candidate_count, candidate_count)
    if vetoed < 0:
        raise ValueError("veto_candidate_count must be >= 0")
    rows = []
    for spec in specs:
        price_of(spec.model)  # refuse to compile a plan whose cost cannot be stated
        if spec.per == "candidate":
            calls = candidate_count
        elif spec.per == "probe":
            calls = candidate_count * PROBES_PER_CANDIDATE
        else:
            calls = math.ceil(vetoed / veto_batch_size)
        rows.append(
            PlanRow(
                slot_class=spec.slot_class,
                model=spec.model,
                calls=calls,
                max_in=spec.max_in,
                max_out=spec.max_out,
                effort=spec.effort,
                committed_microusd=calls * call_bound_microusd(spec.model, spec.max_in, spec.max_out),
            )
        )
    total = sum(r.committed_microusd for r in rows)
    if ceiling_microusd is not None and total > ceiling_microusd:
        raise SpendViolation(
            f"plan commits {format_usd(total)}, above the operator ceiling {format_usd(ceiling_microusd)}; "
            "narrow the screen or lower max_candidates"
        )
    return SpendPlan(
        plan_version=PLAN_VERSION,
        pricing_version=PRICING_VERSION,
        snapshot_hash=snapshot_hash,
        screen_config_hash=screen_config_hash,
        candidate_count=candidate_count,
        rows=rows,
        committed_total_microusd=total,
        hard_stop_microusd=total,
        compiled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def assert_plan_approved(plan: SpendPlan, approved_plan_hash: str | None) -> None:
    """Raise :class:`PlanNotApproved` unless the operator approved exactly this plan's hash."""
    if not approved_plan_hash or approved_plan_hash != plan.plan_hash:
        raise PlanNotApproved(
            f"run requires approval of plan {plan.plan_hash} (committed {format_usd(plan.committed_total_microusd)})"
        )


@dataclass(frozen=True)
class SlotTicket:
    """A one-use permission to make one model call in one slot class. Issued only by a ledger."""

    ticket_id: str
    slot_class: SlotClass
    ledger_id: int


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported for one completed model call."""

    input_tokens: int
    output_tokens: int


class SlotLedger:
    """Runtime counterpart of a plan: issues, redeems and settles slot tickets."""

    _ids = itertools.count(1)

    def __init__(self, plan: SpendPlan, audit: AuditLog) -> None:
        """Initialise remaining calls from the plan; spending starts at zero."""
        self._plan = plan
        self._audit = audit
        self._lock = threading.Lock()
        self._ledger_id = next(self._ids)
        self._remaining = {row.slot_class: row.calls for row in plan.rows}
        self._issued: set[str] = set()
        self._redeemed: set[str] = set()
        self._settled: set[str] = set()
        self._spent_microusd = 0
        self._counter = itertools.count(1)

    @property
    def plan(self) -> SpendPlan:
        """The plan this ledger enforces."""
        return self._plan

    def remaining(self, slot_class: SlotClass) -> int:
        """Calls still available in ``slot_class``."""
        return self._remaining.get(slot_class, 0)

    def acquire(self, slot_class: SlotClass) -> SlotTicket:
        """Take one call from ``slot_class``; raises :class:`NoSlotAvailable` when the pool is empty."""
        with self._lock:
            if self._remaining.get(slot_class, 0) <= 0:
                raise NoSlotAvailable(f"slot class {slot_class.value} is exhausted")
            self._remaining[slot_class] -= 1
            ticket = SlotTicket(f"{slot_class.value}#{next(self._counter)}", slot_class, self._ledger_id)
            self._issued.add(ticket.ticket_id)
        return ticket

    def redeem(self, ticket: SlotTicket) -> PlanRow:
        """Validate and consume a ticket at dispatch time; returns the row with its model and limits."""
        with self._lock:
            if ticket.ledger_id != self._ledger_id or ticket.ticket_id not in self._issued:
                raise InvalidSlotTicket(f"ticket {ticket.ticket_id} was not issued by this ledger")
            if ticket.ticket_id in self._redeemed:
                raise InvalidSlotTicket(f"ticket {ticket.ticket_id} was already redeemed")
            self._redeemed.add(ticket.ticket_id)
        return self._plan.row(ticket.slot_class)

    def settle(self, ticket: SlotTicket, usage: TokenUsage) -> int:
        """Record actual usage for a redeemed ticket and return its cost in µ$.

        A cost above the per-call bound is impossible if the gateway enforced its limits, so
        it is treated as a governance violation rather than a warning.
        """
        row = self._plan.row(ticket.slot_class)
        input_price, output_price = price_of(row.model)
        cost = usage.input_tokens * input_price + usage.output_tokens * output_price
        with self._lock:
            if ticket.ticket_id not in self._redeemed or ticket.ticket_id in self._settled:
                raise InvalidSlotTicket(f"ticket {ticket.ticket_id} cannot be settled")
            self._settled.add(ticket.ticket_id)
            self._spent_microusd += cost
        if cost > row.per_call_bound_microusd:
            raise SpendViolation(f"call {ticket.ticket_id} cost {cost} µ$ above its bound {row.per_call_bound_microusd}")
        return cost

    def summary(self) -> dict[str, object]:
        """JSON-safe spend status for the UI and dossier: committed, spent, remaining per class."""
        with self._lock:
            return {
                "plan_hash": self._plan.plan_hash,
                "committed_microusd": self._plan.committed_total_microusd,
                "spent_microusd": self._spent_microusd,
                "committed": format_usd(self._plan.committed_total_microusd),
                "spent": format_usd(self._spent_microusd),
                "slots": {
                    row.slot_class.value: {"planned": row.calls, "remaining": self._remaining[row.slot_class]}
                    for row in self._plan.rows
                },
            }
