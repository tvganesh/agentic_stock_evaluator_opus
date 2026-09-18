"""Spend plan, slot ledger and LLM gateway tests (build-order gates P3 and P5).

The committed total equals the hard stop, approval binds to an exact plan hash, the slot
counter only goes down, forged or reused tickets are refused, oversize prompts are never
dispatched, models only run in model phases, and actual cost can never exceed its bound.
"""

from __future__ import annotations

import dataclasses

import pytest

from sealed_window.claims.schema import ClaimBatch
from sealed_window.governance.audit import AuditLog
from sealed_window.governance.errors import (
    InvalidSlotTicket,
    NoSlotAvailable,
    PhaseViolation,
    PlanNotApproved,
    PromptOverBudget,
    SpendViolation,
)
from sealed_window.governance.llm_gateway import LLMGateway, ModelRequest, ModelResult
from sealed_window.governance.seal import SEAL, PhaseMachine, RunPhase
from sealed_window.governance.spend import (
    SlotClass,
    SlotLedger,
    SlotTicket,
    assert_plan_approved,
    call_bound_microusd,
    compile_plan,
)

SNAP, CFG = "a" * 64, "b" * 64


def _plan(candidates: int = 10, **kwargs):
    """Compile a plan for fake snapshot/config hashes."""
    return compile_plan(snapshot_hash=SNAP, screen_config_hash=CFG, candidate_count=candidates, **kwargs)


class FakeClient:
    """Model client double that records dispatched requests and returns a fixed result."""

    label = "fake"

    def __init__(self, counted: int = 100, result: ModelResult | None = None) -> None:
        """Configure the token count and the result to return."""
        self.counted = counted
        self.result = result or ModelResult(ClaimBatch(claims=[]), "end_turn", 90, 50)
        self.dispatched: list[ModelRequest] = []

    def count_input_tokens(self, request: ModelRequest) -> int:
        """Return the configured count."""
        return self.counted

    def complete(self, request: ModelRequest) -> ModelResult:
        """Record the request and return the configured result."""
        self.dispatched.append(request)
        return self.result


def _gateway(client: FakeClient, candidates: int = 2, in_model_phase: bool = True):
    """Gateway wired to a fresh ledger, audit log and phase machine (sealed process state)."""
    SEAL.seal()
    audit = AuditLog()
    ledger = SlotLedger(_plan(candidates), audit)
    phases = PhaseMachine()
    phases.advance(RunPhase.SCREEN)
    if in_model_phase:
        phases.advance(RunPhase.CLAIM)
    return LLMGateway(ledger=ledger, audit=audit, client=client, phases=phases), ledger, audit


def test_committed_total_is_the_hard_stop():
    """Committed equals the sum of per-call bounds times calls, and equals the hard stop."""
    plan = _plan(10)
    # three analysts per candidate, two prepaid probes per candidate, one veto per three candidates
    assert [r.calls for r in plan.rows] == [10, 10, 10, 20, 4]
    assert plan.committed_total_microusd == sum(r.calls * r.per_call_bound_microusd for r in plan.rows)
    assert plan.hard_stop_microusd == plan.committed_total_microusd
    assert call_bound_microusd("claude-sonnet-5", 11_000, 4_000) == 11_000 * 2 + 4_000 * 10
    assert call_bound_microusd("qwen3:8b", 24_000, 4_000) == 0, "a locally served model costs nothing"


def test_plan_hash_binds_inputs_not_compile_time():
    """Recompiling the same inputs gives the same hash; changing the candidate count changes it."""
    plan = _plan(10)
    assert plan.model_copy(update={"compiled_at": "later"}).plan_hash == plan.plan_hash
    assert _plan(11).plan_hash != plan.plan_hash


def test_ceiling_and_approval():
    """A plan above the operator ceiling is refused; approval must match the exact hash."""
    with pytest.raises(SpendViolation):
        _plan(10, ceiling_microusd=1)
    plan = _plan(10)
    for bad in (None, "0" * 64):
        with pytest.raises(PlanNotApproved):
            assert_plan_approved(plan, bad)
    assert_plan_approved(plan, plan.plan_hash)


def test_slot_counter_only_goes_down_and_tickets_cannot_be_forged():
    """Exhausted classes refuse; forged or reused tickets are refused."""
    ledger = SlotLedger(_plan(1), AuditLog())
    ticket = ledger.acquire(SlotClass.DEEP_FUNDAMENTAL)
    with pytest.raises(NoSlotAvailable):
        ledger.acquire(SlotClass.DEEP_FUNDAMENTAL)
    with pytest.raises(InvalidSlotTicket):
        ledger.redeem(SlotTicket(ticket.ticket_id, SlotClass.DEEP_FUNDAMENTAL, ledger_id=-1))
    ledger.redeem(ticket)
    with pytest.raises(InvalidSlotTicket):
        ledger.redeem(ticket)


def test_gateway_uses_plan_limits_and_settles():
    """The plan, not the caller, fixes model, max_tokens, thinking and effort; spend is settled and audited."""
    client = FakeClient()
    gateway, ledger, audit = _gateway(client)
    result = gateway.call(ledger.acquire(SlotClass.DEEP_FUNDAMENTAL), system="s", user="u",
                          output_type=ClaimBatch, purpose={"agent": "test"})
    assert isinstance(result, ClaimBatch)
    request = client.dispatched[0]
    assert (request.model, request.max_tokens, request.thinking, request.effort) == \
        ("claude-sonnet-5", 4_000, {"type": "adaptive"}, "medium")
    assert ledger.summary()["spent_microusd"] == 90 * 2 + 50 * 10
    assert audit.entries("model.call")[0]["detail"]["slot_class"] == "deep.fundamental"


def test_haiku_slots_get_no_thinking_or_effort():
    """Haiku 4.5 slots are dispatched without adaptive thinking or effort parameters."""
    client = FakeClient()
    gateway, ledger, _ = _gateway(client)
    gateway.call(ledger.acquire(SlotClass.DEEP_NEWS), system="s", user="u", output_type=ClaimBatch, purpose={})
    assert client.dispatched[0].thinking is None and client.dispatched[0].effort is None


def test_oversize_prompt_is_never_dispatched():
    """A prompt above max_in raises PromptOverBudget before the model is called."""
    client = FakeClient(counted=24_001)  # one token past the fundamental slot's allowance
    gateway, ledger, _ = _gateway(client)
    with pytest.raises(PromptOverBudget):
        gateway.call(ledger.acquire(SlotClass.DEEP_FUNDAMENTAL), system="s", user="u",
                     output_type=ClaimBatch, purpose={})
    assert client.dispatched == []


def test_model_calls_are_refused_outside_model_phases():
    """A model call during SCREEN fails closed without dispatch."""
    client = FakeClient()
    gateway, ledger, _ = _gateway(client, in_model_phase=False)
    with pytest.raises(PhaseViolation):
        gateway.call(ledger.acquire(SlotClass.DEEP_FUNDAMENTAL), system="s", user="u",
                     output_type=ClaimBatch, purpose={})
    assert client.dispatched == []


def test_unknown_usage_settles_at_the_bound_and_overruns_are_violations():
    """Estimated usage is charged at worst case; a reported cost above the bound is a violation."""
    client = FakeClient(counted=1_000, result=ModelResult(None, "connection_error", 0, 0, usage_estimated=True))
    gateway, ledger, _ = _gateway(client)
    assert gateway.call(ledger.acquire(SlotClass.DEEP_FUNDAMENTAL), system="s", user="u",
                        output_type=ClaimBatch, purpose={}) is None
    assert ledger.summary()["spent_microusd"] == 1_000 * 2 + 4_000 * 10

    client = FakeClient(result=ModelResult(ClaimBatch(claims=[]), "end_turn", 500_000, 10))
    gateway, ledger, _ = _gateway(client)
    with pytest.raises(SpendViolation):
        gateway.call(ledger.acquire(SlotClass.DEEP_FUNDAMENTAL), system="s", user="u",
                     output_type=ClaimBatch, purpose={})


def test_degraded_calls_are_surfaced_and_clean_ones_are_not():
    """A call that failed must reach the operator; an empty answer from a healthy call must not.

    Regression for the local runs of 17 Sep 2026: a truncated veto prompt produced an empty
    refutation list that read in the dossier as an auditor finding nothing to object to.
    """
    client = FakeClient(result=ModelResult(None, "context_truncated", 2050, 0,
                                           detail="server evaluated 2050 prompt tokens of ~6290 sent"))
    gateway, ledger, _ = _gateway(client)
    assert gateway.call(ledger.acquire(SlotClass.VETO), system="s", user="u",
                        output_type=ClaimBatch, purpose={"agent": "veto"}) is None
    notes = gateway.drain_degraded()
    assert len(notes) == 1 and "veto" in notes[0] and "context_truncated" in notes[0]
    assert "6290" in notes[0], "the note carries the detail that explains the failure"
    assert gateway.drain_degraded() == [], "draining twice must not repeat a note"

    healthy, ledger2, _ = _gateway(FakeClient())  # end_turn with an empty batch is a real finding
    healthy.call(ledger2.acquire(SlotClass.DEEP_NEWS), system="s", user="u",
                 output_type=ClaimBatch, purpose={"agent": "claim.news"})
    assert healthy.drain_degraded() == []


def test_model_request_surface_has_no_tools():
    """Agents hold no tools: the request type has no field through which tools could be passed."""
    fields = {f.name for f in dataclasses.fields(ModelRequest)}
    assert "tools" not in fields and "tool_choice" not in fields and "mcp_servers" not in fields
