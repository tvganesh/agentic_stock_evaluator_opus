"""Phase 3 claim agents: fundamental, technical and news.

Each agent run is exactly one governed model call for one (instrument, dimension) pair:

1. take a slot ticket for the dimension's slot class from the ledger (no ticket, no call);
2. build the frozen system prompt and the enveloped snapshot slice;
3. call the LLM gateway, which returns a validated :class:`ClaimBatch` or ``None``;
4. bind each draft to its subject and dimension, producing :class:`Claim` records -- and, when a
   draft copies a signal-table falsifier, to that rule's direction (see :func:`direction_from_table`).

The agent cannot choose its model, limits or subject, cannot see other instruments, and
cannot trigger further calls. Claims it returns are unverified hypotheses until phase 4.
"""

from __future__ import annotations

from ..claims.schema import Claim, ClaimBatch, ClaimDraft, Direction
from ..claims.signals import rule_for_falsifier
from ..claims.validator import sanitised_justification
from ..governance.audit import AuditLog
from ..governance.llm_gateway import LLMGateway
from ..governance.spend import SlotClass, SlotLedger
from ..snapshot.columns import Dimension
from ..snapshot.store import SealedSnapshot
from .prompts import claim_system_prompt, claim_user_prompt

CLAIM_SLOT_CLASS: dict[Dimension, SlotClass] = {
    Dimension.FUNDAMENTAL: SlotClass.DEEP_FUNDAMENTAL,
    Dimension.TECHNICAL: SlotClass.DEEP_TECHNICAL,
    Dimension.NEWS: SlotClass.DEEP_NEWS,
}
"""Which slot class in the spend plan pays for each dimension's claim agent."""

CLAIM_DIMENSIONS: tuple[Dimension, ...] = (Dimension.FUNDAMENTAL, Dimension.TECHNICAL, Dimension.NEWS)


def direction_from_table(draft: ClaimDraft, dimension: Dimension) -> Direction | None:
    """The direction of the table rule whose falsifier ``draft`` copies, if it differs from the draft's.

    A copied falsifier names the rule the model chose -- "expensive on book value" -- and that rule has
    one direction by definition. The label is not a judgement left over for the model to make; it is
    bookkeeping, like the subject and claim ID the harness already assigns. It needs assigning because
    ``ClaimDraft`` puts ``direction`` before ``statement`` and ``falsifier``, so a model writing token by
    token commits to it before it has compared a figure. On 26 Sep 2026 (run 20260926T131512Z-bf9cb1)
    qwen3:14b wrote five claims reading "expensive on book value", each with the matching falsifier,
    each labelled positive -- all survived and all added to the score instead of subtracting.

    Returns ``None`` when the falsifier is the model's own composition or the labels already agree.
    """
    match = rule_for_falsifier(draft.falsifier, dimension)
    if match is None or Direction(match[1].direction) is draft.direction:
        return None
    return Direction(match[1].direction)


def run_claim_agent(
    *,
    gateway: LLMGateway,
    ledger: SlotLedger,
    snapshot: SealedSnapshot,
    instrument_key: str,
    dimension: Dimension,
    audit: AuditLog | None = None,
) -> list[Claim]:
    """Run one claim agent for one instrument and return its claims (possibly empty).

    Raises ``NoSlotAvailable`` if the plan has no call left for this dimension and
    ``PromptOverBudget`` if the assembled prompt exceeds the slot's ``max_in``.
    """
    slot_class = CLAIM_SLOT_CLASS[dimension]
    ticket = ledger.acquire(slot_class)
    batch = gateway.call(
        ticket,
        system=claim_system_prompt(dimension),
        user=claim_user_prompt(snapshot, instrument_key, dimension),
        output_type=ClaimBatch,
        purpose={"agent": f"claim.{dimension.value}", "instrument_key": instrument_key},
    )
    if batch is None:
        return []
    claims: dict[str, Claim] = {}
    for draft in batch.claims:
        # An unsupported figure in the argument withholds the argument; it does not kill the claim,
        # whose substance is carried by its statement, falsifier and evidence. Done here, at bind
        # time, so everything downstream sees a justification that is safe to publish.
        text = sanitised_justification(draft.justification, snapshot, draft.evidence, instrument_key)
        update: dict = {"justification": text}
        source = "model"
        if (table_direction := direction_from_table(draft, dimension)) is not None:
            update["direction"], source = table_direction, "signal_table"
        claim = Claim.from_draft(draft.model_copy(update=update), subject=instrument_key, dimension=dimension,
                                 slot_class=slot_class.value, direction_source=source)
        if source == "signal_table" and audit is not None:
            audit.record("claim.direction_from_table", {
                "claim_id": claim.claim_id, "instrument_key": instrument_key, "dimension": dimension.value,
                "model_direction": draft.direction.value, "table_direction": claim.direction.value,
                "falsifier": draft.falsifier})
        claims.setdefault(claim.claim_id, claim)
    return list(claims.values())
