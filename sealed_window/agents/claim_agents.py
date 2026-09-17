"""Phase 3 claim agents: fundamental, technical and news.

Each agent run is exactly one governed model call for one (instrument, dimension) pair:

1. take a slot ticket for the dimension's slot class from the ledger (no ticket, no call);
2. build the frozen system prompt and the enveloped snapshot slice;
3. call the LLM gateway, which returns a validated :class:`ClaimBatch` or ``None``;
4. bind each draft to its subject and dimension, producing :class:`Claim` records.

The agent cannot choose its model, limits or subject, cannot see other instruments, and
cannot trigger further calls. Claims it returns are unverified hypotheses until phase 4.
"""

from __future__ import annotations

from ..claims.schema import Claim, ClaimBatch
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


def run_claim_agent(
    *,
    gateway: LLMGateway,
    ledger: SlotLedger,
    snapshot: SealedSnapshot,
    instrument_key: str,
    dimension: Dimension,
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
        claim = Claim.from_draft(draft, subject=instrument_key, dimension=dimension, slot_class=slot_class.value)
        claims.setdefault(claim.claim_id, claim)
    return list(claims.values())
