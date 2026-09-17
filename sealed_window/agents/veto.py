"""Phase 5 veto auditor: a model whose only power is to refute.

Deliberately asymmetric (ARCHITECTURE_OPUS.md, clause 06):

* it receives surviving claims and the snapshot evidence for their companies, *not* the
  original agents' reasoning, so it cannot be anchored by it;
* its output schema (:class:`RefutationBatch`) has no field to endorse, approve or raise a
  confidence -- there is nothing to return except attacks;
* each refutation carries its own falsifier and is adjudicated like any claim, so an
  unfalsifiable attack is ignored on the same rule as everybody else.

Fail-safe property: the veto can only *remove* surviving claims. A compromised or overzealous
auditor therefore produces abstention (fewer recommendations), never an extra pick.

Claims are audited in batches of ``VETO_BATCH_SIZE`` instruments per slot so each prompt fits
the plan's ``max_in``. Refutations naming a claim outside their batch are dropped and counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..claims.schema import Claim, Refutation, RefutationBatch
from ..governance.llm_gateway import LLMGateway
from ..governance.spend import VETO_BATCH_SIZE, SlotClass, SlotLedger
from ..snapshot.store import SealedSnapshot
from .prompts import VETO_SYSTEM_PROMPT, veto_user_prompt


@dataclass
class VetoOutcome:
    """Refutations produced by the auditor plus bookkeeping for the dossier."""

    refutations: list[Refutation] = field(default_factory=list)
    batches_audited: int = 0
    dropped_out_of_batch: int = 0


def batch_subjects(subjects: list[str], batch_size: int = VETO_BATCH_SIZE) -> list[list[str]]:
    """Split instruments into consecutive batches of ``batch_size`` (order preserved)."""
    return [subjects[i : i + batch_size] for i in range(0, len(subjects), batch_size)]


def run_veto_batch(
    *,
    gateway: LLMGateway,
    ledger: SlotLedger,
    snapshot: SealedSnapshot,
    claims: list[Claim],
) -> VetoOutcome:
    """Audit one batch of surviving claims with one veto slot; returns bound refutations."""
    outcome = VetoOutcome()
    if not claims:
        return outcome
    ticket = ledger.acquire(SlotClass.VETO)
    batch = gateway.call(
        ticket,
        system=VETO_SYSTEM_PROMPT,
        user=veto_user_prompt(snapshot, claims),
        output_type=RefutationBatch,
        purpose={"agent": "veto", "instruments": sorted({c.subject for c in claims}), "claims": len(claims)},
    )
    outcome.batches_audited = 1
    if batch is None:
        return outcome
    by_id = {claim.claim_id: claim for claim in claims}
    seen: set[str] = set()
    for draft in batch.refutations:
        target = by_id.get(draft.target_claim_id)
        if target is None:
            outcome.dropped_out_of_batch += 1
            continue
        refutation = Refutation.from_draft(draft, target=target, slot_class=SlotClass.VETO.value)
        if refutation.refutation_id not in seen:
            seen.add(refutation.refutation_id)
            outcome.refutations.append(refutation)
    return outcome
