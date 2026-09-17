"""Claim and refutation schemas: the only shapes a model's output can take.

Two layers:

* **Drafts** (:class:`ClaimDraft`, :class:`RefutationDraft`) are what a model returns via
  structured outputs. They deliberately lack ``subject``, ``dimension`` and IDs -- the
  harness assigns those, so a model analysing stock A cannot emit a claim about stock B.
  There is no score field. A refutation draft has no confidence and no way to endorse: the
  auditor can only attack.
* **Records** (:class:`Claim`, :class:`Refutation`, :class:`Adjudication`) are what the
  harness stores in the claim ledger after assigning identity and provenance.

``direction`` is an addition to the architecture's example claim: a claim may support a
position (positive) or undermine it (negative). Ranking sums signed confidences, which is
what lets the dossier say AVOID as well as BUY.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..snapshot.columns import Dimension
from ..snapshot.hashing import hash_object

EvidenceRef = Annotated[str, StringConstraints(pattern=r"^ev:[0-9a-f]{16}$")]
ClaimRef = Annotated[str, StringConstraints(pattern=r"^cl:[0-9a-f]{16}$")]


class Direction(str, Enum):
    """Whether a claim supports (positive) or undermines (negative) owning the stock."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class ClaimDraft(BaseModel):
    """One falsifiable claim as emitted by a claim agent (structured output)."""

    model_config = ConfigDict(extra="forbid")

    predicate: str = Field(
        pattern=r"^[a-z][a-z0-9_]{2,63}$",
        description="snake_case name of the property asserted, e.g. capital_efficiency_improving",
    )
    direction: Direction = Field(description="positive if it supports owning the stock, negative if it undermines it")
    statement: str = Field(min_length=10, max_length=400, description="One or two plain sentences; cite only figures present in the evidence")
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=8, description="Evidence IDs from the snapshot slice")
    confidence: float = Field(ge=0.05, le=0.95, description="Probability the claim is true given the evidence")
    falsifier: str = Field(min_length=3, max_length=300, description="DSL predicate over snapshot columns that, if TRUE, refutes the claim")


class ClaimBatch(BaseModel):
    """A claim agent's complete output for one instrument: at most six claims, possibly none."""

    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimDraft] = Field(max_length=6)


class RefutationDraft(BaseModel):
    """One refutation as emitted by the veto auditor. No confidence, no endorsement field."""

    model_config = ConfigDict(extra="forbid")

    target_claim_id: ClaimRef = Field(description="ID of the surviving claim being attacked")
    statement: str = Field(min_length=10, max_length=400, description="Why the target claim is wrong")
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=8)
    falsifier: str = Field(min_length=3, max_length=300, description="DSL predicate that, if TRUE, shows this refutation is wrong")


class RefutationBatch(BaseModel):
    """The veto auditor's complete output for one batch of surviving claims."""

    model_config = ConfigDict(extra="forbid")

    refutations: list[RefutationDraft] = Field(max_length=24)


class Claim(BaseModel):
    """A claim recorded in the ledger, with harness-assigned identity and provenance."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    subject: str
    dimension: Dimension
    slot_class: str
    predicate: str
    direction: Direction
    statement: str
    evidence: list[str]
    confidence: float
    falsifier: str

    @classmethod
    def from_draft(cls, draft: ClaimDraft, *, subject: str, dimension: Dimension, slot_class: str) -> "Claim":
        """Bind a model draft to its subject and dimension and derive a content-based ``claim_id``."""
        body = draft.model_dump(mode="json")
        claim_id = "cl:" + hash_object({"subject": subject, "dimension": dimension.value, **body})[:16]
        return cls(claim_id=claim_id, subject=subject, dimension=dimension, slot_class=slot_class, **body)


class Refutation(BaseModel):
    """A refutation recorded in the ledger, bound to its target claim's subject and dimension."""

    model_config = ConfigDict(frozen=True)

    refutation_id: str
    target_claim_id: str
    subject: str
    dimension: Dimension
    slot_class: str
    statement: str
    evidence: list[str]
    falsifier: str

    @classmethod
    def from_draft(cls, draft: RefutationDraft, *, target: Claim, slot_class: str) -> "Refutation":
        """Bind an auditor draft to its target claim and derive a content-based ``refutation_id``."""
        body = draft.model_dump(mode="json")
        refutation_id = "rf:" + hash_object(body)[:16]
        return cls(refutation_id=refutation_id, subject=target.subject, dimension=target.dimension,
                   slot_class=slot_class, **body)


class Verdict(str, Enum):
    """Adjudication outcomes. Only SURVIVED contributes to ranking."""

    SURVIVED = "survived"  # falsifier evaluable and did not fire
    REFUTED = "refuted"  # its own falsifier fired
    UNEVALUABLE = "unevaluable"  # missing/stale data, or evaluation error
    REJECTED = "rejected"  # failed parse-time validation (evidence, subject, columns, figures)
    VACUOUS = "vacuous"  # falsifier cannot fire on any value observed in this market
    VETOED = "vetoed"  # a surviving refutation killed it


class Adjudication(BaseModel):
    """The recorded verdict for one claim or refutation, with a human-readable reason."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    kind: str  # "claim" | "refutation"
    verdict: Verdict
    reason: str
    phase: str
