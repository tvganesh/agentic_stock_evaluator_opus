"""Parse-time validation of claims and refutations, run before adjudication.

A claim is REJECTED here (and never adjudicated) if:

* it cites an evidence ID that is not in the snapshot's evidence index;
* it cites evidence about a different instrument than its subject;
* none of its evidence belongs to its own dimension;
* its falsifier does not parse, uses an unknown column, exceeds DSL limits, or references
  no column from the dimensions that claim type may be checked against;
* its statement contains a decimal figure whose magnitude does not appear (within rounding)
  anywhere in the evidence it cites -- a model may quote numbers, never invent them. Magnitudes
  are compared because prose carries direction in words ("4.7% below its average" quotes an
  evidence value of -4.71); the first Claude run lost 7 correct claims to a signed comparison.

Refutations get the same evidence and falsifier checks, plus: the target must be a claim
that survived phase 4, and evidence must concern the target's subject.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..snapshot.columns import COLUMNS, Dimension
from ..snapshot.store import SealedSnapshot
from . import dsl
from .schema import Claim, Refutation

ALLOWED_FALSIFIER_DIMENSIONS: dict[Dimension, frozenset[Dimension]] = {
    Dimension.FUNDAMENTAL: frozenset({Dimension.FUNDAMENTAL}),
    Dimension.TECHNICAL: frozenset({Dimension.TECHNICAL}),
    # News is text; its claims are checked against news counts and the market's reaction.
    Dimension.NEWS: frozenset({Dimension.NEWS, Dimension.TECHNICAL}),
}

_DECIMAL_FIGURE_RE = re.compile(r"(?<![\w.])-?\d+\.\d+")


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validation: ``ok`` with the parsed falsifier, or a rejection reason."""

    ok: bool
    reason: str
    tree: Any = None


def _numbers_in(value: Any) -> Iterable[float]:
    """Yield every number found in a nested evidence ``fields`` structure (including numeric strings)."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, str):
        for match in re.findall(r"-?\d+(?:\.\d+)?", value.replace(",", "")):
            yield float(match)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numbers_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _numbers_in(item)


def _figure_supported(figure: float, known: list[float]) -> bool:
    """True if ``figure``'s magnitude matches a known number's magnitude within 1% or 0.051.

    Signs are ignored because the statement's wording carries direction; the claim's direction
    field and falsifier are still checked against the snapshot.
    """
    magnitude = abs(figure)
    return any(abs(magnitude - abs(k)) <= max(0.051, abs(k) * 0.01) for k in known)


def unsupported_figures(statement: str, records: list[dict[str, Any]]) -> list[str]:
    """Decimal figures in ``statement`` that match no number in the cited evidence records' fields."""
    known = [n for rec in records for n in _numbers_in(rec["fields"])]
    return [figure for figure in _DECIMAL_FIGURE_RE.findall(statement) if not _figure_supported(float(figure), known)]


def _check_evidence(snapshot: SealedSnapshot, evidence: list[str], subject: str) -> tuple[str | None, list[dict]]:
    """Resolve cited evidence; return an error reason (or None) and the resolved records."""
    records = []
    for ev_id in evidence:
        if not snapshot.has_evidence(ev_id):
            return f"cites unknown evidence {ev_id}", []
        record = snapshot.evidence_record(ev_id)
        if record["instrument_key"] != subject:
            return f"evidence {ev_id} is about {record['instrument_key']}, not {subject}", []
        records.append(record)
    return None, records


def _check_falsifier(text: str, dimension: Dimension) -> ValidationResult:
    """Parse a falsifier and require at least one column from the dimensions allowed for ``dimension``."""
    try:
        tree = dsl.parse(text)
    except dsl.FalsifierError as exc:
        return ValidationResult(False, f"falsifier invalid: {exc}")
    allowed = ALLOWED_FALSIFIER_DIMENSIONS[dimension]
    if not any(COLUMNS[name].dimension in allowed for name in dsl.columns(tree)):
        return ValidationResult(False, f"falsifier references no {'/'.join(sorted(d.value for d in allowed))} column")
    return ValidationResult(True, "ok", tree)


def validate_claim(claim: Claim, snapshot: SealedSnapshot) -> ValidationResult:
    """Run every parse-time check on a claim; the adjudicator only sees claims that pass."""
    error, records = _check_evidence(snapshot, claim.evidence, claim.subject)
    if error:
        return ValidationResult(False, error)
    if not any(rec["dimension"] == claim.dimension.value for rec in records):
        return ValidationResult(False, f"no cited evidence belongs to the {claim.dimension.value} dimension")
    missing = unsupported_figures(claim.statement, records)
    if missing:
        return ValidationResult(False, f"statement figure {missing[0]} does not appear in cited evidence")
    return _check_falsifier(claim.falsifier, claim.dimension)


def validate_refutation(
    refutation: Refutation, snapshot: SealedSnapshot, surviving: dict[str, Claim]
) -> ValidationResult:
    """Check a refutation targets a surviving claim and carries valid evidence and falsifier."""
    if refutation.target_claim_id not in surviving:
        return ValidationResult(False, f"target {refutation.target_claim_id} is not a surviving claim")
    error, _ = _check_evidence(snapshot, refutation.evidence, refutation.subject)
    if error:
        return ValidationResult(False, error)
    return _check_falsifier(refutation.falsifier, refutation.dimension)
