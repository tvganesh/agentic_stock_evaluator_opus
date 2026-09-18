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
    # The justification is NOT checked here. An unsupported figure in the argument withholds the
    # argument at bind time (see :func:`sanitised_justification`); it does not kill the claim.
    return _check_falsifier(claim.falsifier, claim.dimension)


WITHHELD_JUSTIFICATION = (
    "(Justification withheld: it cited a figure that does not appear in the evidence this claim "
    "relies on. The claim itself is unaffected -- it stands or falls on its falsifier.)"
)
"""Replaces an argument that reached for a figure the cited evidence does not contain.

Deliberately does not name the figure. This notice is published in the dossier beside the claim,
and the rule the whole check exists to enforce is that published text traces to evidence -- so
printing the unevidenced number here, even labelled as unsupported, would commit the fault it is
reporting. The figure is not lost: the audit log records which one triggered the withholding, which
is where unverified content belongs."""


def sanitised_justification(
    justification: str, snapshot: SealedSnapshot, evidence: Iterable[str], subject: str
) -> str:
    """The claim's justification, or a withholding notice if it quotes an unsupported figure.

    Withholding rather than rejecting, because the two faults are not the same size. A figure in the
    *statement* is the claim asserting something the evidence does not support, and that is fatal. A
    figure in the *justification* is a fault in the commentary around a claim whose statement,
    falsifier and evidence have all passed; killing it throws away a sound argument over a footnote.

    That distinction is not theoretical. In the run of 18 Sep 2026 the fatal version rejected 22 of
    157 claims and dropped survival from 92% to 77%. Among them were two negative fundamental claims
    -- "P/B of 8.82x versus sector" and "P/B of 19.8x versus the sector average of 8.04x" -- both
    correct, and both the only thing standing between their companies and a BUY rating, since the BUY
    rule bars any surviving negative fundamental claim above 0.6 confidence. Losing them manufactured
    two BUYs that the evidence did not support.

    The unsupported figure never reaches print: the argument is replaced, not merely flagged.
    """
    error, records = _check_evidence(snapshot, evidence, subject)
    if error:
        return justification  # the evidence itself is unusable; validate_claim rejects the claim
    return WITHHELD_JUSTIFICATION if unsupported_figures(justification, records) else justification


def validate_refutation(
    refutation: Refutation, snapshot: SealedSnapshot, surviving: dict[str, Claim]
) -> ValidationResult:
    """Check a refutation targets a surviving claim and carries valid evidence and falsifier."""
    if refutation.target_claim_id not in surviving:
        return ValidationResult(False, f"target {refutation.target_claim_id} is not a surviving claim")
    if _reuses_target_falsifier(refutation, surviving[refutation.target_claim_id]):
        return ValidationResult(False, "refutation reuses its target's falsifier, so it attacks nothing")
    error, _ = _check_evidence(snapshot, refutation.evidence, refutation.subject)
    if error:
        return ValidationResult(False, error)
    return _check_falsifier(refutation.falsifier, refutation.dimension)


def _reuses_target_falsifier(refutation: Refutation, target: Claim) -> bool:
    """True if the refutation's disproof condition is the same test its target already passed.

    A surviving claim survived *because* its falsifier is false on the snapshot. A refutation whose
    falsifier is that same condition is therefore also false, so the attack "stands" without ever
    having asserted anything -- and the harness records a veto that deletes a claim the auditor in
    fact agreed with. That is worse than an auditor which finds nothing, because it destroys correct
    claims, and no later check can catch it: the predicate is well formed, evaluable and reachable.

    Measured on the qwen3:8b audit of 17 Sep 2026: of 14 refutations, the 8 that negated their
    target's falsifier all fired and self-destructed harmlessly, while all 6 that reused it verbatim
    stood as false vetoes. Comparison is on the parsed tree, so spacing and formatting do not matter;
    an unparseable falsifier is left to :func:`_check_falsifier` to reject with its own reason.
    """
    try:
        return dsl.parse(refutation.falsifier) == dsl.parse(target.falsifier)
    except dsl.FalsifierError:
        return False
