"""The adjudicator: deterministic verdicts on claims and refutations, and the ranking formula.

Claim verdicts, checked in this order:

1. **REJECTED** -- failed parse-time validation (unknown or foreign evidence, invented figures,
   invalid falsifier, wrong column dimensions). See ``claims.validator``.
2. **UNEVALUABLE** -- a referenced column is missing, or its dimension's data is outside the
   freshness SLA, or evaluation hit e.g. division by zero. Unevaluable is not true: discarded.
3. **REFUTED** -- the falsifier fires on the snapshot. Discarded, not down-weighted.
4. **VACUOUS** -- the falsifier cannot fire for any plausible combination of values observed
   across this snapshot's universe (5th-95th percentiles). An unkillable claim is not a claim;
   this is the check that stops an injected "falsifier: close < 0" from surviving.
5. **SURVIVED** -- evaluable, did not fire, and could have.

Refutations (from the veto auditor) go through the same checks. A SURVIVED refutation marks
its target claim **VETOED**. A refutation that is itself refuted, unevaluable or vacuous is
ignored -- the auditor gets no special trust.

Ranking (versioned constants, no model involved)::

    score(stock) = sum over dimensions d of  w[d] * sum over surviving claims c of  sign(c) * confidence(c)

where ``sign`` is +1 for positive and -1 for negative claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ..claims import dsl
from ..claims.schema import Adjudication, Claim, Direction, Refutation, Verdict
from ..claims.validator import validate_claim, validate_refutation
from ..governance.audit import AuditLog
from ..snapshot.columns import COLUMNS, FRESHNESS_SLA, Dimension
from ..snapshot.hashing import stable_float
from ..snapshot.store import SealedSnapshot

RANKING_VERSION = "ranking-v1"

DIMENSION_WEIGHTS: Mapping[Dimension, float] = MappingProxyType(
    {Dimension.FUNDAMENTAL: 1.0, Dimension.TECHNICAL: 0.8, Dimension.NEWS: 0.5}
)
"""Versioned weights: news counts least because it is the most injectable dimension."""

SCENARIO_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
MIN_UNIVERSE_FOR_QUANTILES = 5


def universe_scenarios(snapshot: SealedSnapshot) -> dict[str, list[float]]:
    """Per-column percentile values across the snapshot universe, used by the vacuity check."""
    table = snapshot.derived_table()
    scenarios: dict[str, list[float]] = {}
    for name in COLUMNS:
        values = sorted(v for row in table.values() if (v := row.get(name)) is not None)
        if len(values) >= MIN_UNIVERSE_FOR_QUANTILES:
            scenarios[name] = sorted({values[round(q * (len(values) - 1))] for q in SCENARIO_QUANTILES})
    return scenarios


def _fallback_scenarios(value: float) -> list[float]:
    """Scenario values for a column when the universe is too small for percentiles (+/-25%, min 1 unit)."""
    delta = max(abs(value) * 0.25, 1.0)
    return [value - delta, value + delta, value * 0.5, value * 1.5]


def _values_text(tree, row: Mapping[str, float | None]) -> str:
    """Render the snapshot values of the falsifier's columns, for human-readable reasons."""
    return ", ".join(f"{name}={row.get(name)}" for name in sorted(dsl.columns(tree)))


def _stale_reason(tree, row: Mapping[str, float | None]) -> str | None:
    """Return why the falsifier's data is outside its dimension SLA, or ``None`` if fresh."""
    for dimension in sorted({COLUMNS[name].dimension for name in dsl.columns(tree)}, key=lambda d: d.value):
        sla = FRESHNESS_SLA[dimension]
        if sla is None:
            continue
        age_column, max_days = sla
        age = row.get(age_column)
        if age is None or age > max_days:
            return f"{dimension.value} data outside freshness SLA ({age_column}={age}, max {max_days})"
    return None


class Adjudicator:
    """Evaluates claims and refutations against one sealed snapshot and records every verdict."""

    def __init__(self, snapshot: SealedSnapshot, audit: AuditLog) -> None:
        """Precompute universe percentiles once; they are a pure function of the snapshot."""
        self._snapshot = snapshot
        self._audit = audit
        self._scenarios = universe_scenarios(snapshot)

    def _scenarios_for(self, tree, row: Mapping[str, float | None]) -> dict[str, list[float]]:
        """Scenario values for the falsifier's columns, falling back to local perturbations."""
        out = {}
        for name in dsl.columns(tree):
            if name in self._scenarios:
                out[name] = self._scenarios[name]
            elif row.get(name) is not None:
                out[name] = _fallback_scenarios(float(row[name]))
        return out

    def _check(self, tree, subject: str) -> tuple[Verdict, str]:
        """Run freshness, evaluation and vacuity checks for a parsed falsifier on one subject."""
        row = self._snapshot.derived_row(subject)
        stale = _stale_reason(tree, row)
        if stale:
            return Verdict.UNEVALUABLE, stale
        try:
            fired = dsl.evaluate(tree, row)
        except dsl.Unevaluable as exc:
            return Verdict.UNEVALUABLE, str(exc)
        if fired:
            return Verdict.REFUTED, f"falsifier fired on snapshot values ({_values_text(tree, row)})"
        if not dsl.is_reachable(tree, row, self._scenarios_for(tree, row)):
            return Verdict.VACUOUS, "falsifier cannot fire on any value observed in this snapshot's universe"
        return Verdict.SURVIVED, f"falsifier did not fire ({_values_text(tree, row)})"

    def _record(self, item: Adjudication, subject: str) -> Adjudication:
        """Write one verdict to the audit log and return it."""
        self._audit.record("adjudication", {"item_id": item.item_id, "kind": item.kind, "subject": subject,
                                            "verdict": item.verdict.value, "reason": item.reason,
                                            "phase": item.phase})
        return item

    def adjudicate_claims(self, claims: list[Claim], *, phase: str) -> dict[str, Adjudication]:
        """Return a verdict for every claim, keyed by ``claim_id``."""
        verdicts: dict[str, Adjudication] = {}
        for claim in claims:
            validation = validate_claim(claim, self._snapshot)
            if not validation.ok:
                verdict, reason = Verdict.REJECTED, validation.reason
            else:
                verdict, reason = self._check(validation.tree, claim.subject)
            verdicts[claim.claim_id] = self._record(
                Adjudication(item_id=claim.claim_id, kind="claim", verdict=verdict, reason=reason, phase=phase),
                claim.subject,
            )
        self._audit.record("adjudication.summary", {"phase": phase, **_counts(verdicts)})
        return verdicts

    def adjudicate_refutations(
        self, refutations: list[Refutation], surviving: dict[str, Claim], *, phase: str
    ) -> tuple[dict[str, Adjudication], dict[str, Adjudication]]:
        """Adjudicate refutations; return (refutation verdicts, VETOED verdicts for their targets)."""
        ref_verdicts: dict[str, Adjudication] = {}
        vetoed: dict[str, Adjudication] = {}
        for refutation in refutations:
            validation = validate_refutation(refutation, self._snapshot, surviving)
            if not validation.ok:
                verdict, reason = Verdict.REJECTED, validation.reason
            else:
                verdict, reason = self._check(validation.tree, refutation.subject)
            ref_verdicts[refutation.refutation_id] = self._record(
                Adjudication(item_id=refutation.refutation_id, kind="refutation", verdict=verdict,
                             reason=reason, phase=phase),
                refutation.subject,
            )
            if verdict is Verdict.SURVIVED and refutation.target_claim_id not in vetoed:
                vetoed[refutation.target_claim_id] = self._record(
                    Adjudication(item_id=refutation.target_claim_id, kind="claim", verdict=Verdict.VETOED,
                                 reason=f"vetoed by {refutation.refutation_id}: {refutation.statement}", phase=phase),
                    refutation.subject,
                )
        self._audit.record("adjudication.summary", {"phase": phase, **_counts(ref_verdicts),
                                                    "claims_vetoed": len(vetoed)})
        return ref_verdicts, vetoed


def _counts(verdicts: Mapping[str, Adjudication]) -> dict[str, int]:
    """Count verdicts by type for audit summaries."""
    counts = {verdict.value: 0 for verdict in Verdict}
    for item in verdicts.values():
        counts[item.verdict.value] += 1
    return counts


@dataclass
class StockScore:
    """Ranking arithmetic for one candidate: per-dimension tallies and the weighted score."""

    instrument_key: str
    score: float
    dimensions: dict[str, dict[str, float]] = field(default_factory=dict)
    max_negative_fundamental_confidence: float = 0.0

    @property
    def surviving_claims(self) -> int:
        """Total surviving claims across dimensions."""
        return int(sum(d["positive"] + d["negative"] for d in self.dimensions.values()))

    @property
    def positive_dimensions(self) -> list[str]:
        """Dimensions with at least one surviving positive claim."""
        return [name for name, d in self.dimensions.items() if d["positive"] > 0]


def score_candidates(
    candidates: list[str], claims: list[Claim], verdicts: Mapping[str, Adjudication]
) -> list[StockScore]:
    """Apply the ranking formula to every candidate and return scores sorted best first."""
    scores = []
    for key in candidates:
        tallies = {d.value: {"positive": 0, "negative": 0, "signed_confidence": 0.0} for d in Dimension}
        max_negative_fundamental = 0.0
        for claim in claims:
            verdict = verdicts.get(claim.claim_id)
            if claim.subject != key or verdict is None or verdict.verdict is not Verdict.SURVIVED:
                continue
            tally = tallies[claim.dimension.value]
            if claim.direction is Direction.POSITIVE:
                tally["positive"] += 1
                tally["signed_confidence"] += claim.confidence
            else:
                tally["negative"] += 1
                tally["signed_confidence"] -= claim.confidence
                if claim.dimension is Dimension.FUNDAMENTAL:
                    max_negative_fundamental = max(max_negative_fundamental, claim.confidence)
        score = sum(DIMENSION_WEIGHTS[d] * tallies[d.value]["signed_confidence"] for d in Dimension)
        for tally in tallies.values():
            tally["signed_confidence"] = stable_float(tally["signed_confidence"])
        scores.append(StockScore(key, stable_float(score), tallies, max_negative_fundamental))
    return sorted(scores, key=lambda s: (-s.score, s.instrument_key))
