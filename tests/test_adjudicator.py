"""Adversarial claim corpus (build-order gate P4), veto adjudication and ranking arithmetic.

Every hand-written false, unfalsifiable or ungrounded claim must be killed with the right
verdict, a true claim must survive, and the auditor's refutations must be held to the same
standard as claims.
"""

from __future__ import annotations

import pytest

from sealed_window.adjudicate.adjudicator import Adjudicator, _stale_reason, score_candidates
from sealed_window.claims import dsl
from sealed_window.claims.schema import (
    Adjudication,
    Claim,
    ClaimDraft,
    Refutation,
    RefutationDraft,
    Verdict,
)
from sealed_window.governance.audit import AuditLog
from sealed_window.snapshot.columns import Dimension

FUND, TECH, NEWS = Dimension.FUNDAMENTAL, Dimension.TECHNICAL, Dimension.NEWS


def _ev(snapshot, key: str, kind: str) -> str:
    """Evidence ID of the first record of ``kind`` for an instrument."""
    return next(r["evidence_id"] for r in snapshot.evidence_for(key) if r["kind"] == kind)


def _claim(key, dimension, falsifier, *, evidence, statement="Capital efficiency compares well with peers.",
           direction="positive", confidence=0.7) -> Claim:
    """Build a harness-bound claim from draft fields."""
    draft = ClaimDraft(predicate="test_claim", direction=direction, statement=statement, evidence=evidence,
                       confidence=confidence, falsifier=falsifier)
    return Claim.from_draft(draft, subject=key, dimension=dimension, slot_class="deep.test")


@pytest.fixture
def pair(snapshot) -> tuple[str, str]:
    """(strong, weak): the lowest-ROE instrument above its sector ROE, and one below it."""
    table = snapshot.derived_table()
    above = sorted((row["roe_pct"], key) for key, row in table.items() if row["roe_pct"] > row["sector_roe_pct"])
    below = sorted((row["roe_pct"], key) for key, row in table.items() if row["roe_pct"] < row["sector_roe_pct"])
    assert above and below, "fixture market should contain both strong and weak ROE names"
    return above[0][1], below[0][1]


def test_adversarial_corpus_every_false_claim_is_killed(snapshot, pair):
    """P4 gate: a true claim survives; every adversarial claim gets the expected fatal verdict."""
    strong, weak = pair
    ev_strong, ev_weak = _ev(snapshot, strong, "fundamental_derived"), _ev(snapshot, weak, "fundamental_derived")
    ev_weak_tech = _ev(snapshot, weak, "technical_derived")
    corpus = {
        "true claim": (_claim(strong, FUND, "roe_pct < sector_roe_pct", evidence=[ev_strong]), Verdict.SURVIVED),
        "falsifier fires": (_claim(weak, FUND, "roe_pct < sector_roe_pct", evidence=[ev_weak]), Verdict.REFUTED),
        "unkillable falsifier": (_claim(weak, FUND, "roe_pct < -1000", evidence=[ev_weak]), Verdict.VACUOUS),
        "unknown evidence": (_claim(weak, FUND, "roe_pct < 1", evidence=["ev:0000000000000000"]), Verdict.REJECTED),
        "another stock's evidence": (_claim(weak, FUND, "roe_pct < 1", evidence=[ev_strong]), Verdict.REJECTED),
        "invented figure": (_claim(weak, FUND, "roe_pct < 1", evidence=[ev_weak],
                                   statement="ROE is a remarkable 9999.99% this year."), Verdict.REJECTED),
        "wrong-dimension falsifier": (_claim(weak, FUND, "rsi_14 > 99", evidence=[ev_weak]), Verdict.REJECTED),
        "no own-dimension evidence": (_claim(weak, FUND, "roe_pct < 1", evidence=[ev_weak_tech]), Verdict.REJECTED),
        "unparseable falsifier": (_claim(weak, FUND, "roe_pct <<< 3", evidence=[ev_weak]), Verdict.REJECTED),
    }
    verdicts = Adjudicator(snapshot, AuditLog()).adjudicate_claims([c for c, _ in corpus.values()], phase="4")
    for name, (claim, expected) in corpus.items():
        assert verdicts[claim.claim_id].verdict is expected, (name, verdicts[claim.claim_id].reason)


def test_freshness_sla_makes_claims_unevaluable():
    """Stale or missing dimension ages make a falsifier unevaluable."""
    tree = dsl.parse("roe_pct < 10")
    assert _stale_reason(tree, {"fundamentals_age_days": 50.0}) is None
    assert "freshness" in _stale_reason(tree, {"fundamentals_age_days": 900.0})
    assert "freshness" in _stale_reason(dsl.parse("rsi_14 > 70"), {"last_candle_age_days": None})


def test_refutations_are_held_to_the_same_standard(snapshot, pair):
    """Only an evaluable, non-firing, reachable refutation vetoes; bad attacks are ignored."""
    strong, weak = pair
    ev = _ev(snapshot, strong, "fundamental_derived")
    target = _claim(strong, FUND, "roe_pct < sector_roe_pct", evidence=[ev])
    adjudicator = Adjudicator(snapshot, AuditLog())
    assert adjudicator.adjudicate_claims([target], phase="4")[target.claim_id].verdict is Verdict.SURVIVED
    roe = snapshot.derived_row(strong)["roe_pct"]

    def refutation(falsifier: str, target_id: str = target.claim_id) -> Refutation:
        """Build a refutation of ``target`` with the given falsifier."""
        draft = RefutationDraft(target_claim_id=target_id, statement="The ROE advantage is too thin to rely on.",
                                evidence=[ev], falsifier=falsifier)
        return Refutation.from_draft(draft, target=target, slot_class="veto")

    sound = refutation(f"roe_pct > {roe + 0.5:.2f}")
    fires = refutation("roe_pct > 0")
    vacuous = refutation("roe_pct > 100000")
    wrong_target = refutation("roe_pct > 1", target_id="cl:ffffffffffffffff")
    # Attacking a claim with the claim's own test asserts nothing: the target survived because that
    # condition is false, so the "attack" stands unchallenged and vetoes a claim it agrees with.
    circular = refutation("roe_pct < sector_roe_pct")
    spaced = refutation("roe_pct<sector_roe_pct")  # same test, different formatting
    ref_verdicts, vetoed = adjudicator.adjudicate_refutations(
        [fires, vacuous, wrong_target, circular, spaced, sound], {target.claim_id: target}, phase="4b")
    assert ref_verdicts[fires.refutation_id].verdict is Verdict.REFUTED
    assert ref_verdicts[vacuous.refutation_id].verdict is Verdict.VACUOUS
    assert ref_verdicts[wrong_target.refutation_id].verdict is Verdict.REJECTED
    assert ref_verdicts[circular.refutation_id].verdict is Verdict.REJECTED
    assert "attacks nothing" in ref_verdicts[circular.refutation_id].reason
    assert ref_verdicts[spaced.refutation_id].verdict is Verdict.REJECTED, "compared as parsed trees, not text"
    assert ref_verdicts[sound.refutation_id].verdict is Verdict.SURVIVED
    assert vetoed[target.claim_id].verdict is Verdict.VETOED and sound.refutation_id in vetoed[target.claim_id].reason


def test_score_is_weighted_signed_confidence_over_survivors(snapshot, pair):
    """score = 1.0 * fundamental + 0.8 * technical + 0.5 * news, counting only survivors."""
    key = pair[0]
    ev = _ev(snapshot, key, "fundamental_derived")
    claims = [
        _claim(key, FUND, "roe_pct < 1", evidence=[ev], confidence=0.7),
        _claim(key, TECH, "rsi_14 < 1", evidence=[ev], direction="negative", confidence=0.5),
        _claim(key, NEWS, "news_count_window < 1", evidence=[ev], confidence=0.4),
    ]
    outcome = [Verdict.SURVIVED, Verdict.SURVIVED, Verdict.REFUTED]
    verdicts = {c.claim_id: Adjudication(item_id=c.claim_id, kind="claim", verdict=v, reason="", phase="4")
                for c, v in zip(claims, outcome)}
    score = score_candidates([key], claims, verdicts)[0]
    assert score.score == pytest.approx(1.0 * 0.7 - 0.8 * 0.5)
    assert score.positive_dimensions == ["fundamental"]
