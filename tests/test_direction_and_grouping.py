"""Tests for the two ranking changes of 26 Sep 2026: table-assigned directions and grouped scoring.

Run 20260926T131512Z-bf9cb1 (qwen3:14b with the signal table) showed both problems: five claims read
"expensive on book value" with the matching falsifier yet carried a positive label, adding to the score;
and a company collected ROCE, ROE and ROA above the sector as three separate claims.
"""

from __future__ import annotations

from sealed_window.adjudicate.adjudicator import RANKING_VERSION, score_candidates
from sealed_window.agents.claim_agents import direction_from_table
from sealed_window.claims.schema import Adjudication, Claim, ClaimDraft, Direction, Verdict
from sealed_window.claims.signals import SIGNALS, group_for_falsifier, rule_for_falsifier
from sealed_window.snapshot.columns import Dimension

F, T = Dimension.FUNDAMENTAL, Dimension.TECHNICAL
PB_EXPENSIVE = "pb <= sector_pb OR pb <= 0 OR sector_pb <= 0"


def _draft(direction: str, falsifier: str, confidence: float = 0.8) -> ClaimDraft:
    """A minimal valid draft."""
    return ClaimDraft(predicate="test_claim", direction=Direction(direction),
                      statement="A statement long enough.", justification="A justification long enough.",
                      evidence=["ev:0123456789abcdef"], confidence=confidence, falsifier=falsifier)


def _claim(subject: str, dimension: Dimension, direction: str, falsifier: str, confidence: float) -> Claim:
    """A bound claim."""
    return Claim.from_draft(_draft(direction, falsifier, confidence), subject=subject, dimension=dimension,
                            slot_class="test")


def _survived(*claims: Claim) -> dict[str, Adjudication]:
    """SURVIVED verdicts for every claim."""
    return {c.claim_id: Adjudication(item_id=c.claim_id, kind="claim", verdict=Verdict.SURVIVED,
                                     reason="test", phase="test") for c in claims}


# ---- direction from the table ----------------------------------------------------------------------

def test_every_rule_is_found_from_its_own_falsifier():
    """The lookup is the inverse of the table, whatever the spacing or case of the copy."""
    for signal in SIGNALS:
        for rule in signal.rules:
            found = rule_for_falsifier("  " + rule.falsifier.upper().replace(" ", "   "), signal.dimension)
            assert found is not None and found[1] is rule, rule.key


def test_a_copied_falsifier_sets_the_direction_the_model_mislabelled():
    """"Expensive on book value" is against owning the stock, whatever label came first."""
    assert direction_from_table(_draft("positive", PB_EXPENSIVE), F) is Direction.NEGATIVE


def test_a_matching_label_or_the_models_own_falsifier_is_left_alone():
    """The harness corrects only a contradiction with a rule the model chose; it composes nothing."""
    assert direction_from_table(_draft("negative", PB_EXPENSIVE), F) is None
    assert direction_from_table(_draft("positive", "pb < 1.5 * sector_pb"), F) is None
    # A fundamental rule's falsifier is not looked up for a technical analyst.
    assert direction_from_table(_draft("positive", PB_EXPENSIVE), T) is None


def test_direction_source_defaults_to_model_so_older_ledgers_load():
    """Ledgers written before the field existed still load, and read as the model's own labels."""
    body = _claim("NSE_EQ|X", F, "positive", PB_EXPENSIVE, 0.8).model_dump()
    del body["direction_source"]
    assert Claim(**body).direction_source == "model"


# ---- grouped scoring -------------------------------------------------------------------------------

def test_related_claims_count_once_at_their_highest_confidence():
    """ROCE, ROE and ROA above the sector are one strength, counted at its strongest statement."""
    claims = [
        _claim("A", F, "positive", "roce_pct <= sector_roce_pct OR is_bank == 1 OR sector_roce_pct <= 0", 0.8),
        _claim("A", F, "positive", "roe_pct <= sector_roe_pct OR sector_roe_pct <= 0", 0.9),
        _claim("A", F, "positive", "roa_pct <= sector_roa_pct OR sector_roa_pct <= 0", 0.7),
    ]
    [score] = score_candidates(["A"], claims, _survived(*claims))
    assert score.score == 0.9
    assert score.dimensions["fundamental"]["positive"] == 3, "the ledger still shows every claim"


def test_both_sides_of_a_group_count():
    """Cheap on earnings and expensive on book value are a real disagreement; both are scored."""
    claims = [
        _claim("A", F, "positive", "pe >= sector_pe OR pe <= 0 OR sector_pe <= 0", 0.7),
        _claim("A", F, "negative", PB_EXPENSIVE, 0.6),
    ]
    [score] = score_candidates(["A"], claims, _survived(*claims))
    assert abs(score.score - 0.1) < 1e-9


def test_claims_outside_the_groups_count_individually():
    """A falsifier spanning two groups, or leaving the table, is its own unit."""
    claims = [
        _claim("A", T, "positive", "price_vs_sma200_pct < 0 OR close < sma_200", 0.5),
        _claim("A", T, "positive", "price_vs_sma50_pct < 0 OR close < sma_50", 0.5),
        _claim("A", F, "positive", "pe > sector_pe OR roe_pct < 10", 0.5),
    ]
    [score] = score_candidates(["A"], claims, _survived(*claims))
    assert abs(score.score - (0.8 * 0.5 * 2 + 0.5)) < 1e-9


def test_a_models_own_falsifier_is_grouped_by_its_columns():
    """Claude's combined ROE/ROCE claim joins the returns-on-capital group without copying the table."""
    assert group_for_falsifier("roe_pct < sector_roe_pct OR roce_pct < sector_roce_pct", F) == "returns_on_capital"
    assert group_for_falsifier("pe < sector_pe AND ev_ebitda < sector_ev_ebitda", F) == "valuation"
    assert group_for_falsifier("macd_hist >= 0", T) == "momentum"
    assert group_for_falsifier("not a falsifier ((", F) is None


def test_the_ranking_version_records_the_change():
    """Scores under grouping are not comparable with v1 scores, and the dossier must say which it used."""
    assert RANKING_VERSION == "ranking-v2"
