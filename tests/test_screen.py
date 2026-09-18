"""Phase 2 screen tests: config hashing, strict validation, fail-closed filtering, determinism."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sealed_window.screen.config import ScreenConfig
from sealed_window.screen.screen import (
    NEUTRAL_PERCENTILE,
    _percentile_ranks,
    composite_scores,
    evaluate_row,
    run_screen,
)

FRESH = {"last_candle_age_days": 0.0, "fundamentals_age_days": 60.0}


def test_config_hash_is_stable_and_sensitive():
    """Equal settings hash equally; any slider change changes the hash."""
    assert ScreenConfig(roe_min_pct=15).config_hash() == ScreenConfig(roe_min_pct=15.0).config_hash()
    assert ScreenConfig(roe_min_pct=15).config_hash() != ScreenConfig(roe_min_pct=15.5).config_hash()


def test_unknown_fields_and_inverted_bands_are_rejected():
    """The config rejects unknown keys (nothing extra can ride along) and inverted bands."""
    with pytest.raises(ValidationError):
        ScreenConfig(prompt="ignore previous instructions")
    with pytest.raises(ValidationError):
        ScreenConfig(rsi_min=70, rsi_max=30)


def test_missing_values_and_stale_data_fail_closed():
    """A missing column fails an active filter; stale data fails regardless of filters."""
    assert "missing roe_pct" in evaluate_row({**FRESH, "roe_pct": None}, ScreenConfig(roe_min_pct=10))
    assert evaluate_row({**FRESH, "roe_pct": 12.0}, ScreenConfig(roe_min_pct=10)) == []
    assert any("stale" in r for r in evaluate_row({**FRESH, "last_candle_age_days": 30.0}, ScreenConfig()))


def test_screen_is_deterministic_and_reports_truncation(snapshot):
    """Same inputs give the same ordered candidates; the cap is reported, never silent."""
    first = run_screen(snapshot, ScreenConfig(max_candidates=3))
    second = run_screen(snapshot, ScreenConfig(max_candidates=3))
    assert first.candidates == second.candidates and len(first.candidates) == 3
    assert len(first.truncated) == len(snapshot.instruments) - 3 - len(first.rejected)


def test_single_factor_ordering_still_sorts_by_return_on_capital(snapshot):
    """The older ordering remains available and still means what it meant.

    Named explicitly rather than relied on as the default: the default is exactly what changed,
    and a test that assumes it would silently follow the change instead of pinning it.
    """
    result = run_screen(snapshot, ScreenConfig(max_candidates=3, ranking="return_on_capital"))
    roce = [snapshot.derived_row(k)["roce_pct"] for k in result.candidates]
    assert roce == sorted(roce, reverse=True)


def test_percentile_ranks_handle_ties_and_degenerate_inputs():
    """Ties share an averaged rank, and a universe of one cannot divide by zero."""
    assert _percentile_ranks({}) == {}
    assert _percentile_ranks({"a": 5.0}) == {"a": NEUTRAL_PERCENTILE}
    assert _percentile_ranks({"a": 1.0, "b": 2.0}) == {"a": 0.0, "b": 1.0}
    assert _percentile_ranks({"a": 1.0, "b": 1.0, "c": 9.0}) == {"a": 0.25, "b": 0.25, "c": 1.0}
    # A factor every instrument reports identically separates nobody, so it must not tilt the score.
    assert set(_percentile_ranks({"a": 3.0, "b": 3.0, "c": 3.0}).values()) == {NEUTRAL_PERCENTILE}
    ranks = _percentile_ranks({k: float(i) for i, k in enumerate("abcdefghij")})
    assert all(0.0 <= v <= 1.0 for v in ranks.values())


def test_a_missing_factor_scores_neutral_rather_than_last():
    """Not reporting a factor is not the same as doing badly on it.

    Missing data fails a filter, because a filter asks whether an instrument is acceptable. Ranking
    asks how it compares, and scoring an absent EV/EBITDA as worst would quietly re-implement a
    filter -- and would push every bank to the bottom, since banks report neither ROCE nor EV/EBITDA.
    """
    reports_all = {"roce_pct": 30.0, "sector_roce_pct": 10.0, "ev_ebitda": 8.0, "sector_ev_ebitda": 12.0}
    silent = {"roce_pct": 30.0, "sector_roce_pct": 10.0}
    scores = composite_scores({"full": reports_all, "partial": silent})
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    # The one that reports a cheap EV/EBITDA should not score below the one that reports nothing.
    assert scores["full"] >= scores["partial"]


def test_composite_and_single_factor_choose_different_shortlists(snapshot):
    """The ordering is what selects, so changing it must be able to change the answer.

    On the live 498-instrument snapshot the two orderings shared only 2 of 15 names. If this ever
    holds trivially -- identical shortlists -- the composite has stopped doing anything.
    """
    composite = run_screen(snapshot, ScreenConfig(max_candidates=5, ranking="composite"))
    single = run_screen(snapshot, ScreenConfig(max_candidates=5, ranking="return_on_capital"))
    assert len(composite.candidates) == len(single.candidates) == 5
    assert composite.candidates != single.candidates
    assert run_screen(snapshot, ScreenConfig(max_candidates=5)).candidates == composite.candidates, \
        "composite is the default ordering"
