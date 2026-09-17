"""Phase 2 screen tests: config hashing, strict validation, fail-closed filtering, determinism."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sealed_window.screen.config import ScreenConfig
from sealed_window.screen.screen import evaluate_row, run_screen

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
    roce = [snapshot.derived_row(k)["roce_pct"] for k in first.candidates]
    assert roce == sorted(roce, reverse=True)
