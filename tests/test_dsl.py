"""Falsifier DSL tests: grammar, safety limits, evaluation semantics and the vacuity check."""

from __future__ import annotations

import pytest

from sealed_window.claims import dsl

ROW = {"roe_pct": 18.0, "sector_roe_pct": 15.0, "price_vs_sma50_pct": 3.0, "macd_hist": -0.5, "pe": None,
       "close": 100.0}


@pytest.mark.parametrize("text,expected", [
    ("roe_pct < sector_roe_pct", False),
    ("roe_pct < sector_roe_pct + 5", True),
    ("price_vs_sma50_pct < 0 OR macd_hist < 0", True),
    ("price_vs_sma50_pct < 0 AND macd_hist < 0", False),
    ("NOT (roe_pct > 20)", True),
    ("(roe_pct - sector_roe_pct) / sector_roe_pct < 0.1", False),
    ("abs(macd_hist) >= 0.5 and max(roe_pct, 1) == 18", True),
    ("roe_pct > 100 OR roe_pct > 10 AND macd_hist > 0", False),  # AND binds tighter than OR
])
def test_evaluation(text, expected):
    """Falsifiers evaluate with standard precedence, grouping and functions."""
    assert dsl.evaluate(dsl.parse(text), ROW) is expected


@pytest.mark.parametrize("text", [
    "unknown_column > 1",
    "__import__('os')",
    "roe_pct.real > 1",
    "roe_pct > 1; close < 0",
    "1 < 2",
    "roe_pct >",
    "roe_pct > 1 " + "OR roe_pct > 1 " * 7,
    "roe_pct + close + pe + macd_hist + sector_roe_pct > 1",
    "(" * 10 + "roe_pct > 1" + ")" * 10,
    "roe_pct > " + "1" * 400,
])
def test_rejected_falsifiers(text):
    """Unknown columns, code, attribute access, constants-only and over-limit expressions are rejected."""
    with pytest.raises(dsl.FalsifierError):
        dsl.parse(text)


def test_missing_value_and_division_by_zero_are_unevaluable():
    """Unevaluable is not true: None columns and division by zero raise Unevaluable."""
    with pytest.raises(dsl.Unevaluable):
        dsl.evaluate(dsl.parse("pe > 10"), ROW)
    with pytest.raises(dsl.Unevaluable):
        dsl.evaluate(dsl.parse("roe_pct / (close - 100) > 1"), ROW)


def test_vacuity_check():
    """A falsifier that can never fire on observed values is unreachable; a real one is reachable."""
    scenarios = {"close": [50.0, 80.0, 100.0, 150.0, 300.0], "roe_pct": [4.0, 10.0, 15.0, 22.0, 30.0]}
    assert not dsl.is_reachable(dsl.parse("close < 0"), ROW, scenarios)
    assert not dsl.is_reachable(dsl.parse("roe_pct < 10 AND roe_pct > 1000"), ROW, scenarios)
    assert dsl.is_reachable(dsl.parse("roe_pct < 15"), ROW, scenarios)
