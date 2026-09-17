"""Walk-forward backtest tests (build-order phase P8).

Covers the leakage guards (no undated fundamentals, no future candles), the metric arithmetic,
a synthetic market where momentum ranking demonstrably beats the screen, determinism, and the
acceptance gates that keep an unproven method from being called validated.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from sealed_window.evaluate.walk_forward import (
    BacktestConfig,
    _drawdown_pct,
    _jaccard_turnover,
    _net_return_pct,
    assert_price_only,
    backtest_snapshot,
    run_walk_forward,
)
from sealed_window.governance.errors import LookaheadError
from sealed_window.screen.config import ScreenConfig

TECHNICAL_ONLY = ScreenConfig(max_candidates=20)
FAST = BacktestConfig(horizons=(5,), cutoffs=(2,), step_sessions=10, warmup_sessions=70, min_windows=5)


def _series(start: float, drift: float, sessions: int = 200) -> list[list[object]]:
    """A deterministic candle series compounding at ``drift`` per session from ``start``."""
    rows = []
    day = date(2025, 1, 1)
    price = start
    while len(rows) < sessions:
        if day.weekday() < 5:
            close = price * (1 + drift)
            rows.append([day.isoformat(), price, max(price, close) * 1.001, min(price, close) * 0.999,
                         close, 100_000.0])
            price = close
        day += timedelta(days=1)
    return rows


@pytest.fixture
def momentum_market() -> dict[str, list[list[object]]]:
    """Three persistent winners and three persistent losers: momentum ranking should win."""
    winners = {f"NSE_EQ|WIN{i}": _series(100 + i, 0.004) for i in range(3)}
    losers = {f"NSE_EQ|LOSE{i}": _series(100 + i, -0.002) for i in range(3)}
    return {**winners, **losers}


def test_fundamental_filters_are_refused():
    """Undated Upstox fundamentals cannot enter a past window, so the screen must be technical-only."""
    with pytest.raises(LookaheadError) as info:
        assert_price_only(ScreenConfig(roe_min_pct=12, rsi_min=30))
    assert "roe_min_pct" in str(info.value)
    assert_price_only(ScreenConfig(rsi_min=30, atr_pct_max=5, price_above_sma50=True))


def test_short_history_is_refused(momentum_market):
    """A snapshot without enough sessions for warmup plus forward horizon raises rather than guessing."""
    with pytest.raises(LookaheadError):
        run_walk_forward(candles=momentum_market, screen_config=TECHNICAL_ONLY,
                         config=BacktestConfig(warmup_sessions=190, horizons=(60,)))


def test_metric_helpers():
    """Net return subtracts costs; drawdown and turnover match hand-computed values."""
    assert _net_return_pct(100.0, 110.0, 20.0) == pytest.approx(9.8)
    assert _drawdown_pct([5.0, 10.0, 2.0]) == pytest.approx((1.02 / 1.10 - 1) * 100)
    assert _jaccard_turnover({"a", "b"}, {"a", "b"}) == 0.0
    assert _jaccard_turnover({"a", "b"}, {"c", "d"}) == 1.0
    assert _jaccard_turnover({"a", "b"}, {"b", "c"}) == pytest.approx(1 - 1 / 3)


def test_momentum_ranking_beats_the_screen_on_a_market_built_for_it(momentum_market):
    """On persistent trends the top-2 by momentum beat the equal-weight basket of all survivors."""
    report = run_walk_forward(candles=momentum_market, screen_config=TECHNICAL_ONLY, config=FAST)
    assert report.as_of_dates, "expected at least one window"
    metric = report.metrics[0]
    assert metric.precision == 1.0
    assert metric.mean_excess_return_pct > 0
    assert metric.mean_net_return_pct > metric.mean_benchmark_return_pct
    assert metric.mean_coverage == 1.0
    assert metric.mean_turnover == 0.0  # the same winners stay on top


def test_late_listing_does_not_truncate_everyone_else(momentum_market):
    """A recently listed stock is skipped in early windows rather than shortening the whole calendar."""
    sessions = len(next(iter(momentum_market.values())))
    with_newcomer = {**momentum_market, "NSE_EQ|NEWLIST": _series(100, 0.003, sessions)[-40:]}
    baseline = run_walk_forward(candles=momentum_market, screen_config=TECHNICAL_ONLY, config=FAST)
    widened = run_walk_forward(candles=with_newcomer, screen_config=TECHNICAL_ONLY, config=FAST)
    assert widened.as_of_dates == baseline.as_of_dates, "one late listing must not cost everyone history"

    # A strict threshold is what the loose default exists to avoid: demanding every instrument be
    # present collapses the calendar to the newcomer's 40 sessions, leaving no window at all.
    from sealed_window.evaluate.walk_forward import _trading_calendar

    assert len(_trading_calendar(with_newcomer, 0.6)) == len(_trading_calendar(momentum_market, 0.6))
    assert len(_trading_calendar(with_newcomer, 0.95)) == 40
    with pytest.raises(LookaheadError):
        run_walk_forward(candles=with_newcomer, screen_config=TECHNICAL_ONLY,
                         config=BacktestConfig(horizons=(5,), cutoffs=(2,), step_sessions=10,
                                               warmup_sessions=70, calendar_coverage=0.95))


def test_report_is_deterministic_and_identifies_its_inputs(momentum_market):
    """Same candles, screen and config give an identical report, stamped with all three hashes."""
    first = run_walk_forward(candles=momentum_market, screen_config=TECHNICAL_ONLY, config=FAST,
                             snapshot_hash="a" * 64)
    second = run_walk_forward(candles=momentum_market, screen_config=TECHNICAL_ONLY, config=FAST,
                              snapshot_hash="a" * 64)
    assert first.as_dict() == second.as_dict()
    assert first.as_dict()["snapshot"] == "a" * 64
    assert first.as_dict()["screen_config"] == TECHNICAL_ONLY.config_hash()
    assert first.as_dict()["backtest_config"] == FAST.config_hash()
    assert len(first.limitations) >= 5 and any("Survivorship" in item for item in first.limitations)


def test_gates_fail_loudly_when_evidence_is_thin(momentum_market):
    """Too few windows blocks promotion even when every measured metric looks good."""
    strict = BacktestConfig(horizons=(5,), cutoffs=(2,), step_sessions=10, warmup_sessions=70,
                            min_windows=500)
    report = run_walk_forward(candles=momentum_market, screen_config=TECHNICAL_ONLY, config=strict)
    assert report.promotion_ready is False
    assert any(failure.startswith("insufficient_windows") for failure in report.acceptance_failures)


def test_ranking_rules_are_named_and_validated(momentum_market):
    """An unknown ranking rule is rejected; the three known rules all run."""
    with pytest.raises(ValueError):
        run_walk_forward(candles=momentum_market, screen_config=TECHNICAL_ONLY,
                         config=BacktestConfig(ranking="hunch", warmup_sessions=70, horizons=(5,),
                                               cutoffs=(2,), step_sessions=10))
    for ranking in ("momentum_90d", "risk_adjusted_momentum", "trend_quality"):
        config = BacktestConfig(ranking=ranking, warmup_sessions=70, horizons=(5,), cutoffs=(2,),
                                step_sessions=10, min_windows=5)
        assert run_walk_forward(candles=momentum_market, screen_config=TECHNICAL_ONLY, config=config).metrics


def test_backtest_runs_over_a_sealed_snapshot(snapshot):
    """The snapshot entry point works end to end on the synthetic fixture market."""
    report = backtest_snapshot(snapshot, TECHNICAL_ONLY,
                               BacktestConfig(horizons=(20,), cutoffs=(5,), step_sessions=40,
                                              warmup_sessions=260, min_windows=2))
    assert report.universe == len(snapshot.instruments)
    assert report.snapshot_hash == snapshot.root_hash
    assert report.metrics[0].windows == len(report.as_of_dates)
