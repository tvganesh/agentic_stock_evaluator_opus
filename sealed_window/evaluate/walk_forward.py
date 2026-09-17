"""Price-only walk-forward evaluation of the technical screen and its ranking rules.

At each as-of date the evaluator rebuilds exactly what the system could have known from prices
alone, applies the screen, ranks the survivors, then measures the following sessions:

    for each window:
        rows      = technical indicators from candles up to and including as_of   (no future data)
        survivors = screen(rows)                                                  (technical filters only)
        selected  = top-k survivors by the chosen ranking rule
        benchmark = equal-weight basket of all survivors
        outcome   = net return of each selection over each horizon, minus costs

Metrics per (horizon, cutoff): precision (share of selections beating the benchmark), mean net and
excess return, worst portfolio drawdown, annualised volatility, selection turnover (1 - Jaccard) and
coverage. Acceptance gates then decide ``promotion_ready``; failures are listed explicitly, so a
method that has not proven itself cannot be described as validated.

Leakage controls (each raises :class:`LookaheadError`):

* the screen config may only use technical filters -- fundamentals from Upstox carry no date;
* every candle used for a window is dated on or before its as_of;
* every forward path starts strictly after the as_of;
* entry price is the as-of close, so a selection is bought at a price that existed when decided.

Determinism: pure arithmetic over the snapshot, so the same snapshot, screen and config always give
the same report, identified by their three hashes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Mapping, Sequence

from ..snapshot import indicators
from ..governance.errors import LookaheadError
from ..screen.config import ScreenConfig
from ..screen.screen import evaluate_row
from ..snapshot.columns import COLUMNS, Dimension
from ..snapshot.hashing import hash_object, stable_float
from ..snapshot.store import SealedSnapshot

BACKTEST_VERSION = "walk-forward-v1"
PRICE_ONLY_DIMENSIONS = frozenset({Dimension.TECHNICAL})

INDICATOR_LOOKBACK_SESSIONS = 300
"""Sessions of history handed to the indicator engine per window.

The longest indicator looks back 250 sessions (52-week high, 1-year return), so 300 leaves margin
while keeping a decade-long snapshot fast: work per window is bounded instead of growing with history."""

RankingRule = Callable[[Mapping[str, float | None]], float | None]


def _momentum_90d(row: Mapping[str, float | None]) -> float | None:
    """Rank by 90-session return: the simplest momentum baseline."""
    return row.get("return_90d_pct")


def _risk_adjusted_momentum(row: Mapping[str, float | None]) -> float | None:
    """Rank by 90-session return divided by annualised volatility (return per unit of risk)."""
    ret, vol = row.get("return_90d_pct"), row.get("volatility_20d_ann_pct")
    return None if ret is None or not vol else ret / vol


def _trend_quality(row: Mapping[str, float | None]) -> float | None:
    """Rank by distance above the 50-session average plus MACD histogram relative to price."""
    above, hist, close = row.get("price_vs_sma50_pct"), row.get("macd_hist"), row.get("close")
    if above is None or hist is None or not close:
        return None
    return above + 100.0 * hist / close


RANKING_RULES: dict[str, RankingRule] = {
    "momentum_90d": _momentum_90d,
    "risk_adjusted_momentum": _risk_adjusted_momentum,
    "trend_quality": _trend_quality,
}
"""Named, versioned ranking rules. The live system ranks by surviving claims, which do not exist for
past dates, so a backtest must name the deterministic rule it is testing."""


@dataclass(frozen=True)
class BacktestConfig:
    """Frozen parameters of a walk-forward run; hashed into the report."""

    horizons: tuple[int, ...] = (20, 60)
    cutoffs: tuple[int, ...] = (5, 10)
    step_sessions: int = 20
    warmup_sessions: int = 250
    calendar_coverage: float = 0.6
    round_trip_cost_bps: float = 20.0
    ranking: str = "momentum_90d"
    min_windows: int = 20
    min_precision: float = 0.5
    min_mean_excess_return_pct: float = 0.0
    max_drawdown_pct: float = -20.0

    def rule(self) -> RankingRule:
        """Return the configured ranking rule, or raise if the name is unknown."""
        if self.ranking not in RANKING_RULES:
            raise ValueError(f"unknown ranking rule {self.ranking!r}; known: {sorted(RANKING_RULES)}")
        return RANKING_RULES[self.ranking]

    def config_hash(self) -> str:
        """Content hash of the backtest parameters (third element of this report's identity)."""
        return hash_object({"version": BACKTEST_VERSION, **self.__dict__})


@dataclass
class WindowOutcome:
    """One as-of date: who passed the screen, who was selected, and what followed."""

    as_of: str
    survivors: list[str]
    selected: dict[int, list[str]] = field(default_factory=dict)
    net_returns: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    benchmark_returns: dict[int, float] = field(default_factory=dict)
    portfolio_paths: dict[tuple[int, int], list[float]] = field(default_factory=dict)


@dataclass
class HorizonMetrics:
    """Measured outcome for one (horizon, cutoff) pair across every window."""

    horizon: int
    cutoff: int
    windows: int
    selections: int
    precision: float | None
    mean_net_return_pct: float | None
    mean_benchmark_return_pct: float | None
    mean_excess_return_pct: float | None
    worst_drawdown_pct: float | None
    annualised_volatility_pct: float | None
    mean_turnover: float | None
    mean_coverage: float | None

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe view for the report."""
        return dict(self.__dict__)


@dataclass
class BacktestReport:
    """The full result: identity, windows, metrics, gate failures and stated limitations."""

    snapshot_hash: str
    screen_config_hash: str
    backtest_config_hash: str
    ranking: str
    as_of_dates: list[str]
    universe: int
    metrics: list[HorizonMetrics]
    promotion_ready: bool
    acceptance_failures: list[str]
    limitations: list[str]

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe view for writing the report to disk."""
        return {
            "version": BACKTEST_VERSION,
            "snapshot": self.snapshot_hash,
            "screen_config": self.screen_config_hash,
            "backtest_config": self.backtest_config_hash,
            "ranking": self.ranking,
            "windows": len(self.as_of_dates),
            "as_of_dates": self.as_of_dates,
            "universe": self.universe,
            "metrics": [m.as_dict() for m in self.metrics],
            "promotion_ready": self.promotion_ready,
            "acceptance_failures": self.acceptance_failures,
            "limitations": self.limitations,
        }

    def as_table(self) -> str:
        """Render the metrics as a fixed-width table for the CLI."""
        lines = [
            f"walk-forward {BACKTEST_VERSION} · ranking {self.ranking} · {len(self.as_of_dates)} windows "
            f"· universe {self.universe}",
            f"snapshot {self.snapshot_hash[:12]}...  screen {self.screen_config_hash[:12]}...  "
            f"config {self.backtest_config_hash[:12]}...",
            "",
            f"{'horizon':>8}{'top-k':>7}{'sel':>6}{'precision':>11}{'net %':>9}{'bench %':>9}"
            f"{'excess %':>10}{'drawdn %':>10}{'vol %':>8}{'turnover':>10}",
            "-" * 88,
        ]

        def cell(value: float | None, places: int = 2) -> str:
            """Format an optional number for the table."""
            return "n/a" if value is None else f"{value:.{places}f}"

        for m in self.metrics:
            lines.append(
                f"{m.horizon:>8}{m.cutoff:>7}{m.selections:>6}{cell(m.precision, 3):>11}"
                f"{cell(m.mean_net_return_pct):>9}{cell(m.mean_benchmark_return_pct):>9}"
                f"{cell(m.mean_excess_return_pct):>10}{cell(m.worst_drawdown_pct):>10}"
                f"{cell(m.annualised_volatility_pct):>8}{cell(m.mean_turnover, 3):>10}"
            )
        lines += ["-" * 88, f"promotion_ready: {self.promotion_ready}"]
        lines += [f"  gate failed: {failure}" for failure in self.acceptance_failures]
        lines += ["", "Limitations:"] + [f"  - {item}" for item in self.limitations]
        return "\n".join(lines)


def assert_price_only(config: ScreenConfig) -> None:
    """Raise :class:`LookaheadError` if the screen filters on anything but technical columns.

    Upstox fundamentals carry no as-of date, so using them in a past window would leak the future.
    """
    leaking = []
    for field_name, value in config.model_dump(mode="json").items():
        if value is None or field_name == "max_candidates":
            continue
        column = _config_field_column(field_name)
        if column is not None and COLUMNS[column].dimension is not Dimension.TECHNICAL:
            leaking.append(field_name)
    if leaking:
        raise LookaheadError(
            "price-only backtest refuses non-technical filters (Upstox fundamentals are undated): "
            + ", ".join(sorted(leaking))
        )


def _config_field_column(field_name: str) -> str | None:
    """Map a screen-config field to the column it filters, or ``None`` if it filters nothing."""
    from ..screen.screen import rules_for

    for rule in rules_for(None):
        if rule.config_field == field_name:
            return rule.column
    return None


def _net_return_pct(entry: float, exit_price: float, cost_bps: float) -> float:
    """Percentage return from entry to exit, less a round-trip cost in basis points."""
    return (exit_price / entry - 1.0) * 100.0 - cost_bps / 100.0


def _mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or ``None`` for an empty sequence."""
    return sum(values) / len(values) if values else None


def _sample_volatility_pct(period_returns: Sequence[float], periods_per_year: int = 252) -> float | None:
    """Annualised sample standard deviation of per-session portfolio returns, in percent."""
    if len(period_returns) < 2:
        return None
    mean = sum(period_returns) / len(period_returns)
    variance = sum((value - mean) ** 2 for value in period_returns) / (len(period_returns) - 1)
    return math.sqrt(variance) * math.sqrt(periods_per_year)


def _drawdown_pct(path: Sequence[float]) -> float:
    """Worst peak-to-trough decline of a cumulative-return path, in percent (<= 0)."""
    peak, worst = 1.0, 0.0
    for cumulative in path:
        wealth = 1.0 + cumulative / 100.0
        peak = max(peak, wealth)
        worst = min(worst, (wealth / peak - 1.0) * 100.0)
    return worst


def _jaccard_turnover(previous: set[str], current: set[str]) -> float:
    """1 - Jaccard similarity between consecutive selections (0 = identical, 1 = no overlap)."""
    union = previous | current
    return 0.0 if not union else 1.0 - len(previous & current) / len(union)


def _trading_calendar(candles: Mapping[str, Sequence[Sequence[Any]]], coverage: float = 0.6) -> list[str]:
    """Dates present for at least ``coverage`` of instruments: the calendar windows step through.

    The threshold is deliberately loose. A strict one (90%) lets a handful of recent listings delete
    years of history for everyone: on the 17 Sep 2026 snapshot it cut a 992-session collection to a
    585-session calendar starting May 2024. Stocks without a candle on a given date are skipped in
    that window instead, which is also what a real screen would have done before they listed.
    """
    counts: dict[str, int] = {}
    for rows in candles.values():
        for row in rows:
            counts[str(row[0])] = counts.get(str(row[0]), 0) + 1
    # Round up: "at least 60% of instruments" must not quietly become 59% through truncation.
    needed = max(1, math.ceil(len(candles) * coverage))
    return sorted(day for day, count in counts.items() if count >= needed)


def run_walk_forward(
    *,
    candles: Mapping[str, Sequence[Sequence[Any]]],
    screen_config: ScreenConfig,
    config: BacktestConfig,
    snapshot_hash: str = "",
) -> BacktestReport:
    """Run the walk-forward over raw candle series and return the measured report.

    ``candles`` maps instrument key to ascending ``[date, open, high, low, close, volume]`` rows.
    """
    assert_price_only(screen_config)
    rule = config.rule()
    calendar = _trading_calendar(candles, config.calendar_coverage)
    max_horizon = max(config.horizons)
    first = config.warmup_sessions
    last = len(calendar) - max_horizon
    if last <= first:
        raise LookaheadError(
            f"snapshot has {len(calendar)} sessions: too few for {config.warmup_sessions} warmup "
            f"+ {max_horizon} forward sessions"
        )
    window_dates = [calendar[i] for i in range(first, last, config.step_sessions)]
    by_key = {key: {str(row[0]): index for index, row in enumerate(rows)} for key, rows in candles.items()}

    outcomes: list[WindowOutcome] = []
    for as_of_text in window_dates:
        as_of = date.fromisoformat(as_of_text)
        survivors: list[tuple[str, float, dict[str, float | None]]] = []
        for key, rows in candles.items():
            index = by_key[key].get(as_of_text)
            if index is None or index + 1 < config.warmup_sessions or index + max_horizon >= len(rows):
                continue  # no candle that day, too little history, or not enough forward data
            history = rows[max(0, index + 1 - INDICATOR_LOOKBACK_SESSIONS) : index + 1]
            if str(history[-1][0]) > as_of_text:
                raise LookaheadError(f"{key}: candle {history[-1][0]} is after as_of {as_of_text}")
            row = indicators.technical_row(history, as_of)
            if evaluate_row(row, screen_config, PRICE_ONLY_DIMENSIONS):
                continue
            score = rule(row)
            if score is None:
                continue
            survivors.append((key, float(score), row))
        if not survivors:
            continue

        survivors.sort(key=lambda item: (-item[1], item[0]))
        outcome = WindowOutcome(as_of=as_of_text, survivors=[key for key, _, _ in survivors])
        forward: dict[str, list[float]] = {}
        for key, _, _ in survivors:
            index = by_key[key][as_of_text]
            entry = float(candles[key][index][4])
            future = candles[key][index + 1 : index + 1 + max_horizon]
            if str(future[0][0]) <= as_of_text:
                raise LookaheadError(f"{key}: forward path starts at {future[0][0]}, not after {as_of_text}")
            forward[key] = [_net_return_pct(entry, float(row[4]), config.round_trip_cost_bps) for row in future]

        for horizon in config.horizons:
            outcome.benchmark_returns[horizon] = sum(
                forward[key][horizon - 1] for key in forward
            ) / len(forward)
        for cutoff in config.cutoffs:
            chosen = [key for key, _, _ in survivors[:cutoff]]
            outcome.selected[cutoff] = chosen
            for horizon in config.horizons:
                outcome.net_returns[(horizon, cutoff)] = [forward[key][horizon - 1] for key in chosen]
                outcome.portfolio_paths[(horizon, cutoff)] = [
                    sum(forward[key][step] for key in chosen) / len(chosen) for step in range(horizon)
                ]
        outcomes.append(outcome)

    metrics = _metrics(outcomes, config)
    failures = _acceptance_failures(metrics, len(outcomes), config)
    return BacktestReport(
        snapshot_hash=snapshot_hash,
        screen_config_hash=screen_config.config_hash(),
        backtest_config_hash=config.config_hash(),
        ranking=config.ranking,
        as_of_dates=[outcome.as_of for outcome in outcomes],
        universe=len(candles),
        metrics=metrics,
        promotion_ready=not failures,
        acceptance_failures=failures,
        limitations=LIMITATIONS,
    )


LIMITATIONS = [
    "Price-only: fundamentals are excluded because Upstox ratios carry no as-of date.",
    "Survivorship bias: the universe is today's index membership, so delisted and demoted names are absent.",
    "The benchmark is an equal-weight basket of stocks that passed the same screen, not a market index.",
    "The claim and veto layers are not measured; this tests the deterministic screen and ranking only.",
    "Overlapping windows share sessions, so window outcomes are not independent observations.",
    "Costs are a flat round-trip assumption; no slippage, impact, taxes or dividends are modelled.",
]


def _metrics(outcomes: Sequence[WindowOutcome], config: BacktestConfig) -> list[HorizonMetrics]:
    """Aggregate per-window outcomes into one metrics row per (horizon, cutoff)."""
    metrics: list[HorizonMetrics] = []
    for horizon in config.horizons:
        for cutoff in config.cutoffs:
            nets: list[float] = []
            benches: list[float] = []
            excesses: list[float] = []
            drawdowns: list[float] = []
            period_returns: list[float] = []
            coverages: list[float] = []
            turnovers: list[float] = []
            previous: set[str] | None = None
            for outcome in outcomes:
                chosen = outcome.selected.get(cutoff, [])
                coverages.append(len(chosen) / cutoff)
                if previous is not None:
                    turnovers.append(_jaccard_turnover(previous, set(chosen)))
                previous = set(chosen)
                window_nets = outcome.net_returns.get((horizon, cutoff), [])
                if not window_nets:
                    continue
                benchmark = outcome.benchmark_returns[horizon]
                nets.extend(window_nets)
                benches.extend([benchmark] * len(window_nets))
                excesses.extend(value - benchmark for value in window_nets)
                path = outcome.portfolio_paths[(horizon, cutoff)]
                drawdowns.append(_drawdown_pct(path))
                wealth = [1.0 + value / 100.0 for value in path]
                period_returns.extend(
                    (current / previous_wealth - 1.0)
                    for previous_wealth, current in zip([1.0] + wealth[:-1], wealth)
                )
            metrics.append(
                HorizonMetrics(
                    horizon=horizon,
                    cutoff=cutoff,
                    windows=len(outcomes),
                    selections=len(nets),
                    precision=stable_float(sum(1 for value in excesses if value >= 0) / len(excesses))
                    if excesses
                    else None,
                    mean_net_return_pct=stable_float(_mean(nets)),
                    mean_benchmark_return_pct=stable_float(_mean(benches)),
                    mean_excess_return_pct=stable_float(_mean(excesses)),
                    worst_drawdown_pct=stable_float(min(drawdowns)) if drawdowns else None,
                    annualised_volatility_pct=stable_float(
                        (_sample_volatility_pct(period_returns) or 0.0) * 100.0
                    )
                    if len(period_returns) > 1
                    else None,
                    mean_turnover=stable_float(_mean(turnovers)),
                    mean_coverage=stable_float(_mean(coverages)),
                )
            )
    return metrics


def _acceptance_failures(
    metrics: Sequence[HorizonMetrics], window_count: int, config: BacktestConfig
) -> list[str]:
    """List every gate a result fails; an empty list is the only route to ``promotion_ready``."""
    failures: list[str] = []
    if window_count < config.min_windows:
        failures.append(f"insufficient_windows:{window_count}<{config.min_windows}")
    for metric in metrics:
        label = f"h{metric.horizon}_k{metric.cutoff}"
        if metric.precision is None or metric.precision < config.min_precision:
            failures.append(f"precision_below_threshold:{label}")
        if (
            metric.mean_excess_return_pct is None
            or metric.mean_excess_return_pct < config.min_mean_excess_return_pct
        ):
            failures.append(f"excess_return_below_threshold:{label}")
        if metric.worst_drawdown_pct is None or metric.worst_drawdown_pct < config.max_drawdown_pct:
            failures.append(f"drawdown_below_threshold:{label}")
    return failures


def backtest_snapshot(
    snapshot: SealedSnapshot, screen_config: ScreenConfig, config: BacktestConfig
) -> BacktestReport:
    """Run the walk-forward over a sealed snapshot's candles (the CLI entry point)."""
    directory_candles = {
        instrument["instrument_key"]: snapshot.candles_for(instrument["instrument_key"])
        for instrument in snapshot.instruments
    }
    return run_walk_forward(
        candles={key: rows for key, rows in directory_candles.items() if rows},
        screen_config=screen_config,
        config=config,
        snapshot_hash=snapshot.root_hash,
    )
