"""Deterministic screen: snapshot + screen config -> candidate set. No model, no network.

Fail-closed semantics:

* an active filter over a column that is ``None`` rejects the instrument ("missing data is
  not a pass");
* every candidate must have technical and fundamental data inside the freshness SLAs from
  ``snapshot.columns``, whether or not a filter touches them, because the claim phase will
  reason over both;
* if more instruments pass than ``max_candidates``, they are ordered by return on capital
  (ROCE, or ROE where ROCE is not reported, descending; ties by instrument key) and truncated,
  and the truncation is reported -- never silent.

Bank-aware filters: Upstox reports NIM, Net NPA and CASA for banks instead of ROCE, and a
bank's liabilities-to-equity is structurally near 10x. So the ROCE floor and leverage ceiling
apply only to non-banks, and the Net NPA ceiling only to banks. An instrument whose bank status
is unknown (no key ratios at all) is treated as a non-bank, so it still fails closed on ROCE.

The result, together with the snapshot hash and config hash, fully determines the candidate
list, so a disputed shortlist can be reproduced before any question of model behaviour arises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from ..snapshot.columns import COLUMNS, FRESHNESS_SLA, Dimension
from ..snapshot.store import SealedSnapshot
from .config import ScreenConfig

Row = dict[str, float | None]


@dataclass(frozen=True)
class _Rule:
    """One filter: activating config field, column read, test, and which instruments it applies to."""

    config_field: str
    column: str
    label: str
    test: Callable[[float, object], bool]
    scope: str = "all"  # "all" | "non_bank" | "bank"


_RULES: tuple[_Rule, ...] = (
    _Rule("roe_min_pct", "roe_pct", "ROE below floor", lambda v, t: v >= t),
    _Rule("roce_min_pct", "roce_pct", "ROCE below floor", lambda v, t: v >= t, scope="non_bank"),
    _Rule("liabilities_to_equity_max", "liabilities_to_equity", "leverage above ceiling", lambda v, t: v <= t,
          scope="non_bank"),
    _Rule("net_npa_max_pct", "net_npa_pct", "net NPA above ceiling", lambda v, t: v <= t, scope="bank"),
    _Rule("revenue_growth_min_pct", "revenue_growth_1y_pct", "revenue growth below floor", lambda v, t: v >= t),
    _Rule("earnings_growth_min_pct", "net_profit_growth_1y_pct", "earnings growth below floor", lambda v, t: v >= t),
    _Rule("pe_min", "pe", "P/E below band", lambda v, t: v >= t),
    _Rule("pe_max", "pe", "P/E above band", lambda v, t: v <= t),
    _Rule("rsi_min", "rsi_14", "RSI below band", lambda v, t: v >= t),
    _Rule("rsi_max", "rsi_14", "RSI above band", lambda v, t: v <= t),
    _Rule("price_above_sma20", "price_vs_sma20_pct", "price vs MA20 mismatch", lambda v, t: (v > 0) == t),
    _Rule("price_above_sma50", "price_vs_sma50_pct", "price vs MA50 mismatch", lambda v, t: (v > 0) == t),
    _Rule("volume_ratio_min", "volume_ratio_20d", "volume ratio below floor", lambda v, t: v >= t),
    _Rule("atr_pct_max", "atr_pct", "ATR% above ceiling", lambda v, t: v <= t),
    _Rule("return_30d_min_pct", "return_30d_pct", "30d return below band", lambda v, t: v >= t),
    _Rule("return_30d_max_pct", "return_30d_pct", "30d return above band", lambda v, t: v <= t),
    _Rule("return_90d_min_pct", "return_90d_pct", "90d return below band", lambda v, t: v >= t),
    _Rule("return_90d_max_pct", "return_90d_pct", "90d return above band", lambda v, t: v <= t),
)


@dataclass
class ScreenResult:
    """Outcome of phase 2: candidates in claim-phase order, reasons for every rejection."""

    snapshot_hash: str
    config_hash: str
    candidates: list[str]
    rejected: dict[str, list[str]] = field(default_factory=dict)
    truncated: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        """JSON-safe summary for the audit log, CLI and UI funnel view."""
        return {
            "snapshot_hash": self.snapshot_hash,
            "config_hash": self.config_hash,
            "candidates": len(self.candidates),
            "rejected": len(self.rejected),
            "truncated": len(self.truncated),
        }


def is_bank(row: Row) -> bool:
    """True only when the snapshot positively identifies the instrument as a bank."""
    return row.get("is_bank") == 1.0


def return_on_capital(row: Row) -> float | None:
    """ROCE where reported, otherwise ROE (banks); the single-factor ordering."""
    roce = row.get("roce_pct")
    return roce if roce is not None else row.get("roe_pct")


# ---- composite ranking ------------------------------------------------------------------
#
# The filters answer "is this acceptable?"; the ordering answers "is this best?". Only the
# ordering actually selects, because far more instruments pass the filters than fit the
# candidate cap -- in the run of 18 Sep 2026, 177 passed every filter and were cut by the sort
# alone. Ordering on one factor therefore let a single number make the selection.
#
# Each factor below is converted to a percentile rank across the whole universe, so quantities
# in different units (a ratio, a percentage, a growth rate) become comparable, and an outlier
# cannot dominate. Comparisons are sector-relative where a benchmark exists: a P/E of 70 means
# something different for a packaged-foods company than for a bank.


@dataclass(frozen=True)
class _Factor:
    """One ranked factor: how to read it from a row, its group, and which direction is better."""

    name: str
    group: str
    read: Callable[[Row], float | None]
    higher_is_better: bool = True


def _spread(column: str, benchmark: str) -> Callable[[Row], float | None]:
    """Value minus its sector benchmark, in the column's own units."""
    def read(row: Row) -> float | None:
        value, base = row.get(column), row.get(benchmark)
        return None if value is None or base is None else value - base
    return read


def _ratio(column: str, benchmark: str) -> Callable[[Row], float | None]:
    """Value as a multiple of its sector benchmark (1.0 = at the sector)."""
    def read(row: Row) -> float | None:
        value, base = row.get(column), row.get(benchmark)
        return None if value is None or not base else value / base
    return read


def _plain(column: str) -> Callable[[Row], float | None]:
    """The column as it stands."""
    return lambda row: row.get(column)


def _macd_hist_pct(row: Row) -> float | None:
    """MACD histogram as a percentage of price, so it compares across price levels."""
    hist, close = row.get("macd_hist"), row.get("close")
    return None if hist is None or not close else 100.0 * hist / close


FACTORS: tuple[_Factor, ...] = (
    _Factor("roce_vs_sector", "quality", _spread("roce_pct", "sector_roce_pct")),
    _Factor("roe_vs_sector", "quality", _spread("roe_pct", "sector_roe_pct")),
    _Factor("margin_trend", "quality", _plain("operating_margin_delta_1y_pp")),
    _Factor("pe_vs_sector", "value", _ratio("pe", "sector_pe"), higher_is_better=False),
    _Factor("pb_vs_sector", "value", _ratio("pb", "sector_pb"), higher_is_better=False),
    _Factor("ev_ebitda_vs_sector", "value", _ratio("ev_ebitda", "sector_ev_ebitda"), higher_is_better=False),
    _Factor("revenue_growth", "growth", _plain("revenue_growth_1y_pct")),
    _Factor("profit_growth", "growth", _plain("net_profit_growth_1y_pct")),
    _Factor("above_sma200", "trend", _plain("price_vs_sma200_pct")),
    _Factor("macd_hist_pct", "trend", _macd_hist_pct),
    _Factor("return_90d", "trend", _plain("return_90d_pct")),
    _Factor("atr_pct", "risk", _plain("atr_pct"), higher_is_better=False),
    _Factor("volatility", "risk", _plain("volatility_20d_ann_pct"), higher_is_better=False),
    # Drawdown is negative or zero, so nearer zero is already the higher value.
    _Factor("drawdown", "risk", _plain("drawdown_from_52w_high_pct")),
)

GROUP_WEIGHTS: dict[str, float] = {
    "quality": 0.30, "value": 0.20, "growth": 0.20, "trend": 0.20, "risk": 0.10,
}
"""How much each group contributes to the composite.

A reasoned prior, not a measured edge. The walk-forward cannot settle it: Upstox ratios carry no
as-of date, so no fundamental factor can be evaluated at a past date without leaking the future,
and the three price-only rankings that *can* be tested all failed their acceptance gates. Treat
these weights as an argument to be disagreed with, and keep them in one place so disagreeing is
a one-line change."""

NEUTRAL_PERCENTILE = 0.5
"""Score for a factor an instrument does not report.

Missing data fails a *filter* -- "missing data is not a pass" -- because a filter asks whether an
instrument is acceptable. Ranking asks how it compares, and an instrument that simply does not
report EV/EBITDA has not thereby done badly on it. Scoring the gap as neutral keeps the ordering
from quietly re-implementing a filter; banks, which report neither ROCE nor EV/EBITDA, land here
on those factors rather than being pushed to the bottom."""


def _percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    """Percentile rank in [0, 1] for each key, ties sharing the average rank."""
    if not values:
        return {}
    if len(values) == 1:
        return {key: NEUTRAL_PERCENTILE for key in values}
    ordered = sorted(values.items(), key=lambda item: item[1])
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][1] == ordered[index][1]:
            stop += 1
        shared = (index + stop) / 2.0 / (len(ordered) - 1)
        for position in range(index, stop + 1):
            ranks[ordered[position][0]] = shared
        index = stop + 1
    return ranks


def composite_scores(table: Mapping[str, Row]) -> dict[str, float]:
    """Composite percentile score in [0, 1] per instrument, over the whole universe.

    Percentiles are taken across every instrument in the snapshot, not only those that passed the
    filters, so a score means "this good relative to the market" rather than "relative to whoever
    survived today's settings" -- which keeps scores comparable as the sliders move.
    """
    per_factor: dict[str, dict[str, float]] = {}
    for factor in FACTORS:
        observed = {}
        for key, row in table.items():
            value = factor.read(row)
            if value is not None:
                observed[key] = value if factor.higher_is_better else -value
        per_factor[factor.name] = _percentile_ranks(observed)

    by_group: dict[str, list[_Factor]] = {}
    for factor in FACTORS:
        by_group.setdefault(factor.group, []).append(factor)

    scores: dict[str, float] = {}
    for key in table:
        total = 0.0
        for group, factors in by_group.items():
            marks = [per_factor[f.name].get(key, NEUTRAL_PERCENTILE) for f in factors]
            total += GROUP_WEIGHTS.get(group, 0.0) * (sum(marks) / len(marks))
        scores[key] = total
    return scores


def rules_for(dimensions: frozenset[Dimension] | None = None) -> tuple[_Rule, ...]:
    """Filters whose column belongs to one of ``dimensions`` (all rules when ``None``).

    The walk-forward backtest passes ``{TECHNICAL}`` so no undated fundamental can enter a past window.
    """
    if dimensions is None:
        return _RULES
    return tuple(rule for rule in _RULES if COLUMNS[rule.column].dimension in dimensions)


def freshness_failures(row: Row, dimensions: frozenset[Dimension] | None = None) -> list[str]:
    """Return reasons a row violates the freshness SLAs of ``dimensions`` (all dimensions by default)."""
    reasons = []
    for dimension in (Dimension.TECHNICAL, Dimension.FUNDAMENTAL):
        if dimensions is not None and dimension not in dimensions:
            continue
        sla = FRESHNESS_SLA[dimension]
        if sla is None:
            continue
        column, max_days = sla
        age = row.get(column)
        if age is None or age > max_days:
            reasons.append(f"stale or missing {dimension.value} data ({column}={age})")
    return reasons


def _applies(rule: _Rule, row: Row) -> bool:
    """Whether a rule's scope covers this instrument (bank-only, non-bank-only or all)."""
    if rule.scope == "bank":
        return is_bank(row)
    if rule.scope == "non_bank":
        return not is_bank(row)
    return True


def evaluate_row(row: Row, config: ScreenConfig, dimensions: frozenset[Dimension] | None = None) -> list[str]:
    """Apply every active, in-scope filter to one derived row; return rejection reasons (empty = pass).

    ``dimensions`` restricts which columns may be filtered on (used by the price-only backtest).
    """
    reasons = freshness_failures(row, dimensions)
    for rule in rules_for(dimensions):
        threshold = getattr(config, rule.config_field)
        if threshold is None or not _applies(rule, row):
            continue
        value = row.get(rule.column)
        if value is None:
            reasons.append(f"missing {rule.column}")
        elif not rule.test(value, threshold):
            reasons.append(f"{rule.label} ({rule.column}={value}, limit={threshold})")
    return reasons


def run_screen(snapshot: SealedSnapshot, config: ScreenConfig) -> ScreenResult:
    """Filter the snapshot universe with ``config`` and return the ordered candidate set."""
    passed: list[str] = []
    rejected: dict[str, list[str]] = {}
    table = snapshot.derived_table()
    for instrument in snapshot.instruments:
        key = instrument["instrument_key"]
        reasons = evaluate_row(table.get(key, {}), config)
        if reasons:
            rejected[key] = reasons
        else:
            passed.append(key)

    # The ordering is what selects: far more instruments pass the filters than fit the cap, so this
    # sort decides the shortlist. Composite by default; the single-factor ordering stays available
    # because every run before this change used it and their configs must still mean what they meant.
    if config.ranking == "composite":
        scores = composite_scores(table)

        def order(key: str) -> tuple[float, str]:
            """Sort by composite percentile descending, then instrument key for determinism."""
            return (-scores.get(key, NEUTRAL_PERCENTILE), key)
    else:
        def order(key: str) -> tuple[float, str]:
            """Sort by return on capital descending (missing last), then key for determinism."""
            value = return_on_capital(table[key])
            return (-(value if value is not None else float("-inf")), key)

    passed.sort(key=order)
    return ScreenResult(
        snapshot_hash=snapshot.root_hash,
        config_hash=config.config_hash(),
        candidates=passed[: config.max_candidates],
        rejected=rejected,
        truncated=passed[config.max_candidates :],
    )
