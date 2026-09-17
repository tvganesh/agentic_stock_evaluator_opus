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
from typing import Callable

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
    """ROCE where reported, otherwise ROE (banks); used to order candidates before truncation."""
    roce = row.get("roce_pct")
    return roce if roce is not None else row.get("roe_pct")


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

    def order(key: str) -> tuple[float, str]:
        """Sort by return on capital descending (missing last), then instrument key for determinism."""
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
