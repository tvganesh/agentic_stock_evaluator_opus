"""The derived-column catalogue: the shared vocabulary of screens, prompts and falsifiers.

Every number the system reasons about is a column in a snapshot's derived table, computed
in phase 1 by deterministic Python (``sealed_window.snapshot.indicators``). This catalogue
is what ties the phases together:

* the screen (phase 2) filters on these columns;
* agent prompts (phase 3) list them, so models write falsifiers in this vocabulary;
* the falsifier DSL (phase 4) accepts only these identifiers -- an unknown column makes a
  claim unevaluable, and unevaluable claims are discarded;
* freshness SLAs are declared per dimension here, so stale data forces abstention.

Changing a column's meaning requires bumping :data:`DERIVED_COLUMNS_VERSION`, which changes
the query-set hash and therefore every snapshot hash built afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

DERIVED_COLUMNS_VERSION = "derived-v3"
"""v2 (15 Sep 2026): removed quarterly year-on-year growth (Upstox returns only four quarters, so it
could never be computed) and added bank ratios (NIM, Net NPA, CASA, ``is_bank``), which Upstox
reports for banks in place of ROCE and EV/EBITDA.
v3 (15 Sep 2026): added sector P/B, ROA and EV/EBITDA benchmarks (the first Claude run reached for a
sector P/B column that did not exist), and renamed ``news_count_30d`` to ``news_count_window``
because Upstox returns only about a week of headlines and Claude read the old name literally.
The sealed process refuses snapshots built with any other version."""


class Dimension(str, Enum):
    """The three analysis dimensions; each claim, column and agent belongs to exactly one."""

    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    NEWS = "news"


@dataclass(frozen=True)
class Column:
    """One derived column: its name, dimension, unit and a precise one-line definition."""

    name: str
    dimension: Dimension
    unit: str
    description: str


def _columns(dimension: Dimension, rows: list[tuple[str, str, str]]) -> dict[str, Column]:
    """Build ``{name: Column}`` for one dimension from ``(name, unit, description)`` rows."""
    return {name: Column(name, dimension, unit, description) for name, unit, description in rows}


_TECHNICAL = _columns(
    Dimension.TECHNICAL,
    [
        ("close", "INR", "Latest daily close."),
        ("sma_20", "INR", "20-session simple moving average of close."),
        ("sma_50", "INR", "50-session simple moving average of close."),
        ("sma_200", "INR", "200-session simple moving average of close."),
        ("price_vs_sma20_pct", "%", "(close / sma_20 - 1) * 100."),
        ("price_vs_sma50_pct", "%", "(close / sma_50 - 1) * 100."),
        ("price_vs_sma200_pct", "%", "(close / sma_200 - 1) * 100."),
        ("rsi_14", "0-100", "14-session RSI with Wilder smoothing."),
        ("macd", "INR", "EMA(12) - EMA(26) of close."),
        ("macd_signal", "INR", "9-session EMA of macd."),
        ("macd_hist", "INR", "macd - macd_signal."),
        ("atr_14", "INR", "14-session average true range with Wilder smoothing."),
        ("atr_pct", "%", "atr_14 / close * 100."),
        ("volatility_20d_ann_pct", "%", "Annualised sample stdev of the last 20 daily log returns * 100."),
        ("volume_ratio_20d", "x", "Mean volume of the last 5 sessions / mean volume of the last 20 sessions."),
        ("return_5d_pct", "%", "Close-to-close return over 5 sessions."),
        ("return_30d_pct", "%", "Close-to-close return over 21 sessions (~30 calendar days)."),
        ("return_90d_pct", "%", "Close-to-close return over 63 sessions (~90 calendar days)."),
        ("return_1y_pct", "%", "Close-to-close return over 250 sessions."),
        ("drawdown_from_52w_high_pct", "%", "(close / max close of last 250 sessions - 1) * 100; always <= 0."),
        ("last_candle_age_days", "days", "Calendar days between the last candle and the snapshot as_of."),
    ],
)

_FUNDAMENTAL = _columns(
    Dimension.FUNDAMENTAL,
    [
        ("pe", "x", "Price / earnings (Upstox key ratios)."),
        ("pb", "x", "Price / book (Upstox key ratios)."),
        ("roe_pct", "%", "Return on equity (Upstox key ratios)."),
        ("roa_pct", "%", "Return on assets (Upstox key ratios)."),
        ("roce_pct", "%", "Return on capital employed (Upstox key ratios); used as the ROIC proxy."),
        ("ev_ebitda", "x", "Enterprise value / EBITDA (Upstox key ratios)."),
        ("sector_pe", "x", "Sector benchmark P/E."),
        ("sector_roe_pct", "%", "Sector benchmark ROE."),
        ("sector_roce_pct", "%", "Sector benchmark ROCE."),
        ("sector_pb", "x", "Sector benchmark P/B."),
        ("sector_roa_pct", "%", "Sector benchmark ROA."),
        ("sector_ev_ebitda", "x", "Sector benchmark EV/EBITDA."),
        ("revenue_growth_1y_pct", "%", "Latest fiscal-year revenue vs prior year, consolidated."),
        ("net_profit_growth_1y_pct", "%", "Latest fiscal-year net profit vs prior year; null if prior year <= 0."),
        ("operating_margin_pct", "%", "Latest fiscal-year operating profit / revenue * 100."),
        ("operating_margin_delta_1y_pp", "pp", "operating_margin_pct minus the prior year's margin."),
        ("is_bank", "flag", "1 if Upstox reports bank ratios (NIM or Net NPA) instead of ROCE, 0 otherwise."),
        ("nim_pct", "%", "Net interest margin (banks; Upstox key ratios)."),
        ("sector_nim_pct", "%", "Sector benchmark net interest margin."),
        ("net_npa_pct", "%", "Net non-performing assets as % of advances (banks); lower is better."),
        ("sector_net_npa_pct", "%", "Sector benchmark net NPA."),
        ("casa_pct", "%", "Current and savings deposits as % of total deposits (banks)."),
        ("sector_casa_pct", "%", "Sector benchmark CASA ratio."),
        ("liabilities_to_equity", "x", "Total liabilities / (total assets - total liabilities), latest year."),
        ("liabilities_to_equity_delta_1y", "x", "liabilities_to_equity minus the prior year's value."),
        ("fundamentals_age_days", "days", "Calendar days between the latest reported period end and as_of."),
    ],
)

_NEWS = _columns(
    Dimension.NEWS,
    [
        ("news_count_7d", "count", "Headlines published in the 7 days up to and including as_of."),
        ("news_count_window", "count",
         "Headlines in the acquired news window ending at as_of (30 days requested; Upstox currently "
         "returns only about the last 7 days, so zero does not mean a month without news)."),
    ],
)

COLUMNS: Mapping[str, Column] = MappingProxyType({**_TECHNICAL, **_FUNDAMENTAL, **_NEWS})
"""Every column a screen, prompt or falsifier may reference."""

FRESHNESS_SLA: Mapping[Dimension, tuple[str, float] | None] = MappingProxyType(
    {
        Dimension.TECHNICAL: ("last_candle_age_days", 5.0),
        Dimension.FUNDAMENTAL: ("fundamentals_age_days", 400.0),
        Dimension.NEWS: None,
    }
)
"""Per-dimension ``(age column, max days)``: a falsifier touching a stale dimension is unevaluable."""


def columns_for(dimension: Dimension) -> list[Column]:
    """Return the catalogue entries for one dimension, in declaration order (used by prompts)."""
    return [column for column in COLUMNS.values() if column.dimension is dimension]
