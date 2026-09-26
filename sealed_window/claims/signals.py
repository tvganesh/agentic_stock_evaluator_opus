"""The signal table: every indicator's rule for, rule against, falsifier and worked examples.

One table, read by everything that needs to know what an indicator means:

* the claim prompts render it (``agents.prompts``), so a model is told, for each indicator, which
  reading supports owning the stock, which undermines it, and the exact falsifier for each;
* the tests prove it (``tests/test_signal_table.py``): every falsifier is the exact negation of its
  rule over the whole value space, every column is in the catalogue, and every example's numbers
  satisfy its rule.

Why it exists
-------------
qwen3:14b, given a checklist of conditions to check (26 Sep 2026, run 20260926T111431Z-1ff470), wrote
"cheap on earnings" for 7 of the 8 shortlisted stocks whose P/E was *above* their sector's. It chose a
direction and wrote the sentence before it had written any number -- ``ClaimDraft`` puts ``direction``
and ``statement`` ahead of ``falsifier`` -- so the comparison never happened. Where it did reach the
falsifier, it sometimes wrote it backwards (ACUTAAS: "P/E below sector", falsifier ``pe <= sector_pe``
with P/E 70.7 against 6.0, which survived), or joined the parts of a compound condition with AND where
the negation needs OR. The table fixes the part a prompt can fix: the rule and its falsifier are written
once, here, correctly, and the model is asked to copy the falsifier rather than compose one.

Comparability guards are folded into the falsifier. A sector P/E of -142.9 (LLOYDSME's, from Upstox)
makes "above the sector" meaningless, so "expensive on earnings" is also wrong when ``sector_pe <= 0``:
a claim made against a meaningless benchmark is refuted rather than counted.

The examples are rendered from real companies' readings in the snapshot of 17 Sep 2026, chosen from
outside that day's 15-stock shortlist, and state the figures before the conclusion -- the order that
kept Claude's claims consistent with their data. They are frozen text, like the rest of the system
prompt: an example's figures belong to its company on that date and are never evidence for another.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..snapshot.columns import Dimension

POSITIVE = "positive"
NEGATIVE = "negative"


@dataclass(frozen=True)
class Example:
    """One worked example: a real company's readings on 17 Sep 2026 that satisfy a rule."""

    symbol: str
    values: Mapping[str, float]


@dataclass(frozen=True)
class Rule:
    """One reading of an indicator: the condition under which it holds, its direction, and how to say it.

    ``falsifier`` is TRUE exactly when ``condition`` (together with the signal's guard) is FALSE; the
    test suite proves this over the value space rather than trusting the text.
    """

    key: str
    direction: str
    label: str
    condition: str
    falsifier: str
    template: str
    """Statement text with ``{symbol}`` and ``{column}`` placeholders; figures come before the conclusion."""
    magnitude: bool = False
    """Render figures unsigned, for templates that carry the sign in words ("7.03% below")."""

    def render(self, example: Example) -> str:
        """The example's statement, with its figures formatted to two decimals."""
        figures = {name: f"{abs(value) if self.magnitude else value:.2f}" for name, value in example.values.items()}
        return self.template.format(symbol=example.symbol, **figures)


@dataclass(frozen=True)
class Signal:
    """One indicator: what it measures, when it is comparable, and its rules for and against."""

    key: str
    dimension: Dimension
    group: str
    """Signals that describe the same thing share a group, and a group counts once per side in ranking."""
    title: str
    meaning: str
    rules: tuple[Rule, ...]
    guard: str | None = None
    """Condition under which the comparison means anything; folded into every rule's falsifier."""
    guard_note: str | None = None

    def full_condition(self, rule: Rule) -> str:
        """``rule.condition`` joined with the guard: the exact predicate ``rule.falsifier`` negates."""
        return rule.condition if self.guard is None else f"({self.guard}) AND ({rule.condition})"


def _ex(symbol: str, **values: float) -> Example:
    """Shorthand for an :class:`Example`."""
    return Example(symbol, MappingProxyType(values))


# ---- fundamental -------------------------------------------------------------------------------

_FUNDAMENTAL: tuple[Signal, ...] = (
    Signal(
        "pe", Dimension.FUNDAMENTAL, "valuation", "Valuation on earnings (P/E)",
        "Price paid per rupee of earnings, against the sector. Lower is cheaper.",
        guard="pe > 0 AND sector_pe > 0",
        guard_note="only when pe and sector_pe are both above 0",
        rules=(
            Rule("pe.cheap", POSITIVE, "cheap on earnings", "pe < sector_pe",
                 "pe >= sector_pe OR pe <= 0 OR sector_pe <= 0",
                 "{symbol} trades at a P/E of {pe}x against a sector P/E of {sector_pe}x, so it is cheap on earnings."),
            Rule("pe.expensive", NEGATIVE, "expensive on earnings", "pe > sector_pe",
                 "pe <= sector_pe OR pe <= 0 OR sector_pe <= 0",
                 "{symbol} trades at a P/E of {pe}x against a sector P/E of {sector_pe}x, so it is expensive on earnings."),
        ),
    ),
    Signal(
        "ev_ebitda", Dimension.FUNDAMENTAL, "valuation", "Valuation on cash profit (EV/EBITDA)",
        "Enterprise value per rupee of operating cash profit, against the sector. Lower is cheaper.",
        guard="ev_ebitda > 0 AND sector_ev_ebitda > 0",
        guard_note="only when ev_ebitda and sector_ev_ebitda are both above 0",
        rules=(
            Rule("ev_ebitda.cheap", POSITIVE, "cheap on cash profit", "ev_ebitda < sector_ev_ebitda",
                 "ev_ebitda >= sector_ev_ebitda OR ev_ebitda <= 0 OR sector_ev_ebitda <= 0",
                 "{symbol} trades at an EV/EBITDA of {ev_ebitda}x against a sector {sector_ev_ebitda}x, so it is cheap on cash profit."),
            Rule("ev_ebitda.expensive", NEGATIVE, "expensive on cash profit", "ev_ebitda > sector_ev_ebitda",
                 "ev_ebitda <= sector_ev_ebitda OR ev_ebitda <= 0 OR sector_ev_ebitda <= 0",
                 "{symbol} trades at an EV/EBITDA of {ev_ebitda}x against a sector {sector_ev_ebitda}x, so it is expensive on cash profit."),
        ),
    ),
    Signal(
        "pb", Dimension.FUNDAMENTAL, "valuation", "Valuation on book (P/B)",
        "Price paid per rupee of book value, against the sector. Lower is cheaper.",
        guard="pb > 0 AND sector_pb > 0",
        guard_note="only when pb and sector_pb are both above 0",
        rules=(
            Rule("pb.cheap", POSITIVE, "cheap on book value", "pb < sector_pb",
                 "pb >= sector_pb OR pb <= 0 OR sector_pb <= 0",
                 "{symbol} trades at a P/B of {pb}x against a sector P/B of {sector_pb}x, so it is cheap on book value."),
            Rule("pb.expensive", NEGATIVE, "expensive on book value", "pb > sector_pb",
                 "pb <= sector_pb OR pb <= 0 OR sector_pb <= 0",
                 "{symbol} trades at a P/B of {pb}x against a sector P/B of {sector_pb}x, so it is expensive on book value."),
        ),
    ),
    Signal(
        "roce", Dimension.FUNDAMENTAL, "returns_on_capital", "Return on capital employed (ROCE)",
        "Operating profit per rupee of all capital, debt and equity, against the sector. Higher is better.",
        guard="is_bank == 0 AND sector_roce_pct > 0",
        guard_note="not for banks, and only when sector_roce_pct is above 0",
        rules=(
            Rule("roce.above", POSITIVE, "returns on capital above the sector", "roce_pct > sector_roce_pct",
                 "roce_pct <= sector_roce_pct OR is_bank == 1 OR sector_roce_pct <= 0",
                 "{symbol} earns a ROCE of {roce_pct}% against a sector {sector_roce_pct}%, above its peers."),
            Rule("roce.below", NEGATIVE, "returns on capital below the sector", "roce_pct < sector_roce_pct",
                 "roce_pct >= sector_roce_pct OR is_bank == 1 OR sector_roce_pct <= 0",
                 "{symbol} earns a ROCE of {roce_pct}% against a sector {sector_roce_pct}%, below its peers."),
        ),
    ),
    Signal(
        "roe", Dimension.FUNDAMENTAL, "returns_on_capital", "Return on equity (ROE)",
        "Profit per rupee of shareholders' money, against the sector. Higher is better.",
        guard="sector_roe_pct > 0",
        guard_note="only when sector_roe_pct is above 0",
        rules=(
            Rule("roe.above", POSITIVE, "returns to shareholders above the sector", "roe_pct > sector_roe_pct",
                 "roe_pct <= sector_roe_pct OR sector_roe_pct <= 0",
                 "{symbol} earns a ROE of {roe_pct}% against a sector {sector_roe_pct}%, above its peers."),
            Rule("roe.below", NEGATIVE, "returns to shareholders below the sector", "roe_pct < sector_roe_pct",
                 "roe_pct >= sector_roe_pct OR sector_roe_pct <= 0",
                 "{symbol} earns a ROE of {roe_pct}% against a sector {sector_roe_pct}%, below its peers."),
        ),
    ),
    Signal(
        "roa", Dimension.FUNDAMENTAL, "returns_on_capital", "Return on assets (ROA)",
        "Profit per rupee of assets, against the sector. Higher is better.",
        guard="sector_roa_pct > 0",
        guard_note="only when sector_roa_pct is above 0",
        rules=(
            Rule("roa.above", POSITIVE, "returns on assets above the sector", "roa_pct > sector_roa_pct",
                 "roa_pct <= sector_roa_pct OR sector_roa_pct <= 0",
                 "{symbol} earns a ROA of {roa_pct}% against a sector {sector_roa_pct}%, above its peers."),
            Rule("roa.below", NEGATIVE, "returns on assets below the sector", "roa_pct < sector_roa_pct",
                 "roa_pct >= sector_roa_pct OR sector_roa_pct <= 0",
                 "{symbol} earns a ROA of {roa_pct}% against a sector {sector_roa_pct}%, below its peers."),
        ),
    ),
    Signal(
        "revenue_growth", Dimension.FUNDAMENTAL, "growth", "Revenue growth",
        "Latest fiscal-year revenue against the year before.",
        rules=(
            Rule("revenue_growth.growing", POSITIVE, "revenue growing", "revenue_growth_1y_pct > 0",
                 "revenue_growth_1y_pct <= 0",
                 "{symbol} grew revenue {revenue_growth_1y_pct}% in the latest year, so the business is expanding."),
            Rule("revenue_growth.shrinking", NEGATIVE, "revenue shrinking", "revenue_growth_1y_pct < 0",
                 "revenue_growth_1y_pct >= 0",
                 "{symbol}'s revenue changed {revenue_growth_1y_pct}% in the latest year, so the business is shrinking."),
        ),
    ),
    Signal(
        "profit_growth", Dimension.FUNDAMENTAL, "growth", "Net profit growth",
        "Latest fiscal-year net profit against the year before (null when the prior year was a loss).",
        rules=(
            Rule("profit_growth.growing", POSITIVE, "profit growing", "net_profit_growth_1y_pct > 0",
                 "net_profit_growth_1y_pct <= 0",
                 "{symbol} grew net profit {net_profit_growth_1y_pct}% in the latest year."),
            Rule("profit_growth.shrinking", NEGATIVE, "profit shrinking", "net_profit_growth_1y_pct < 0",
                 "net_profit_growth_1y_pct >= 0",
                 "{symbol}'s net profit changed {net_profit_growth_1y_pct}% in the latest year, so profit is falling."),
        ),
    ),
    Signal(
        "margin_trend", Dimension.FUNDAMENTAL, "margins", "Operating margin trend",
        "Change in operating margin over the latest year, in percentage points.",
        rules=(
            Rule("margin_trend.improving", POSITIVE, "margins improving", "operating_margin_delta_1y_pp > 0",
                 "operating_margin_delta_1y_pp <= 0",
                 "{symbol}'s operating margin changed by {operating_margin_delta_1y_pp} percentage points over the year, so margins are improving."),
            Rule("margin_trend.falling", NEGATIVE, "margins falling", "operating_margin_delta_1y_pp < 0",
                 "operating_margin_delta_1y_pp >= 0",
                 "{symbol}'s operating margin changed by {operating_margin_delta_1y_pp} percentage points over the year, so margins are falling."),
        ),
    ),
    Signal(
        "leverage_trend", Dimension.FUNDAMENTAL, "leverage", "Leverage trend",
        "Change in liabilities-to-equity over the latest year. Falling is better.",
        guard="is_bank == 0",
        guard_note="not for banks, whose leverage is structurally high",
        rules=(
            Rule("leverage_trend.falling", POSITIVE, "leverage falling", "liabilities_to_equity_delta_1y < 0",
                 "liabilities_to_equity_delta_1y >= 0 OR is_bank == 1",
                 "{symbol}'s liabilities-to-equity changed by {liabilities_to_equity_delta_1y}x over the year, so leverage is falling."),
            Rule("leverage_trend.rising", NEGATIVE, "leverage rising", "liabilities_to_equity_delta_1y > 0",
                 "liabilities_to_equity_delta_1y <= 0 OR is_bank == 1",
                 "{symbol}'s liabilities-to-equity rose by {liabilities_to_equity_delta_1y}x over the year, so leverage is rising."),
        ),
    ),
    Signal(
        "bank_nim", Dimension.FUNDAMENTAL, "bank_quality", "Bank net interest margin (NIM)",
        "Lending margin, against the sector. Higher is better. Banks only.",
        guard="is_bank == 1",
        guard_note="banks only",
        rules=(
            Rule("bank_nim.above", POSITIVE, "lending margin above the sector", "nim_pct > sector_nim_pct",
                 "nim_pct <= sector_nim_pct OR is_bank == 0",
                 "{symbol} earns a net interest margin of {nim_pct}% against a sector {sector_nim_pct}%, above its peers."),
            Rule("bank_nim.below", NEGATIVE, "lending margin below the sector", "nim_pct < sector_nim_pct",
                 "nim_pct >= sector_nim_pct OR is_bank == 0",
                 "{symbol} earns a net interest margin of {nim_pct}% against a sector {sector_nim_pct}%, below its peers."),
        ),
    ),
    Signal(
        "bank_npa", Dimension.FUNDAMENTAL, "bank_quality", "Bank net non-performing assets (Net NPA)",
        "Bad loans as a share of advances, against the sector. Lower is better. Banks only.",
        guard="is_bank == 1",
        guard_note="banks only",
        rules=(
            Rule("bank_npa.better", POSITIVE, "fewer bad loans than the sector", "net_npa_pct < sector_net_npa_pct",
                 "net_npa_pct >= sector_net_npa_pct OR is_bank == 0",
                 "{symbol}'s net NPA is {net_npa_pct}% against a sector {sector_net_npa_pct}%, so its loan book is cleaner than its peers'."),
            Rule("bank_npa.worse", NEGATIVE, "more bad loans than the sector", "net_npa_pct > sector_net_npa_pct",
                 "net_npa_pct <= sector_net_npa_pct OR is_bank == 0",
                 "{symbol}'s net NPA is {net_npa_pct}% against a sector {sector_net_npa_pct}%, so its loan book is weaker than its peers'."),
        ),
    ),
    Signal(
        "bank_casa", Dimension.FUNDAMENTAL, "bank_quality", "Bank CASA ratio",
        "Cheap current and savings deposits as a share of all deposits, against the sector. Higher is better. "
        "Banks only.",
        guard="is_bank == 1",
        guard_note="banks only",
        rules=(
            Rule("bank_casa.above", POSITIVE, "cheaper deposit base than the sector", "casa_pct > sector_casa_pct",
                 "casa_pct <= sector_casa_pct OR is_bank == 0",
                 "{symbol}'s CASA ratio is {casa_pct}% against a sector {sector_casa_pct}%, a cheaper deposit base than its peers."),
            Rule("bank_casa.below", NEGATIVE, "costlier deposit base than the sector", "casa_pct < sector_casa_pct",
                 "casa_pct >= sector_casa_pct OR is_bank == 0",
                 "{symbol}'s CASA ratio is {casa_pct}% against a sector {sector_casa_pct}%, a costlier deposit base than its peers."),
        ),
    ),
)

# ---- technical ---------------------------------------------------------------------------------

_TECHNICAL: tuple[Signal, ...] = (
    Signal(
        "rsi", Dimension.TECHNICAL, "momentum", "RSI-14 zones",
        "Size of recent gains against recent losses over 14 sessions, 0-100. Zones: at or below 30 oversold; "
        "30-45 weak (a pullback if price_vs_sma200_pct is above 0, bearish if below); 45-55 neutral, make no "
        "claim; 55-70 bullish; at or above 70 overbought.",
        rules=(
            Rule("rsi.oversold", POSITIVE, "oversold, a possible rebound", "rsi_14 <= 30",
                 "rsi_14 > 30",
                 "{symbol}'s RSI-14 is {rsi_14}, at or below 30, so the stock is oversold and may rebound."),
            Rule("rsi.pullback", POSITIVE, "pullback in an uptrend",
                 "rsi_14 > 30 AND rsi_14 < 45 AND price_vs_sma200_pct > 0",
                 "rsi_14 <= 30 OR rsi_14 >= 45 OR price_vs_sma200_pct <= 0",
                 "{symbol}'s RSI-14 is {rsi_14} while the price is {price_vs_sma200_pct}% above its 200-day average, "
                 "a pullback within an uptrend."),
            Rule("rsi.bullish", POSITIVE, "bullish momentum", "rsi_14 >= 55 AND rsi_14 < 70",
                 "rsi_14 < 55 OR rsi_14 >= 70",
                 "{symbol}'s RSI-14 is {rsi_14}, between 55 and 70, so momentum is bullish without being overbought."),
            Rule("rsi.overbought", NEGATIVE, "overbought, at risk of a pullback", "rsi_14 >= 70",
                 "rsi_14 < 70",
                 "{symbol}'s RSI-14 is {rsi_14}, at or above 70, so the stock is overbought."),
            Rule("rsi.bearish", NEGATIVE, "weak momentum in a downtrend",
                 "rsi_14 > 30 AND rsi_14 < 45 AND price_vs_sma200_pct < 0",
                 "rsi_14 <= 30 OR rsi_14 >= 45 OR price_vs_sma200_pct >= 0",
                 "{symbol}'s RSI-14 is {rsi_14} while the price is {price_vs_sma200_pct}% below its 200-day average, "
                 "weak momentum in a downtrend.", magnitude=True),
        ),
    ),
    Signal(
        "trend_200", Dimension.TECHNICAL, "trend", "Long-term trend (200-day average)",
        "Price against its 200-session average. Above means a long-term uptrend.",
        rules=(
            Rule("trend_200.up", POSITIVE, "long-term uptrend", "price_vs_sma200_pct > 0",
                 "price_vs_sma200_pct <= 0",
                 "{symbol} trades {price_vs_sma200_pct}% above its 200-day average, a long-term uptrend."),
            Rule("trend_200.down", NEGATIVE, "long-term downtrend", "price_vs_sma200_pct < 0",
                 "price_vs_sma200_pct >= 0",
                 "{symbol} trades {price_vs_sma200_pct}% below its 200-day average, a long-term downtrend.", magnitude=True),
        ),
    ),
    Signal(
        "trend_50", Dimension.TECHNICAL, "trend", "Medium-term trend (50-day average)",
        "Price against its 50-session average.",
        rules=(
            Rule("trend_50.up", POSITIVE, "medium-term uptrend", "price_vs_sma50_pct > 0",
                 "price_vs_sma50_pct <= 0",
                 "{symbol} trades {price_vs_sma50_pct}% above its 50-day average, a medium-term uptrend."),
            Rule("trend_50.down", NEGATIVE, "medium-term downtrend", "price_vs_sma50_pct < 0",
                 "price_vs_sma50_pct >= 0",
                 "{symbol} trades {price_vs_sma50_pct}% below its 50-day average, a medium-term downtrend.", magnitude=True),
        ),
    ),
    Signal(
        "trend_20", Dimension.TECHNICAL, "trend", "Short-term trend (20-day average)",
        "Price against its 20-session average.",
        rules=(
            Rule("trend_20.up", POSITIVE, "short-term strength", "price_vs_sma20_pct > 0",
                 "price_vs_sma20_pct <= 0",
                 "{symbol} trades {price_vs_sma20_pct}% above its 20-day average, short-term strength."),
            Rule("trend_20.down", NEGATIVE, "short-term weakness", "price_vs_sma20_pct < 0",
                 "price_vs_sma20_pct >= 0",
                 "{symbol} trades {price_vs_sma20_pct}% below its 20-day average, short-term weakness.", magnitude=True),
        ),
    ),
    Signal(
        "macd_cross", Dimension.TECHNICAL, "momentum", "MACD against its signal line",
        "macd_hist is the MACD line minus its signal line. Above 0: the MACD line is above the signal line, "
        "momentum improving. Below 0: the signal line is above the MACD line, momentum fading.",
        rules=(
            Rule("macd_cross.bullish", POSITIVE, "MACD bullish", "macd_hist > 0",
                 "macd_hist <= 0",
                 "{symbol}'s MACD histogram is {macd_hist}, so the MACD line is above its signal line and momentum is improving."),
            Rule("macd_cross.bearish", NEGATIVE, "MACD bearish", "macd_hist < 0",
                 "macd_hist >= 0",
                 "{symbol}'s MACD histogram is {macd_hist}, so the signal line is above the MACD line and momentum is fading."),
        ),
    ),
    Signal(
        "macd_zero", Dimension.TECHNICAL, "trend", "MACD against zero",
        "macd is the 12-session average minus the 26-session average. Above 0 means the short average is above "
        "the long: an uptrend.",
        rules=(
            Rule("macd_zero.up", POSITIVE, "short average above long", "macd > 0",
                 "macd <= 0",
                 "{symbol}'s MACD line is {macd}, above zero, so its 12-day average is above its 26-day average."),
            Rule("macd_zero.down", NEGATIVE, "short average below long", "macd < 0",
                 "macd >= 0",
                 "{symbol}'s MACD line is {macd}, below zero, so its 12-day average is below its 26-day average."),
        ),
    ),
    Signal(
        "return_30d", Dimension.TECHNICAL, "price_returns", "30-day return",
        "Price change over about a month.",
        rules=(
            Rule("return_30d.up", POSITIVE, "rising over the month", "return_30d_pct > 0",
                 "return_30d_pct <= 0",
                 "{symbol} returned {return_30d_pct}% over the last 30 days."),
            Rule("return_30d.down", NEGATIVE, "falling over the month", "return_30d_pct < 0",
                 "return_30d_pct >= 0",
                 "{symbol} returned {return_30d_pct}% over the last 30 days, a decline."),
        ),
    ),
    Signal(
        "return_90d", Dimension.TECHNICAL, "price_returns", "90-day return",
        "Price change over about three months.",
        rules=(
            Rule("return_90d.up", POSITIVE, "rising over three months", "return_90d_pct > 0",
                 "return_90d_pct <= 0",
                 "{symbol} returned {return_90d_pct}% over the last 90 days."),
            Rule("return_90d.down", NEGATIVE, "falling over three months", "return_90d_pct < 0",
                 "return_90d_pct >= 0",
                 "{symbol} returned {return_90d_pct}% over the last 90 days, a decline."),
        ),
    ),
    Signal(
        "return_1y", Dimension.TECHNICAL, "price_returns", "1-year return",
        "Price change over about a year.",
        rules=(
            Rule("return_1y.up", POSITIVE, "rising over the year", "return_1y_pct > 0",
                 "return_1y_pct <= 0",
                 "{symbol} returned {return_1y_pct}% over the last year."),
            Rule("return_1y.down", NEGATIVE, "falling over the year", "return_1y_pct < 0",
                 "return_1y_pct >= 0",
                 "{symbol} returned {return_1y_pct}% over the last year, a decline."),
        ),
    ),
    Signal(
        "drawdown", Dimension.TECHNICAL, "risk", "Distance from the 52-week high",
        "How far below its highest close of the last year the stock is; always 0 or below. 0 to -10 near the "
        "high; -10 to -20 a correction, make no claim; below -20 significant damage.",
        rules=(
            Rule("drawdown.near_high", POSITIVE, "trading near its high", "drawdown_from_52w_high_pct > -10",
                 "drawdown_from_52w_high_pct <= -10",
                 "{symbol} is {drawdown_from_52w_high_pct}% below its 52-week high, trading near its peak.", magnitude=True),
            Rule("drawdown.damaged", NEGATIVE, "significant damage", "drawdown_from_52w_high_pct < -20",
                 "drawdown_from_52w_high_pct >= -20",
                 "{symbol} is {drawdown_from_52w_high_pct}% below its 52-week high, significant damage to the price.", magnitude=True),
        ),
    ),
    Signal(
        "volatility", Dimension.TECHNICAL, "risk", "Volatility",
        "Annualised swing of daily returns over 20 sessions. Below 20 calm; 20-35 typical for NSE, make no "
        "claim; above 35 high risk.",
        rules=(
            Rule("volatility.calm", POSITIVE, "calm trading", "volatility_20d_ann_pct < 20",
                 "volatility_20d_ann_pct >= 20",
                 "{symbol}'s 20-day annualised volatility is {volatility_20d_ann_pct}%, below 20, so the stock trades calmly."),
            Rule("volatility.high", NEGATIVE, "high volatility", "volatility_20d_ann_pct > 35",
                 "volatility_20d_ann_pct <= 35",
                 "{symbol}'s 20-day annualised volatility is {volatility_20d_ann_pct}%, above 35, a high-risk level."),
        ),
    ),
    Signal(
        "atr", Dimension.TECHNICAL, "risk", "Daily range (ATR %)",
        "Typical daily high-to-low range as a share of the price. Below 2 calm; 2-4 typical for NSE (the "
        "Nifty 500 median is about 2.8), make no claim; above 4 jumpy.",
        rules=(
            Rule("atr.calm", POSITIVE, "narrow daily range", "atr_pct < 2",
                 "atr_pct >= 2",
                 "{symbol}'s average true range is {atr_pct}% of its price, a narrow daily range."),
            Rule("atr.jumpy", NEGATIVE, "wide daily range", "atr_pct > 4",
                 "atr_pct <= 4",
                 "{symbol}'s average true range is {atr_pct}% of its price, a wide daily range."),
        ),
    ),
    Signal(
        "volume", Dimension.TECHNICAL, "volume", "Volume behind the move",
        "volume_ratio_20d is the last 5 sessions' average volume against the last 20 sessions'. Above 1 means "
        "rising participation. Read it with the direction of return_5d_pct.",
        rules=(
            Rule("volume.buying", POSITIVE, "a rise on rising volume", "volume_ratio_20d > 1 AND return_5d_pct > 0",
                 "volume_ratio_20d <= 1 OR return_5d_pct <= 0",
                 "{symbol} rose {return_5d_pct}% over 5 days on volume {volume_ratio_20d}x its 20-day average, "
                 "a rise with growing participation."),
            Rule("volume.thin_rally", NEGATIVE, "a rise on thinning volume",
                 "volume_ratio_20d < 1 AND return_5d_pct > 0",
                 "volume_ratio_20d >= 1 OR return_5d_pct <= 0",
                 "{symbol} rose {return_5d_pct}% over 5 days on volume only {volume_ratio_20d}x its 20-day average, "
                 "a rise with thinning participation."),
            Rule("volume.selling", NEGATIVE, "a fall on rising volume",
                 "volume_ratio_20d > 1 AND return_5d_pct < 0",
                 "volume_ratio_20d <= 1 OR return_5d_pct >= 0",
                 "{symbol} fell {return_5d_pct}% over 5 days on volume {volume_ratio_20d}x its 20-day average, "
                 "selling with growing participation.", magnitude=True),
        ),
    ),
)

SIGNALS: tuple[Signal, ...] = _FUNDAMENTAL + _TECHNICAL
"""Every signal, fundamental first."""


def signals_for(dimension: Dimension) -> tuple[Signal, ...]:
    """The signals an analyst of ``dimension`` is given, in table order."""
    return tuple(signal for signal in SIGNALS if signal.dimension is dimension)


def rules() -> tuple[Rule, ...]:
    """Every rule in the table, in order."""
    return tuple(rule for signal in SIGNALS for rule in signal.rules)


# ---- worked examples: real readings from the snapshot of 17 Sep 2026 (528d7570), off that day's shortlist ----
# Generated by picking, per rule, stocks that satisfy the full condition clearly (not on its edge) with
# readings inside the universe's 1st-99th percentile, no stock used more than twice. Frozen: regenerating
# them would change every claim prompt.

_EXAMPLES: Mapping[str, tuple[Example, ...]] = {
    'pe.cheap': (_ex('MAXHEALTH', pe=68.98, sector_pe=86.5), _ex('CHAMBLFERT', pe=8.5, sector_pe=71.68)),
    'pe.expensive': (_ex('GMRAIRPORT', pe=128.31, sector_pe=56.75), _ex('SBFC', pe=27.77, sector_pe=24.29)),
    'ev_ebitda.cheap': (_ex('INDIAMART', ev_ebitda=14.04, sector_ev_ebitda=46.06), _ex('TATACONSUM', ev_ebitda=31.45, sector_ev_ebitda=35.83)),
    'ev_ebitda.expensive': (_ex('INDHOTEL', ev_ebitda=26.57, sector_ev_ebitda=12.33), _ex('HYUNDAI', ev_ebitda=18.41, sector_ev_ebitda=2.53)),
    'pb.cheap': (_ex('FIRSTCRY', pb=1.84, sector_pb=2.65), _ex('BPCL', pb=1.31, sector_pb=1.43)),
    'pb.expensive': (_ex('PTCIL', pb=22.42, sector_pb=4.17), _ex('JSWINFRA', pb=4.35, sector_pb=3.28)),
    'roce.above': (_ex('ICICIGI', is_bank=0.0, roce_pct=18.79, sector_roce_pct=6.33), _ex('TVSMOTOR', is_bank=0.0, roce_pct=16.64, sector_roce_pct=13.35)),
    'roce.below': (_ex('PCBL', is_bank=0.0, roce_pct=7.55, sector_roce_pct=8.75), _ex('ANURAS', is_bank=0.0, roce_pct=7.68, sector_roce_pct=69.3)),
    'roe.above': (_ex('JKCEMENT', roe_pct=14.1, sector_roe_pct=11.4), _ex('FEDERALBNK', roe_pct=12.07, sector_roe_pct=8.84)),
    'roe.below': (_ex('JSL', roe_pct=16.14, sector_roe_pct=17.51), _ex('SAPPHIRE', roe_pct=-1.05, sector_roe_pct=6.3)),
    'roa.above': (_ex('HEROMOTOCO', roa_pct=16.41, sector_roa_pct=1.52), _ex('EIHOTEL', roa_pct=11.3, sector_roa_pct=3.66)),
    'roa.below': (_ex('BHEL', roa_pct=3.19, sector_roa_pct=5.29), _ex('CLEAN', roa_pct=13.06, sector_roa_pct=55.92)),
    'revenue_growth.growing': (_ex('COFORGE', revenue_growth_1y_pct=34.629024), _ex('GROWW', revenue_growth_1y_pct=19.039295)),
    'revenue_growth.shrinking': (_ex('ANGELONE', revenue_growth_1y_pct=-1.942776), _ex('PETRONET', revenue_growth_1y_pct=-14.290445)),
    'profit_growth.growing': (_ex('WIPRO', net_profit_growth_1y_pct=0.359358), _ex('IRCTC', net_profit_growth_1y_pct=5.973838)),
    'profit_growth.shrinking': (_ex('RVNL', net_profit_growth_1y_pct=-31.862043), _ex('MOTHERSON', net_profit_growth_1y_pct=-1.450901)),
    'margin_trend.improving': (_ex('FACT', operating_margin_delta_1y_pp=0.432022), _ex('UNIONBANK', operating_margin_delta_1y_pp=0.636491)),
    'margin_trend.falling': (_ex('NAM-INDIA', operating_margin_delta_1y_pp=-3.155074), _ex('JINDALSTEL', operating_margin_delta_1y_pp=-0.458154)),
    'leverage_trend.falling': (_ex('TMCV', is_bank=0.0, liabilities_to_equity_delta_1y=-0.340199), _ex('BBTC', is_bank=0.0, liabilities_to_equity_delta_1y=-0.113795)),
    'leverage_trend.rising': (_ex('ADANIPOWER', is_bank=0.0, liabilities_to_equity_delta_1y=0.184834), _ex('RVNL', is_bank=0.0, liabilities_to_equity_delta_1y=0.069958)),
    'bank_nim.above': (_ex('CENTRALBK', is_bank=1.0, nim_pct=2.78, sector_nim_pct=2.6), _ex('BANDHANBNK', is_bank=1.0, nim_pct=5.37, sector_nim_pct=4.18)),
    'bank_nim.below': (_ex('RBLBANK', is_bank=1.0, nim_pct=3.88, sector_nim_pct=4.18), _ex('SBIN', is_bank=1.0, nim_pct=2.42, sector_nim_pct=2.6)),
    'bank_npa.better': (_ex('PNB', is_bank=1.0, net_npa_pct=0.29, sector_net_npa_pct=0.41), _ex('KARURVYSYA', is_bank=1.0, net_npa_pct=0.19, sector_net_npa_pct=0.79)),
    'bank_npa.worse': (_ex('UNIONBANK', is_bank=1.0, net_npa_pct=0.48, sector_net_npa_pct=0.41), _ex('BANDHANBNK', is_bank=1.0, net_npa_pct=0.97, sector_net_npa_pct=0.79)),
    'bank_casa.above': (_ex('ICICIBANK', casa_pct=38.6, is_bank=1.0, sector_casa_pct=30.96), _ex('CENTRALBK', casa_pct=47.3, is_bank=1.0, sector_casa_pct=39.04)),
    'bank_casa.below': (_ex('KARURVYSYA', casa_pct=26.91, is_bank=1.0, sector_casa_pct=30.96), _ex('CANBK', casa_pct=29.84, is_bank=1.0, sector_casa_pct=39.04)),
    'rsi.oversold': (_ex('ZEEL', rsi_14=28.018133),),
    'rsi.pullback': (_ex('SCI', price_vs_sma200_pct=0.743825, rsi_14=34.37361),),
    'rsi.bullish': (_ex('FINCABLES', rsi_14=62.474852),),
    'rsi.overbought': (_ex('TATACHEM', rsi_14=77.097841),),
    'rsi.bearish': (_ex('LTM', price_vs_sma200_pct=-10.507768, rsi_14=39.971884),),
    'trend_200.up': (_ex('UNOMINDA', price_vs_sma200_pct=3.019013), _ex('MEDANTA', price_vs_sma200_pct=16.567969)),
    'trend_200.down': (_ex('MGL', price_vs_sma200_pct=-2.796572), _ex('HCLTECH', price_vs_sma200_pct=-8.532242)),
    'trend_50.up': (_ex('RBLBANK', price_vs_sma50_pct=5.023674), _ex('THELEELA', price_vs_sma50_pct=2.422673)),
    'trend_50.down': (_ex('NMDC', price_vs_sma50_pct=-4.448807), _ex('BHARTIARTL', price_vs_sma50_pct=-3.930177)),
    'trend_20.up': (_ex('CYIENT', price_vs_sma20_pct=0.536937), _ex('INOXWIND', price_vs_sma20_pct=0.473895)),
    'trend_20.down': (_ex('GICRE', price_vs_sma20_pct=-3.121882), _ex('AWL', price_vs_sma20_pct=-3.355564)),
    'macd_cross.bullish': (_ex('ANGELONE', macd_hist=1.196453), _ex('KEC', macd_hist=1.287517)),
    'macd_cross.bearish': (_ex('PARADEEP', macd_hist=-1.370583), _ex('KPITTECH', macd_hist=-1.427089)),
    'macd_zero.up': (_ex('LTFOODS', macd=2.681794), _ex('KPRMILL', macd=1.552282)),
    'macd_zero.down': (_ex('CHAMBLFERT', macd=-6.929297), _ex('NBCC', macd=-2.744432)),
    'return_30d.up': (_ex('TEGA', return_30d_pct=10.73303), _ex('BLUESTARCO', return_30d_pct=4.018754)),
    'return_30d.down': (_ex('AUBANK', return_30d_pct=-3.383843), _ex('POLYMED', return_30d_pct=-4.353673)),
    'return_90d.up': (_ex('HEXT', return_90d_pct=6.876268), _ex('FSL', return_90d_pct=13.791667)),
    'return_90d.down': (_ex('TRITURBINE', return_90d_pct=-17.87436), _ex('INDIGO', return_90d_pct=-3.328146)),
    'return_1y.up': (_ex('CARBORUNIV', return_1y_pct=9.143087), _ex('CGCL', return_1y_pct=41.018061)),
    'return_1y.down': (_ex('EMAMILTD', return_1y_pct=-37.519728), _ex('NHPC', return_1y_pct=-9.94012)),
    'drawdown.near_high': (_ex('GRAPHITE', drawdown_from_52w_high_pct=-3.616883), _ex('INDGN', drawdown_from_52w_high_pct=-3.253495)),
    'drawdown.damaged': (_ex('IRCTC', drawdown_from_52w_high_pct=-37.958186), _ex('NCC', drawdown_from_52w_high_pct=-37.508641)),
    'volatility.calm': (_ex('ONGC', volatility_20d_ann_pct=16.638198), _ex('MRF', volatility_20d_ann_pct=13.773894)),
    'volatility.high': (_ex('KIRLOSENG', volatility_20d_ann_pct=44.905956), _ex('FINCABLES', volatility_20d_ann_pct=56.65077)),
    'atr.calm': (_ex('EICHERMOT', atr_pct=1.781115), _ex('ACC', atr_pct=1.844975)),
    'atr.jumpy': (_ex('TEJASNET', atr_pct=5.035744), _ex('TEGA', atr_pct=4.383877)),
    'volume.buying': (_ex('CLEAN', return_5d_pct=0.278552, volume_ratio_20d=1.191672), _ex('MANKIND', return_5d_pct=1.545254, volume_ratio_20d=1.445356)),
    'volume.thin_rally': (_ex('SUNPHARMA', return_5d_pct=0.128693, volume_ratio_20d=0.93114),),
    'volume.selling': (_ex('SUZLON', return_5d_pct=-6.40264, volume_ratio_20d=1.17132),),
}
"""Worked examples per rule key; each one's values satisfy its rule, which the tests re-check."""


def examples_for(rule: Rule) -> tuple[Example, ...]:
    """The worked examples for ``rule`` (empty if none were found in the snapshot)."""
    return _EXAMPLES.get(rule.key, ())


# ---- lookups used after the model has answered ------------------------------------------------------

def _normalised(text: str) -> str:
    """A falsifier with whitespace collapsed and case folded, so copies match however they were spaced."""
    return " ".join(text.split()).lower()


_RULE_BY_FALSIFIER: Mapping[tuple[Dimension, str], tuple[Signal, Rule]] = MappingProxyType({
    (signal.dimension, _normalised(rule.falsifier)): (signal, rule) for signal in SIGNALS for rule in signal.rules
})


def rule_for_falsifier(falsifier: str, dimension: Dimension) -> tuple[Signal, Rule] | None:
    """The table rule whose falsifier ``falsifier`` copies, for an analyst of ``dimension``; else ``None``.

    Only an exact copy (up to spacing and case) counts. A falsifier the model composed itself, however
    similar, is the model's own and is left to it.
    """
    return _RULE_BY_FALSIFIER.get((dimension, _normalised(falsifier)))


def _column_groups() -> Mapping[tuple[Dimension, str], str]:
    """Each column that every rule of a signal tests, mapped to that signal's group.

    Columns a signal only sometimes tests (``price_vs_sma200_pct`` in two RSI zones) are left out, so they
    keep the group of the signal that is about them; ``is_bank`` is a guard, not a measure, and has none.
    """
    from . import dsl  # local: dsl is only needed to read column names out of the rules

    mapping: dict[tuple[Dimension, str], str] = {}
    for signal in SIGNALS:
        shared = set.intersection(*(dsl.columns(dsl.parse(rule.condition)) for rule in signal.rules))
        for column in shared - {"is_bank"}:
            mapping[(signal.dimension, column)] = signal.group
    return MappingProxyType(mapping)


_COLUMN_GROUP = _column_groups()


def group_for_falsifier(falsifier: str, dimension: Dimension) -> str | None:
    """The signal group a claim's falsifier tests, or ``None`` if it spans groups or leaves the table.

    A copied table falsifier takes its rule's group. Otherwise the falsifier's columns decide: if every
    column (``is_bank`` aside) belongs to one group, so does the claim -- which groups Claude's
    ``roe_pct < sector_roe_pct OR roce_pct < sector_roce_pct`` with the other returns on capital. A
    falsifier touching two groups, or a column the table does not cover, is not grouped and counts alone.
    """
    match = rule_for_falsifier(falsifier, dimension)
    if match is not None:
        return match[0].group
    from . import dsl

    try:
        names = dsl.columns(dsl.parse(falsifier)) - {"is_bank"}
    except dsl.FalsifierError:
        return None
    groups = {_COLUMN_GROUP.get((dimension, name)) for name in names}
    return groups.pop() if len(groups) == 1 and None not in groups else None
