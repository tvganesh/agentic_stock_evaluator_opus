"""Derived-indicator engine (build-order phase P2): every number, computed before any model exists.

It lives in ``snapshot`` rather than ``acquire`` because it is pure arithmetic with no network,
credential or vendor dependency, and both process roles need it: the ETL computes a snapshot's
columns with it, and the sealed walk-forward backtest recomputes them at past dates. The ``acquire``
package stays import-forbidden in the sealed role.

"Every number the system will ever reason about is computed in phase 1" -- this module is
where that happens. Functions are pure, dependency-free and deterministic; outputs are
rounded with :func:`stable_float` so snapshot hashes are reproducible. Column definitions
match ``sealed_window.snapshot.columns`` exactly; the catalogue is the specification and
this file is the implementation.

Three row builders feed ``derived.json``:

* :func:`technical_row`   from daily candles  -> technical columns
* :func:`fundamental_row` from key ratios, income statements and balance sheet
* :func:`news_row`        from normalised headlines

Missing or insufficient data yields ``None`` for that column (never a guess). The screen
treats ``None`` as failing any active filter; the adjudicator treats a falsifier over a
``None`` column as unevaluable, which discards the claim.
"""

from __future__ import annotations

import calendar
import math
from datetime import date, timedelta
from typing import Any, Sequence

from .columns import COLUMNS, Dimension, columns_for
from .hashing import stable_float

# --------------------------------------------------------------------------------------
# Generic numeric helpers
# --------------------------------------------------------------------------------------


def sma(values: Sequence[float], window: int) -> float | None:
    """Simple moving average of the last ``window`` values, or ``None`` if too short."""
    if window <= 0 or len(values) < window:
        return None
    return sum(values[-window:]) / window


def ema_series(values: Sequence[float], window: int) -> list[float]:
    """Exponential moving average series seeded with the first ``window``-value SMA.

    ``result[i]`` corresponds to ``values[window - 1 + i]``; empty if there is too little data.
    """
    if window <= 0 or len(values) < window:
        return []
    k = 2.0 / (window + 1)
    out = [sum(values[:window]) / window]
    for value in values[window:]:
        out.append(value * k + out[-1] * (1 - k))
    return out


def rsi_wilder(closes: Sequence[float], period: int = 14) -> float | None:
    """Relative Strength Index with Wilder smoothing over the full series (column ``rsi_14``)."""
    if len(closes) < period + 1:
        return None
    diffs = [b - a for a, b in zip(closes[:-1], closes[1:])]
    avg_gain = sum(max(d, 0.0) for d in diffs[:period]) / period
    avg_loss = sum(max(-d, 0.0) for d in diffs[:period]) / period
    for d in diffs[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def macd(
    closes: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float | None, float | None, float | None]:
    """MACD line, signal line and histogram (columns ``macd``, ``macd_signal``, ``macd_hist``)."""
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = ema_series(closes, fast)
    ema_slow = ema_series(closes, slow)
    line = [f - s for f, s in zip(ema_fast[slow - fast :], ema_slow)]
    signal_line = ema_series(line, signal)
    if not signal_line:
        return None, None, None
    return line[-1], signal_line[-1], line[-1] - signal_line[-1]


def atr_wilder(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14
) -> float | None:
    """Average True Range with Wilder smoothing (column ``atr_14``)."""
    if len(closes) < period + 1:
        return None
    true_ranges = [
        max(h - l, abs(h - prev_close), abs(l - prev_close))
        for h, l, prev_close in zip(highs[1:], lows[1:], closes[:-1])
    ]
    atr = sum(true_ranges[:period]) / period
    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def annualised_volatility_pct(closes: Sequence[float], window: int = 20) -> float | None:
    """Annualised sample stdev of the last ``window`` daily log returns, in percent."""
    if len(closes) < window + 1:
        return None
    tail = closes[-(window + 1) :]
    if any(c <= 0 for c in tail):
        return None
    returns = [math.log(b / a) for a, b in zip(tail[:-1], tail[1:])]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(252) * 100.0


def pct_return(closes: Sequence[float], sessions: int) -> float | None:
    """Close-to-close percentage return over ``sessions`` sessions."""
    if len(closes) <= sessions or closes[-1 - sessions] <= 0:
        return None
    return (closes[-1] / closes[-1 - sessions] - 1.0) * 100.0


def volume_ratio(volumes: Sequence[float], short: int = 5, long: int = 20) -> float | None:
    """Mean volume of the last ``short`` sessions over the mean of the last ``long`` sessions."""
    short_avg, long_avg = sma(volumes, short), sma(volumes, long)
    if short_avg is None or not long_avg:
        return None
    return short_avg / long_avg


def drawdown_from_high_pct(closes: Sequence[float], window: int = 250) -> float | None:
    """Percentage below the highest close of the last ``window`` sessions (<= 0)."""
    if not closes:
        return None
    high = max(closes[-window:])
    return None if high <= 0 else (closes[-1] / high - 1.0) * 100.0


def pct_vs(value: float | None, reference: float | None) -> float | None:
    """``(value / reference - 1) * 100``, or ``None`` when either side is missing or zero."""
    if value is None or not reference:
        return None
    return (value / reference - 1.0) * 100.0


def parse_number(raw: Any) -> float | None:
    """Parse vendor numbers like ``"8.94%"``, ``"1,234.5"`` or ``20.15``; ``None`` for blanks/dashes."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if math.isfinite(raw) else None
    text = str(raw).strip().replace(",", "").rstrip("%").strip()
    if text in ("", "-", "--", "NA", "N/A", "null"):
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_period(label: str) -> date | None:
    """Turn a reporting period label like ``"Mar 2026"`` into its month-end date."""
    try:
        month_name, year_text = label.strip().split()
        month = list(calendar.month_abbr).index(month_name[:3].title())
        year = int(year_text)
    except (ValueError, AttributeError):
        return None
    if month == 0:
        return None
    return date(year, month, calendar.monthrange(year, month)[1])


def _empty_row(dimension: Dimension) -> dict[str, float | None]:
    """A row with every column of ``dimension`` set to ``None``."""
    return {column.name: None for column in columns_for(dimension)}


def _rounded(row: dict[str, float | None]) -> dict[str, float | None]:
    """Round every value for byte-stable storage and assert the row matches the catalogue."""
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"indicator row has columns missing from the catalogue: {sorted(unknown)}")
    return {name: stable_float(value) for name, value in row.items()}


# --------------------------------------------------------------------------------------
# Row builders
# --------------------------------------------------------------------------------------


def technical_row(candles: Sequence[Sequence[Any]], as_of: date) -> dict[str, float | None]:
    """Compute all technical columns from ascending ``[date, open, high, low, close, volume]`` rows."""
    row = _empty_row(Dimension.TECHNICAL)
    if not candles:
        return row
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]
    volumes = [float(c[5]) for c in candles]
    close = closes[-1]
    sma20, sma50, sma200 = sma(closes, 20), sma(closes, 50), sma(closes, 200)
    macd_line, macd_signal, macd_hist = macd(closes)
    atr = atr_wilder(highs, lows, closes)
    last_date = date.fromisoformat(str(candles[-1][0])[:10])
    row.update(
        close=close,
        sma_20=sma20,
        sma_50=sma50,
        sma_200=sma200,
        price_vs_sma20_pct=pct_vs(close, sma20),
        price_vs_sma50_pct=pct_vs(close, sma50),
        price_vs_sma200_pct=pct_vs(close, sma200),
        rsi_14=rsi_wilder(closes),
        macd=macd_line,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        atr_14=atr,
        atr_pct=None if atr is None or close <= 0 else atr / close * 100.0,
        volatility_20d_ann_pct=annualised_volatility_pct(closes),
        volume_ratio_20d=volume_ratio(volumes),
        return_5d_pct=pct_return(closes, 5),
        return_30d_pct=pct_return(closes, 21),
        return_90d_pct=pct_return(closes, 63),
        return_1y_pct=pct_return(closes, 250),
        drawdown_from_52w_high_pct=drawdown_from_high_pct(closes),
        last_candle_age_days=float((as_of - last_date).days),
    )
    return _rounded(row)


def statement_series(statement: dict[str, Any] | None, category: str) -> list[tuple[date, float]]:
    """Extract ``(period_end, value)`` pairs for one income-statement category, newest first."""
    if not statement:
        return []
    for block in statement.get("income_statement", []) or []:
        if block.get("category") != category:
            continue
        pairs = []
        for item in block.get("history", []) or []:
            period, value = parse_period(str(item.get("period", ""))), parse_number(item.get("value"))
            if period is not None and value is not None:
                pairs.append((period, value))
        return sorted(pairs, reverse=True)
    return []


def _growth(latest: float, base: float) -> float | None:
    """Percentage growth from ``base`` to ``latest``; ``None`` when the base is not positive."""
    return None if base <= 0 else (latest / base - 1.0) * 100.0


def fundamental_row(
    key_ratios: list[dict[str, Any]] | None,
    income_yearly: dict[str, Any] | None,
    income_quarterly: dict[str, Any] | None,
    balance_sheet: dict[str, Any] | None,
    as_of: date,
) -> dict[str, float | None]:
    """Compute all fundamental columns from Upstox fundamentals payloads (any may be missing)."""
    row = _empty_row(Dimension.FUNDAMENTAL)
    ratios = {str(item.get("name", "")).upper(): item for item in key_ratios or []}

    def ratio(name: str, side: str = "company_value") -> float | None:
        """Look up one key ratio's company or sector value."""
        return parse_number(ratios.get(name, {}).get(side))

    row.update(
        pe=ratio("P/E"),
        pb=ratio("P/B"),
        roe_pct=ratio("ROE"),
        roa_pct=ratio("ROA"),
        roce_pct=ratio("ROCE"),
        ev_ebitda=ratio("EV/EBITDA"),
        sector_pe=ratio("P/E", "sector_value"),
        sector_roe_pct=ratio("ROE", "sector_value"),
        sector_roce_pct=ratio("ROCE", "sector_value"),
        sector_pb=ratio("P/B", "sector_value"),
        sector_roa_pct=ratio("ROA", "sector_value"),
        sector_ev_ebitda=ratio("EV/EBITDA", "sector_value"),
        nim_pct=ratio("NIM"),
        sector_nim_pct=ratio("NIM", "sector_value"),
        net_npa_pct=ratio("NET NPA"),
        sector_net_npa_pct=ratio("NET NPA", "sector_value"),
        casa_pct=ratio("CASA"),
        sector_casa_pct=ratio("CASA", "sector_value"),
    )
    if key_ratios:
        row["is_bank"] = 1.0 if ("NIM" in ratios or "NET NPA" in ratios) else 0.0

    period_ends: list[date] = []
    revenue = statement_series(income_yearly, "revenue")
    profit = statement_series(income_yearly, "net_profit")
    operating = statement_series(income_yearly, "operating_profit")
    if len(revenue) >= 2:
        row["revenue_growth_1y_pct"] = _growth(revenue[0][1], revenue[1][1])
    if len(profit) >= 2:
        row["net_profit_growth_1y_pct"] = _growth(profit[0][1], profit[1][1])
    margins = {
        period: op / rev * 100.0
        for (period, rev), (op_period, op) in zip(revenue, operating)
        if period == op_period and rev > 0
    }
    margin_dates = sorted(margins, reverse=True)
    if margin_dates:
        row["operating_margin_pct"] = margins[margin_dates[0]]
    if len(margin_dates) >= 2:
        row["operating_margin_delta_1y_pp"] = margins[margin_dates[0]] - margins[margin_dates[1]]
    period_ends += [p for p, _ in revenue[:1]]

    # Quarterly statements only date the latest report; Upstox's four quarters cannot give year-on-year growth.
    q_revenue = statement_series(income_quarterly, "revenue")
    period_ends += [p for p, _ in q_revenue[:1]]

    leverage: list[tuple[date, float]] = []
    for item in (balance_sheet or {}).get("history", []) or []:
        period = parse_period(str(item.get("period", "")))
        assets, liabilities = parse_number(item.get("total_asset")), parse_number(item.get("total_liability"))
        if period is None or assets is None or liabilities is None or assets - liabilities <= 0:
            continue
        leverage.append((period, liabilities / (assets - liabilities)))
    leverage.sort(reverse=True)
    if leverage:
        row["liabilities_to_equity"] = leverage[0][1]
        period_ends.append(leverage[0][0])
    if len(leverage) >= 2:
        row["liabilities_to_equity_delta_1y"] = leverage[0][1] - leverage[1][1]

    past_period_ends = [p for p in period_ends if p <= as_of]
    if past_period_ends:
        row["fundamentals_age_days"] = float((as_of - max(past_period_ends)).days)
    return _rounded(row)


def news_row(articles: Sequence[dict[str, Any]], as_of: date) -> dict[str, float | None]:
    """Count headlines in the 7- and 30-day windows ending at ``as_of`` (future dates excluded)."""
    row = _empty_row(Dimension.NEWS)
    dates = [date.fromisoformat(a["published_date"]) for a in articles if a.get("published_date")]
    row["news_count_7d"] = float(sum(1 for d in dates if as_of - timedelta(days=6) <= d <= as_of))
    row["news_count_window"] = float(sum(1 for d in dates if as_of - timedelta(days=29) <= d <= as_of))
    return _rounded(row)
