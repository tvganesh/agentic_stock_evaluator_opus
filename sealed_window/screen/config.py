"""Screen configuration: the left-rail sliders as a validated, hashable object.

Two tabs, as in the application section of ARCHITECTURE_OPUS.md:

* Fundamental -- ROE floor, ROCE floor (ROIC proxy), leverage ceiling, revenue growth,
  earnings growth, P/E band.
* Technical -- RSI band, price vs MA20 / MA50, volume-ratio floor, ATR% ceiling, 30- and
  90-day return bands.

Every field is optional (``None`` = filter off). The config is frozen, rejects unknown
fields, and is identified by :meth:`ScreenConfig.config_hash`, the second element of the
reproducibility triple.

Banks: Upstox reports NIM, Net NPA and CASA for banks instead of ROCE, and any bank's
liabilities-to-equity is structurally near 10x. The ROCE floor and leverage ceiling therefore
apply to non-banks only, and the Net NPA ceiling applies to banks only (see ``screen.screen``).

Not yet available: the market-cap floor and promoter-pledge ceiling from the architecture
need the Upstox company-profile and share-holdings endpoints, which are not in the
allowlist. Adding them is a reviewed change to ``governance.policy`` plus new columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..snapshot.hashing import hash_object

SCREEN_CONFIG_VERSION = "screen-v3"
"""v3: composite percentile ordering by default. Bumped so identical slider values under different
semantics never share a config hash -- which also means runs recorded under v2 no longer reproduce
their config hash, and ``readjudicate`` will correctly refuse them. Their stored dossiers remain
readable; they simply cannot be rebuilt under rules they were not produced by."""


class ScreenConfig(BaseModel):
    """User screen settings. Frozen, strictly validated and content-hashed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ---- Fundamental tab ---------------------------------------------------------------
    roe_min_pct: float | None = Field(None, ge=-50, le=100, description="ROE floor (%)")
    roce_min_pct: float | None = Field(None, ge=-50, le=100, description="ROCE floor (%) - ROIC proxy; non-banks")
    liabilities_to_equity_max: float | None = Field(None, ge=0, le=20, description="Leverage ceiling (x); non-banks")
    net_npa_max_pct: float | None = Field(None, ge=0, le=20, description="Net NPA ceiling (%); banks only")
    revenue_growth_min_pct: float | None = Field(None, ge=-100, le=200, description="1y revenue growth floor (%)")
    earnings_growth_min_pct: float | None = Field(None, ge=-100, le=500, description="1y net profit growth floor (%)")
    pe_min: float | None = Field(None, ge=0, le=500, description="P/E lower bound")
    pe_max: float | None = Field(None, ge=0, le=500, description="P/E upper bound")

    # ---- Technical tab -----------------------------------------------------------------
    rsi_min: float | None = Field(None, ge=0, le=100, description="RSI(14) lower bound")
    rsi_max: float | None = Field(None, ge=0, le=100, description="RSI(14) upper bound")
    price_above_sma20: bool | None = Field(None, description="True: close above MA20; False: below")
    price_above_sma50: bool | None = Field(None, description="True: close above MA50; False: below")
    volume_ratio_min: float | None = Field(None, ge=0, le=10, description="Volume ratio floor (x)")
    atr_pct_max: float | None = Field(None, ge=0, le=50, description="ATR as % of price ceiling")
    return_30d_min_pct: float | None = Field(None, ge=-100, le=500)
    return_30d_max_pct: float | None = Field(None, ge=-100, le=500)
    return_90d_min_pct: float | None = Field(None, ge=-100, le=500)
    return_90d_max_pct: float | None = Field(None, ge=-100, le=500)

    # ---- Run shape ---------------------------------------------------------------------
    max_candidates: int = Field(30, ge=1, le=150, description="Cap on candidates sent to the claim phase")
    ranking: Literal["composite", "return_on_capital"] = Field(
        "composite",
        description="How survivors are ordered before truncation: composite percentile across "
                    "quality, value, growth, trend and risk, or the older single-factor ordering",
    )
    veto_top_n: int | None = Field(
        None, ge=1, le=150,
        description="Cap on candidates whose claims reach the veto (None: audit every candidate)",
    )
    """How far down the shortlist the auditor reads.

    Analysts are cheap and the veto is not -- measured on 18 Sep 2026, analysts cost $0.036 a
    candidate against the veto's $0.049 -- so the cheap stage can cover a wide field while the
    expensive one covers only the front of it. That lets the models, rather than a deterministic
    sort, decide which companies deserve scrutiny.

    The cost is real and is recorded in the dossier: claims on candidates below the cut are
    adjudicated but never attacked by an auditor."""

    @model_validator(mode="after")
    def _bands_are_ordered(self) -> "ScreenConfig":
        """Reject bands whose lower bound exceeds the upper bound (e.g. RSI 70..30)."""
        for low, high in (
            ("pe_min", "pe_max"),
            ("rsi_min", "rsi_max"),
            ("return_30d_min_pct", "return_30d_max_pct"),
            ("return_90d_min_pct", "return_90d_max_pct"),
        ):
            lo, hi = getattr(self, low), getattr(self, high)
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(f"{low} ({lo}) must not exceed {high} ({hi})")
        return self

    def config_hash(self) -> str:
        """Content hash of the config (with its version); part of the reproducibility triple."""
        return hash_object({"version": SCREEN_CONFIG_VERSION, "config": self.model_dump(mode="json")})


@dataclass(frozen=True)
class SliderSpec:
    """UI metadata for one slider: which tab, label, bounds, step and suggested default."""

    field: str
    tab: str
    label: str
    minimum: float
    maximum: float
    step: float
    suggested: float | None
    unit: str


SLIDERS: tuple[SliderSpec, ...] = (
    SliderSpec("roe_min_pct", "fundamental", "ROE floor", 0, 40, 0.5, 15, "%"),
    SliderSpec("roce_min_pct", "fundamental", "ROCE floor (non-banks)", 0, 50, 0.5, 15, "%"),
    SliderSpec("liabilities_to_equity_max", "fundamental", "Liabilities / equity ceiling (non-banks)", 0, 5, 0.1, 1.5, "x"),
    SliderSpec("net_npa_max_pct", "fundamental", "Net NPA ceiling (banks only)", 0, 5, 0.1, None, "%"),
    SliderSpec("revenue_growth_min_pct", "fundamental", "Revenue growth floor (1y)", -20, 50, 1, 5, "%"),
    SliderSpec("earnings_growth_min_pct", "fundamental", "Earnings growth floor (1y)", -30, 80, 1, 5, "%"),
    SliderSpec("pe_min", "fundamental", "P/E minimum", 0, 100, 1, None, "x"),
    SliderSpec("pe_max", "fundamental", "P/E maximum", 0, 150, 1, 60, "x"),
    SliderSpec("rsi_min", "technical", "RSI minimum", 0, 100, 1, 35, ""),
    SliderSpec("rsi_max", "technical", "RSI maximum", 0, 100, 1, 75, ""),
    SliderSpec("volume_ratio_min", "technical", "Volume ratio floor", 0, 3, 0.05, None, "x"),
    SliderSpec("atr_pct_max", "technical", "ATR % ceiling", 0, 10, 0.1, 4, "%"),
    SliderSpec("return_30d_min_pct", "technical", "30-day return minimum", -30, 30, 1, None, "%"),
    SliderSpec("return_30d_max_pct", "technical", "30-day return maximum", -30, 60, 1, None, "%"),
    SliderSpec("return_90d_min_pct", "technical", "90-day return minimum", -50, 50, 1, None, "%"),
    SliderSpec("return_90d_max_pct", "technical", "90-day return maximum", -50, 100, 1, None, "%"),
)
"""Slider definitions served to the front end; ``price_above_sma20/50`` are rendered as toggles."""
