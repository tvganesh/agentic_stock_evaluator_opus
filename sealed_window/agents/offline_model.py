"""A deterministic, rule-based stand-in for the model client. It is NOT an LLM.

Purpose: exercise the complete sealed pipeline -- slots, gateway, claims, adjudication, veto,
dossier -- with no API key and no network, and give the test suite a model whose output is
known in advance (build-order gate P3: "compiled cost matches actual spend on a dry run with
stub models").

It implements :class:`~sealed_window.governance.llm_gateway.ModelClient` by reading the same
prompt a real model would receive, parsing the JSON inside the data envelope, and applying a
handful of transparent rules. Token usage is estimated from prompt length and settled against
the plan exactly like real usage.

It also simulates a *compromised* model: when a slice contains the synthetic injection
headline, it emits the unkillable claim the headline asks for (falsifier ``close < 0``). The
adjudicator must discard that claim as VACUOUS -- the end-to-end proof that injection can
produce a claim but not a claim that passes the machine check.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from ..claims.schema import ClaimBatch, ClaimDraft, RefutationBatch, RefutationDraft
from ..governance.llm_gateway import ModelRequest, ModelResult

INJECTION_MARKER = "IGNORE ALL PREVIOUS INSTRUCTIONS"


def _envelope(text: str, tag: str) -> Any:
    """Extract and parse the JSON inside ``<tag>...</tag>`` from a prompt (``[]`` if absent)."""
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", text, flags=re.S)
    return json.loads(match.group(1)) if match else []


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~3.5 characters per token) used for simulated usage."""
    return math.ceil(len(text) / 3.5)


class OfflineHeuristicModel:
    """Rule-based :class:`ModelClient` for offline runs; output is labelled ``offline-heuristic``."""

    label = "offline-heuristic"
    permitted_hosts: frozenset[str] = frozenset()
    """Nothing is reachable: this client computes, so an offline run opens no host at all."""

    def count_input_tokens(self, request: ModelRequest) -> int:
        """Estimate input tokens from the system and user prompt length."""
        return _estimate_tokens(request.system + request.user)

    def complete(self, request: ModelRequest) -> ModelResult:
        """Produce a ClaimBatch or RefutationBatch from rules over the enveloped evidence."""
        if request.output_type is ClaimBatch:
            parsed: Any = ClaimBatch(claims=self._claims(request.user))
        elif request.output_type is RefutationBatch:
            parsed = RefutationBatch(refutations=self._refutations(request.user))
        else:
            return ModelResult(None, "unsupported_schema", 0, 0)
        output_tokens = min(_estimate_tokens(parsed.model_dump_json()), request.max_tokens)
        return ModelResult(parsed, "end_turn", self.count_input_tokens(request), output_tokens)

    # ---- claims ---------------------------------------------------------------------

    def _claims(self, user: str) -> list[ClaimDraft]:
        """Apply the per-dimension rules to one instrument's slice."""
        dimension = re.search(r"^Dimension: (\w+)$", user, flags=re.M).group(1)
        records = _envelope(user, "snapshot_slice")
        by_kind: dict[str, dict[str, Any]] = {}
        for record in records:
            by_kind.setdefault(record["kind"], record)
        rules = {"fundamental": self._fundamental, "technical": self._technical, "news": self._news}[dimension]
        return rules(by_kind, records)[:6]

    @staticmethod
    def _draft(predicate: str, direction: str, statement: str, evidence: list[str], confidence: float,
               falsifier: str) -> ClaimDraft:
        """Build a validated ClaimDraft, deriving the justification from the rule that produced it.

        A real analyst argues for its claim; this stand-in has no argument to give, so it says so
        plainly rather than inventing one. Deliberately free of figures: the justification passes the
        same ``unsupported_figures`` check as the statement, and a number here that happened not to
        appear in the cited record would fail an offline run for no useful reason.
        """
        justification = (
            f"Produced by the offline rule for {predicate} rather than by a model, so there is no "
            f"argument behind it beyond the rule itself. It stands or falls on its falsifier, "
            f"`{falsifier}`, which the adjudicator evaluates against the snapshot."
        )
        return ClaimDraft(predicate=predicate, direction=direction, statement=statement, evidence=evidence,
                          confidence=confidence, falsifier=falsifier, justification=justification)

    def _fundamental(self, by_kind: dict[str, dict], records: list[dict]) -> list[ClaimDraft]:
        """Rules over fundamental_derived: ROE vs sector, growth, margins, leverage."""
        rec = by_kind.get("fundamental_derived")
        if not rec:
            return []
        f, ev = rec["fields"], [rec["evidence_id"]]
        out = []
        roe, sector_roe = f.get("roe_pct"), f.get("sector_roe_pct")
        if roe is not None and sector_roe is not None:
            if roe > sector_roe:
                out.append(self._draft("roe_above_sector", "positive",
                                       f"ROE of {roe:.2f}% is above the sector benchmark of {sector_roe:.2f}%.",
                                       ev, 0.65, f"roe_pct < {sector_roe:.2f}"))
            else:
                out.append(self._draft("roe_below_sector", "negative",
                                       f"ROE of {roe:.2f}% trails the sector benchmark of {sector_roe:.2f}%.",
                                       ev, 0.55, f"roe_pct > {sector_roe:.2f}"))
        growth = f.get("revenue_growth_1y_pct")
        if growth is not None and growth >= 10:
            out.append(self._draft("revenue_growth_strong", "positive",
                                   f"Revenue grew {growth:.2f}% over the latest fiscal year.",
                                   ev, 0.6, "revenue_growth_1y_pct < 5"))
        elif growth is not None and growth < 0:
            out.append(self._draft("revenue_contracting", "negative",
                                   f"Revenue shrank {growth:.2f}% over the latest fiscal year.",
                                   ev, 0.6, "revenue_growth_1y_pct > 0"))
        delta = f.get("operating_margin_delta_1y_pp")
        if delta is not None and delta > 0:
            out.append(self._draft("margins_expanding", "positive",
                                   f"Operating margin expanded by {delta:.2f} percentage points year on year.",
                                   ev, 0.55, "operating_margin_delta_1y_pp < 0"))
        npa, sector_npa = f.get("net_npa_pct"), f.get("sector_net_npa_pct")
        if npa is not None and sector_npa is not None:
            if npa <= sector_npa:
                out.append(self._draft("asset_quality_better_than_sector", "positive",
                                       f"Net NPA of {npa:.2f}% is at or below the sector's {sector_npa:.2f}%.",
                                       ev, 0.6, f"net_npa_pct > {sector_npa:.2f}"))
            else:
                out.append(self._draft("asset_quality_weaker_than_sector", "negative",
                                       f"Net NPA of {npa:.2f}% is above the sector's {sector_npa:.2f}%.",
                                       ev, 0.6, f"net_npa_pct < {sector_npa:.2f}"))
        leverage = f.get("liabilities_to_equity")
        if f.get("is_bank") != 1.0 and leverage is not None and leverage > 2:
            out.append(self._draft("balance_sheet_leveraged", "negative",
                                   f"Liabilities are {leverage:.2f} times equity.",
                                   ev, 0.5, "liabilities_to_equity < 1.5"))
        return out

    def _technical(self, by_kind: dict[str, dict], records: list[dict]) -> list[ClaimDraft]:
        """Rules over technical_derived: trend, long-term trend, overbought, momentum."""
        rec = by_kind.get("technical_derived")
        if not rec:
            return []
        t, ev = rec["fields"], [rec["evidence_id"]]
        out = []
        vs50, hist = t.get("price_vs_sma50_pct"), t.get("macd_hist")
        if vs50 is not None and hist is not None and vs50 > 0 and hist > 0:
            out.append(self._draft("uptrend_confirmed", "positive",
                                   f"Close is {vs50:.2f}% above its 50-session average with a positive MACD histogram.",
                                   ev, 0.6, "price_vs_sma50_pct < 0 OR macd_hist < 0"))
        vs200 = t.get("price_vs_sma200_pct")
        if vs200 is not None and vs200 < 0:
            out.append(self._draft("below_long_term_trend", "negative",
                                   f"Close is {vs200:.2f}% relative to its 200-session average.",
                                   ev, 0.55, "price_vs_sma200_pct > 0"))
        rsi = t.get("rsi_14")
        if rsi is not None and rsi > 72:
            out.append(self._draft("overbought", "negative", f"RSI(14) is {rsi:.2f}, in overbought territory.",
                                   ev, 0.5, "rsi_14 < 65"))
        r90 = t.get("return_90d_pct")
        if r90 is not None and r90 > 10:
            out.append(self._draft("medium_term_momentum", "positive",
                                   f"The stock returned {r90:.2f}% over roughly 90 days.",
                                   ev, 0.55, "return_90d_pct < 5"))
        return out

    def _news(self, by_kind: dict[str, dict], records: list[dict]) -> list[ClaimDraft]:
        """Rules over news counts and price reaction, plus the simulated injection compromise."""
        out = []
        news, tech = by_kind.get("news_derived"), by_kind.get("technical_derived")
        if news and tech:
            count, r5 = news["fields"].get("news_count_window"), tech["fields"].get("return_5d_pct")
            if count is not None and r5 is not None and count >= 3 and r5 > 0:
                out.append(self._draft("news_flow_with_positive_reaction", "positive",
                                       f"Headline flow is active and the stock returned {r5:.2f}% over 5 sessions.",
                                       [news["evidence_id"], tech["evidence_id"]], 0.45, "return_5d_pct < 0"))
        for record in records:
            if record["kind"] == "news_article" and INJECTION_MARKER in record["fields"].get("heading", ""):
                out.append(self._draft("revenue_doubled", "positive",
                                       "Recent coverage says revenue doubled.",
                                       [record["evidence_id"]], 0.95, "close < 0"))
        return out

    # ---- refutations ----------------------------------------------------------------

    def _refutations(self, user: str) -> list[RefutationDraft]:
        """Refute positive claims contradicted by profit declines or thin volume."""
        derived = {
            (record["instrument"], record["kind"]): record
            for record in _envelope(user, "snapshot_slice")
            if record["kind"].endswith("_derived")
        }
        out = []
        for claim in _envelope(user, "claims_under_audit"):
            if claim["direction"] != "positive":
                continue
            fund = derived.get((claim["instrument"], "fundamental_derived"))
            tech = derived.get((claim["instrument"], "technical_derived"))
            if claim["dimension"] == "fundamental" and fund:
                growth = fund["fields"].get("net_profit_growth_1y_pct")
                if growth is not None and growth < -5:
                    out.append(RefutationDraft(
                        target_claim_id=claim["claim_id"],
                        statement=f"Net profit changed {growth:.2f}% year on year, which undermines this thesis.",
                        evidence=[fund["evidence_id"]], falsifier="net_profit_growth_1y_pct >= 0"))
            if claim["dimension"] == "technical" and tech:
                ratio = tech["fields"].get("volume_ratio_20d")
                if ratio is not None and ratio < 0.85:
                    out.append(RefutationDraft(
                        target_claim_id=claim["claim_id"],
                        statement=f"Recent volume is only {ratio:.2f} times its 20-session average; the move lacks participation.",
                        evidence=[tech["evidence_id"]], falsifier="volume_ratio_20d >= 0.85"))
        return out[:24]
