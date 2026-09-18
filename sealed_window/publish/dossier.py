"""Dossier assembly: recommendations and write-ups built only from adjudicated claims.

Recommendation rules (versioned, deterministic):

* **BUY** -- score >= ``BUY_MIN_SCORE``, surviving positive claims in at least two dimensions
  including fundamental, and no surviving negative fundamental claim with confidence >= 0.6.
* **AVOID** -- score <= ``AVOID_MAX_SCORE``.
* **WATCH** -- anything else with at least one surviving claim.
* **Abstain** -- no surviving claims: not published at all.

Published picks are BUY then WATCH, by score, capped at ``TARGET_MAX_PICKS``. If fewer than
``TARGET_MIN_PICKS`` qualify, the dossier publishes fewer and says so; nothing can pad the list,
because nothing in this phase can write an entry.

Each entry carries the write-up the brief asks for, but assembled rather than generated: the
surviving claims in plain prose with evidence and falsifier attached ("what would make this
wrong"), the claims that were refuted or vetoed and by what, the counts of discarded claims and
why, and key metrics. The header carries ``as_of`` prominently, the reproducibility triple
(snapshot, screen config, plan), spend, the funnel, and warnings for synthetic data or an
offline stand-in model.
"""

from __future__ import annotations

from typing import Any

from ..adjudicate.adjudicator import DIMENSION_WEIGHTS, RANKING_VERSION, StockScore, score_candidates
from ..claims.schema import Adjudication, Claim, Refutation, Verdict
from ..governance.spend import SpendPlan
from ..screen.screen import ScreenResult
from ..snapshot.columns import DERIVED_COLUMNS_VERSION, Dimension
from ..snapshot.store import SealedSnapshot

RECOMMENDATION_VERSION = "recommend-v1"
BUY_MIN_SCORE = 1.0
BUY_MIN_POSITIVE_DIMENSIONS = 2
BUY_MAX_NEGATIVE_FUNDAMENTAL_CONFIDENCE = 0.6
AVOID_MAX_SCORE = -0.5
TARGET_MIN_PICKS = 5
TARGET_MAX_PICKS = 10

DISCLAIMER = (
    "Research output only; not investment advice. This system cannot place, modify or cancel orders "
    "and has no access to any portfolio, holdings, positions or funds."
)

KEY_METRICS = (
    "close", "roe_pct", "roce_pct", "pe", "revenue_growth_1y_pct", "net_profit_growth_1y_pct",
    "liabilities_to_equity", "net_npa_pct", "rsi_14", "price_vs_sma50_pct", "return_90d_pct", "news_count_window",
)


def recommend(score: StockScore) -> str | None:
    """Map a candidate's score and tallies to BUY / WATCH / AVOID, or ``None`` to abstain."""
    if score.surviving_claims == 0:
        return None
    if (
        score.score >= BUY_MIN_SCORE
        and len(score.positive_dimensions) >= BUY_MIN_POSITIVE_DIMENSIONS
        and Dimension.FUNDAMENTAL.value in score.positive_dimensions
        and score.max_negative_fundamental_confidence < BUY_MAX_NEGATIVE_FUNDAMENTAL_CONFIDENCE
    ):
        return "BUY"
    if score.score <= AVOID_MAX_SCORE:
        return "AVOID"
    return "WATCH"


def _claim_view(claim: Claim, verdict: Adjudication | None) -> dict[str, Any]:
    """Serialise a claim with its verdict for the dossier."""
    return {
        "claim_id": claim.claim_id,
        "dimension": claim.dimension.value,
        "direction": claim.direction.value,
        "predicate": claim.predicate,
        "statement": claim.statement,
        "confidence": claim.confidence,
        "evidence": claim.evidence,
        "falsifier": claim.falsifier,
        "verdict": verdict.verdict.value if verdict else "not_adjudicated",
        "reason": verdict.reason if verdict else "",
    }


def build_dossier(
    *,
    run_id: str,
    snapshot: SealedSnapshot,
    screen: ScreenResult,
    plan: SpendPlan,
    spend_summary: dict[str, Any],
    claims: list[Claim],
    refutations: list[Refutation],
    verdicts: dict[str, Adjudication],
    model_client: str,
    notes: list[str],
) -> dict[str, Any]:
    """Assemble the dossier dictionary (written as JSON and rendered to Markdown by the orchestrator)."""
    refutations_by_target: dict[str, list[Refutation]] = {}
    for refutation in refutations:
        refutations_by_target.setdefault(refutation.target_claim_id, []).append(refutation)

    dimension_order = {d.value: i for i, d in enumerate(Dimension)}
    entries: list[dict[str, Any]] = []
    for score in score_candidates(screen.candidates, claims, verdicts):
        key = score.instrument_key
        instrument = snapshot.instrument(key)
        own = [c for c in claims if c.subject == key]
        survivors, attacked, discarded = [], [], []
        for claim in own:
            verdict = verdicts.get(claim.claim_id)
            view = _claim_view(claim, verdict)
            view["refutations"] = [
                {"refutation_id": r.refutation_id, "statement": r.statement, "falsifier": r.falsifier,
                 "evidence": r.evidence,
                 "verdict": verdicts[r.refutation_id].verdict.value if r.refutation_id in verdicts else "n/a",
                 "reason": verdicts[r.refutation_id].reason if r.refutation_id in verdicts else ""}
                for r in refutations_by_target.get(claim.claim_id, [])
            ]
            if verdict and verdict.verdict is Verdict.SURVIVED:
                survivors.append(view)
            elif verdict and verdict.verdict in (Verdict.REFUTED, Verdict.VETOED):
                attacked.append(view)
            else:
                discarded.append(view)
        survivors.sort(key=lambda v: (dimension_order[v["dimension"]], -v["confidence"], v["claim_id"]))
        row = snapshot.derived_row(key)
        entries.append({
            "instrument_key": key,
            "trading_symbol": instrument.get("trading_symbol"),
            "name": instrument.get("name"),
            "recommendation": recommend(score) or "ABSTAIN",
            "score": score.score,
            "claim_tallies": score.dimensions,
            "surviving_claims": survivors,
            "refuted_or_vetoed_claims": attacked,
            "discarded_claims": discarded,
            "key_metrics": {name: row.get(name) for name in KEY_METRICS},
        })

    ranked_picks = [e for e in entries if e["recommendation"] == "BUY"] + [
        e for e in entries if e["recommendation"] == "WATCH"
    ]
    picks = ranked_picks[:TARGET_MAX_PICKS]
    avoid = [e for e in entries if e["recommendation"] == "AVOID"]
    picked_keys = {e["instrument_key"] for e in picks + avoid}
    not_published = [
        {"trading_symbol": e["trading_symbol"], "instrument_key": e["instrument_key"],
         "recommendation": e["recommendation"], "score": e["score"],
         "reason": "no surviving claims" if e["recommendation"] == "ABSTAIN" else "below the top-pick cutoff"}
        for e in entries if e["instrument_key"] not in picked_keys
    ]

    abstention = None
    if len(picks) < TARGET_MIN_PICKS:
        abstention = (
            f"Only {len(picks)} candidate(s) accumulated enough surviving evidence to publish "
            f"(target {TARGET_MIN_PICKS}-{TARGET_MAX_PICKS}). The list is not padded."
        )

    all_verdicts = [v for v in verdicts.values() if v.kind == "claim"]
    header = {
        "run_id": run_id,
        "as_of": snapshot.as_of,
        "prices_as_of": snapshot.prices_as_of,
        "source": snapshot.source,
        "synthetic_data": snapshot.is_synthetic,
        "model_client": model_client,
        "reproducibility": {
            "snapshot": snapshot.root_hash,
            "screen_config": screen.config_hash,
            "plan": plan.plan_hash,
        },
        "versions": {
            "ranking": RANKING_VERSION,
            "recommendation": RECOMMENDATION_VERSION,
            "derived_columns": DERIVED_COLUMNS_VERSION,
            "dimension_weights": {d.value: w for d, w in DIMENSION_WEIGHTS.items()},
        },
        "spend": spend_summary,
        "funnel": {
            "universe": len(snapshot.instruments),
            "screened_candidates": len(screen.candidates),
            "truncated_by_max_candidates": len(screen.truncated),
            "claims": len(claims),
            "claims_survived": sum(1 for v in all_verdicts if v.verdict is Verdict.SURVIVED),
            "claims_vetoed": sum(1 for v in all_verdicts if v.verdict is Verdict.VETOED),
            "refutations": len(refutations),
            "published_picks": len(picks),
            "avoid": len(avoid),
        },
        "abstention": abstention,
        "notes": notes,
        "disclaimer": DISCLAIMER,
    }
    return {"header": header, "picks": picks, "avoid": avoid, "not_published": not_published}


def _render_claim(view: dict[str, Any], struck: bool = False) -> list[str]:
    """Markdown lines for one claim, with evidence and falsifier (and refutations if any)."""
    statement = f"~~{view['statement']}~~" if struck else view["statement"]
    lines = [
        f"- **[{view['dimension']} · {view['direction']} · conf {view['confidence']:.2f}]** {statement}",
        f"  - would be wrong if: `{view['falsifier']}`",
        f"  - evidence: {', '.join(view['evidence'])}",
    ]
    if struck:
        lines.append(f"  - {view['verdict']}: {view['reason']}")
    for refutation in view.get("refutations", []):
        lines.append(f"  - attack {refutation['refutation_id']} ({refutation['verdict']}): {refutation['statement']} "
                     f"— refutation wrong if `{refutation['falsifier']}`")
    return lines


def render_markdown(dossier: dict[str, Any]) -> str:
    """Render the dossier as Markdown; every sentence comes from a claim or a fixed template."""
    h = dossier["header"]
    prices = h.get("prices_as_of") or "unknown"
    out = ["# Sealed Window dossier", "",
           f"> **As of {h['as_of']} · prices to {prices}** — a recommendation is a statement about a moment."]
    if h.get("prices_as_of") and h["prices_as_of"] != h["as_of"]:
        out.append(f"> **Prices end {h['prices_as_of']}, before as_of.** Normal for weekends and holidays; "
                   "otherwise a trading session is missing from the evidence.")
    if h["synthetic_data"]:
        out.append("> **SYNTHETIC DATA.** Built from the offline fixture market; these are not real companies or prices.")
    if h["model_client"] == "offline-heuristic":
        out.append(f"> **Model client: {h['model_client']}.** Claims came from a rule-based stand-in, not an LLM.")
    elif h["model_client"] != "anthropic":
        out.append(f"> **Model client: {h['model_client']}.** Claims came from a model served on this "
                   "machine, not the Anthropic API; treat its judgement accordingly.")
    r = h["reproducibility"]
    f = h["funnel"]
    out += [
        "",
        "| Reproducibility triple | Hash |",
        "|---|---|",
        f"| snapshot (what was true, and when) | `{r['snapshot']}` |",
        f"| screen config (what you asked for) | `{r['screen_config']}` |",
        f"| plan (what was spent, on which models) | `{r['plan']}` |",
        "",
        f"**Funnel:** universe {f['universe']} → screened {f['screened_candidates']} → claims {f['claims']} "
        f"→ survived {f['claims_survived']} (vetoed {f['claims_vetoed']}) → published {f['published_picks']}",
        "",
        f"**Spend:** committed {h['spend']['committed']}, spent {h['spend']['spent']} (hard stop = committed)",
    ]
    if h["abstention"]:
        out += ["", f"**Abstention:** {h['abstention']}"]
    for note in h["notes"]:
        out.append(f"- note: {note}")

    def section(title: str, entries: list[dict[str, Any]]) -> None:
        """Append one section of recommendation entries."""
        out.extend(["", f"## {title}"])
        if not entries:
            out.append("_None._")
        for rank, e in enumerate(entries, start=1):
            tallies = " · ".join(
                f"{dim} +{int(t['positive'])}/−{int(t['negative'])}" for dim, t in e["claim_tallies"].items()
            )
            out.extend(["", f"### {rank}. {e['trading_symbol']} — {e['recommendation']} (score {e['score']:.2f})",
                        f"{e['name']} · `{e['instrument_key']}` · surviving claims: {tallies}", "",
                        "**Why (surviving claims)**"])
            for view in e["surviving_claims"]:
                out.extend(_render_claim(view))
            if e["refuted_or_vetoed_claims"]:
                out.extend(["", "**Claims that did not survive**"])
                for view in e["refuted_or_vetoed_claims"]:
                    out.extend(_render_claim(view, struck=True))
            if e["discarded_claims"]:
                reasons = sorted({f"{v['verdict']}: {v['reason']}" for v in e["discarded_claims"]})
                out.extend(["", f"**Discarded before ranking:** {len(e['discarded_claims'])}"])
                out.extend(f"- {x}" for x in reasons)
            metrics = ", ".join(f"{k}={v}" for k, v in e["key_metrics"].items())
            out.extend(["", f"_Key metrics:_ {metrics}"])

    section("Picks (BUY / WATCH)", dossier["picks"])
    section("Avoid", dossier["avoid"])
    out += ["", "## Not published", ""]
    out += [f"- {n['trading_symbol']}: {n['recommendation']} ({n['reason']})" for n in dossier["not_published"]] or ["_None._"]
    out += ["", "---", h["disclaimer"], ""]
    return "\n".join(out)
