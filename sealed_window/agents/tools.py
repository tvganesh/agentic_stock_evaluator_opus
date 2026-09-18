"""Snapshot tools: what a claim agent may look up before committing to claims.

Today an agent sees roughly 700 tokens -- a summary row and 30 sessions of prices -- while the
snapshot holds ~1,000 sessions, four years of statements, and the derived numbers of every other
company. These tools close that gap without weakening anything: they read the *sealed snapshot*, so
no network opens, no credential exists, and every value returned already carries the evidence ID a
claim must cite.

Four tools, each bound to one instrument for the life of an agent call:

    price_history(sessions)        up to 250 sessions of closes and volumes
    statement_detail(statement)    the full annual, quarterly or balance-sheet table
    peer_comparison(column)        this stock's rank and percentile among the universe
    news_articles(page)            headlines beyond the 25 in the opening slice

Bounds that keep an agent loop from becoming an open-ended one:

* **Instrument scope.** The toolbox closes over one instrument key. There is no argument through
  which an agent can read another company's data; ``peer_comparison`` returns aggregates and this
  stock's position among them, never another company's name or figures.
* **A prepaid probe pool.** Each candidate carries a fixed allowance shared by its agents. When it
  is empty every tool call is refused with an explanation, so the agent must conclude. The counter
  only goes down; nothing can top it up mid-run.
* **Bounded payloads.** Every tool caps what it returns, so a probe cannot blow the slot's input
  allowance on the following turn.
* **Audited.** Every call and refusal is recorded with its arguments and the remaining allowance.

Invalid arguments return an error payload rather than raising: the model is told what it did wrong
and may try once more at the cost of another probe, which is the behaviour that keeps a loop honest
without letting it spin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..governance.audit import AuditLog
from ..snapshot.columns import COLUMNS
from ..snapshot.store import SealedSnapshot

MAX_PRICE_SESSIONS = 250
"""Sessions one ``price_history`` call may return; 250 covers every indicator's lookback."""

MAX_NEWS_PER_PAGE = 25
STATEMENT_KINDS = ("annual", "quarterly", "balance_sheet")
_STATEMENT_EVIDENCE = {
    "annual": "income_statement_yearly",
    "quarterly": "income_statement_quarterly",
    "balance_sheet": "balance_sheet",
}


@dataclass(frozen=True)
class ToolDefinition:
    """One tool as the model sees it: a name, what it is for, and a strict argument schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ProbePool:
    """A prepaid allowance of tool calls, shared by the agents analysing one candidate."""

    def __init__(self, allowance: int) -> None:
        """Start with ``allowance`` probes; the count only ever decreases."""
        self._remaining = max(0, int(allowance))

    @property
    def remaining(self) -> int:
        """Probes still available to this candidate."""
        return self._remaining

    def take(self) -> bool:
        """Consume one probe, returning False when the allowance is exhausted."""
        if self._remaining <= 0:
            return False
        self._remaining -= 1
        return True


def tool_definitions() -> list[ToolDefinition]:
    """The four tools offered to every claim agent, with strict argument schemas."""
    return [
        ToolDefinition(
            name="price_history",
            description=(
                "Daily closes and volumes for this instrument, most recent last. Use when the "
                f"summary indicators are ambiguous and the shape of the move matters. Max "
                f"{MAX_PRICE_SESSIONS} sessions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sessions": {"type": "integer", "minimum": 5, "maximum": MAX_PRICE_SESSIONS,
                                 "description": "How many recent sessions to return."}
                },
                "required": ["sessions"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="statement_detail",
            description=(
                "The full reported table for this company: annual or quarterly revenue, operating "
                "profit and net profit by period, or the balance sheet. Use to check whether a "
                "growth or margin figure is a trend or a single period."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "enum": list(STATEMENT_KINDS),
                                  "description": "Which statement to return."}
                },
                "required": ["statement"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="peer_comparison",
            description=(
                "Where this company stands on one measure against every company in the snapshot: "
                "its value, rank, percentile and the universe's quartiles. Use to judge whether a "
                "number is genuinely unusual. Returns no other company's identity."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "column": {"type": "string",
                               "description": "A derived column name, e.g. roce_pct or return_90d_pct."}
                },
                "required": ["column"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="news_articles",
            description=(
                f"A page of {MAX_NEWS_PER_PAGE} headlines for this company, oldest pages first. Use "
                "when the opening slice was truncated and the remaining headlines might matter."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "minimum": 1, "maximum": 10,
                             "description": "1 is the most recent page."}
                },
                "required": ["page"],
                "additionalProperties": False,
            },
        ),
    ]


class SnapshotToolbox:
    """Executes tool calls against one instrument's data in the sealed snapshot."""

    def __init__(
        self,
        snapshot: SealedSnapshot,
        instrument_key: str,
        pool: ProbePool,
        audit: AuditLog | None = None,
    ) -> None:
        """Bind the toolbox to one instrument and one prepaid probe allowance."""
        self._snapshot = snapshot
        self._key = instrument_key
        self._pool = pool
        self._audit = audit

    @property
    def remaining(self) -> int:
        """Probes left for this candidate."""
        return self._pool.remaining

    def execute(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Run one tool call, consuming a probe; returns the payload or an explanatory error."""
        if not self._pool.take():
            return self._record(name, arguments, {
                "error": "probe allowance exhausted for this company; answer with the evidence you have"
            })
        handlers = {
            "price_history": self._price_history,
            "statement_detail": self._statement_detail,
            "peer_comparison": self._peer_comparison,
            "news_articles": self._news_articles,
        }
        handler = handlers.get(name)
        if handler is None:
            return self._record(name, arguments, {"error": f"unknown tool {name!r}"})
        try:
            return self._record(name, arguments, handler(arguments))
        except (KeyError, TypeError, ValueError) as exc:
            return self._record(name, arguments, {"error": f"invalid arguments: {exc}"})

    def _record(self, name: str, arguments: Mapping[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        """Audit one probe with its arguments, outcome and the allowance left."""
        if self._audit is not None:
            self._audit.record("agent.probe", {
                "instrument_key": self._key,
                "tool": name,
                "arguments": dict(arguments),
                "refused": "error" in payload,
                "detail": payload.get("error"),
                "remaining_probes": self._pool.remaining,
            })
        return payload

    def _price_history(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Return recent closes and volumes, capped at :data:`MAX_PRICE_SESSIONS`."""
        sessions = int(arguments["sessions"])
        if not 5 <= sessions <= MAX_PRICE_SESSIONS:
            raise ValueError(f"sessions must be between 5 and {MAX_PRICE_SESSIONS}")
        rows = self._snapshot.candles_for(self._key)[-sessions:]
        evidence = next(
            (r["evidence_id"] for r in self._snapshot.evidence_for(self._key) if r["kind"] == "price_history"),
            None,
        )
        return {
            "instrument": self._key,
            "sessions": len(rows),
            "evidence": [evidence] if evidence else [],
            "candles": [{"date": r[0], "close": r[4], "volume": r[5]} for r in rows],
        }

    def _statement_detail(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Return one full reported statement table with its evidence ID."""
        kind = str(arguments["statement"])
        if kind not in STATEMENT_KINDS:
            raise ValueError(f"statement must be one of {STATEMENT_KINDS}")
        wanted = _STATEMENT_EVIDENCE[kind]
        record = next(
            (r for r in self._snapshot.evidence_for(self._key) if r["kind"] == wanted), None
        )
        if record is None:
            return {"instrument": self._key, "statement": kind, "periods": [],
                    "note": "this company reported no such statement in the snapshot"}
        return {"instrument": self._key, "statement": kind,
                "evidence": [record["evidence_id"]], "fields": record["fields"]}

    def _peer_comparison(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Return this stock's value, rank and percentile for one column across the universe."""
        column = str(arguments["column"])
        if column not in COLUMNS:
            raise ValueError(f"unknown column {column!r}")
        table = self._snapshot.derived_table()
        values = sorted(v for row in table.values() if (v := row.get(column)) is not None)
        mine = table.get(self._key, {}).get(column)
        if mine is None or not values:
            return {"instrument": self._key, "column": column, "value": None,
                    "note": "not reported for this company"}
        below = sum(1 for v in values if v < mine)
        quartile = lambda q: values[min(len(values) - 1, int(q * (len(values) - 1)))]  # noqa: E731
        return {
            "instrument": self._key,
            "column": column,
            "value": mine,
            "rank_from_top": len(values) - below,
            "universe_size": len(values),
            "percentile": round(100.0 * below / len(values), 1),
            "quartiles": {"p25": quartile(0.25), "median": quartile(0.5), "p75": quartile(0.75)},
        }

    def _news_articles(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Return one page of headlines with their evidence IDs, newest page first."""
        page = int(arguments["page"])
        if not 1 <= page <= 10:
            raise ValueError("page must be between 1 and 10")
        records = [r for r in self._snapshot.evidence_for(self._key) if r["kind"] == "news_article"]
        records.sort(key=lambda r: r["fields"].get("published_time_ms", 0), reverse=True)
        chunk = records[(page - 1) * MAX_NEWS_PER_PAGE : page * MAX_NEWS_PER_PAGE]
        return {
            "instrument": self._key,
            "page": page,
            "total_articles": len(records),
            "articles": [
                {"evidence_id": r["evidence_id"], "published_date": r["fields"].get("published_date"),
                 "heading": r["fields"].get("heading"), "summary": r["fields"].get("summary")}
                for r in chunk
            ],
        }
