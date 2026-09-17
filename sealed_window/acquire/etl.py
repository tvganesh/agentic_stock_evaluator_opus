"""Phase 1 ETL: acquire, normalise, derive, assign evidence IDs, seal.

Flow of :func:`run_acquisition`:

1. Record ``phase.enter 1_acquire`` with the seal status (network live, no model).
2. Build the market source: the live :class:`UpstoxAdapter` behind the egress gate (the
   analytics token is loaded and scrubbed from the environment here), or the
   :class:`SyntheticMarket` fixture. Optionally wrap it in a :class:`CheckpointedSource` so an
   interrupted acquisition resumes without re-fetching.
3. :func:`build_snapshot_content` resolves the universe against the instrument master (by
   ISIN when the universe file provides one, otherwise by trading symbol) and fetches, per
   instrument, daily candles, key ratios, yearly and quarterly income statements and the
   balance sheet, plus headlines in batches. Vendor failures are recorded as acquisition
   notes (the instrument then has ``None`` columns). Governance denials and HTTP 429 rate
   limiting are *not* caught: they abort the run.
4. Every derived number is computed by ``indicators``; every fact becomes an evidence
   record ``evidence_id = hash(query, params, as_of, row_key)`` with lineage to its inputs.
5. The gate is closed (token dropped), the process seal drops, and only then is the
   snapshot written read-only to disk under its root hash; checkpoints are then discarded.

Universe files are either plain text (one symbol per line) or an index constituents CSV such
as NSE's ``ind_nifty500list.csv`` (columns ``Symbol`` and ``ISIN Code``). The file's name and
content hash are recorded in the manifest, so the snapshot says exactly which list it covers.

The query set -- which capabilities, lookback windows and column version -- is itself hashed
into the manifest, so a change to what the ETL fetches changes every snapshot hash.

Price date integrity: Upstox's historical candle API excludes the current day, so when ``as_of`` is
today (and the history lacks that session) the ETL fetches the session's daily candle from the
intraday endpoint -- one extra request per instrument. Every snapshot then records
``prices_as_of`` (the newest candle date) and ``instruments_behind_as_of``, and a lag is audited
as ``snapshot.price_lag``. A lag is normal for weekends and holidays; otherwise a session is missing.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from ..governance import policy
from ..governance.audit import AuditLog
from ..governance.egress import EgressHTTPError, EgressRateLimited
from ..governance.process_roles import ProcessRole, current_role
from ..governance.seal import SEAL
from ..snapshot.columns import DERIVED_COLUMNS_VERSION, Dimension, columns_for
from ..snapshot.hashing import content_hash, evidence_id, hash_object, stable_float
from ..snapshot.store import SnapshotContent, write_snapshot
from ..snapshot import indicators
from .checkpoint import CheckpointedSource
from .fixture_source import SyntheticMarket
from .upstox_adapter import AdapterError

CANDLE_LOOKBACK_DAYS = 1460
"""Calendar days of daily candles requested (~1,000 sessions, four years).

Widened from 760 on 17 Sep 2026: two years gave the walk-forward only ~10 windows at the default
step, below the 20 its gates require; four years gives ~34. Upstox allows a decade, but the backtest
uses *today's* index membership, so reaching further back compounds survivorship bias faster than it
adds evidence -- companies that failed out of the index are simply absent. Deeper history is worth
collecting only alongside point-in-time membership. The request count is unchanged (one call per
instrument); only the payload grows, to roughly 35 MB per snapshot."""

NEWS_WINDOW_DAYS = 30
PRICE_HISTORY_SESSIONS = 30
"""Recent closes/volumes exposed to agents as a compact evidence record."""

MAX_SUMMARY_CHARS = 600
"""Headline summaries are truncated; long vendor text only widens the injection surface."""

_IST = timezone(timedelta(hours=5, minutes=30))
_SOURCE_ERRORS = (EgressHTTPError, AdapterError, ValueError, KeyError, TypeError)

ProgressCallback = Callable[[int, int, str], None]
"""``progress(done, total, trading_symbol)`` called after each instrument is acquired."""

SESSION_FINAL_AFTER_IST = time(16, 0)
"""NSE closes at 15:30 IST; a date's daily data is treated as final only from 16:00 IST that day."""


def session_final_at(as_of: date) -> datetime:
    """The timezone-aware moment after which market data for ``as_of`` is considered final."""
    return datetime.combine(as_of, SESSION_FINAL_AFTER_IST, tzinfo=_IST)


def session_is_final(as_of: date, now: datetime | None = None) -> bool:
    """True once the ``as_of`` session has closed; acquiring earlier would capture partial data."""
    return (now or datetime.now(timezone.utc)) >= session_final_at(as_of)


class MarketSource(Protocol):
    """Interface shared by the live adapter, the synthetic fixture and the checkpoint wrapper."""

    def instrument_master(self) -> list[dict[str, Any]]:
        """Return NSE cash-equity instrument records."""

    def daily_candles(self, instrument_key: str, from_date: date, to_date: date) -> list[list[Any]]:
        """Return vendor-shaped daily candles."""

    def intraday_daily_candle(self, instrument_key: str) -> list[list[Any]]:
        """Return the current session's vendor-shaped daily candle (empty if unavailable)."""

    def key_ratios(self, isin: str) -> list[dict[str, Any]]:
        """Return vendor-shaped key ratios."""

    def income_statement(self, isin: str, time_period: str) -> dict[str, Any]:
        """Return a vendor-shaped income statement."""

    def balance_sheet(self, isin: str) -> dict[str, Any]:
        """Return a vendor-shaped balance sheet."""

    def news(self, instrument_keys: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Return vendor-shaped headlines keyed by instrument."""


@dataclass(frozen=True)
class UniverseEntry:
    """One requested instrument: its trading symbol and, when the universe file has it, its ISIN."""

    symbol: str
    isin: str | None = None


def query_set_spec() -> dict[str, Any]:
    """Describe exactly what the ETL fetches and derives; hashed into every manifest."""
    return {
        "capabilities": {
            cap.name.value: {"host": cap.host, "path": cap.path_template, "query": sorted(cap.query_params)}
            for cap in policy.CAPABILITIES.values()
        },
        "candle_lookback_days": CANDLE_LOOKBACK_DAYS,
        "news_window_days": NEWS_WINDOW_DAYS,
        "price_history_sessions": PRICE_HISTORY_SESSIONS,
        "income_periods": ["yearly", "quarterly"],
        "derived_columns_version": DERIVED_COLUMNS_VERSION,
    }


def query_set_hash() -> str:
    """Hash of :func:`query_set_spec`; recorded in the snapshot manifest."""
    return hash_object(query_set_spec())


def read_universe(path: Path) -> list[UniverseEntry]:
    """Read a universe file: an index constituents CSV (Symbol, ISIN Code) or one symbol per line.

    Entries are de-duplicated by symbol, keeping file order.
    """
    text = path.read_text(encoding="utf-8-sig")
    lines = [line for line in text.splitlines() if line.strip()]
    header = [cell.strip().lower() for cell in lines[0].split(",")] if lines else []
    entries: list[UniverseEntry] = []
    if "symbol" in header:
        for row in csv.DictReader(io.StringIO(text)):
            normalised = {str(k).strip().lower(): (v or "").strip() for k, v in row.items() if k}
            symbol = normalised.get("symbol", "").upper()
            isin = normalised.get("isin code") or normalised.get("isin") or None
            if symbol:
                entries.append(UniverseEntry(symbol, isin.upper() if isin else None))
    else:
        for line in lines:
            symbol = line.split("#", 1)[0].strip().upper()
            if symbol:
                entries.append(UniverseEntry(symbol))
    seen: set[str] = set()
    unique = []
    for entry in entries:
        if entry.symbol not in seen:
            seen.add(entry.symbol)
            unique.append(entry)
    return unique


def universe_source(path: Path, entries: Sequence[UniverseEntry]) -> dict[str, Any]:
    """Provenance of a universe file for the manifest: file name, content hash and entry count."""
    return {"file": path.name, "hash": content_hash(path.read_bytes()), "entries": len(entries)}


def _normalise_candles(raw: list[list[Any]]) -> list[list[Any]]:
    """Convert vendor candles to ascending ``[date, open, high, low, close, volume]`` rows."""
    rows = []
    for candle in raw:
        if len(candle) < 6:
            continue
        rows.append([str(candle[0])[:10], *(stable_float(float(v)) for v in candle[1:6])])
    rows.sort(key=lambda r: r[0])
    deduped: dict[str, list[Any]] = {r[0]: r for r in rows}
    return list(deduped.values())


def _normalise_news(articles: list[dict[str, Any]], as_of: date) -> list[dict[str, Any]]:
    """Keep headlines inside the window ending at ``as_of`` (no lookahead), truncated and de-duplicated."""
    kept: dict[tuple[int, str], dict[str, Any]] = {}
    for article in articles:
        try:
            published_ms = int(article["published_time"])
        except (KeyError, TypeError, ValueError):
            continue
        published = datetime.fromtimestamp(published_ms / 1000, tz=_IST).date()
        if not (as_of - timedelta(days=NEWS_WINDOW_DAYS - 1) <= published <= as_of):
            continue
        heading = str(article.get("heading") or "").strip()
        if not heading:
            continue
        kept[(published_ms, heading)] = {
            "published_date": published.isoformat(),
            "published_time_ms": published_ms,
            "heading": heading[:300],
            "summary": str(article.get("summary") or "").strip()[:MAX_SUMMARY_CHARS],
        }
    return [kept[k] for k in sorted(kept)]


def _add_evidence(
    evidence: dict[str, dict[str, Any]],
    *,
    instrument_key: str,
    dimension: Dimension,
    kind: str,
    query: str,
    params: dict[str, Any],
    as_of: str,
    row_key: str,
    fields: dict[str, Any],
    inputs: list[str] | None = None,
) -> str:
    """Create one evidence record with a content-derived ID and lineage; return the ID."""
    ev_id = evidence_id(query, params, as_of, row_key)
    evidence[ev_id] = {
        "evidence_id": ev_id,
        "instrument_key": instrument_key,
        "dimension": dimension.value,
        "kind": kind,
        "query": query,
        "params": params,
        "as_of": as_of,
        "row_key": row_key,
        "fields": fields,
        "inputs": sorted(inputs or []),
    }
    return ev_id


def _statement_fields(statement: dict[str, Any] | None, limit: int) -> dict[str, Any]:
    """Compact an income statement into per-period revenue/operating/net rows for evidence."""
    series = {
        category: dict(indicators.statement_series(statement, category))
        for category in ("revenue", "operating_profit", "net_profit")
    }
    periods = sorted(series["revenue"], reverse=True)[:limit]
    return {
        "units_in": (statement or {}).get("units_in"),
        "periods": [
            {"period_end": p.isoformat(), **{c: series[c].get(p) for c in series}} for p in periods
        ],
    }


def _fetch(notes: list[dict[str, Any]], symbol: str, what: str, call):
    """Run one source call; on a vendor/data error record an acquisition note and return ``None``.

    Governance violations and HTTP 429 are deliberately not caught: they abort the acquisition.
    """
    try:
        return call()
    except EgressRateLimited:
        raise
    except _SOURCE_ERRORS as exc:
        notes.append({"symbol": symbol, "what": what, "issue": type(exc).__name__, "detail": str(exc)[:200]})
        return None


def _resolve_universe(
    master: list[dict[str, Any]], entries: Sequence[UniverseEntry], notes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Match universe entries to NSE_EQ instruments (ISIN first, then symbol), noting anything skipped."""
    by_symbol = {str(row.get("trading_symbol", "")).upper(): row for row in master}
    by_isin = {str(row.get("isin", "")).upper(): row for row in master if row.get("isin")}
    key_rule = policy.ParamRule(policy.NSE_EQ_KEY_RE)
    instruments: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for entry in entries:
        record = by_isin.get(entry.isin) if entry.isin else None
        if record is None:
            record = by_symbol.get(entry.symbol)
            if record is not None and entry.isin and str(record.get("isin", "")).upper() != entry.isin:
                notes.append({"symbol": entry.symbol, "what": "instrument_master",
                              "issue": f"symbol found but ISIN differs from universe file ({entry.isin})"})
                continue
        if record is None or not key_rule.matches(str(record.get("instrument_key"))):
            notes.append({"symbol": entry.symbol, "what": "instrument_master", "issue": "not an NSE_EQ equity"})
            continue
        key = record["instrument_key"]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        instruments.append(
            {k: record.get(k) for k in ("instrument_key", "isin", "trading_symbol", "name", "exchange", "segment")}
        )
    return instruments


def build_snapshot_content(
    source: MarketSource,
    universe: Sequence[str | UniverseEntry],
    as_of: date,
    source_label: str,
    *,
    universe_provenance: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
    fetch_intraday_for: date | None = None,
) -> SnapshotContent:
    """Fetch, normalise and derive everything for ``universe`` as of ``as_of`` (no disk writes).

    ``fetch_intraday_for`` is today's date for live runs: when it equals ``as_of``, instruments whose
    history lacks that session get it from the intraday endpoint.
    """
    entries = [e if isinstance(e, UniverseEntry) else UniverseEntry(str(e).upper()) for e in universe]
    as_of_text = as_of.isoformat()
    notes: list[dict[str, Any]] = []
    instruments = _resolve_universe(source.instrument_master(), entries, notes)

    keys = [inst["instrument_key"] for inst in instruments]
    raw_news = _fetch(notes, "*", "news", lambda: source.news(keys)) or {}

    candles: dict[str, list[list[Any]]] = {}
    fundamentals: dict[str, dict[str, Any]] = {}
    news: dict[str, list[dict[str, Any]]] = {}
    derived: dict[str, dict[str, float | None]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    from_date = as_of - timedelta(days=CANDLE_LOOKBACK_DAYS)
    expected_columns = {c.name for d in Dimension for c in columns_for(d)}

    for index, inst in enumerate(instruments, start=1):
        key, isin, symbol = inst["instrument_key"], inst["isin"], inst["trading_symbol"]

        # ---- technical --------------------------------------------------------------
        raw_candles = _fetch(notes, symbol, "daily_candles", lambda: source.daily_candles(key, from_date, as_of))
        intraday_used = False
        if fetch_intraday_for == as_of and not any(str(c[0])[:10] == as_of_text for c in raw_candles or []):
            session = _fetch(notes, symbol, "intraday_daily_candle", lambda: source.intraday_daily_candle(key))
            session_rows = [c for c in session or [] if len(c) >= 6 and str(c[0])[:10] == as_of_text]
            if session_rows:
                raw_candles = list(raw_candles or []) + session_rows[:1]
                intraday_used = True
        rows = _normalise_candles(raw_candles or [])
        rows = [r for r in rows if r[0] <= as_of_text]  # lookahead guard
        candles[key] = rows
        tech = indicators.technical_row(rows, as_of)
        tail = rows[-PRICE_HISTORY_SESSIONS:]
        price_ev = _add_evidence(
            evidence, instrument_key=key, dimension=Dimension.TECHNICAL, kind="price_history",
            query=policy.CapabilityName.DAILY_CANDLES.value,
            params={"instrument_key": key, "from_date": from_date.isoformat(), "to_date": as_of_text,
                    "intraday_session": intraday_used},
            as_of=as_of_text, row_key=key,
            fields={"sessions": len(rows), "recent": [{"date": r[0], "close": r[4], "volume": r[5]} for r in tail]},
        )
        _add_evidence(
            evidence, instrument_key=key, dimension=Dimension.TECHNICAL, kind="technical_derived",
            query="derived:technical", params={"version": DERIVED_COLUMNS_VERSION},
            as_of=as_of_text, row_key=key, fields=tech, inputs=[price_ev],
        )

        # ---- fundamental ------------------------------------------------------------
        ratios = _fetch(notes, symbol, "key_ratios", lambda: source.key_ratios(isin))
        yearly = _fetch(notes, symbol, "income_statement_yearly", lambda: source.income_statement(isin, "yearly"))
        quarterly = _fetch(notes, symbol, "income_statement_quarterly",
                           lambda: source.income_statement(isin, "quarterly"))
        balance = _fetch(notes, symbol, "balance_sheet", lambda: source.balance_sheet(isin))
        fundamentals[key] = {"key_ratios": ratios, "income_yearly": yearly,
                             "income_quarterly": quarterly, "balance_sheet": balance}
        fund = indicators.fundamental_row(ratios, yearly, quarterly, balance, as_of)
        fund_inputs = [
            _add_evidence(
                evidence, instrument_key=key, dimension=Dimension.FUNDAMENTAL, kind="key_ratios",
                query=policy.CapabilityName.KEY_RATIOS.value, params={"isin": isin}, as_of=as_of_text,
                row_key=key,
                fields={str(r.get("name")): {"company": r.get("company_value"), "sector": r.get("sector_value")}
                        for r in ratios or []},
            ),
            _add_evidence(
                evidence, instrument_key=key, dimension=Dimension.FUNDAMENTAL, kind="income_statement_yearly",
                query=policy.CapabilityName.INCOME_STATEMENT.value,
                params={"isin": isin, "type": "consolidated", "time_period": "yearly"}, as_of=as_of_text,
                row_key=key, fields=_statement_fields(yearly, 5),
            ),
            _add_evidence(
                evidence, instrument_key=key, dimension=Dimension.FUNDAMENTAL, kind="income_statement_quarterly",
                query=policy.CapabilityName.INCOME_STATEMENT.value,
                params={"isin": isin, "type": "consolidated", "time_period": "quarterly"}, as_of=as_of_text,
                row_key=key, fields=_statement_fields(quarterly, 8),
            ),
            _add_evidence(
                evidence, instrument_key=key, dimension=Dimension.FUNDAMENTAL, kind="balance_sheet",
                query=policy.CapabilityName.BALANCE_SHEET.value, params={"isin": isin, "type": "consolidated"},
                as_of=as_of_text, row_key=key,
                fields={"history": [
                    {k: item.get(k) for k in ("period", "total_asset", "total_liability")}
                    for item in (balance or {}).get("history", [])[:5]
                ]},
            ),
        ]
        _add_evidence(
            evidence, instrument_key=key, dimension=Dimension.FUNDAMENTAL, kind="fundamental_derived",
            query="derived:fundamental", params={"version": DERIVED_COLUMNS_VERSION},
            as_of=as_of_text, row_key=key, fields=fund, inputs=fund_inputs,
        )

        # ---- news -------------------------------------------------------------------
        articles = _normalise_news(raw_news.get(key, []), as_of)
        news[key] = articles
        article_ids = [
            _add_evidence(
                evidence, instrument_key=key, dimension=Dimension.NEWS, kind="news_article",
                query=policy.CapabilityName.NEWS_BY_INSTRUMENT.value,
                params={"category": "instrument_keys", "instrument_key": key}, as_of=as_of_text,
                row_key=f"{key}:{a['published_time_ms']}:{a['heading']}", fields=a,
            )
            for a in articles
        ]
        news_cols = indicators.news_row(articles, as_of)
        _add_evidence(
            evidence, instrument_key=key, dimension=Dimension.NEWS, kind="news_derived",
            query="derived:news", params={"version": DERIVED_COLUMNS_VERSION},
            as_of=as_of_text, row_key=key, fields=news_cols, inputs=article_ids,
        )

        derived[key] = {**tech, **fund, **news_cols}
        if set(derived[key]) != expected_columns:
            raise ValueError(f"derived row for {symbol} does not match the column catalogue")
        if progress is not None:
            progress(index, len(instruments), symbol)

    last_candle = {key: (rows[-1][0] if rows else None) for key, rows in candles.items()}
    known_dates = [d for d in last_candle.values() if d]
    return SnapshotContent(
        prices_as_of=max(known_dates) if known_dates else None,
        instruments_behind_as_of=sum(1 for d in last_candle.values() if d is None or d < as_of_text),
        as_of=as_of_text,
        source=source_label,
        query_set_hash=query_set_hash(),
        universe=keys,
        instruments=instruments,
        candles=candles,
        fundamentals=fundamentals,
        news=news,
        derived=derived,
        evidence=evidence,
        acquisition_notes=notes,
        universe_source=universe_provenance,
    )


def run_acquisition(
    *,
    universe: Sequence[str | UniverseEntry],
    as_of: date,
    snapshot_root: Path,
    source_kind: str,
    audit: AuditLog,
    universe_provenance: dict[str, Any] | None = None,
    checkpoint_root: Path | None = None,
    progress: ProgressCallback | None = None,
    prior_request_ages: Sequence[float] = (),
) -> str:
    """Run phase 1 end to end and return the sealed snapshot's root hash.

    ``source_kind`` is ``"upstox"`` (live, requires ``UPSTOX_ANALYTICS_TOKEN``) or ``"fixture"``.
    With ``checkpoint_root`` set, completed reads are saved so a re-run with the same universe
    and ``as_of`` resumes where an interrupted one stopped.
    """
    gate = None
    if source_kind == "upstox":
        if not session_is_final(as_of):
            raise ValueError(f"as_of {as_of} is not final until {session_final_at(as_of):%Y-%m-%d %H:%M} IST")
        from ..governance.credentials import load_analytics_token
        from ..governance.egress import EgressGate
        from .upstox_adapter import UpstoxAdapter

        gate = EgressGate(load_analytics_token(), audit, prior_request_ages=prior_request_ages)
        source: MarketSource = UpstoxAdapter(gate, audit=audit)
        label = "upstox-analytics"
    elif source_kind == "fixture":
        source = SyntheticMarket(as_of)
        label = "synthetic-fixture"
    else:
        raise ValueError(f"unknown source {source_kind!r}")

    checkpoint = None
    if checkpoint_root is not None:
        not_before = session_final_at(as_of) if source_kind == "upstox" else None
        checkpoint = CheckpointedSource(source, checkpoint_root / f"{as_of.isoformat()}-{label}", not_before=not_before)
        if checkpoint.discarded_stale:
            audit.record("checkpoint.discarded_stale", {"directory": str(checkpoint.directory),
                                                        "not_before": not_before.isoformat() if not_before else None})
        source = checkpoint

    audit.record("phase.enter", {"phase": "1_acquire", "source": label, "universe_entries": len(universe),
                                 "universe_source": universe_provenance, "seal": SEAL.status(),
                                 "query_set_hash": query_set_hash(),
                                 "checkpoint": str(checkpoint.directory) if checkpoint else None})
    try:
        today_ist = datetime.now(_IST).date() if source_kind == "upstox" else None
        content = build_snapshot_content(source, universe, as_of, label,
                                         universe_provenance=universe_provenance, progress=progress,
                                         fetch_intraday_for=today_ist)
        content.request_count = gate.request_count if gate else 0
    except EgressRateLimited as exc:
        audit.record("acquire.rate_limited", {"capability": exc.capability.value,
                                              "requests": gate.request_count if gate else 0,
                                              "checkpointed_reads": checkpoint.misses + checkpoint.hits if checkpoint else 0})
        raise
    finally:
        if gate is not None:
            gate.close()
        if current_role() is ProcessRole.ACQUIRE:
            SEAL.seal()
            audit.record("seal.dropped", SEAL.status())

    root_hash = write_snapshot(content, snapshot_root)
    audit.record("snapshot.sealed", {"root_hash": root_hash, "as_of": content.as_of, "source": label,
                                     "instruments": len(content.instruments),
                                     "acquisition_notes": len(content.acquisition_notes),
                                     "checkpoint_hits": checkpoint.hits if checkpoint else 0,
                                     "prices_as_of": content.prices_as_of})
    if content.prices_as_of != content.as_of:
        audit.record("snapshot.price_lag", {"as_of": content.as_of, "prices_as_of": content.prices_as_of,
                                            "instruments_behind_as_of": content.instruments_behind_as_of})
    if checkpoint is not None:
        checkpoint.discard()
    return root_hash
