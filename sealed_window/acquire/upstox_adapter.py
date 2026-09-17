"""Upstox adapter: typed wrappers over the seven allowlisted Analytics-token reads.

Each method names a capability and passes typed parameters to the egress gate; none of
them ever sees or builds a URL, holds the token, or can issue anything but the gate's
``GET``. The adapter's job is purely to unwrap Upstox's ``{"status": ..., "data": ...}``
envelope and hand plain Python structures to the ETL.

News pagination is audited page by page (``news.page`` events with article counts and the
vendor's pagination metadata), because the first Nifty 500 run returned headlines covering
only about five days and the audit is how we tell a vendor limit from a pagination bug.

There is intentionally no method for orders, portfolio, holdings, positions, funds,
profile, P&L, GTT, mutual funds, option chains or the WebSocket feed. Adding one requires
a new entry in ``policy.CAPABILITIES`` (reviewed) *and* a method here (reviewed).
"""

from __future__ import annotations

import json
import zlib
from datetime import date
from typing import Any

from ..governance.audit import AuditLog
from ..governance.egress import EgressGate
from ..governance.policy import CapabilityName

MAX_DECOMPRESSED_BYTES = 256 * 1024 * 1024
"""Guard against a decompression bomb in the instrument master download."""


class AdapterError(RuntimeError):
    """A permitted response that was malformed or reported a non-success status."""


def _bounded_gunzip(body: bytes, limit: int = MAX_DECOMPRESSED_BYTES) -> bytes:
    """Decompress gzip ``body`` but refuse to inflate beyond ``limit`` bytes."""
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = decompressor.decompress(body, limit)
    if decompressor.unconsumed_tail:
        raise AdapterError("instrument master exceeds the decompression limit")
    return out + decompressor.flush()


def _payload(body: bytes, capability: CapabilityName) -> dict[str, Any]:
    """Parse an Upstox JSON envelope and return it whole; raise :class:`AdapterError` unless it succeeded."""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"{capability.value}: response is not JSON") from exc
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise AdapterError(f"{capability.value}: non-success status")
    return payload


def _unwrap(body: bytes, capability: CapabilityName) -> Any:
    """Parse an Upstox JSON envelope and return its ``data`` member."""
    return _payload(body, capability).get("data")


class UpstoxAdapter:
    """The ETL's market source backed by the live Upstox Analytics API (through the egress gate)."""

    NEWS_BATCH_SIZE = 30
    NEWS_PAGE_SIZE = 100
    NEWS_MAX_PAGES = 3

    def __init__(self, gate: EgressGate, audit: AuditLog | None = None) -> None:
        """Bind to the egress gate (and optionally the acquisition audit log for pagination records)."""
        self._gate = gate
        self._audit = audit

    def instrument_master(self) -> list[dict[str, Any]]:
        """Download the NSE instrument master and keep only cash-market equities (``NSE_EQ``/``EQ``)."""
        body = self._gate.get(CapabilityName.INSTRUMENT_MASTER)
        try:
            # The file is gzip; if the server sent it with Content-Encoding the client already decoded it.
            rows = json.loads(_bounded_gunzip(body) if body[:2] == b"\x1f\x8b" else body)
        except (zlib.error, json.JSONDecodeError) as exc:
            raise AdapterError("instrument master could not be decoded") from exc
        return [
            row
            for row in rows
            if isinstance(row, dict) and row.get("segment") == "NSE_EQ" and row.get("instrument_type") == "EQ"
        ]

    def daily_candles(self, instrument_key: str, from_date: date, to_date: date) -> list[list[Any]]:
        """Fetch daily candles ``[timestamp, open, high, low, close, volume, oi]`` for a date range."""
        data = _unwrap(
            self._gate.get(
                CapabilityName.DAILY_CANDLES,
                path_params={
                    "instrument_key": instrument_key,
                    "to_date": to_date.isoformat(),
                    "from_date": from_date.isoformat(),
                },
            ),
            CapabilityName.DAILY_CANDLES,
        )
        return list((data or {}).get("candles", []))

    def intraday_daily_candle(self, instrument_key: str) -> list[list[Any]]:
        """Fetch the current session's daily candle, which the historical candle API does not include."""
        data = _unwrap(
            self._gate.get(CapabilityName.INTRADAY_DAILY_CANDLE, path_params={"instrument_key": instrument_key}),
            CapabilityName.INTRADAY_DAILY_CANDLE,
        )
        return list((data or {}).get("candles", []))

    def key_ratios(self, isin: str) -> list[dict[str, Any]]:
        """Fetch key ratios with sector benchmarks (P/E, ROE, ROCE...; NIM, Net NPA, CASA for banks)."""
        data = _unwrap(
            self._gate.get(CapabilityName.KEY_RATIOS, path_params={"isin": isin}), CapabilityName.KEY_RATIOS
        )
        return list(data or [])

    def income_statement(self, isin: str, time_period: str) -> dict[str, Any]:
        """Fetch the consolidated income statement, ``yearly`` or ``quarterly``."""
        data = _unwrap(
            self._gate.get(
                CapabilityName.INCOME_STATEMENT,
                path_params={"isin": isin},
                query_params={"type": "consolidated", "time_period": time_period},
            ),
            CapabilityName.INCOME_STATEMENT,
        )
        return dict(data or {})

    def balance_sheet(self, isin: str) -> dict[str, Any]:
        """Fetch the consolidated yearly balance sheet summary."""
        data = _unwrap(
            self._gate.get(
                CapabilityName.BALANCE_SHEET, path_params={"isin": isin}, query_params={"type": "consolidated"}
            ),
            CapabilityName.BALANCE_SHEET,
        )
        return dict(data or {})

    def news(self, instrument_keys: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Fetch headlines in batches of 30 keys (``category=instrument_keys`` only), auditing every page."""
        merged: dict[str, list[dict[str, Any]]] = {}
        for batch_index, start in enumerate(range(0, len(instrument_keys), self.NEWS_BATCH_SIZE)):
            batch = instrument_keys[start : start + self.NEWS_BATCH_SIZE]
            for page in range(1, self.NEWS_MAX_PAGES + 1):
                payload = _payload(
                    self._gate.get(
                        CapabilityName.NEWS_BY_INSTRUMENT,
                        query_params={
                            "category": "instrument_keys",
                            "instrument_keys": ",".join(batch),
                            "page_number": str(page),
                            "page_size": str(self.NEWS_PAGE_SIZE),
                        },
                    ),
                    CapabilityName.NEWS_BY_INSTRUMENT,
                )
                data = payload.get("data")
                page_items = 0
                with_articles = 0
                blocks = data if isinstance(data, list) else [data or {}]
                for block in blocks:
                    for key, articles in (block or {}).items():
                        if key in batch and isinstance(articles, list):
                            merged.setdefault(key, []).extend(a for a in articles if isinstance(a, dict))
                            page_items += len(articles)
                            with_articles += 1 if articles else 0
                if self._audit is not None:
                    self._audit.record("news.page", {
                        "batch": batch_index,
                        "page": page,
                        "instruments_requested": len(batch),
                        "instruments_with_articles": with_articles,
                        "articles": page_items,
                        # Serialised so vendor key names cannot trip the audit log's secret-key guard.
                        "metadata": json.dumps(payload.get("metadata"), sort_keys=True)[:500],
                    })
                if page_items < self.NEWS_PAGE_SIZE:
                    break
        return merged
