"""Price-date integrity tests.

The live 15 Sep 2026 acquisition sealed a snapshot "as of 15 Sep" whose prices stopped at 11 Sep,
because Upstox's historical candle API excludes the current day. These tests pin the two fixes:
today's session is filled from the intraday endpoint, and every snapshot records how far its
prices lag ``as_of``.
"""

from __future__ import annotations

from datetime import timedelta

from conftest import AS_OF
from sealed_window.acquire.etl import build_snapshot_content
from sealed_window.acquire.fixture_source import SyntheticMarket
from sealed_window.snapshot.store import SealedSnapshot, write_snapshot

SYMBOLS = ["SYNTH01", "SYNTH02"]
SESSION_CANDLE = [f"{AS_OF.isoformat()}T09:15:00+05:30", 100.0, 105.0, 99.0, 104.0, 12345, 0]


class SessionMissingMarket(SyntheticMarket):
    """Fixture market that behaves like Upstox on the day: history stops before as_of."""

    def __init__(self, as_of) -> None:
        """Start with no intraday requests counted."""
        super().__init__(as_of)
        self.intraday_calls = 0

    def daily_candles(self, instrument_key, from_date, to_date):
        """Synthetic history with the to_date session removed."""
        return [c for c in super().daily_candles(instrument_key, from_date, to_date)
                if not c[0].startswith(to_date.isoformat())]

    def intraday_daily_candle(self, instrument_key):
        """Return one current-session daily candle and count the request."""
        self.intraday_calls += 1
        return [SESSION_CANDLE]


class CountingMarket(SyntheticMarket):
    """Fixture market with complete history that counts intraday requests (there should be none)."""

    def __init__(self, as_of) -> None:
        """Start with no intraday requests counted."""
        super().__init__(as_of)
        self.intraday_calls = 0

    def intraday_daily_candle(self, instrument_key):
        """Count the request and return nothing."""
        self.intraday_calls += 1
        return []


def test_price_lag_is_recorded_in_the_manifest(tmp_path):
    """Without the intraday fill, prices_as_of is the previous session and every instrument is behind."""
    market = SessionMissingMarket(AS_OF)
    content = build_snapshot_content(market, SYMBOLS, AS_OF, "synthetic-fixture")
    assert market.intraday_calls == 0
    assert content.prices_as_of == (AS_OF - timedelta(days=1)).isoformat()
    assert content.instruments_behind_as_of == 2
    snapshot = SealedSnapshot.load(tmp_path, write_snapshot(content, tmp_path))
    assert snapshot.prices_as_of == content.prices_as_of
    assert snapshot.manifest["instruments_behind_as_of"] == 2


def test_intraday_candle_fills_todays_session():
    """When as_of is today, the missing session comes from the intraday endpoint and is traceable in evidence."""
    market = SessionMissingMarket(AS_OF)
    content = build_snapshot_content(market, SYMBOLS, AS_OF, "synthetic-fixture", fetch_intraday_for=AS_OF)
    assert market.intraday_calls == len(SYMBOLS)
    assert content.prices_as_of == AS_OF.isoformat() and content.instruments_behind_as_of == 0
    key = content.instruments[0]["instrument_key"]
    assert content.candles[key][-1] == [AS_OF.isoformat(), 100.0, 105.0, 99.0, 104.0, 12345.0]
    price = next(r for r in content.evidence.values() if r["kind"] == "price_history" and r["instrument_key"] == key)
    assert price["params"]["intraday_session"] is True


def test_no_intraday_request_when_history_has_the_session_or_date_is_not_today():
    """The extra request is spent only when as_of is today and the history lacks that session."""
    complete = CountingMarket(AS_OF)
    build_snapshot_content(complete, SYMBOLS, AS_OF, "synthetic-fixture", fetch_intraday_for=AS_OF)
    assert complete.intraday_calls == 0
    missing = SessionMissingMarket(AS_OF)
    build_snapshot_content(missing, SYMBOLS, AS_OF, "synthetic-fixture", fetch_intraday_for=AS_OF + timedelta(days=3))
    assert missing.intraday_calls == 0
