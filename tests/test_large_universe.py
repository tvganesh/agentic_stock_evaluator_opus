"""Tests for large-universe acquisition (Nifty 500): rate limiting, 429 handling, universe CSV
parsing with ISIN resolution, checkpoint resume, and the request budget.
"""

from __future__ import annotations

import math
import re

import httpx
import pytest

from conftest import AS_OF, PROJECT_ROOT
from sealed_window.acquire.checkpoint import CheckpointedSource
from sealed_window.acquire.etl import UniverseEntry, build_snapshot_content, read_universe
from sealed_window.acquire.fixture_source import SyntheticMarket, synthetic_isin
from sealed_window.acquire.upstox_adapter import UpstoxAdapter
from sealed_window.governance import policy
from sealed_window.governance.audit import AuditLog
from sealed_window.governance.credentials import AnalyticsToken
from sealed_window.governance.egress import EgressGate, EgressRateLimited
from sealed_window.governance.policy import CapabilityName as C
from sealed_window.governance.ratelimit import RollingRateLimiter
from sealed_window.governance.seal import SEAL
from sealed_window.snapshot.store import write_snapshot


class FakeTime:
    """A controllable clock whose sleep advances time instantly."""

    def __init__(self) -> None:
        """Start the clock at zero with no sleeps recorded."""
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        """Return the current fake time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance fake time and record the sleep."""
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_waits_for_the_tightest_window():
    """With 2/second and 3/minute, the 4th request waits until the minute window frees a slot."""
    t = FakeTime()
    limiter = RollingRateLimiter([(1.0, 2), (60.0, 3)], clock=t.clock, sleep=t.sleep)
    assert [limiter.acquire() for _ in range(2)] == [0.0, 0.0]
    assert limiter.acquire() == pytest.approx(1.0)  # per-second window full
    assert limiter.acquire() == pytest.approx(59.0)  # per-minute window full until t=60
    assert t.now == pytest.approx(60.0)


def test_policy_limits_stay_below_upstox_published_limits():
    """Our client-side ceilings are strictly below Upstox's 50/s, 500/min, 2000/30min."""
    published = {1.0: 50, 60.0: 500, 1800.0: 2000}
    for seconds, count in policy.UPSTOX_RATE_LIMITS:
        assert count < published[seconds]


def test_nifty500_request_budget_fits_the_cap():
    """A same-day Nifty 500 acquisition (6 reads per stock incl. intraday + all news pages + master) fits the cap."""
    stocks = 505
    worst_case = stocks * 6 + math.ceil(stocks / UpstoxAdapter.NEWS_BATCH_SIZE) * UpstoxAdapter.NEWS_MAX_PAGES + 1
    assert worst_case <= policy.MAX_REQUESTS_PER_ACQUISITION


def test_http_429_raises_rate_limited_and_aborts_the_build():
    """A 429 is not swallowed as a per-stock note: it stops the acquisition."""
    SEAL.open_data_network()
    audit = AuditLog()
    gate = EgressGate(AnalyticsToken("t" * 40), audit, sleep=lambda _: None,
                      transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    with pytest.raises(EgressRateLimited):
        gate.get(C.KEY_RATIOS, path_params={"isin": "INE002A01018"})
    assert audit.entries("egress.http_error")[0]["detail"]["status"] == 429

    class RateLimitedMarket(SyntheticMarket):
        """Fixture market whose key-ratios endpoint is rate limited."""

        def key_ratios(self, isin):
            """Simulate Upstox answering 429."""
            raise EgressRateLimited(C.KEY_RATIOS, 429)

    with pytest.raises(EgressRateLimited):
        build_snapshot_content(RateLimitedMarket(AS_OF), ["SYNTH01"], AS_OF, "synthetic-fixture")


def test_universe_csv_resolves_by_isin_and_refuses_mismatches(tmp_path):
    """ISIN wins over symbol; a symbol whose ISIN disagrees with the file is skipped with a note."""
    csv_file = tmp_path / "index.csv"
    csv_file.write_text(
        "Company Name,Industry,Symbol,Series,ISIN Code\n"
        f"Synthetic One,Test,SYNTH01,EQ,{synthetic_isin(1)}\n"
        f"Renamed Two,Test,RENAMED,EQ,{synthetic_isin(2)}\n"
        "Mismatch Three,Test,SYNTH03,EQ,INE999Z01019\n",
        encoding="utf-8",
    )
    entries = read_universe(csv_file)
    assert entries[1] == UniverseEntry("RENAMED", synthetic_isin(2))
    content = build_snapshot_content(SyntheticMarket(AS_OF), entries, AS_OF, "synthetic-fixture")
    assert [i["trading_symbol"] for i in content.instruments] == ["SYNTH01", "SYNTH02"]
    assert any(n["symbol"] == "SYNTH03" and "ISIN differs" in n["issue"] for n in content.acquisition_notes)


def test_checkpoint_resume_needs_no_requests_and_gives_the_same_snapshot(tmp_path):
    """A resumed build serves every read from checkpoints and produces an identical snapshot hash."""
    symbols = ["SYNTH01", "SYNTH02", "SYNTH03"]
    first_source = CheckpointedSource(SyntheticMarket(AS_OF), tmp_path / "ckpt")
    first = write_snapshot(build_snapshot_content(first_source, symbols, AS_OF, "synthetic-fixture"), tmp_path / "a")

    class Offline(SyntheticMarket):
        """A source that fails if anything is actually fetched."""

        def __getattribute__(self, name):
            """Refuse every data method so only checkpoint hits can succeed."""
            if name in {"instrument_master", "daily_candles", "key_ratios", "income_statement",
                        "balance_sheet", "news"}:
                raise AssertionError(f"{name} should have come from the checkpoint")
            return super().__getattribute__(name)

    resumed_source = CheckpointedSource(Offline(AS_OF), tmp_path / "ckpt")
    resumed = write_snapshot(build_snapshot_content(resumed_source, symbols, AS_OF, "synthetic-fixture"), tmp_path / "b")
    assert resumed == first and resumed_source.misses == 0 and resumed_source.hits > 0


def test_bundled_nifty500_file_is_well_formed():
    """config/nifty500.csv has ~500 unique constituents with valid ISINs (NSE lists can briefly hold 501)."""
    entries = read_universe(PROJECT_ROOT / "config" / "nifty500.csv")
    assert 500 <= len(entries) <= 505
    assert len({e.isin for e in entries}) == len(entries)
    assert all(e.isin and re.fullmatch(policy.ISIN_RE, e.isin) for e in entries)
