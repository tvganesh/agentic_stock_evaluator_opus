"""Snapshot (P0) and indicator engine (P2) tests.

P0: identical inputs produce identical snapshot hashes; tampering is detected; files are
read-only; evidence IDs are deterministic. P2: indicators match hand-verifiable series and the
growth calculation reconciles with the change Upstox itself reports in its documentation.
"""

from __future__ import annotations

import os
import shutil
from datetime import date

import pytest

from conftest import AS_OF
from sealed_window.snapshot import indicators as ind
from sealed_window.acquire.etl import _normalise_news, build_snapshot_content
from sealed_window.acquire.fixture_source import SYNTHETIC_SYMBOLS, SyntheticMarket
from sealed_window.governance.errors import SnapshotIntegrityError
from sealed_window.snapshot.hashing import evidence_id
from sealed_window.snapshot.store import SealedSnapshot, write_snapshot


def test_identical_inputs_produce_identical_hashes(tmp_path, sealed_snapshot):
    """Two independent builds over the same inputs yield the same root hash."""
    content = build_snapshot_content(SyntheticMarket(AS_OF), list(SYNTHETIC_SYMBOLS), AS_OF, "synthetic-fixture")
    assert write_snapshot(content, tmp_path) == sealed_snapshot[1]


def test_tampering_is_detected(tmp_path, sealed_snapshot):
    """Changing one byte of a sealed file makes loading fail."""
    root, root_hash = sealed_snapshot
    copy = tmp_path / root_hash
    shutil.copytree(root / root_hash, copy)
    target = copy / "derived.json"
    os.chmod(copy, 0o755)
    os.chmod(target, 0o644)
    target.write_bytes(target.read_bytes().replace(b'"roe_pct":', b'"roe_pct": ', 1))
    with pytest.raises(SnapshotIntegrityError):
        SealedSnapshot.load(tmp_path, root_hash)


def test_snapshot_files_are_read_only(sealed_snapshot):
    """Sealed files carry no write permission."""
    root, root_hash = sealed_snapshot
    for path in (root / root_hash).iterdir():
        assert not os.stat(path).st_mode & 0o222


def test_evidence_ids_are_deterministic_and_indexed(snapshot):
    """Evidence IDs are stable functions of their inputs and every derived row has evidence."""
    assert evidence_id("q", {"a": 1}, "2026-09-11", "k") == evidence_id("q", {"a": 1}, "2026-09-11", "k")
    key = snapshot.instruments[0]["instrument_key"]
    kinds = {rec["kind"] for rec in snapshot.evidence_for(key)}
    assert {"technical_derived", "fundamental_derived", "news_derived", "key_ratios"} <= kinds


def test_news_after_as_of_is_excluded():
    """Lookahead guard: headlines published after as_of never enter a snapshot."""
    future_ms = 1_789_000_000_000  # 2026-09-09 IST is ~1.7894e12; this is in the past window
    articles = [
        {"heading": "inside window", "published_time": future_ms},
        {"heading": "future", "published_time": 1_900_000_000_000},
    ]
    kept = _normalise_news(articles, AS_OF)
    assert [a["heading"] for a in kept] == ["inside window"]


def test_indicators_on_hand_verifiable_series():
    """RSI, MACD, ATR, volatility and returns behave exactly on trivial series."""
    rising = [float(i) for i in range(1, 60)]
    assert ind.rsi_wilder(rising) == 100.0
    assert ind.rsi_wilder(list(reversed(rising))) == 0.0
    assert ind.macd([10.0] * 60) == (0.0, 0.0, 0.0)
    assert ind.atr_wilder([12.0] * 30, [10.0] * 30, [11.0] * 30) == 2.0
    assert ind.annualised_volatility_pct([100.0 * 1.01 ** i for i in range(30)]) == pytest.approx(0.0, abs=1e-9)
    assert ind.pct_return([100.0, 110.0], 1) == pytest.approx(10.0)
    assert ind.sma([1, 2, 3, 4], 2) == 3.5


def test_vendor_parsing_and_growth_reconcile_with_upstox_example():
    """Revenue 1,086,181 vs 982,671 gives +10.53%, the change Upstox's own example reports."""
    statement = {"income_statement": [{"category": "revenue", "history": [
        {"value": 1086181, "period": "Mar 2026", "change": "+10.53%"},
        {"value": 982671, "period": "Mar 2025", "change": "+7.15%"},
    ]}]}
    row = ind.fundamental_row([{"name": "ROE", "company_value": "8.94%", "sector_value": "16.46%"}],
                              statement, None, None, date(2026, 9, 11))
    assert round(row["revenue_growth_1y_pct"], 2) == 10.53
    assert row["roe_pct"] == 8.94 and row["sector_roe_pct"] == 16.46
    assert ind.parse_period("Mar 2026") == date(2026, 3, 31)
    assert ind.parse_number("-") is None
