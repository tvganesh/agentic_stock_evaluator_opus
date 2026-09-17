"""Tests for the longer-history change (17 Sep 2026).

Two years of candles gave the walk-forward only ~10 windows at the default step, below the 20 its
gates require; four years gives ~34. Upstox allows a decade, but the backtest uses today's index
membership, so deeper history compounds survivorship bias faster than it adds evidence. Larger
candle payloads are kept safe by three properties pinned here: candles are hash-verified but parsed
only when read, per-window indicator work stays bounded, and requests per instrument are unchanged.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

from sealed_window.acquire.etl import CANDLE_LOOKBACK_DAYS
from sealed_window.evaluate.walk_forward import INDICATOR_LOOKBACK_SESSIONS
from sealed_window.governance.errors import SnapshotIntegrityError
from sealed_window.snapshot.store import SealedSnapshot


def test_lookback_gives_enough_windows_without_extra_requests():
    """Four years (~1,000 sessions) clears the 20-window gate, in one call per instrument as before."""
    assert 1400 <= CANDLE_LOOKBACK_DAYS <= 1500
    sessions = CANDLE_LOOKBACK_DAYS * 5 / 7  # trading days per calendar day
    windows = (sessions - 250 - 60) / 20  # warmup, longest horizon, default step
    assert windows >= 20, f"only {windows:.0f} windows would fit the gate"
    assert INDICATOR_LOOKBACK_SESSIONS >= 251  # 52-week high and 1-year return look back 250


def test_candles_are_verified_at_load_but_parsed_lazily(sealed_snapshot):
    """Loading does not parse candles; the first read does, and returns the ascending series."""
    root, root_hash = sealed_snapshot
    snapshot = SealedSnapshot.load(root, root_hash)
    assert snapshot._candles is None, "candles should not be parsed until something reads them"
    key = snapshot.instruments[0]["instrument_key"]
    rows = snapshot.candles_for(key)
    assert snapshot._candles is not None and rows and rows[0][0] < rows[-1][0]


def test_tampered_candles_are_still_detected_at_load(tmp_path, sealed_snapshot):
    """Lazy parsing does not weaken integrity: edited candle bytes fail the hash check at load."""
    root, root_hash = sealed_snapshot
    copy = tmp_path / root_hash
    shutil.copytree(root / root_hash, copy)
    os.chmod(copy, 0o755)
    os.chmod(copy / "candles.json", 0o644)
    candles = json.loads((copy / "candles.json").read_bytes())
    first = next(iter(candles))
    candles[first][0][4] = 999999.0  # rewrite one close price
    (copy / "candles.json").write_text(json.dumps(candles, sort_keys=True, separators=(",", ":")))
    with pytest.raises(SnapshotIntegrityError):
        SealedSnapshot.load(tmp_path, root_hash)


def test_indicator_window_is_capped(snapshot):
    """Each window hands the indicator engine at most INDICATOR_LOOKBACK_SESSIONS candles."""
    from sealed_window.evaluate import walk_forward

    seen: list[int] = []
    original = walk_forward.indicators.technical_row

    def recording(history, as_of):
        """Record how much history each window passes to the indicator engine."""
        seen.append(len(history))
        return original(history, as_of)

    walk_forward.indicators.technical_row = recording
    try:
        walk_forward.backtest_snapshot(
            snapshot,
            __import__("sealed_window.screen.config", fromlist=["ScreenConfig"]).ScreenConfig(),
            walk_forward.BacktestConfig(horizons=(20,), cutoffs=(5,), step_sessions=40,
                                        warmup_sessions=260, min_windows=2),
        )
    finally:
        walk_forward.indicators.technical_row = original
    assert seen and max(seen) <= INDICATOR_LOOKBACK_SESSIONS
