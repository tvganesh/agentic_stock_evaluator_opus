"""Tests for fixes prompted by the first real Claude run (15 Sep 2026, run 20260915T144336Z-4dd5da).

Covers: figures compared by magnitude (7 correct claims were rejected for quoting "4.7% below"
against an evidence value of -4.71), sector P/B, ROA and EV/EBITDA benchmark columns, the
``news_count_window`` rename, the larger veto output limit, and refusal of snapshots built with a
different derived-column version.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from conftest import AS_OF
from sealed_window.snapshot import indicators as ind
from sealed_window.acquire.etl import build_snapshot_content
from sealed_window.acquire.fixture_source import SyntheticMarket
from sealed_window.agents.offline_model import OfflineHeuristicModel
from sealed_window.app.server import create_app
from sealed_window.claims.validator import unsupported_figures
from sealed_window.governance.errors import SnapshotIncompatible
from sealed_window.governance.seal import SEAL
from sealed_window.governance.spend import DEFAULT_SLOT_SPECS, SlotClass, compile_plan
from sealed_window.orchestrator import prepare_run
from sealed_window.screen.config import ScreenConfig
from sealed_window.snapshot.columns import COLUMNS
from sealed_window.snapshot.store import write_snapshot


def test_figures_are_compared_by_magnitude():
    """Direction words carry the sign, so "3.91% below" matches -3.913; invented numbers still fail."""
    records = [{"fields": {"price_vs_sma20_pct": -3.913074, "drawdown_from_52w_high_pct": -21.26,
                           "liabilities_to_equity_delta_1y": -0.42}}]
    assert unsupported_figures("Price is 3.91% below its 20-day average and 21.26% below its high.", records) == []
    assert unsupported_figures("Leverage fell by 0.42x year over year.", records) == []
    assert unsupported_figures("Price is -3.91% versus its 20-day average.", records) == []
    assert unsupported_figures("ROE is a remarkable 9999.99% this year.", records) == ["9999.99"]


def test_sector_benchmarks_are_columns():
    """Sector P/B, ROA and EV/EBITDA come from the Upstox key-ratios example payload."""
    ratios = [
        {"name": "P/E", "company_value": "20.15", "sector_value": "12.46"},
        {"name": "P/B", "company_value": "2.13", "sector_value": "1.53"},
        {"name": "ROA", "company_value": "4.39%", "sector_value": "7.54%"},
        {"name": "ROE", "company_value": "8.94%", "sector_value": "16.46%"},
        {"name": "ROCE", "company_value": "10.39%", "sector_value": "16.9%"},
        {"name": "EV/EBITDA", "company_value": "10.25", "sector_value": "6.94"},
    ]
    row = ind.fundamental_row(ratios, None, None, None, date(2026, 9, 15))
    assert (row["sector_pb"], row["sector_roa_pct"], row["sector_ev_ebitda"]) == (1.53, 7.54, 6.94)


def test_news_column_is_named_for_what_it_counts():
    """The misleading 30-day name is gone; the window column documents Upstox's short news history."""
    assert "news_count_30d" not in COLUMNS
    assert "7 days" in COLUMNS["news_count_window"].description


def test_veto_output_limit_and_commitment():
    """Veto slots allow 16,000 output tokens; 20 candidates commit claim slots plus 5 veto batches."""
    veto = next(spec for spec in DEFAULT_SLOT_SPECS if spec.slot_class is SlotClass.VETO)
    assert veto.max_out == 16_000
    plan = compile_plan(snapshot_hash="a" * 64, screen_config_hash="b" * 64, candidate_count=20)
    per_candidate = (11_000 * 2 + 4_000 * 10) + (8_000 * 2 + 3_000 * 10) + (9_000 * 1 + 1_500 * 5)
    assert plan.committed_total_microusd == 20 * per_candidate + 5 * (40_000 * 2 + 16_000 * 10)


@pytest.fixture
def old_version_snapshot(tmp_path, monkeypatch):
    """A snapshot sealed under a different derived-column version label."""
    monkeypatch.setattr("sealed_window.snapshot.store.DERIVED_COLUMNS_VERSION", "derived-v0")
    content = build_snapshot_content(SyntheticMarket(AS_OF), ["SYNTH01", "SYNTH02"], AS_OF, "synthetic-fixture")
    return tmp_path, write_snapshot(content, tmp_path)


def test_snapshot_from_another_column_version_is_refused(old_version_snapshot):
    """plan/evaluate refuse a mismatched snapshot rather than run with disagreeing column names."""
    root, root_hash = old_version_snapshot
    with pytest.raises(SnapshotIncompatible):
        prepare_run(root, root_hash, ScreenConfig())


def test_dashboard_reports_version_mismatch_as_conflict(old_version_snapshot, tmp_path):
    """The dashboard answers 409 with a message telling the operator to re-acquire."""
    SEAL.seal()
    root, root_hash = old_version_snapshot
    client = TestClient(create_app(snapshot_root=root, runs_root=tmp_path / "runs",
                                   client_factory=lambda mode: OfflineHeuristicModel()))
    response = client.post("/api/plan", json={"snapshot_hash": root_hash, "config": {}})
    assert response.status_code == 409 and "fresh acquisition" in response.json()["detail"]
