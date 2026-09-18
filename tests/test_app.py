"""Web app tests: governance endpoint, strict request validation, approve-a-number semantics, and
reading a finished run back from disk so a command-line run can be viewed without paying twice."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from sealed_window.agents.offline_model import OfflineHeuristicModel
from sealed_window.app.server import create_app
from sealed_window.governance.seal import SEAL
from sealed_window.orchestrator import Orchestrator, RunRequest, RunState, prepare_run
from sealed_window.screen.config import ScreenConfig


@pytest.fixture
def app_client(sealed_snapshot, tmp_path):
    """A TestClient over the app with the offline model, in sealed process state."""
    SEAL.seal()
    root, root_hash = sealed_snapshot
    app = create_app(snapshot_root=root, runs_root=tmp_path, client_factory=lambda mode, local_model=None: OfflineHeuristicModel())
    return TestClient(app), root_hash


def test_a_finished_run_is_readable_by_a_server_that_never_ran_it(sealed_snapshot, tmp_path):
    """A run written by the orchestrator opens in a fresh server, from disk alone.

    The CLI and the dashboard write the same artefacts, so a run started from the shell has to be
    viewable here. Without this the only way to see a result in the UI is to run it again, which
    means paying for the same answer twice.
    """
    SEAL.seal()
    root, root_hash = sealed_snapshot
    runs_root = tmp_path / "runs"
    config = ScreenConfig()
    prepared = prepare_run(root, root_hash, config)
    state = RunState()
    Orchestrator(snapshot_root=root, runs_root=runs_root,
                 client_factory=lambda mode, local_model=None: OfflineHeuristicModel()).run(
        RunRequest(snapshot_hash=root_hash, screen_config=config,
                   approved_plan_hash=prepared.plan.plan_hash), state)

    # A brand-new app: this run exists nowhere in its memory, only on disk.
    client = TestClient(create_app(snapshot_root=root, runs_root=runs_root,
                                   client_factory=lambda mode, local_model=None: OfflineHeuristicModel()))

    assert [r["run_id"] for r in client.get("/api/runs").json()] == [state.run_id]

    loaded = client.get(f"/api/runs/{state.run_id}").json()
    assert loaded["status"] == "done" and loaded["historical"] is True
    assert loaded["seal"] is None, "a finished run has no live seal; inventing one would imply it is running"
    assert loaded["dossier"]["header"]["funnel"]["universe"] > 0
    assert loaded["claims"], "the claim ledger rebuilds from the stored dossier, without loading a snapshot"
    assert {"symbol", "dimension", "statement", "falsifier", "verdict"} <= set(loaded["claims"][0])
    assert "# Sealed Window dossier" in client.get(f"/api/runs/{state.run_id}/dossier.md").text


def test_unknown_and_malformed_run_ids_fail_closed(app_client):
    """A run id that names nothing is a 404, and one that tries to escape runs_root never resolves."""
    client, _ = app_client
    assert client.get("/api/runs/20260101T000000Z-abcdef").status_code == 404
    for hostile in ("..", "..%2F..%2Fetc", "a/b"):
        assert client.get(f"/api/runs/{hostile}").status_code in (400, 404), hostile


def test_index_and_governance(app_client):
    """The page loads, and governance lists exactly the seven allowlisted capabilities and no account paths."""
    client, _ = app_client
    assert "SEALED WINDOW" in client.get("/").text
    gov = client.get("/api/governance").json()
    paths = [c["path"] for c in gov["upstox"]["capabilities"]]
    assert len(paths) == 7 and not any(word in p for p in paths for word in ("portfolio", "order", "user", "funds"))
    assert gov["upstox"]["method"] == "GET" and gov["seal"]["network_live"] is False


def test_unknown_config_fields_are_rejected(app_client):
    """Extra fields in the screen config (e.g. text aimed at a prompt) are a 422."""
    client, root_hash = app_client
    response = client.post("/api/plan", json={"snapshot_hash": root_hash, "config": {"note": "buy everything"}})
    assert response.status_code == 422


def test_run_requires_approval_of_the_exact_plan(app_client):
    """A stale or wrong approval is refused with 409; the matching approval runs to a dossier."""
    client, root_hash = app_client
    config = {"roe_min_pct": 12, "max_candidates": 8}
    plan = client.post("/api/plan", json={"snapshot_hash": root_hash, "config": config}).json()
    body = {"snapshot_hash": root_hash, "config": config, "model_mode": "offline"}

    refused = client.post("/api/runs", json={**body, "approved_plan_hash": "0" * 64})
    assert refused.status_code == 409 and refused.json()["detail"]["compiled_plan_hash"] == plan["plan_hash"]

    started = client.post("/api/runs", json={**body, "approved_plan_hash": plan["plan_hash"]})
    assert started.status_code == 202
    run_id = started.json()["run_id"]
    deadline = time.monotonic() + 120
    while (state := client.get(f"/api/runs/{run_id}").json())["status"] == "running":
        assert time.monotonic() < deadline, "run did not finish"
        time.sleep(0.2)
    assert state["status"] == "done", state["error"]
    assert "Sealed Window dossier" in client.get(f"/api/runs/{run_id}/dossier.md").text
