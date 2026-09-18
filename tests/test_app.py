"""Web app tests: governance endpoint, strict request validation, and approve-a-number semantics."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from sealed_window.agents.offline_model import OfflineHeuristicModel
from sealed_window.app.server import create_app
from sealed_window.governance.seal import SEAL


@pytest.fixture
def app_client(sealed_snapshot, tmp_path):
    """A TestClient over the app with the offline model, in sealed process state."""
    SEAL.seal()
    root, root_hash = sealed_snapshot
    app = create_app(snapshot_root=root, runs_root=tmp_path, client_factory=lambda mode, local_model=None: OfflineHeuristicModel())
    return TestClient(app), root_hash


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
