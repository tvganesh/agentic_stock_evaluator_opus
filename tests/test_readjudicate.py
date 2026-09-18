"""Tests for ``readjudicate``: re-running phases 4-6 over a stored run's claims.

Adjudication rules change, and when they do a stored run disagrees with what the same claims would
yield today. This command recomputes only the part that never involved a model -- adjudication,
scoring and the report -- so the question can be answered without paying for a second run that would
return different claims anyway.

Two properties matter enough to pin:

* **It reproduces.** The same claims and snapshot yield the same verdicts, and the source run is left
  exactly as it was.
* **It refuses.** Rebuilding a dossier from settings other than the ones that produced the run would
  yield a report that looks authoritative and is not, so a hash mismatch must stop the command dead.
  That guard is the reason this file exists: it was shipped unexercised, and a typo in the comparison
  would have let a mismatched rebuild through in silence.

Run through a subprocess rather than by calling ``cmd_readjudicate`` directly. The command claims the
SEALED process role, roles are taken once per process, and the test suite has already imported
``sealed_window.acquire.etl`` (forbidden in that role) via conftest -- so an in-process call would
fail for reasons unrelated to what is being tested. The subprocess also exercises the real entry
point: argument parsing, exit codes and all.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from conftest import PROJECT_ROOT
from sealed_window.agents.offline_model import OfflineHeuristicModel
from sealed_window.governance.audit import AuditLog
from sealed_window.governance.seal import SEAL
from sealed_window.orchestrator import Orchestrator, RunRequest, RunState, prepare_run
from sealed_window.screen.config import ScreenConfig


def _readjudicate(tmp_path, snapshot_root, run_id, screen="config/screen.default.json"):
    """Invoke the CLI against a temp tree and return the completed process."""
    return subprocess.run(
        [sys.executable, "-m", "sealed_window",
         "--snapshots", str(snapshot_root), "--runs", str(tmp_path / "runs"),
         "readjudicate", "--run", run_id, "--screen", screen, "--model", "offline"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180,
    )


@pytest.fixture
def stored_run(sealed_snapshot, tmp_path):
    """A finished offline run on disk, as the orchestrator writes one.

    The screen comes from the committed config file rather than ``ScreenConfig()``: the two have
    drifted apart (the defaults hash to 1ea46d5e… and the file to 92cd0bf0…), and the command
    reconciles against whatever ``--screen`` names. Loading the same file the CLI is pointed at is
    both correct here and a guard against that file silently diverging again.
    """
    SEAL.seal()  # clean_seal resets the process seal per test; a model window requires a sealed one
    root, root_hash = sealed_snapshot
    config = ScreenConfig.model_validate(
        json.loads((PROJECT_ROOT / "config" / "screen.default.json").read_text(encoding="utf-8")))
    prepared = prepare_run(root, root_hash, config)
    state = RunState()
    Orchestrator(snapshot_root=root, runs_root=tmp_path / "runs",
                 client_factory=lambda mode, local_model=None: OfflineHeuristicModel()).run(
        RunRequest(snapshot_hash=root_hash, screen_config=config,
                   approved_plan_hash=prepared.plan.plan_hash), state)
    return root, state.run_id


def test_readjudication_reproduces_the_run_and_leaves_it_untouched(stored_run, tmp_path):
    """The same claims and snapshot yield the same verdicts, written to a new run directory."""
    snapshot_root, run_id = stored_run
    source = tmp_path / "runs" / run_id
    before = {name: (source / name).read_bytes() for name in ("dossier.json", "claims.json")}

    result = _readjudicate(tmp_path, snapshot_root, run_id)
    assert result.returncode == 0, result.stderr

    rebuilt_id = next(line.split("-> ")[1].strip() for line in result.stdout.splitlines() if "-> " in line)
    rebuilt = tmp_path / "runs" / rebuilt_id
    assert rebuilt != source and rebuilt.is_dir()

    assert {name: (source / name).read_bytes() for name in before} == before, \
        "the source run must never be modified"

    old = json.loads(before["claims.json"])
    new = json.loads((rebuilt / "claims.json").read_text(encoding="utf-8"))
    assert [c["claim_id"] for c in new["claims"]] == [c["claim_id"] for c in old["claims"]]
    assert {k: v["verdict"] for k, v in new["verdicts"].items()} == \
           {k: v["verdict"] for k, v in old["verdicts"].items()}, "verdicts must reproduce exactly"

    rebuilt_dossier = json.loads((rebuilt / "dossier.json").read_text(encoding="utf-8"))
    assert rebuilt_dossier["header"]["funnel"] == json.loads(before["dossier.json"])["header"]["funnel"]
    assert any(run_id in note for note in rebuilt_dossier["header"]["notes"]), \
        "the rebuilt dossier states which run it derives from"
    assert (rebuilt / "dossier.md").read_text(encoding="utf-8").startswith("# Sealed Window dossier")


def test_readjudication_audit_chain_verifies(stored_run, tmp_path):
    """Every verdict it reaches is recorded and hash-chained, as in a full run."""
    snapshot_root, run_id = stored_run
    result = _readjudicate(tmp_path, snapshot_root, run_id)
    assert result.returncode == 0, result.stderr

    rebuilt_id = next(line.split("-> ")[1].strip() for line in result.stdout.splitlines() if "-> " in line)
    log = tmp_path / "runs" / rebuilt_id / "audit.jsonl"
    assert log.is_file()
    assert AuditLog.verify(log) > 0
    events = [json.loads(line)["event"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert "readjudicate.start" in events and "readjudicate.published" in events


def test_a_mismatched_screen_config_is_refused(stored_run, tmp_path):
    """Rebuilding from settings other than the ones that produced the run stops the command dead.

    The guard this pins was shipped unexercised. A report assembled from a different screen would
    look exactly as authoritative as a correct one, which is why the failure has to be loud.
    """
    snapshot_root, run_id = stored_run
    before = sorted(p.name for p in (tmp_path / "runs").iterdir())

    result = _readjudicate(tmp_path, snapshot_root, run_id, screen="config/screen.smoke.json")
    assert result.returncode == 2, result.stdout
    assert "refusing" in result.stderr and "does not match" in result.stderr

    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == before, \
        "a refused re-adjudication writes nothing"
