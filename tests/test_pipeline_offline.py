"""End-to-end sealed pipeline over the fixture snapshot with the offline model client.

Covers the build-order gates that need a whole run: the plan is never exceeded and every call
had a slot (P3/P5), the veto demonstrably changes outcomes (P6), a prompt-injection claim is
discarded, slider values never reach a prompt, every published claim traces to snapshot
evidence, the audit chain is intact with phases in order, the dossier is reproducible, and an
unapproved plan aborts before any model call.
"""

from __future__ import annotations

import json

import pytest

from sealed_window.agents.offline_model import OfflineHeuristicModel
from sealed_window.governance.audit import AuditLog
from sealed_window.governance.errors import PlanNotApproved
from sealed_window.governance.llm_gateway import ModelRequest
from sealed_window.governance.seal import SEAL
from sealed_window.orchestrator import Orchestrator, RunRequest, RunState, prepare_run
from sealed_window.screen.config import ScreenConfig

DISTINCTIVE_SLIDER = 1.2345678  # 7 decimals: snapshot numbers are rounded to 6, so no false substring match
CONFIG = ScreenConfig(roe_min_pct=DISTINCTIVE_SLIDER, max_candidates=40)


class RecordingModel(OfflineHeuristicModel):
    """Offline model that also records every prompt it is sent."""

    def __init__(self) -> None:
        """Start with no recorded prompts."""
        self.prompts: list[str] = []

    def complete(self, request: ModelRequest):
        """Record system + user prompt, then answer like the offline model."""
        self.prompts.append(request.system + "\n" + request.user)
        return super().complete(request)


def _run(sealed_snapshot, runs_root, approved: str | None = None):
    """Execute one full run and return (dossier, run_dir, prepared, model)."""
    SEAL.seal()
    root, root_hash = sealed_snapshot
    prepared = prepare_run(root, root_hash, CONFIG)
    model = RecordingModel()
    state = RunState()
    orchestrator = Orchestrator(snapshot_root=root, runs_root=runs_root, client_factory=lambda mode: model)
    request = RunRequest(root_hash, CONFIG, approved or prepared.plan.plan_hash, "offline")
    dossier = orchestrator.run(request, state)
    return dossier, runs_root / state.run_id, prepared, model


@pytest.fixture(scope="module")
def run(sealed_snapshot, tmp_path_factory):
    """One shared offline run for the read-only assertions in this module."""
    SEAL._reset_for_tests()
    result = _run(sealed_snapshot, tmp_path_factory.mktemp("runs"))
    SEAL._reset_for_tests()
    return result


def _ledger(run_dir):
    """Load the run's claims.json."""
    return json.loads((run_dir / "claims.json").read_text())


def test_plan_never_exceeded_and_every_call_had_a_slot(run):
    """Spend stays within the committed total and model calls equal slots consumed."""
    dossier, run_dir, prepared, _ = run
    spend = dossier["header"]["spend"]
    assert spend["spent_microusd"] <= spend["committed_microusd"] == prepared.plan.committed_total_microusd
    consumed = sum(s["planned"] - s["remaining"] for s in spend["slots"].values())
    calls = AuditLog(run_dir / "audit.jsonl").entries("model.call")
    assert len(calls) == consumed and all(c["detail"]["ticket"] for c in calls)


def test_audit_chain_intact_and_models_only_in_model_phases(run):
    """The hash chain verifies, phases follow the documented order, and model calls sit inside CLAIM/VETO."""
    _, run_dir, _, _ = run
    AuditLog.verify(run_dir / "audit.jsonl")
    phase = None
    order = []
    for entry in AuditLog(run_dir / "audit.jsonl").entries():
        if entry["event"] == "phase.transition":
            phase = entry["detail"]["to"]
            order.append(phase)
            assert entry["detail"]["seal"]["network_live"] is False
        if entry["event"] == "model.call":
            assert phase in ("3_claim", "5_veto")
    assert order == ["2_screen", "3_claim", "4_adjudicate", "5_veto", "4b_adjudicate_veto", "6_publish", "done"]


def test_injection_claim_is_discarded(run):
    """The synthetic injection headline yields an unkillable claim, which is discarded as vacuous."""
    dossier, run_dir, _, _ = run
    ledger = _ledger(run_dir)
    injected = [c for c in ledger["claims"] if c["predicate"] == "revenue_doubled"]
    assert injected, "the offline model should have been 'persuaded' by the injection headline"
    for claim in injected:
        assert ledger["verdicts"][claim["claim_id"]]["verdict"] == "vacuous"
    published = {c["claim_id"] for p in dossier["picks"] + dossier["avoid"] for c in p["surviving_claims"]}
    assert not published & {c["claim_id"] for c in injected}


def test_veto_changes_outcomes(run):
    """P6 gate: the auditor runs, finds the planted weak stock, and its surviving attack removes a claim.

    Asserted as a property rather than "some claim was vetoed": the fixture guarantees an attackable
    stock (``PROFIT_SHOCK_SYMBOL``), so a silent failure of the veto path shows up here instead of
    depending on which synthetic prices happen to trip the offline auditor's rules.
    """
    _, run_dir, _, _ = run
    ledger = _ledger(run_dir)
    audit = AuditLog(run_dir / "audit.jsonl")
    assert [c for c in audit.entries("model.call") if c["detail"]["slot_class"] == "veto"], "auditor never ran"

    surviving = [r for r in ledger["refutations"]
                 if ledger["verdicts"][r["refutation_id"]]["verdict"] == "survived"]
    assert surviving, "the auditor produced no refutation that passed its own machine check"
    for refutation in surviving:
        target = ledger["verdicts"][refutation["target_claim_id"]]
        assert target["verdict"] == "vetoed", "a surviving refutation must remove its target claim"
        assert refutation["refutation_id"] in target["reason"]


def test_published_claims_trace_to_snapshot_evidence(run, snapshot):
    """Every surviving claim in the dossier cites evidence that exists for its own instrument."""
    dossier, _, _, _ = run
    assert dossier["picks"], "fixture run should publish at least one pick"
    for entry in dossier["picks"] + dossier["avoid"]:
        for claim in entry["surviving_claims"]:
            for ev in claim["evidence"]:
                assert snapshot.evidence_record(ev)["instrument_key"] == entry["instrument_key"]


def test_slider_values_never_reach_prompts(run):
    """The distinctive slider value appears in no prompt sent to any model."""
    _, _, _, model = run
    assert model.prompts and not any(str(DISTINCTIVE_SLIDER) in p for p in model.prompts)


def test_dossier_is_reproducible(run, sealed_snapshot, tmp_path):
    """Same snapshot, config and plan give the same picks and triple."""
    first = run[0]
    SEAL._reset_for_tests()
    second = _run(sealed_snapshot, tmp_path)[0]
    assert json.dumps(first["picks"], sort_keys=True) == json.dumps(second["picks"], sort_keys=True)
    assert first["header"]["reproducibility"] == second["header"]["reproducibility"]


def test_markdown_dossier_carries_as_of_triple_and_warnings(run):
    """The rendered dossier shows as_of, the reproducibility triple and synthetic/offline warnings."""
    dossier, run_dir, _, _ = run
    text = (run_dir / "dossier.md").read_text()
    assert "As of 2026-09-11" in text and dossier["header"]["reproducibility"]["snapshot"] in text
    assert "SYNTHETIC DATA" in text and "offline-heuristic" in text


def test_unapproved_plan_aborts_before_any_model_call(sealed_snapshot, tmp_path):
    """A wrong approval hash aborts the run, is audited, and no model is ever called."""
    SEAL.seal()
    root, root_hash = sealed_snapshot
    model = RecordingModel()
    state = RunState()
    orchestrator = Orchestrator(snapshot_root=root, runs_root=tmp_path, client_factory=lambda mode: model)
    with pytest.raises(PlanNotApproved):
        orchestrator.run(RunRequest(root_hash, CONFIG, "0" * 64, "offline"), state)
    events = [e["event"] for e in AuditLog(tmp_path / state.run_id / "audit.jsonl").entries()]
    assert "plan.rejected" in events and "run.aborted" in events and "model.call" not in events
    assert model.prompts == []
