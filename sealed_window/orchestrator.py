"""The orchestrator: deterministic control flow for phases 2-6 of a sealed analysis run.

It owns the run; agents own nothing. Sequence for :meth:`Orchestrator.run`:

    SCREEN          load + verify snapshot, filter, compile the spend plan, require approval
    CLAIM           one governed call per (candidate, dimension), bounded concurrency
    ADJUDICATE      deterministic verdicts on every claim
    VETO            (if anything survived) auditor batches, one slot each
    ADJUDICATE_VETO deterministic verdicts on refutations; surviving ones veto their targets
    PUBLISH         dossier JSON + Markdown, claim ledger, transcript
    DONE

Termination is structural, not a heuristic: the task list is fixed after SCREEN (candidates x
3 dimensions, then ceil(survivors' instruments / batch) veto calls), agents cannot enqueue
work, probes do not exist in v1, and the slot ledger bounds every call. There is no loop that
could fail to terminate, so no loop detector is needed.

Every run writes to ``runs_root/<run_id>/``: ``audit.jsonl`` (hash chained), ``plan.json``,
``transcript.jsonl``, ``claims.json``, ``dossier.json`` and ``dossier.md``. A
``GovernanceViolation`` aborts the run and is audited; it is never retried around.

This module is import-forbidden in the ACQUIRE process role.
"""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .adjudicate.adjudicator import Adjudicator
from .agents.claim_agents import CLAIM_DIMENSIONS, run_claim_agent
from .agents.veto import batch_subjects, run_veto_batch
from .claims.schema import Adjudication, Claim, Refutation, Verdict
from .governance.audit import AuditLog
from .governance.errors import (
    GovernanceViolation,
    NoSlotAvailable,
    PlanNotApproved,
    PromptOverBudget,
    SnapshotIncompatible,
)
from .governance.llm_gateway import LLMGateway, ModelClient
from .governance.seal import SEAL, PhaseMachine, RunPhase
from .governance.spend import SlotClass, SlotLedger, SpendPlan, assert_plan_approved, compile_plan
from .publish.dossier import build_dossier, render_markdown
from .screen.config import ScreenConfig
from .screen.screen import ScreenResult, run_screen
from .snapshot.columns import DERIVED_COLUMNS_VERSION
from .snapshot.store import SealedSnapshot

MODEL_MODES = ("anthropic", "offline")


def make_model_client(mode: str) -> ModelClient:
    """Construct the model client for ``mode``: the Anthropic API or the offline heuristic stand-in."""
    if mode == "anthropic":
        from .governance.llm_gateway import AnthropicModelClient

        return AnthropicModelClient()
    if mode == "offline":
        from .agents.offline_model import OfflineHeuristicModel

        return OfflineHeuristicModel()
    raise ValueError(f"unknown model mode {mode!r}; expected one of {MODEL_MODES}")


@dataclass
class PreparedRun:
    """The deterministic, model-free part of a run: verified snapshot, screen result and plan."""

    snapshot: SealedSnapshot
    screen: ScreenResult
    plan: SpendPlan


def prepare_run(
    snapshot_root: Path, snapshot_hash: str, config: ScreenConfig, ceiling_microusd: int | None = None
) -> PreparedRun:
    """Load and verify the snapshot, run the screen and compile the plan (used by ``plan`` and ``run``).

    Refuses snapshots built with a different derived-column version (:class:`SnapshotIncompatible`).
    """
    snapshot = SealedSnapshot.load(snapshot_root, snapshot_hash)
    built_with = snapshot.manifest.get("derived_columns_version")
    if built_with != DERIVED_COLUMNS_VERSION:
        raise SnapshotIncompatible(
            f"snapshot {snapshot_hash[:12]}... was built with {built_with}, this code expects "
            f"{DERIVED_COLUMNS_VERSION}; run a fresh acquisition"
        )
    screen = run_screen(snapshot, config)
    plan = compile_plan(
        snapshot_hash=snapshot.root_hash,
        screen_config_hash=screen.config_hash,
        candidate_count=len(screen.candidates),
        ceiling_microusd=ceiling_microusd,
    )
    return PreparedRun(snapshot, screen, plan)


@dataclass
class RunRequest:
    """Inputs for one analysis run, including the operator's approval of a specific plan hash."""

    snapshot_hash: str
    screen_config: ScreenConfig
    approved_plan_hash: str
    model_mode: str = "offline"
    ceiling_microusd: int | None = None
    concurrency: int = 4


class RunState:
    """Thread-safe live view of a run for the UI: phase, seal, spend, funnel and claim ledger."""

    def __init__(self, run_id: str | None = None) -> None:
        """Create a state record with a fresh run ID (UTC timestamp + random suffix)."""
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        self._lock = threading.Lock()
        self._ledger: SlotLedger | None = None
        self._data: dict[str, Any] = {"run_id": self.run_id, "status": "running", "phase": None, "error": None,
                                      "funnel": {}, "plan_hash": None, "claims": [], "dossier": None}

    def set(self, **fields: Any) -> None:
        """Update fields atomically."""
        with self._lock:
            self._data.update(fields)

    def attach_ledger(self, ledger: SlotLedger) -> None:
        """Expose live spend from the run's slot ledger."""
        self._ledger = ledger

    def to_dict(self) -> dict[str, Any]:
        """Snapshot of the state plus the process seal status and live spend."""
        with self._lock:
            data = dict(self._data)
        data["seal"] = SEAL.status()
        data["spend"] = self._ledger.summary() if self._ledger else None
        return data


def claim_ledger_view(
    snapshot: SealedSnapshot, claims: list[Claim], verdicts: dict[str, Adjudication]
) -> list[dict[str, Any]]:
    """Rows for the UI claim ledger: every claim with its current verdict (``pending`` before phase 4)."""
    rows = []
    for claim in claims:
        verdict = verdicts.get(claim.claim_id)
        rows.append({
            "claim_id": claim.claim_id,
            "symbol": snapshot.instrument(claim.subject).get("trading_symbol"),
            "dimension": claim.dimension.value,
            "direction": claim.direction.value,
            "confidence": claim.confidence,
            "statement": claim.statement,
            "falsifier": claim.falsifier,
            "verdict": verdict.verdict.value if verdict else "pending",
            "reason": verdict.reason if verdict else "",
        })
    return rows


class Orchestrator:
    """Runs sealed analysis runs against snapshots on disk and writes their artefacts."""

    def __init__(
        self,
        *,
        snapshot_root: Path,
        runs_root: Path,
        client_factory: Callable[[str], ModelClient] = make_model_client,
    ) -> None:
        """Configure storage locations and how model clients are built."""
        self._snapshot_root = snapshot_root
        self._runs_root = runs_root
        self._client_factory = client_factory

    def run(self, request: RunRequest, state: RunState | None = None) -> dict[str, Any]:
        """Execute phases 2-6 and return the dossier. Raises on governance violations (after auditing)."""
        state = state or RunState()
        run_dir = self._runs_root / state.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        audit = AuditLog(run_dir / "audit.jsonl")

        def on_transition(old: RunPhase | None, new: RunPhase) -> None:
            """Audit every phase change together with the seal status at that moment."""
            audit.record("phase.transition", {"from": old.value if old else None, "to": new.value,
                                              "seal": SEAL.status()})
            state.set(phase=new.value)

        phases = PhaseMachine(SEAL, on_transition=on_transition)
        notes: list[str] = []
        try:
            audit.record("run.start", {"run_id": state.run_id, "snapshot": request.snapshot_hash,
                                       "screen_config_hash": request.screen_config.config_hash(),
                                       "model_mode": request.model_mode, "seal": SEAL.status()})
            phases.advance(RunPhase.SCREEN)
            prepared = prepare_run(self._snapshot_root, request.snapshot_hash, request.screen_config,
                                   request.ceiling_microusd)
            snapshot, screen, plan = prepared.snapshot, prepared.screen, prepared.plan
            audit.record("screen.result", screen.summary())
            state.set(plan_hash=plan.plan_hash, funnel={"universe": len(snapshot.instruments),
                                                        "candidates": len(screen.candidates)})
            try:
                assert_plan_approved(plan, request.approved_plan_hash)
            except PlanNotApproved:
                audit.record("plan.rejected", {"compiled": plan.plan_hash, "approved": request.approved_plan_hash})
                raise
            audit.record("plan.approved", {"plan_hash": plan.plan_hash,
                                           "committed_microusd": plan.committed_total_microusd})
            (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

            ledger = SlotLedger(plan, audit)
            state.attach_ledger(ledger)
            client = self._client_factory(request.model_mode)
            gateway = LLMGateway(ledger=ledger, audit=audit, client=client, phases=phases,
                                 transcript_path=run_dir / "transcript.jsonl")

            phases.advance(RunPhase.CLAIM)
            claims = self._claim_phase(request, gateway, ledger, snapshot, screen, audit, notes)
            state.set(claims=claim_ledger_view(snapshot, claims, {}))

            phases.advance(RunPhase.ADJUDICATE)
            adjudicator = Adjudicator(snapshot, audit)
            verdicts = adjudicator.adjudicate_claims(claims, phase=RunPhase.ADJUDICATE.value)
            state.set(claims=claim_ledger_view(snapshot, claims, verdicts))
            surviving = {c.claim_id: c for c in claims if verdicts[c.claim_id].verdict is Verdict.SURVIVED}

            refutations: list[Refutation] = []
            if surviving and ledger.remaining(SlotClass.VETO) > 0:
                phases.advance(RunPhase.VETO)
                refutations = self._veto_phase(gateway, ledger, snapshot, screen, surviving, audit, notes)
                phases.advance(RunPhase.ADJUDICATE_VETO)
                ref_verdicts, vetoed = adjudicator.adjudicate_refutations(
                    refutations, surviving, phase=RunPhase.ADJUDICATE_VETO.value)
                verdicts.update(ref_verdicts)
                verdicts.update(vetoed)
                state.set(claims=claim_ledger_view(snapshot, claims, verdicts))
            else:
                notes.append("veto phase skipped: no surviving claims to audit")

            phases.advance(RunPhase.PUBLISH)
            dossier = build_dossier(run_id=state.run_id, snapshot=snapshot, screen=screen, plan=plan,
                                    spend_summary=ledger.summary(), claims=claims, refutations=refutations,
                                    verdicts=verdicts, model_client=client.label, notes=notes)
            self._write_outputs(run_dir, dossier, claims, refutations, verdicts)
            audit.record("run.published", {"picks": len(dossier["picks"]), "avoid": len(dossier["avoid"]),
                                           "reproducibility": dossier["header"]["reproducibility"],
                                           "spend": ledger.summary()})
            phases.advance(RunPhase.DONE)
            state.set(status="done", dossier=dossier)
            return dossier
        except GovernanceViolation as exc:
            audit.record("run.aborted", {"violation": type(exc).__name__, "message": str(exc)})
            phases.advance(RunPhase.ABORTED)
            state.set(status="aborted", error=f"{type(exc).__name__}: {exc}")
            raise
        except Exception as exc:
            audit.record("run.failed", {"error": type(exc).__name__, "message": str(exc)[:500]})
            phases.advance(RunPhase.ABORTED)
            state.set(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise

    def _claim_phase(
        self,
        request: RunRequest,
        gateway: LLMGateway,
        ledger: SlotLedger,
        snapshot: SealedSnapshot,
        screen: ScreenResult,
        audit: AuditLog,
        notes: list[str],
    ) -> list[Claim]:
        """Run every (candidate, dimension) claim agent with bounded concurrency; results in stable order."""
        tasks = [(key, dimension) for key in screen.candidates for dimension in CLAIM_DIMENSIONS]

        def work(task: tuple[str, Any]) -> list[Claim]:
            """Run one claim agent; slot exhaustion or an oversize prompt skips it (audited)."""
            key, dimension = task
            try:
                return run_claim_agent(gateway=gateway, ledger=ledger, snapshot=snapshot,
                                       instrument_key=key, dimension=dimension)
            except (NoSlotAvailable, PromptOverBudget) as exc:
                audit.record("claim.skipped", {"instrument_key": key, "dimension": dimension.value,
                                               "violation": type(exc).__name__, "message": str(exc)})
                notes.append(f"{dimension.value} claims skipped for {key}: {type(exc).__name__}")
                return []

        with ThreadPoolExecutor(max_workers=max(1, request.concurrency)) as pool:
            results = list(pool.map(work, tasks))
        ordered: dict[str, Claim] = {}
        for batch in results:
            for claim in batch:
                ordered.setdefault(claim.claim_id, claim)
        return list(ordered.values())

    def _veto_phase(
        self,
        gateway: LLMGateway,
        ledger: SlotLedger,
        snapshot: SealedSnapshot,
        screen: ScreenResult,
        surviving: dict[str, Claim],
        audit: AuditLog,
        notes: list[str],
    ) -> list[Refutation]:
        """Audit surviving claims in instrument batches, one veto slot per batch."""
        subjects = [key for key in screen.candidates if any(c.subject == key for c in surviving.values())]
        refutations: list[Refutation] = []
        for batch in batch_subjects(subjects):
            batch_claims = [c for c in surviving.values() if c.subject in batch]
            try:
                outcome = run_veto_batch(gateway=gateway, ledger=ledger, snapshot=snapshot, claims=batch_claims)
            except (NoSlotAvailable, PromptOverBudget) as exc:
                audit.record("veto.skipped", {"instruments": batch, "violation": type(exc).__name__,
                                              "message": str(exc)})
                notes.append(f"veto not performed for {len(batch)} instrument(s): {type(exc).__name__}")
                continue
            if outcome.dropped_out_of_batch:
                notes.append(f"{outcome.dropped_out_of_batch} refutation(s) named claims outside their batch and were dropped")
            refutations.extend(outcome.refutations)
        return refutations

    @staticmethod
    def _write_outputs(
        run_dir: Path,
        dossier: dict[str, Any],
        claims: list[Claim],
        refutations: list[Refutation],
        verdicts: dict[str, Adjudication],
    ) -> None:
        """Write dossier JSON/Markdown and the full claim ledger for the run."""
        (run_dir / "dossier.json").write_text(json.dumps(dossier, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "dossier.md").write_text(render_markdown(dossier), encoding="utf-8")
        ledger = {
            "claims": [c.model_dump(mode="json") for c in claims],
            "refutations": [r.model_dump(mode="json") for r in refutations],
            "verdicts": {k: v.model_dump(mode="json") for k, v in sorted(verdicts.items())},
        }
        (run_dir / "claims.json").write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
