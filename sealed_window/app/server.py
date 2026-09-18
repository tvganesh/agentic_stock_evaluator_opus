"""FastAPI server for the Sealed Window UI.

Endpoints
---------
GET  /                         the single-page front end
GET  /api/governance           the access policy, process role, seal status and scrubbed env names
GET  /api/snapshots            sealed snapshots available to screen
GET  /api/screen/schema        slider and toggle definitions for the two tabs
POST /api/plan                 run the screen and compile the spend plan (no model calls)
POST /api/runs                 start a run; requires the approved plan hash to match a fresh compile
GET  /api/runs                 completed runs found on disk, newest first
GET  /api/runs/{run_id}        live run state: phase, seal, spend, claim ledger, dossier when done
GET  /api/runs/{run_id}/dossier.md  the rendered dossier

Runs started from the command line are readable here too. The orchestrator writes the same
artefacts whichever way a run is launched, so the two run endpoints fall back to reading
``dossier.json`` from disk when a run is not in this process's memory. Such a payload is marked
``historical`` and carries no seal: the seal describes *this* process now, and reporting a live
one beside a finished run would imply the run were still happening.

Governance properties of the app itself:

* "You approve a number, not a ceiling": the run endpoint recompiles the plan and refuses with
  409 unless the client's approved hash equals it, so a changed slider cannot ride on an old
  approval.
* One run at a time per process, so the seal indicator describes exactly one run.
* Interactive docs (``/docs``, ``/openapi.json``) are disabled; the server binds to localhost by
  default (see the ``serve`` CLI command).
* Model and news text is returned as JSON and rendered with ``textContent`` in the front end,
  never as HTML.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from ..governance import policy
from ..governance.errors import GovernanceViolation, SnapshotIncompatible, SnapshotIntegrityError, SpendViolation
from ..governance.llm_gateway import ModelClient
from ..governance.process_roles import current_role
from ..governance.seal import SEAL
from ..governance.spend import format_usd
from ..orchestrator import (
    DEFAULT_LOCAL_MODEL,
    MODEL_MODES,
    Orchestrator,
    RunRequest,
    RunState,
    make_model_client,
    prepare_run,
)
from ..publish.dossier import render_markdown
from ..screen.config import SLIDERS, ScreenConfig
from ..snapshot.store import list_snapshots

STATIC_DIR = Path(__file__).parent / "static"


class PlanBody(BaseModel):
    """Request body for plan compilation: which snapshot, which sliders, optional ceiling."""

    model_config = ConfigDict(extra="forbid")

    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: ScreenConfig
    ceiling_usd: float | None = Field(None, gt=0, le=1000)
    # The mode selects the plan's slot specs and so changes the plan hash: it must be part of the
    # body the operator compiles a plan with, not only of the body that starts the run.
    model_mode: str = Field("offline", pattern="^(anthropic|local|offline)$")
    local_model: str = Field(DEFAULT_LOCAL_MODEL, min_length=1, max_length=100)


class RunBody(PlanBody):
    """Request body for starting a run: the plan body plus the approval and concurrency."""

    approved_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    concurrency: int = Field(4, ge=1, le=8)


def _ceiling(body: PlanBody) -> int | None:
    """Convert the optional USD ceiling to micro-dollars."""
    return int(body.ceiling_usd * 1_000_000) if body.ceiling_usd is not None else None


def create_app(
    *,
    snapshot_root: Path,
    runs_root: Path,
    scrubbed_env: Iterable[str] = (),
    client_factory: Callable[[str, str], ModelClient] = make_model_client,
) -> FastAPI:
    """Build the FastAPI application bound to snapshot and run directories."""
    app = FastAPI(title="Sealed Window", docs_url=None, redoc_url=None, openapi_url=None)
    orchestrator = Orchestrator(snapshot_root=snapshot_root, runs_root=runs_root, client_factory=client_factory)
    runs: dict[str, RunState] = {}
    runs_lock = threading.Lock()
    scrubbed = sorted(scrubbed_env)

    @app.get("/")
    def index() -> FileResponse:
        """Serve the single-page front end."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/governance")
    def governance() -> dict[str, Any]:
        """Describe what this system may reach, in which process, and the current seal state."""
        role = current_role()
        return {
            "process_role": role.value if role else "unassigned (library/test)",
            "seal": SEAL.status(),
            "scrubbed_environment_variables": scrubbed,
            "upstox": {
                "credential": policy.UPSTOX_TOKEN_ENV + " (ACQUIRE process only)",
                "method": policy.ALLOWED_METHOD,
                "capabilities": [
                    {"name": c.name.value, "host": c.host, "path": c.path_template,
                     "query": {k: r.pattern for k, r in c.query_params.items()},
                     "token_attached": c.attach_token, "consumed_by": c.consumed_by}
                    for c in policy.CAPABILITIES.values()
                ],
            },
            "model_provider_hosts": sorted(policy.MODEL_PROVIDER_HOSTS),
            "forbidden_imports": {"acquire": list(policy.ACQUIRE_ROLE_FORBIDDEN_MODULES),
                                  "sealed": list(policy.SEALED_ROLE_FORBIDDEN_MODULES)},
            "order_placement": "no code path exists",
        }

    @app.get("/api/snapshots")
    def snapshots() -> list[dict[str, Any]]:
        """List sealed snapshots, newest first."""
        return list_snapshots(snapshot_root)

    @app.get("/api/screen/schema")
    def screen_schema() -> dict[str, Any]:
        """Slider and toggle definitions for the Fundamental and Technical tabs."""
        return {
            "sliders": [asdict(s) for s in SLIDERS],
            "toggles": [
                {"field": "price_above_sma20", "tab": "technical", "label": "Price vs MA20"},
                {"field": "price_above_sma50", "tab": "technical", "label": "Price vs MA50"},
            ],
            "max_candidates": {"minimum": 1, "maximum": 150, "default": 30},
            "model_modes": list(MODEL_MODES),
        }

    @app.post("/api/plan")
    def plan(body: PlanBody) -> dict[str, Any]:
        """Run the deterministic screen and compile the spend plan the operator will approve."""
        try:
            prepared = prepare_run(snapshot_root, body.snapshot_hash, body.config, _ceiling(body),
                                   body.model_mode, body.local_model)
        except SnapshotIntegrityError as exc:
            raise HTTPException(404, str(exc)) from exc
        except SnapshotIncompatible as exc:
            raise HTTPException(409, str(exc)) from exc
        except SpendViolation as exc:
            raise HTTPException(422, str(exc)) from exc
        snapshot, screen, compiled = prepared.snapshot, prepared.screen, prepared.plan
        return {
            "as_of": snapshot.as_of,
            "prices_as_of": snapshot.prices_as_of,
            "synthetic_data": snapshot.is_synthetic,
            "screen": screen.summary(),
            "candidates": [snapshot.instrument(k).get("trading_symbol") for k in screen.candidates],
            "plan_hash": compiled.plan_hash,
            "committed": format_usd(compiled.committed_total_microusd),
            "hard_stop": format_usd(compiled.hard_stop_microusd),
            "rows": [dict(r.model_dump(mode="json"), committed=format_usd(r.committed_microusd))
                     for r in compiled.rows],
            "table": compiled.as_table(),
        }

    @app.post("/api/runs", status_code=202)
    def start_run(body: RunBody) -> dict[str, Any]:
        """Start a run in a background thread if the approval matches a freshly compiled plan."""
        try:
            prepared = prepare_run(snapshot_root, body.snapshot_hash, body.config, _ceiling(body),
                                   body.model_mode, body.local_model)
        except SnapshotIntegrityError as exc:
            raise HTTPException(404, str(exc)) from exc
        except SnapshotIncompatible as exc:
            raise HTTPException(409, str(exc)) from exc
        except SpendViolation as exc:
            raise HTTPException(422, str(exc)) from exc
        if prepared.plan.plan_hash != body.approved_plan_hash:
            raise HTTPException(409, {"message": "approved plan does not match the compiled plan; recompile and approve",
                                      "compiled_plan_hash": prepared.plan.plan_hash})
        with runs_lock:
            if any(state.to_dict()["status"] == "running" for state in runs.values()):
                raise HTTPException(409, "a run is already in progress")
            state = RunState()
            runs[state.run_id] = state
        request = RunRequest(snapshot_hash=body.snapshot_hash, screen_config=body.config,
                             approved_plan_hash=body.approved_plan_hash, model_mode=body.model_mode,
                             local_model=body.local_model,
                             ceiling_microusd=_ceiling(body), concurrency=body.concurrency)

        def target() -> None:
            """Run the orchestrator; failures are already recorded in the run state and audit log."""
            try:
                orchestrator.run(request, state)
            except (GovernanceViolation, Exception):  # noqa: BLE001 - state carries the error
                pass

        threading.Thread(target=target, name=f"run-{state.run_id}", daemon=True).start()
        return {"run_id": state.run_id, "plan_hash": prepared.plan.plan_hash}

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        """Completed runs on disk, newest first, with enough detail to choose one."""
        out = []
        for path in sorted(runs_root.glob("*/dossier.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                header = json.loads(path.read_text(encoding="utf-8"))["header"]
            except (OSError, ValueError, KeyError):
                continue  # a half-written or foreign directory is skipped, not fatal
            out.append({
                "run_id": path.parent.name,
                "as_of": header.get("as_of"),
                "model_client": header.get("model_client"),
                "picks": header.get("funnel", {}).get("published_picks"),
                "spent": header.get("spend", {}).get("spent"),
            })
        return out

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str) -> dict[str, Any]:
        """Live state of a run, or a finished run reconstructed from disk."""
        state = runs.get(run_id)
        if state is not None:
            return state.to_dict()
        return _run_from_disk(runs_root, run_id)

    @app.get("/api/runs/{run_id}/dossier.md", response_class=PlainTextResponse)
    def run_dossier(run_id: str) -> str:
        """The rendered Markdown dossier of a finished run, from memory or from disk."""
        state = runs.get(run_id)
        dossier = state.to_dict().get("dossier") if state else None
        if dossier is None:
            dossier = _run_from_disk(runs_root, run_id)["dossier"]
        return render_markdown(dossier)

    return app


def _ledger_from_dossier(dossier: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild the claim-ledger rows from a stored dossier.

    The live ledger comes from :func:`claim_ledger_view`, which needs the snapshot to resolve
    trading symbols. A stored dossier already carries the symbol on every entry, so a finished run
    renders in the same table without loading a snapshot that can run to a hundred megabytes.
    """
    rows = []
    for entry in list(dossier.get("picks", [])) + list(dossier.get("avoid", [])):
        symbol = entry.get("trading_symbol")
        for claim in entry.get("surviving_claims", []):
            rows.append({"claim_id": claim.get("claim_id"), "symbol": symbol,
                         "dimension": claim.get("dimension"), "direction": claim.get("direction"),
                         "confidence": claim.get("confidence"), "statement": claim.get("statement"),
                         "falsifier": claim.get("falsifier"), "verdict": "survived", "reason": ""})
        for claim in entry.get("refuted_or_vetoed_claims", []):
            rows.append({"claim_id": claim.get("claim_id"), "symbol": symbol,
                         "dimension": claim.get("dimension"), "direction": claim.get("direction"),
                         "confidence": claim.get("confidence"), "statement": claim.get("statement"),
                         "falsifier": claim.get("falsifier"), "verdict": claim.get("verdict", "refuted"),
                         "reason": claim.get("reason", "")})
    return rows


def _run_from_disk(runs_root: Path, run_id: str) -> dict[str, Any]:
    """Reconstruct a finished run's UI payload from its stored dossier.

    Shaped like :meth:`RunState.to_dict` so the front end renders a past run through exactly the
    same code as a live one. Two deliberate differences: ``historical`` is true, and ``seal`` is
    ``None`` rather than the current process seal -- a finished run has no live seal, and showing
    one would suggest it were still in flight.
    """
    if not run_id.replace("-", "").isalnum():
        raise HTTPException(400, "malformed run id")  # never let a path segment escape runs_root
    path = runs_root / run_id / "dossier.json"
    try:
        dossier = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(404, f"no stored dossier for run {run_id}") from exc
    header = dossier.get("header", {})
    return {
        "run_id": run_id,
        "status": "done",
        "phase": "done",
        "historical": True,
        "error": None,
        "funnel": header.get("funnel", {}),
        "plan_hash": header.get("reproducibility", {}).get("plan"),
        "spend": header.get("spend"),
        "claims": _ledger_from_dossier(dossier),
        "dossier": dossier,
        "seal": None,
    }
