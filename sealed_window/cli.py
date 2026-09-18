"""Command-line entry points. Each command first takes its process role, then imports its work.

Commands
--------
acquire       Phase 1 in the ACQUIRE role: fetch from Upstox (or the fixture) and seal a snapshot.
snapshots     List sealed snapshots.
plan          Phases 2 (dry): screen a snapshot and print the compiled spend plan and its hash.
evaluate      Phases 2-6: requires ``--approve-plan <hash>`` matching the freshly compiled plan.
backtest      Phase P8: price-only walk-forward over a snapshot's candles. No model, no network.
verify-audit  Verify the hash chain of an audit log.
serve         Start the FastAPI app (SEALED role) with sliders, seal indicator and claim ledger.

Why roles are taken here: ``acquire`` and everything else are different programs as far as
governance is concerned. Taking the role before importing anything else means the import
guards are active before a forbidden module could be loaded -- the ETL cannot load an LLM
client, and the analysis process cannot load the Upstox adapter or credentials.

Exit codes: 0 success, 2 usage error (including an unfinished --as-of session), 3 governance
denial, 4 rate-limited by Upstox (resumable), 5 another acquisition already running.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .governance.errors import GovernanceViolation
from .governance.process_roles import enter_acquire_role, enter_sealed_role
# Safe in either process role: the spend module is pure pricing data and is forbidden to neither.
# The orchestrator, which the ACQUIRE role may not import, must not be reached at parser-build time.
from .governance.spend import DEFAULT_LOCAL_MODEL, MODEL_MODES

DATA_ROOT = Path("data")
DEFAULT_SNAPSHOTS = DATA_ROOT / "snapshots"
DEFAULT_RUNS = DATA_ROOT / "runs"
DEFAULT_UNIVERSE = Path("config/nifty500.csv")
DEFAULT_SCREEN = Path("config/screen.default.json")
_IST = timezone(timedelta(hours=5, minutes=30))


def _latest_final_session_ist() -> date:
    """Default ``as_of``: today (IST) once its session is final at 16:00 IST, otherwise yesterday.

    Weekends and holidays need no special case: the ETL simply finds the last candle before that date.
    """
    now = datetime.now(_IST)
    return now.date() if now.hour >= 16 else now.date() - timedelta(days=1)


def _load_screen_config(path: Path):
    """Read and validate a screen config JSON file into a ScreenConfig."""
    from .screen.config import ScreenConfig

    return ScreenConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))


def cmd_acquire(args: argparse.Namespace) -> int:
    """Run phase 1 in the ACQUIRE role, printing progress, and print the sealed snapshot hash."""
    enter_acquire_role()
    import fcntl
    import math
    import time

    from .acquire.etl import read_universe, run_acquisition, session_final_at, session_is_final, universe_source
    from .acquire.fixture_source import SYNTHETIC_SYMBOLS
    from .acquire.upstox_adapter import UpstoxAdapter
    from .governance import policy
    from .governance.audit import AuditLog
    from .governance.egress import EgressRateLimited
    from .governance.ratelimit import recent_request_ages

    as_of = date.fromisoformat(args.as_of) if args.as_of else _latest_final_session_ist()
    if args.source == "upstox" and not session_is_final(as_of):
        print(f"Refusing --as-of {as_of}: that session's data is not final until "
              f"{session_final_at(as_of):%Y-%m-%d %H:%M} IST. Use an earlier date or wait.", file=sys.stderr)
        return 2

    data_root = args.snapshots.parent
    data_root.mkdir(parents=True, exist_ok=True)
    lock = open(data_root / "acquire.lock", "w")  # held until this process exits
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Another acquisition is already running ({data_root / 'acquire.lock'} is held). "
              "Wait for it to finish; parallel runs would share one Upstox rate limit.", file=sys.stderr)
        return 5

    provenance = None
    if args.source == "fixture" and args.universe is None:
        universe = list(SYNTHETIC_SYMBOLS)
    else:
        path = args.universe or DEFAULT_UNIVERSE
        universe = read_universe(path)
        provenance = universe_source(path, universe)
    audit_dir = data_root / "acquire_audit"
    prior = recent_request_ages(audit_dir) if args.source == "upstox" else []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit = AuditLog(audit_dir / f"{as_of.isoformat()}-{args.source}-{stamp}.jsonl")

    if args.source == "upstox":
        checkpoint_dir = data_root / "acquire_checkpoints" / f"{as_of.isoformat()}-upstox-analytics"
        cached = len(list(checkpoint_dir.glob("*.json"))) if checkpoint_dir.exists() else 0
        per_stock = 6 if as_of == datetime.now(_IST).date() else 5  # +1 intraday candle for today's session
        estimate = len(universe) * per_stock + math.ceil(len(universe) / UpstoxAdapter.NEWS_BATCH_SIZE) + 1
        remaining = max(estimate - cached, 0)
        per_30m = min(count for seconds, count in policy.UPSTOX_RATE_LIMITS if seconds >= 1800)
        minutes = 30 * ((remaining + len(prior)) // per_30m) + remaining * policy.MIN_REQUEST_INTERVAL_S / 60
        print(f"{len(universe)} instruments, ~{estimate:,} requests ({cached:,} already checkpointed, "
              f"{len(prior):,} sent by earlier runs in the last 30 min); at {per_30m:,} per 30 min "
              f"expect roughly {minutes:.0f} minutes. Interrupted runs resume from checkpoints.", file=sys.stderr)

    started = time.monotonic()

    def progress(done: int, total: int, symbol: str) -> None:
        """Print a progress line every 25 instruments and at the end."""
        if done % 25 == 0 or done == total:
            elapsed = int(time.monotonic() - started)
            print(f"  [{done:>4}/{total}] {symbol:<14} elapsed {elapsed // 60:02d}:{elapsed % 60:02d}",
                  file=sys.stderr, flush=True)

    try:
        root_hash = run_acquisition(universe=universe, as_of=as_of, snapshot_root=args.snapshots,
                                    source_kind=args.source, audit=audit, universe_provenance=provenance,
                                    checkpoint_root=data_root / "acquire_checkpoints",
                                    progress=progress, prior_request_ages=prior)
    except EgressRateLimited as exc:
        print(f"Upstox rate-limited the run (HTTP 429 on {exc.capability.value}). Completed reads are "
              "checkpointed; wait a few minutes and re-run the same command to resume.", file=sys.stderr)
        return 4
    from .snapshot.store import SealedSnapshot

    manifest = SealedSnapshot.load(args.snapshots, root_hash).manifest
    print(f"sealed snapshot {root_hash}  (as_of {as_of}, prices to {manifest.get('prices_as_of')}, source {args.source})")
    print(f"audit log {audit.path}")
    if manifest.get("prices_as_of") != manifest["as_of"]:
        print(f"WARNING: as_of is {manifest['as_of']} but the newest price candle is {manifest.get('prices_as_of')} "
              f"({manifest.get('instruments_behind_as_of')}/{len(manifest['universe'])} instruments behind as_of). "
              "Normal for weekends and holidays; otherwise that session is missing from this snapshot.",
              file=sys.stderr)
    return 0


def cmd_snapshots(args: argparse.Namespace) -> int:
    """List sealed snapshots (SEALED role)."""
    enter_sealed_role()
    from .snapshot.store import list_snapshots

    for item in list_snapshots(args.snapshots):
        print(f"{item['root_hash']}  as_of {item['as_of']}  {item['source']}  {item['universe_size']} instruments")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Screen a snapshot and print the compiled spend plan (no model calls)."""
    enter_sealed_role()
    from .orchestrator import prepare_run

    ceiling = int(args.ceiling_usd * 1_000_000) if args.ceiling_usd is not None else None
    prepared = prepare_run(args.snapshots, args.snapshot, _load_screen_config(args.screen), ceiling,
                           args.model, args.local_model)
    print(f"screen: {json.dumps(prepared.screen.summary())}")
    print(prepared.plan.as_table())
    print(f"\nTo run with this exact plan:\n  --approve-plan {prepared.plan.plan_hash}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run phases 2-6 with an approved plan and print where the dossier was written."""
    enter_sealed_role()
    from .orchestrator import Orchestrator, RunRequest, RunState

    ceiling = int(args.ceiling_usd * 1_000_000) if args.ceiling_usd is not None else None
    request = RunRequest(snapshot_hash=args.snapshot, screen_config=_load_screen_config(args.screen),
                         approved_plan_hash=args.approve_plan, model_mode=args.model,
                         local_model=args.local_model,
                         ceiling_microusd=ceiling, concurrency=args.concurrency)
    state = RunState()
    dossier = Orchestrator(snapshot_root=args.snapshots, runs_root=args.runs).run(request, state)
    header = dossier["header"]
    print(f"run {state.run_id}: {header['funnel']['published_picks']} pick(s), "
          f"spend {header['spend']['spent']} of {header['spend']['committed']} committed")
    if header["abstention"]:
        print(header["abstention"])
    print(f"dossier: {args.runs / state.run_id / 'dossier.md'}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run the price-only walk-forward over a snapshot and write its report (no model, no network)."""
    enter_sealed_role()
    import json as json_module

    from .evaluate.walk_forward import BacktestConfig, backtest_snapshot
    from .governance.errors import SnapshotIncompatible
    from .snapshot.columns import DERIVED_COLUMNS_VERSION
    from .snapshot.store import SealedSnapshot

    snapshot = SealedSnapshot.load(args.snapshots, args.snapshot)
    built_with = snapshot.manifest.get("derived_columns_version")
    if built_with != DERIVED_COLUMNS_VERSION:
        raise SnapshotIncompatible(
            f"snapshot {args.snapshot[:12]}... was built with {built_with}, this code expects "
            f"{DERIVED_COLUMNS_VERSION}; run a fresh acquisition"
        )
    config = BacktestConfig(
        horizons=tuple(int(value) for value in args.horizons.split(",")),
        cutoffs=tuple(int(value) for value in args.cutoffs.split(",")),
        step_sessions=args.step,
        warmup_sessions=args.warmup,
        round_trip_cost_bps=args.cost_bps,
        ranking=args.ranking,
        min_windows=args.min_windows,
        calendar_coverage=args.calendar_coverage,
    )
    report = backtest_snapshot(snapshot, _load_screen_config(args.screen), config)
    print(report.as_table())
    out_dir = args.snapshots.parent / "backtests" / f"{snapshot.root_hash[:12]}-{config.config_hash()[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json_module.dumps(report.as_dict(), indent=2, sort_keys=True),
                                         encoding="utf-8")
    (out_dir / "report.txt").write_text(report.as_table() + "\n", encoding="utf-8")
    print(f"\nreport: {out_dir / 'report.json'}")
    return 0


def cmd_verify_audit(args: argparse.Namespace) -> int:
    """Verify an audit log's hash chain and print the entry count."""
    enter_sealed_role()
    from .governance.audit import AuditLog

    print(f"audit chain intact: {AuditLog.verify(args.path)} entries")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the web app in the SEALED role (bound to localhost by default)."""
    scrubbed = enter_sealed_role()
    import uvicorn

    from .app.server import create_app

    app = create_app(snapshot_root=args.snapshots, runs_root=args.runs, scrubbed_env=scrubbed)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI arguments for every command."""
    parser = argparse.ArgumentParser(prog="sealed_window", description=__doc__.splitlines()[0])
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS, help="snapshot root directory")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="run output directory")
    sub = parser.add_subparsers(dest="command", required=True)

    acquire = sub.add_parser("acquire", help="phase 1: acquire and seal a snapshot")
    acquire.add_argument("--source", choices=("upstox", "fixture"), default="upstox")
    acquire.add_argument("--universe", type=Path, default=None,
                         help="index constituents CSV or symbols file (default config/nifty500.csv)")
    acquire.add_argument("--as-of", default=None,
                         help="snapshot date YYYY-MM-DD (default: today after 16:00 IST, else yesterday)")
    acquire.set_defaults(func=cmd_acquire)

    sub.add_parser("snapshots", help="list sealed snapshots").set_defaults(func=cmd_snapshots)

    for name, func, help_text in (("plan", cmd_plan, "screen + compile spend plan"),
                                  ("evaluate", cmd_evaluate, "run phases 2-6 with an approved plan")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--snapshot", required=True, help="snapshot root hash")
        cmd.add_argument("--screen", type=Path, default=DEFAULT_SCREEN, help="screen config JSON")
        cmd.add_argument("--ceiling-usd", type=float, default=None, help="refuse plans committing more than this")
        # --model and --local-model belong to both commands: the mode selects the plan's slot specs,
        # so a plan printed in one mode does not hash-match a run started in another.
        cmd.add_argument("--model", choices=MODEL_MODES, default="offline",
                         help="anthropic (paid API), local (model served on this machine), offline (stand-in)")
        cmd.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL,
                         help=f"model name for --model local (default {DEFAULT_LOCAL_MODEL})")
        if name == "evaluate":
            cmd.add_argument("--approve-plan", required=True, help="plan hash printed by the plan command")
            cmd.add_argument("--concurrency", type=int, default=4)
        cmd.set_defaults(func=func)

    backtest = sub.add_parser("backtest", help="price-only walk-forward over a snapshot's candles")
    backtest.add_argument("--snapshot", required=True, help="snapshot root hash")
    backtest.add_argument("--screen", type=Path, default=Path("config/screen.technical.json"),
                          help="technical-only screen config (fundamental filters are refused)")
    backtest.add_argument("--ranking", default="momentum_90d",
                          help="ranking rule: momentum_90d, risk_adjusted_momentum or trend_quality")
    backtest.add_argument("--horizons", default="20,60", help="forward horizons in sessions")
    backtest.add_argument("--cutoffs", default="5,10", help="top-k selection sizes")
    backtest.add_argument("--step", type=int, default=20, help="sessions between windows")
    backtest.add_argument("--warmup", type=int, default=250, help="sessions of history required before a window")
    backtest.add_argument("--cost-bps", type=float, default=20.0, help="round-trip cost in basis points")
    backtest.add_argument("--min-windows", type=int, default=20, help="windows required to pass the gate")
    backtest.add_argument("--calendar-coverage", type=float, default=0.6,
                          help="share of instruments needing a candle on a date for it to become a window")
    backtest.set_defaults(func=cmd_backtest)

    verify = sub.add_parser("verify-audit", help="verify an audit log hash chain")
    verify.add_argument("path", type=Path)
    verify.set_defaults(func=cmd_verify_audit)

    serve = sub.add_parser("serve", help="start the web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, dispatch the command, and map governance denials to exit code 3."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except GovernanceViolation as exc:
        print(f"GOVERNANCE DENIAL [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 3
