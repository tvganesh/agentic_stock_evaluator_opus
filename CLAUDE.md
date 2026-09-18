# Sealed Window — codebase guide

NSE equity research, research-only. No order, portfolio, holdings or funds code path exists
anywhere in this repository, and none may be added.

**The invariant everything else follows from:** the market-data connection and a running model are
never live at the same time. Data is collected first and frozen; analysis runs against that frozen
copy in a separate process that has no route to the data provider.

A run is six phases. Only two of them contain a model:

| Phase | Folder | Model | Market data |
|---|---|---|---|
| 1 acquire | `acquire/` | none | **live** |
| 2 screen | `screen/` | none | sealed |
| 3 claim | `agents/` | running | sealed |
| 4 adjudicate | `adjudicate/` | none | sealed |
| 5 veto | `agents/veto.py` | running | sealed |
| 6 publish | `publish/` | none | sealed |

Reading order for someone new: `governance/policy.py` → `governance/seal.py` → `acquire/etl.py` →
`screen/screen.py` → `claims/schema.py` → `adjudicate/adjudicator.py` → `orchestrator.py`.

---

## `sealed_window/` — the package root

Holds the two entry points and the control flow, nothing else. `cli.py` (308 lines) defines every
command — `acquire`, `screen`, `plan`, `evaluate`, `backtest`, `serve`, `snapshots`,
`verify-audit` — and its first act in each is to claim a **process role**, before importing anything
that does work. That ordering is deliberate: the import guards must be active before a forbidden
module could be loaded. `__main__.py` just delegates, so `python -m sealed_window` works.

`orchestrator.py` (345 lines) owns phases 2 through 6. It advances the phase machine, compiles and
enforces the spend plan, runs claim agents with bounded concurrency, calls the adjudicator, runs the
veto, assembles the dossier, and writes every artefact of a run: `audit.jsonl`, `plan.json`,
`transcript.jsonl`, `claims.json`, `dossier.json`, `dossier.md`. Termination is structural rather
than heuristic — the task list is fixed once screening finishes, agents cannot enqueue work, and the
slot ledger bounds every call — so there is no loop that could fail to terminate.

## `governance/` — the spine (11 modules, ~2,000 lines)

Every other folder passes through this one. It is the largest package by design.

`policy.py` is pure data and the single reviewed source of what may be touched: GET only, two Upstox
hosts, seven exact path templates with typed path and query parameters, which credentials may exist,
which modules each process role may import, rate limits, and the model-provider hosts. Nothing here
executes; everything that enforces reads from it. `egress.py` is the only code permitted to make an
HTTP request to Upstox — callers name a capability and pass typed parameters, never a URL, and the
finished request is authorised again independently before it goes out. `credentials.py` loads the one
permitted token, scrubs it from the environment, and refuses to start if any other `UPSTOX_*`
variable exists. `ratelimit.py` keeps requests inside Upstox's per-second, per-minute and
per-30-minute limits, seeded from earlier runs' audit logs because the vendor limits the account, not
the process.

`seal.py` implements the invariant at socket level: it patches name resolution and connection so that
outside a model window nothing is reachable, and inside one only the model provider is. It also holds
the phase machine that confines models to phases 3 and 5. `process_roles.py` makes the two programs
genuinely different — the collector cannot import an LLM client, and the analysis process cannot
import the Upstox adapter, the credential loader or a general HTTP client.

`spend.py` compiles the prepaid plan: committed total equals the hard stop, approved by hash before
the first call, with a one-use slot ticket for every request and a prepaid probe pool for tool calls.
`llm_gateway.py` is the only code that may call a model, and enforces slot, phase, token limit,
settlement and audit on each call. `audit.py` is an append-only hash-chained log; editing any entry
breaks the chain. `errors.py` holds the fail-closed exception hierarchy every control raises.

## `snapshot/` — the frozen record

`store.py` writes a snapshot atomically as read-only files named by the hash of their manifest, and
verifies every file's hash on load, so analysis can never run over altered data. Candle bytes are
verified at load but parsed only when something reads them, because four years for ~500 stocks is
large and only the backtest needs them. `hashing.py` provides canonical JSON, the content hashes, and
the evidence IDs that are the only way anything downstream may refer to a fact.

`columns.py` is the catalogue of derived columns — the shared vocabulary of screens, prompts and
falsifiers — plus the freshness limits per dimension. `indicators.py` computes every one of those
numbers: RSI, MACD, ATR, moving averages, volatility, volume ratio, returns, drawdown, growth rates,
margins, leverage and the bank ratios. It lives here rather than in `acquire/` because it is pure
arithmetic with no network or credentials, and both process roles need it — the collector to build a
snapshot, the backtest to recompute columns at past dates.

## `acquire/` — phase 1, the only place with a live connection

`etl.py` (535 lines) drives collection: resolve the universe against the instrument master by ISIN,
fetch candles, ratios, statements and news per company, normalise them, compute derived columns,
assign an evidence ID to every fact, then close the connection, drop the token, seal the process, and
only then write the snapshot. It also guards price-date integrity — today's session comes from a
separate intraday endpoint, and any lag between `as_of` and the newest candle is recorded and warned
about. `upstox_adapter.py` wraps the seven allowlisted reads and nothing else; there is deliberately
no method for orders, holdings or accounts. `checkpoint.py` makes a 40-minute acquisition resumable
and discards checkpoints saved before the session closed. `fixture_source.py` is a deterministic
synthetic market with the same interface, so the whole pipeline runs with no token and no network —
including a planted injection headline and a planted profit collapse that keep the defences honest.

## `screen/` — phase 2, the sliders

`config.py` defines the settings as a frozen, validated, content-hashed object, and the slider
metadata the UI renders. `screen.py` applies them: missing data fails closed, stale data fails
closed, banks are judged on bank measures because ROCE and leverage are meaningless for them, and
truncation to the candidate cap is reported rather than silent. No model exists in this phase, which
is why slider values can never reach a prompt.

## `claims/` — what a model is allowed to say

`schema.py` defines the only shapes a model's output may take. Drafts carry no subject and no IDs —
the harness assigns those, so an agent analysing one company cannot emit a claim about another — and
there is no score field anywhere. `dsl.py` (427 lines) is the falsifier language: a small predicate
grammar over snapshot columns with hard limits on length, nesting and column count, plus the
reachability check that discards a disproof condition no company in the market could ever trigger.
`validator.py` runs the parse-time checks — evidence must exist and belong to this company, quoted
figures must appear in that evidence, the condition must use columns appropriate to the claim.

## `agents/` — phases 3 and 5, the only model code

Agents are single governed calls, not autonomous loops: no filesystem, no shell, no network, no
ability to call each other. `prompts.py` holds the frozen system prompts, the column catalogue shown
to each analyst, and the data envelope that marks snapshot content as data rather than instruction.
`claim_agents.py` runs one analyst for one company and one dimension. `veto.py` runs the auditor,
whose output schema has no field for agreement — it can only refute, and its refutations face the
same machine checks as claims.

`tools.py` (282 lines) is the newest piece: four tools an analyst may call before committing —
deeper price history, the full statement tables, peer comparison across the universe, and further
news pages. They read the sealed snapshot, so agency costs tokens but opens no network. Each toolbox
is bound to one company, draws on a prepaid probe pool, and audits every call. `offline_model.py` is
a rule-based stand-in used for tests and free dry runs; it is explicitly not an LLM and every dossier
it produces says so.

## `adjudicate/` — phase 4, where claims live or die

`adjudicator.py` evaluates each claim's disproof condition against the snapshot and returns one of
six verdicts: survived, refuted, unevaluable, rejected, vacuous, or vetoed. Only survivors count.
Ranking is then fixed arithmetic over them — each surviving claim contributes its confidence, signed
by direction, weighted by dimension (fundamental 1.0, technical 0.8, news 0.5, news lowest because it
is the input written by strangers). No model computes the ranking.

## `publish/` — phase 6, the report

`dossier.py` assembles the output from surviving claims with no generative step, so every sentence
traces to a claim, every claim to evidence, and every evidence ID to a row in a hashed snapshot. It
holds the recommendation rules (BUY / WATCH / AVOID / abstain), the abstention behaviour when fewer
than five companies qualify, and the renderer that keeps each claim's disproof condition attached so
a reader can see what would make it wrong.

## `evaluate/` — does any of this work?

`walk_forward.py` (477 lines) rebuilds the screen at past dates from the snapshot's own candles,
ranks survivors by a named rule, and measures the following 20 and 60 sessions against an
equal-weight basket of everything that passed the same screen. It reports precision, excess return,
drawdown, volatility, turnover and coverage against acceptance gates, refuses any screen config
containing undated fundamentals, and prints its limitations with every result. Its current verdict on
the deterministic ranking is that it does not clear the gates — reported rather than hidden.

## `app/` — the local dashboard

`server.py` exposes the governance policy, snapshots, slider schema, plan compilation, runs and
dossiers over a loopback-only API, with no endpoint that can collect data. `static/index.html` is the
single page: sliders, the seal indicator, spend approval that names the model and its cost, the claim
ledger where claims appear and are struck through as they are killed, and the dossier.

---

## Rules for working in this repository

- **Governance controls are fail-closed.** A `GovernanceViolation` stops the operation. Never catch
  one to retry around it, relax a limit, or fall back to a less-governed path.
- **Adding an endpoint is a policy change**, reviewed in `policy.py` and matched by a test that the
  forbidden neighbours are still refused.
- **Never widen a budget to make a run finish.** The committed total is the hard stop.
- **Numbers are computed in Python, never by a model.** A model interprets; it does not calculate.
- **Every module, class, function and method carries a docstring** saying what it does and how it
  relates to the system. Nothing enforces this automatically — it holds by review, so keep it up.
- **Tests require no network and no credentials.** `.venv/bin/python -m pytest -q` — 168 at present.
- **`data/` is not committed**: snapshots, audit logs, run artefacts and backtests are regenerable.
