# Ring Zero Stock Evaluator

**Architecture proposal — v1 · 13 Sep 2026**

A fundamental, technical and news-driven screener for NSE equities, built so the
agents are *structurally incapable* of trading, reading a portfolio, or spending
past a declared ceiling.

| | |
|---|---|
| **Status** | Proposal — no code written |
| **Data** | Upstox Analytics API (read-only) |
| **Models** | Claude Opus 5 / Sonnet 5 / Haiku 4.5 |
| **Scope** | Research and recommendation only — no order placement |

---

## 01 · Premise — governance is a chokepoint, not a paragraph

> Any design where the rule *"agents may only issue GET requests"* lives in a
> system prompt is not governance. It is a suggestion, and one bad retrieval away
> from being ignored.

The architecture is arranged around a single property: an agent is **physically
incapable** of issuing a non-`GET`, because it never holds a network client, a
credential, a shell, a subprocess handle, or a filesystem handle. It holds a list
of named functions returning records that have already been fetched, validated
and normalised by code it cannot reach.

Everything below — the ring model, the budget ledger, the bounded probe enum — is
a consequence of taking that property seriously, rather than a set of independent
features bolted on afterwards.

---

## 02 · Access — "GET-only" does not mean "no portfolio"

This is the first correction worth making, because it changes the shape of the
control. Upstox exposes holdings, positions, funds, order history, trade history,
P&L and GTT as `GET` endpoints. A method-only rule permits *all of them*.

So the access rule is four-dimensional, each dimension enforced independently so a
failure in one does not open the door.

| Axis | Rule | Enforced at |
|---|---|---|
| **Credential** | Upstox **Analytics token** only — read-only by issue. Never an OAuth trading token; no fallback path exists in code. | Ring 0 config |
| **Method** | `GET` pinned in the transport layer. Any other verb raises before a socket opens. | Ring 0 transport |
| **Host** | One allowlisted host. No generic-URL or raw-HTTP tool exists anywhere in the codebase. | Ring 0 transport |
| **Path** | Deny-by-default allowlist of **exact path templates**. Absence is denial. | Ring 0 router |

### The allowlist, concretely

| Capability | Path family | |
|---|---|---|
| Instrument master | `/instruments` | ALLOW |
| Quotes & LTP | `/market-quote/…` | ALLOW |
| Historical candles | `/historical-candle/…` | ALLOW |
| Market status | `/market/status/…` | ALLOW |
| Fundamentals | `/…/fundamentals` | ALLOW |
| News | `/…/news` | ALLOW |
| Orders | `/order/*` | **DENY** |
| Portfolio & holdings | `/portfolio/*` | **DENY** |
| Funds & margin | `/user/get-funds…` | **DENY** |
| Trades, charges, P&L | `/trade/*`, `/charges/*` | **DENY** |
| GTT & mutual funds | `/gtt/*`, `/mf/*` | **DENY** |
| Profile | `/user/profile` | **DENY** |

The denied rows live in a test suite, not only in this table. Phase 0 ships a case
per denied family asserting the call raises before egress. A governance rule you
have never tried to violate in a test is a rule you do not have.

> **No Upstox MCP server.** MCP hands the model a tool surface whose shape is
> controlled by the server rather than by us. That inverts the direction of trust
> this design exists to establish, so the connector stays out of scope however
> convenient it looks.

---

## 03 · Isolation — three rings, three processes

Privilege decreases as you move toward the model. The credential lives at the
network boundary and never travels inward.

```
┌──────────────────────────────────────────────────────────────────────┐
│ RING 2 · AGENT RING                                                  │
│ Claude calls. Interpretation, judgement, narrative.                  │
│ no env · no network · no filesystem · no shell · no code execution   │
└──────────────────────────────────────────────────────────────────────┘
    │
    │  GATE — SCHEMA VALIDATOR
    ▼  named tools only; model output parsed against a versioned schema
       before it enters shared state
┌──────────────────────────────────────────────────────────────────────┐
│ RING 1 · KERNEL                                                      │
│ Orchestrator FSM · token budget ledger · indicator engine            │
│ Evidence ledger · freshness SLA · hash-chained audit log             │
│ Owns all control flow. Computes every number.                        │
└──────────────────────────────────────────────────────────────────────┘
    │
    │  GATE — METHOD PIN + PATH ALLOWLIST
    ▼  localhost IPC; the request is a capability name and typed params,
       never a URL
┌──────────────────────────────────────────────────────────────────────┐
│ RING 0 · EGRESS PROXY                                                │
│ Sole holder of UPSTOX_ANALYTICS_TOKEN. Rate limiter.                 │
│ Content-addressed response snapshots for replay and backtest.        │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼  HTTPS GET → api.upstox.com
```

Privilege flows downward, data flows upward, and nothing crosses a boundary except
through the gate drawn on it.

### Three properties that earn the process split

- The credential sits in **Ring 0's process environment only**. Rings 1 and 2
  cannot read it even by inspecting their own environment. It never appears in a
  prompt, a tool result, a log line, an exception trace, a fixture or a report.
- Ring 2 has **no arbitrary code execution**. Indicators are computed by tested
  Python in Ring 1, never by a model writing code on the fly.
- Every Ring 0 response is snapshotted content-addressed, so a whole run **replays
  offline** — zero network calls, zero model calls. That is what makes the
  regression suite and the backtest honest rather than decorative.

---

## 04 · Division — the model never computes a number

A hard architectural rule, not a preference. An LLM asked to compute RSI will
produce a plausible RSI — and plausible is the worst available failure mode in a
valuation system, because nothing downstream can detect it.

| Deterministic Python — Ring 1 | Claude — Ring 2 |
|---|---|
| RSI, MACD, ATR, MA20/50/200, realised volatility, volume ratio, trailing returns | Interpreting a computed indicator bundle in context |
| P/E, ROE, ROIC, debt-to-equity, revenue and earnings growth, free cash flow | Weighing signals that genuinely conflict with one another |
| Screen thresholds, component weights, composite score, ranking | The narrative: *why* this stock, and what would falsify the thesis |
| Freshness checks, budget arithmetic, control flow, termination | Choosing which bounded follow-up probe to request |

If a model emits a figure, it must be one it cites from an evidence record. A
validator rejects the output otherwise, and the claim never reaches the report.

---

## 05 · Funnel — cost control and quality control are the same mechanism

Spend expensive reasoning on very few things. Each stage is a pure function of the
previous stage's output, and the first two cost nothing at all.

| Stage | Survivors | Engine | Spend |
|---|---:|---|---:|
| 0 · Universe | 2,000 | deterministic | — |
| 1 · Screen | 150 | deterministic | — |
| 2 · Triage | 35 | `claude-haiku-4-5` | $0.45 |
| 3 · Deep dive | 12 | `claude-sonnet-5` | $1.50 |
| 4 · Synthesis | 5–10 | `claude-opus-5` | $2.00 |
| **Full run** | | *before prompt caching* | **≈ $4–5** |

Stage 1 applies the user's slider thresholds. Stage 2 is one small structured call
per stock, roughly 1.5K in and 300 out. Stage 3 gives each survivor a fundamental,
technical and news pass at around 12K in and 2K out. Stage 4 is Opus at
`effort: high` with adaptive thinking, holding all twelve dossiers at once and
making the actual trade-off.

> **This inverts the "big model plans, small models execute" idea in the brief.**
> I would argue for *small models filter, big model decides*. The reasoning that
> genuinely needs Opus is the final comparison between twelve plausible candidates
> under a risk budget — not the dispatching, which is deterministic anyway and
> costs nothing to do in code.

---

## 06 · Budget — a four-level ledger that fails closed

```
Run budget          hard ceiling — e.g. 3M tokens / $8
└─ Stage budget     triage · deep dive · synthesis
   └─ Per-candidate budget    bounded by the stage's survivor count
      └─ Per-call ceiling     reserve → call → settle
```

1. **Pre-flight reservation.** Before every call, `count_tokens` gives the exact
   input cost; add `max_tokens` as worst-case output. If the reservation exceeds
   what remains, *the call never happens* — the stage degrades by truncating its
   candidate list rather than overrunning.
2. **Post-flight settle.** Record real `usage` and release the unused portion of
   the reservation back to the stage.
3. **Prompt caching** on the frozen system prompt, the tool definitions and the
   shared market-context block, with per-candidate volatile data after the last
   breakpoint. A warm-up assertion checks `cache_read_input_tokens > 0`; if it is
   zero, something is silently invalidating the prefix and every cost figure in
   this document is wrong.
4. **A task budget** on the Opus synthesis stage, so the model paces itself and
   lands gracefully instead of being guillotined mid-sentence by `max_tokens`.
5. **No self-healing.** A run that hits its ceiling terminates with partial
   results and says so on the report. Nothing in the codebase may raise a budget
   or relax a guardrail to let a run finish — that is the single most common way a
   governed system quietly stops being one.

---

## 07 · Agency — genuinely agentic, provably bounded

The interesting behaviour in the brief is the adaptive probe: *why did HDFC Bank
fall 4% today? → check the news → compare volume against the 30-day average → is
this market-wide or company-specific?* That is also precisely where unbounded
loops come from. The resolution is to let the model *choose* while the kernel
*bounds*.

- Control flow is a **DAG owned by the orchestrator**. Agents cannot call agents.
  There is no supervisor agent holding a spawn tool.
- A Stage 3 agent may return `probes: [...]` — at most **K = 3**, drawn from a
  **closed enum**: `sector_comparison`, `news_window`, `volume_anomaly`,
  `peer_valuation`, `index_correlation`. Not free text. Not a URL.
- Probes are **depth-limited to one**. A probe result cannot itself generate
  probes, so `max_probes = candidates × K` and there is no cycle in the graph to
  loop around.

### Belt and braces over the structural bound

Node-visit cap, retry cap, wall-clock cap, and a **no-progress detector**: if no
new symbol has entered the top-N and no top-N score has moved by more than ε
across the last M probes, the run terminates. That is the direct answer to *"make
sure we are actually progressing towards picking reliable stocks instead of going
on a pointless search."*

### Progress needs a declared objective

> Fill **N slots** with candidates scoring at or above the threshold and holding
> **100% evidence completeness**.

Progress is measured against that, not against activity. If only three stocks
qualify, the run returns three and explains why. It never pads the list to reach
five — a recommender that always finds ten picks is telling you about its own
defaults, not about the market.

---

## 08 · Injection — news is untrusted input, not instruction

A news agent reading arbitrary text is the largest injection surface here, and the
one place a compromise could realistically move money — not by trading, which is
impossible in this design, but by corrupting a recommendation you then act on by
hand.

- Retrieved text is wrapped in a data envelope. The agent's contract states that
  content inside the envelope is data and never instruction, regardless of what it
  claims about itself.
- The news summariser is **quarantined**: it holds no tools at all and returns
  only a schema-constrained object — `sentiment`, `event_type`, `materiality`,
  `evidence_ids`. Free text it produces never reaches a stage that holds tools.
- Operator instructions mid-run travel by **mid-conversation system messages**,
  the injection-safe operator channel, rather than as text the model could confuse
  with retrieved content.
- A standing injection corpus runs in CI. If a crafted headline can change a
  recommendation, that is a build failure, not a known issue.

---

## 09 · Evidence — cite it or abstain

Every upstream fact gets `evidence_id = hash(endpoint, params, as_of)`. Every model
claim must cite evidence IDs, and a validator rejects any recommendation citing an
unknown ID or resting on evidence older than that field's freshness SLA — seconds
for a quote, a quarter for a balance-sheet line.

**Stale or missing required evidence forces abstention**, never a hedged
recommendation. "I could not verify Q2 revenue growth, so this stays on WATCH" is a
usable output. A confident thesis resting on a number nobody fetched is not.

The audit log is append-only and hash-chained: config version, prompt hashes, model
IDs, every call's token usage, every allow/deny decision at Ring 0, every schema
rejection. Given a report, you can reconstruct exactly which bytes produced it.

---

## 10 · Application

FastAPI with a thin front end, and a left rail carrying the two tabs from the brief.

| Fundamental tab | Technical tab |
|---|---|
| ROE floor · ROIC floor · debt-to-equity ceiling | RSI band · price against MA20 and MA50 |
| Revenue growth · earnings growth · P/E band | Volume ratio floor · ATR ceiling |
| Market-cap floor · promoter-pledge ceiling | 30-day and 90-day return bands |

> **A governance point about the sliders.** They are parameters to the
> deterministic Stage 1 screen and are *never* interpolated into a prompt. User
> input therefore cannot alter agent instructions — it can only narrow a filter.
> That closes an injection path most UI designs leave wide open.

Also on screen: a live **budget meter** showing spent, reserved and remaining; the
funnel counts collapsing 2,000 → 150 → 35 → 12 → 8; and a kill switch that halts
cleanly at the next stage boundary.

### What each recommendation must carry

- **BUY / WATCH / AVOID**, with the composite score and its four components
- The thesis, in plain prose
- The three to five evidence rows that actually drove it, each with source and
  timestamp
- Key risks, stated as risks rather than softened into caveats
- **Explicit invalidation conditions** — *"this thesis fails if Q3 revenue growth
  comes in below 8%, or if price closes below ₹X on rising volume"*
- Confidence, and specifically what would raise it

Version 1 recommends and explains. It does not trade, and the code path to trade
does not exist — which is also what makes the system debuggable, backtestable and
safe to iterate on.

---

## 11 · Sequence — build order, and the gate on each phase

The ordering is the point: the governance kernel ships and is proven before a
single agent is spun up.

| Phase | Deliverable | Gate — what must be true to proceed |
|---|---|---|
| **P0** | Governance kernel: orchestrator FSM, budget ledger, schema validator, audit log, filesystem sandbox. **No network. No model calls.** | Tests prove the *deny* paths: non-GET raises, `/portfolio` is refused, budget overrun fails closed, cycle detection trips. |
| **P1** | Ring 0 egress proxy and the Upstox adapter, driven by recorded cassettes. | Credential provably absent from Rings 1 and 2. Denied-path suite green against the live host. |
| **P2** | Indicator and screening engine, fully deterministic. | Useful on its own at zero model spend; indicators reconciled against a known-good reference series. |
| **P3** | Stages 2–4: tiered models, structured outputs, prompt caching, budget ledger wired live. | Measured cost per run inside the declared envelope; cache hit rate non-zero. |
| **P4** | News ingestion and the quarantine summariser. | Injection corpus passes — no crafted headline changes a recommendation. |
| **P5** | UI: sliders, funnel view, budget meter, per-stock dossiers. | Slider values provably never reach a prompt. |
| **P6** | Backtest harness over point-in-time snapshots. | Lookahead-bias check clean. Does the ranking beat the naive screen it sits on top of? |

P6 is the phase that tells you whether any of this *works*. Everything before it
tells you whether it is safe.

---

## 12 · Open decisions

These change the work materially, so I would rather ask than assume.

**1. Which folder, and which remote?**
The brief pins work to `agentic_stock_evaluator_codex`; we are sitting in
`agentic_stock_evaluator_opus`, which is empty and not a git repository. Confirm
the scope rule is now "this folder only", and whether you want a fresh GitHub
remote created — it would become the sole permitted push target.

**2. Anthropic, confirmed?**
The brief names GPT-5-Codex; you asked for Anthropic. Everything here is costed
and tiered for Opus 5, Sonnet 5 and Haiku 4.5.
→ *Assumed: Claude*

**3. WebSocket now, or phase two?**
A persistent bidirectional socket is a channel you can *write* to, which sits
awkwardly beside a GET-only rule — and fundamental plus technical ranking does not
need tick data. REST snapshots for v1; the V3 feed later behind a read-only reader
with an allowlisted subscribe payload. Push back if you want live ticks from day
one.
→ *Recommend: defer*

**4. How deep is fundamentals coverage?**
If Upstox fundamentals turn out thin for mid- and small-caps, we need a second
source — another egress allowlist entry, another adapter, and your approval before
any of it is wired. Worth checking coverage before P2 rather than discovering it
during P3.

**5. Python?**
Assumed throughout — FastAPI, pandas for the indicator engine, the Anthropic
Python SDK. Say the word if you would rather this were TypeScript.

### Deferred capability — option chain

Dropped from the P1 allowlist. Open interest, put-call ratio and implied
volatility are real positioning signals, but no stage in this design consumes
them, F&O coverage spans only ~200 of the 2,000-name universe, and a full chain
is one of the largest payloads on the API. It goes back in only when a named
stage needs it — the allowlist is derived from what the pipeline consumes, never
from what the vendor happens to offer.

---

*Ring Zero Stock Evaluator · Architecture proposal v1 · 13 Sep 2026*
*Research and recommendation only — no order placement.*
