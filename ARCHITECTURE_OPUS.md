# Sealed Window Stock Evaluator

**Architecture proposal — v1 · 13 Sep 2026**

An NSE equity research system whose governing property is not that the agents are
sandboxed, but that **the agents run offline**. Acquisition and analysis never
overlap in time. When a model is running, there is no network. When the network is
live, there is no model.

| | |
|---|---|
| **Status** | Proposal — no code written |
| **Data** | Upstox Analytics API (read-only), fetched by a non-agentic ETL |
| **Models** | Claude Opus 5 / Sonnet 5 / Haiku 4.5 |
| **Scope** | Research and recommendation only — no order placement |

---

## The three pillars

1. **Sealed window** — a run is split into phases. Network access exists in
   exactly one of them, and no model is loaded during it. Agents only ever read a
   frozen, content-addressed snapshot.
2. **Falsifiable claims** — agents do not produce scores or recommendations. They
   produce typed claims, each carrying a machine-checkable falsifier. A
   deterministic adjudicator kills the claims whose falsifier fires. Ranking is
   arithmetic over the survivors; no model computes it.
3. **Prepaid spend** — the entire cost of a run is compiled and committed *before
   the first model call*, from counts that are known after the deterministic
   screen. Not a meter that trips at a ceiling — a plan with no headroom.

---

## 01 · The invariant

> **`NETWORK_LIVE` and `MODEL_RUNNING` are never simultaneously true.**

Everything in this document is downstream of that one line. It is worth stating as
a timing diagram, because that is what it is:

```
             ┌─── phase 1 ───┐
NETWORK   ───┘               └──────────────────────────────────────────
              LIVE            SEALED ────────────────────────────────────

MODEL     ────────────────────────┐       ┌───┐       ┌───┐
                                  └───────┘   └───────┘   └─────────────
              none        none    running  none  running   none

          │  1 ACQUIRE  │ 2 SCREEN │ 3 CLAIM │ 4 ADJUDICATE │ 5 VETO │ 6 PUBLISH │
             ETL          determin.  agents    deterministic  auditor  determin.
             no model     no model             no model                no model
```

The seal drops at the end of phase 1 and does not lift for the rest of the run.

### Why this is a stronger claim than "the agent may only issue GET"

- **It is cheaper to prove.** "This process has no network" is a socket-level
  assertion, or a network namespace with no route. "This process may only issue
  `GET` to these seventeen paths" is a reference monitor you have to write,
  configure, keep current, and trust.
- **Exfiltration stops being possible rather than stopping being allowed.** The
  worst outcome of a successful prompt injection is a bad claim — which phase 4
  is built to kill. There is no channel to leak to, so there is no leak to
  prevent.
- **Reproducibility is free.** The snapshot hash fully determines the inputs. Two
  runs over the same snapshot with the same spend plan produce the same dossier,
  bit for bit, which is what makes the backtest and the regression suite mean
  something.
- **"Which endpoints?" becomes a code-review question, not a runtime one.** The
  ETL is a fixed set of queries in reviewed code with no model anywhere near it.
  There is no agent deciding what to fetch, so there is nothing to constrain at
  request time.

### The cost of this design, stated plainly

**It cannot answer an intraday question.** If you want to know why HDFC Bank is
down 4% *right now*, the sealed window fights you: you have to re-seal a fresh
snapshot, which makes the ETL latency-critical and turns a nightly batch into an
on-demand service. That is a real tradeoff and the main thing to weigh before
adopting this over a live-confinement design. See Open Decision 1.

---

## 02 · Phase 1 — Acquire

**Model: none. Network: live.**

A scheduled, deterministic ETL. No LLM is loaded in this process; no agent exists
yet. It reads the Upstox Analytics API and materialises a **sealed snapshot**.

| Property | Detail |
|---|---|
| Credential | `UPSTOX_ANALYTICS_TOKEN`, read-only by issue. Lives only in the ETL process. |
| Queries | A fixed, code-reviewed set. Adding one is a pull request, not a config toggle. |
| Output | Content-addressed, immutable, with a manifest and an `as_of` timestamp. |
| Seal | On completion the snapshot is written read-only and its root hash recorded. |

### What a snapshot contains

```
snapshot/4c1f8a02…8ae2/
  manifest.json          as_of, upstox API version, query set hash, row counts
  instruments.parquet    NSE equity master
  candles/               daily OHLCV, 500 sessions, adjusted + unadjusted
  fundamentals.parquet   quarterly and annual, 12 periods
  news/                  headlines + bodies, 30-day window, per instrument
  derived.parquet        RSI, MACD, ATR, MA20/50/200, volatility, volume ratio,
                         ROE, ROIC, D/E, growth rates — all computed here, in
                         deterministic Python, before any model exists
```

**Every number the system will ever reason about is computed in phase 1.** Not
because a model would be careless with arithmetic, but because a number computed
inside the sealed window would have no provenance — nothing to hash, nothing to
replay, nothing to check a falsifier against.

Each row carries `evidence_id = blake3(query, params, as_of, row_key)`. Evidence
IDs are the only way anything downstream is allowed to refer to a fact.

---

## 03 · Phase 2 — Screen

**Model: none. Network: sealed.**

The snapshot plus the user's screen configuration produces a candidate set, purely
by filtering. This is the phase the UI sliders drive.

The screen config is hashed and recorded. Together with the snapshot hash it
already determines the candidate set with no model involved — so a disputed
shortlist can be reproduced exactly, before any question of model behaviour
arises.

### The spend plan is compiled here

Candidate counts are now known, so the exact shape of the run is known. The
compiler emits a plan and freezes it:

```
plan 3e91b4…  ·  snapshot 4c1f8a02…8ae2  ·  screen cfg 91b07c…
compiled 2026-09-13T09:12:04+05:30

slot class      model             calls   max_in   max_out   committed
──────────────────────────────────────────────────────────────────────
triage          claude-haiku-4-5    148    1,800       320      $0.50
deep.fundamental claude-sonnet-5     31   11,000     1,800      $1.24
deep.technical  claude-sonnet-5      31    6,500     1,200      $0.78
deep.news       claude-haiku-4-5     31    9,000       700      $0.39
probe reserve   claude-haiku-4-5     20    4,000       600      $0.14
synthesis       claude-opus-5         3   74,000    14,000      $2.16
veto            claude-sonnet-5       1   38,000     3,000      $0.11
──────────────────────────────────────────────────────────────────────
                                committed total                 $5.32
                                hard stop                       $5.32
```

Note the last two lines. **The committed total and the hard stop are the same
number.** That is the difference between this and a budget ledger: a ledger is a
runtime control that permits variable behaviour until the money runs out, so the
honest thing it can tell you in advance is "at most $8". A compiled plan tells you
"$5.32", and is right.

Mechanically:

- Every call must cite a **slot** from the plan. A call with no slot cannot be
  issued — there is no code path that constructs a request without one.
- `max_in` is enforced by `count_tokens` before dispatch. An over-budget prompt is
  a bug in the assembler, surfaced immediately, never silently truncated.
- Adaptivity draws from the **probe reserve**: twenty prepaid Haiku slots. An
  agent that wants a follow-up spends one. When the pool is empty, probes stop.
  There is no arithmetic at runtime, just a counter that only goes down.
- A run that exhausts its plan **finishes** — it does not stop dead. The plan is
  ordered so that the last slot to be spent still produces a publishable dossier
  over whatever survived.

---

## 04 · Phase 3 — Claim

**Model: running. Network: sealed.**

This is the first phase in which a model exists, and it is structurally incapable
of doing anything but reading a snapshot slice passed to it in memory and
returning a typed object. It has no filesystem, no network, no shell, no code
execution and no tool that reaches outside the process.

### Agents emit claims, not scores

The central departure from a conventional scoring pipeline: **an agent may not
return a number representing its opinion.** `"technical_score": 78` is
unfalsifiable — there is no fact about the world that could show it to be wrong,
so nothing downstream can check it. Instead:

```json
{
  "subject": "NSE_EQ|INE090A01021",
  "predicate": "capital_efficiency_improving",
  "statement": "ROE has expanded from 15.2% to 18.4% across the last eight quarters while leverage was flat.",
  "evidence": ["ev:7f3a91c0", "ev:22b0c4de", "ev:0ac71b93"],
  "confidence": 0.72,
  "falsifier": "roe_ttm < 16.0 OR roe_slope_8q <= 0 OR debt_to_equity_delta_8q > 0.15"
}
```

Two rules make this work:

1. **Every claim carries a falsifier** — a predicate over snapshot columns which,
   if it evaluates true, refutes the claim. The agent writes the condition under
   which it would be wrong.
2. **Every claim cites evidence IDs**, and a claim citing an ID not in the
   snapshot manifest is rejected at parse time, before adjudication.

A model that cannot state what would refute its claim has not made a claim. It has
made an impression, and impressions do not belong in a valuation.

### What this buys against prompt injection

News text lives in the snapshot and is read offline. A crafted headline can
absolutely persuade an agent to emit a false claim — no envelope wording prevents
that. But the claim then has to survive phase 4, where its falsifier is evaluated
against structured data the headline cannot touch. **Injection can produce a
claim; it cannot produce a claim that passes a machine check.** And with the
network sealed, there is nowhere for it to send anything.

---

## 05 · Phase 4 — Adjudicate

**Model: none. Network: sealed.**

Deterministic. For every claim, evaluate its falsifier against the snapshot.

- Falsifier fires → the claim is **discarded**, not down-weighted. A refuted claim
  contributes nothing, because a claim whose own stated refutation condition holds
  is simply false.
- Falsifier does not fire → the claim survives with its stated confidence.
- Falsifier cannot be evaluated (references a column that does not exist, or data
  outside its freshness SLA) → the claim is **discarded**, and the run records
  why. Unevaluable is not the same as true.

Ranking is then a fixed formula over surviving claims, grouped by dimension:

```
score(stock) = Σ  w[dimension] · Σ confidence(c)
              dim              c ∈ surviving(stock, dim)
```

with `w` a versioned constant. **No model computes the ranking.** The model's job
is to generate hypotheses; the machine's job is to kill the bad ones and count
what is left.

This is what makes the system's output auditable in a way a scoring rubric is not:
you can ask "why is ICICI ranked third?" and the answer is a list of surviving
claims and the falsifiers that did not fire — not a number a model felt was about
right.

---

## 06 · Phase 5 — Veto

**Model: running. Network: sealed.**

A single auditor pass with a deliberately asymmetric power: **it can only refute.**

- It receives the surviving claims and the snapshot slices they cite — **not** the
  original agent's reasoning, so it cannot be anchored by it.
- Its only legal output is a list of refutations, each of which must itself carry
  a falsifier over the snapshot.
- It has no way to endorse, approve or raise a confidence. There is no field for
  it.

Refutations go back through phase 4 and are adjudicated exactly like claims — an
auditor that asserts something unfalsifiable is ignored on the same rule as
everybody else.

The asymmetry is the point. A second model asked "does this look right?" will
mostly say yes, and you will have bought agreement rather than scrutiny. A second
model that can *only* attack, and whose attacks must themselves survive a machine
check, is doing work.

---

## 07 · Phase 6 — Publish

**Model: none. Network: sealed.**

The dossier is assembled deterministically from surviving claims. There is no
generative step here, which means every sentence in the output traces to a claim,
and every claim traces to evidence IDs, and every evidence ID traces to a row in a
hashed snapshot.

### The reproducibility triple

Every published recommendation is stamped with three hashes:

```
snapshot   4c1f8a02…8ae2    what was true, and when
screen cfg 91b07c…          what you asked for
plan       3e91b4…          what was spent, on which models
```

Those three fully determine the output. "Show me exactly why you said this" is
three lookups, not an investigation.

### What each recommendation carries

- **BUY / WATCH / AVOID**, with the surviving-claim count per dimension
- The surviving claims in plain prose, each with its evidence and its falsifier
  still attached — *the reader can see what would make this wrong*
- Claims that were **refuted**, and by what. A thesis that survived three attacks
  is worth more than one nobody tried to kill, and hiding the attacks throws that
  information away.
- The snapshot `as_of`, prominently. A recommendation is a statement about a
  moment, and the moment is part of the statement.

### Abstention

If fewer than N stocks accumulate enough surviving claims, the run publishes fewer
and says so. There is no mechanism to pad the list, because there is no model at
this stage that could write an extra entry.

---

## 08 · The application

FastAPI, with a thin front end. Left rail, two tabs.

| Fundamental tab | Technical tab |
|---|---|
| ROE floor · ROIC floor · debt-to-equity ceiling | RSI band · price against MA20 and MA50 |
| Revenue growth · earnings growth · P/E band | Volume ratio floor · ATR ceiling |
| Market-cap floor · promoter-pledge ceiling | 30-day and 90-day return bands |

Sliders are arguments to the phase-2 filter. They are hashed into the screen
config and never reach a prompt — not as a safety patch, but because phase 2 has
no model in it for them to reach.

Also on screen:

- The **seal indicator** — which phase the run is in, whether the network is live,
  whether a model is running. The invariant from clause 01, rendered as a live
  status rather than a promise in a document.
- The **spend plan**, shown before the run starts, with the committed total. You
  approve a number, not a ceiling.
- A **claim ledger** — claims appearing, then being struck through as phase 4
  refutes them. Watching claims die is the most informative view of a run.

---

## 09 · Build order

| Phase | Deliverable | Gate |
|---|---|---|
| **P0** | Snapshot format, manifest, evidence IDs, content addressing. **No network, no model.** | A snapshot round-trips; two builds over identical inputs produce identical hashes. |
| **P1** | The ETL and the Upstox adapter. Network exists here and nowhere else. | Seal asserted: after phase 1 the analysis process has no route. Test proves a socket attempt raises. |
| **P2** | Derived-indicator engine inside the ETL. | Indicators reconciled against a known-good reference series. |
| **P3** | Screen, screen-config hashing, spend-plan compiler. | Compiled cost matches actual spend to the cent on a dry run with stub models. |
| **P4** | Claim schema, falsifier DSL, the adjudicator. | Adversarial claim corpus: every hand-written false claim is correctly killed. |
| **P5** | Phase 3 agents against the sealed snapshot. | Committed plan is never exceeded; no call without a slot. |
| **P6** | The veto auditor. | Veto demonstrably changes outcomes on a seeded corpus — if it never refutes anything, it is theatre. |
| **P7** | UI: sliders, seal indicator, spend approval, claim ledger. | |
| **P8** | Backtest over historical sealed snapshots. | Lookahead-bias check clean. Does claim-survival ranking beat the raw screen? |

P0 to P2 have no model in them at all and are useful on their own. That is
deliberate: if the agentic layer turns out to add nothing over the deterministic
screen, P8 will say so, and you will have lost nothing but the agentic layer.

---

## 10 · Relationship to the Ring Zero proposal

You have both documents, so this should be explicit.

**Where they agree, and why that is not a coincidence:** read-only Analytics token
only; no trading, portfolio, funds or order endpoints; no generic HTTP tool; the
model never computes a scored number from raw data; evidence with timestamps and
forced abstention on stale or missing data; deterministic code owning control
flow; no silently raising a budget. These are not stylistic choices. Any competent
design for this brief lands on them, and a second design that diverged for the
sake of diverging would be a worse design, not a more original one.

**Where they genuinely differ:**

| | Ring Zero | Sealed Window |
|---|---|---|
| Governing idea | Confine the agent while it reaches out | Never let the agent reach out |
| Enforcement | Reference monitor: method pin, path allowlist, egress proxy | Temporal separation: no network in the analysis process |
| Agent output | Scores and narrative, schema-validated | Typed claims with machine-checkable falsifiers |
| Ranking | Weighted composite of model-assigned scores | Arithmetic over claims that survived adjudication |
| Cost control | Runtime ledger with reserve-and-settle against a ceiling | Plan compiled and committed before the first call |
| Second opinion | None | Auditor with veto-only power |
| Intraday | Supported | Requires re-sealing — the main weakness |

They are not variants of each other. Ring Zero is the right design if you need
live reactivity and are willing to maintain a reference monitor. This one is the
right design if you would rather have reproducibility and a much smaller thing to
trust, and can live with a snapshot cadence.

---

## 11 · Open decisions

**1. Snapshot cadence — the central tradeoff.**
Nightly seal after close is the natural cadence for a fundamentals-led screener,
and makes everything above cheap. Intraday reactivity means re-sealing on demand,
which turns the ETL into a latency-critical service and weakens the "one
acquisition window" story. If intraday is a hard requirement, say so now — it
changes the architecture, not a parameter.

**2. Falsifier expressiveness.**
A small restricted DSL over snapshot columns — comparisons, boolean connectives,
a fixed function set — is auditable and safe to evaluate. Restricted Python is
more expressive and a much larger thing to trust. I would start with the DSL and
widen it only when a real claim cannot be expressed.
→ *Recommend: DSL*

**3. Should the veto auditor be a different model family?**
Using Claude for both claim and veto means correlated blind spots. A genuinely
independent auditor means a second provider — a new dependency, a second
credential, and a second set of governance questions. Defensible either way; I
would ship with Sonnet as the auditor and revisit if P6's gate shows it rarely
refutes.

**4. Snapshot retention.**
Reproducibility is only real for as long as the snapshots exist. Two thousand
names with 500 sessions of candles, twelve periods of fundamentals and a 30-day
news window is not large, but daily seals accumulate. How long do you want to be
able to reproduce a recommendation — a quarter, a year, forever?

**5. Carried forward from the earlier proposal.**
Scope is this folder only; Anthropic confirmed; Python assumed; and Upstox
fundamentals coverage for mid- and small-caps still needs checking before P2,
since thin coverage would mean a second source in the ETL and your approval before
it is wired.

---

*Sealed Window Stock Evaluator · Architecture proposal v1 · 13 Sep 2026*
*Research and recommendation only — no order placement.*
