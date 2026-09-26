# Local model experiments — Qwen 3 on the M3 Max

A record of the work done on 26 Sep 2026 to make locally served Qwen models usable as the claim and
veto agents, and of every code change it led to. Each change is listed with the run that exposed
the need for it, so the reasoning can be checked against the data.

**Summary.** Qwen 3 14B went from agreeing with Claude barely at all (rank correlation 0.28, average
score gap 2.10) to agreeing moderately well (0.62, gap 0.99), at no cost per run. Four changes got
it there:

1. A **signal table** listing a rule for and a rule against for every indicator, each with a tripwire
   proven correct.
2. **Worked examples** that state the figures before the conclusion.
3. **Directions set from the table** when a claim copies a table tripwire.
4. **Grouped scoring**, so related claims count once.

Everything here is uncommitted at the time of writing.

---

## 1. Setup

| Item | Value |
|---|---|
| Machine | Apple M3 Max, 64 GB. Moved from an Intel laptop, so `.venv` was rebuilt with uv and Python 3.12.14 (arm64) |
| Model server | Ollama 0.34.4, loopback only (`127.0.0.1:11434`) |
| Context window | `launchctl setenv OLLAMA_CONTEXT_LENGTH 40960`, then restart Ollama. **Required.** The auditor prompt can reach 40,000 tokens, and `local_client.py` refuses any answer where the server cut the prompt short |
| Models | `qwen3:8b` (5.2 GB, Q4_K_M), `qwen3:14b` (9.3 GB). Both run fully on the GPU |
| Test bed | Snapshot `528d7570…` (17 Sep 2026, 498 stocks), `config/screen.15.json` → the same 15-stock shortlist every run |
| Baseline | Claude run `20260918T162208Z-b1e861` (18 Sep, Sonnet 5 / Haiku 4.5, $1.61) |

Runs are free (the plan commits $0.00) and reproducible:

```bash
.venv/bin/python -m sealed_window plan --snapshot 528d7570159948cd5aa3db05f777f98a1a52b4f708e467c9a5df03b89113e227 \
    --screen config/screen.15.json --model local --local-model qwen3:14b
.venv/bin/python -m sealed_window evaluate --snapshot 528d7570159948cd5aa3db05f777f98a1a52b4f708e467c9a5df03b89113e227 \
    --screen config/screen.15.json --model local --local-model qwen3:14b --approve-plan <plan hash>
```

A side-by-side claims report for four stocks (MCX, LUPIN, LLOYDSME, ANANDRATHI), comparing run 4 with
Claude, is in `data/analysis/qwen_vs_claude_claims.md`. `data/` is not committed.

---

## 2. The runs, in order

All runs use the same 15 candidates. "For : against" counts claims made. "Discarded" means rejected
as invalid or untestable. BUY / WATCH / AVOID are the published recommendations.

| # | Run | Model and prompt | Time | Claims for : against | Survived | Discarded | BUY / WATCH / AVOID |
|---|---|---|---|---|---|---|---|
| 0 | `20260926T044540Z-dc9406` | Offline rule-based stand-in (not an LLM) | seconds | 56 | 48 | 5 | 5 / 5 / 0 |
| 1 | `20260926T060318Z-a82159` | qwen3:8b, original prompt | 21 min | 92 : 72 | 94 | **61** | 1 / 9 / 0 |
| 2 | `20260926T062443Z-e19cfb` | qwen3:14b, original prompt | 32 min | 95 : 48 | 111 | 11 | 3 / 7 / 0 |
| 3 | `20260926T094432Z-aea442` | 14b + risk-only checklist | 21 min | 38 : 64 | 78 | 20 | 1 / 7 / 7 |
| 4 | `20260926T111431Z-1ff470` | 14b + two-sided checklist and reference levels | 34 min | 119 : 57 | 153 | 2 | 8 / 2 / 0 |
| 5 | `20260926T131512Z-bf9cb1` | 14b + signal table and examples | 48 min | 115 : 69 | 153 | 0 | 5 / 5 / 2 |
| 6 | `20260926T141150Z-35eaf2` | 14b + signal table + table directions + grouped scoring (`ranking-v2`) | 82 min | 117 : 62 | 148 | 2 | 4 / 6 / 2 |
| — | `20260918T162208Z-b1e861` | **Claude** baseline | minutes | 75 : 62 | 127 | 6 | 0 / 10 / 4 |

Agreement with Claude, both scored under `ranking-v2`:

| | Run 5 as published (v1) | Run 5 re-scored (v2 + table directions) | **Run 6** |
|---|---|---|---|
| Average score gap vs Claude | 2.10 | 1.30 | **0.99** |
| Rank correlation with Claude (1 = same order) | 0.28 | 0.49 | **0.62** |

Run 6 took longer than run 5 (82 min against 48) for no reason identified; the prompt was the same
length. Treat run times as indicative.

---

## 3. What each run showed, and the change it led to

### Run 1: qwen3:8b is not usable
- **35 pinned tripwires.** The threshold was the stock's own reading, e.g. `rsi_14 > 40.348574` with RSI
  at 40.348574, so the tripwire could never fire. Nearly all were on claims *against* a stock, so the
  negatives were discarded and scores drifted positive (70 : 24 surviving).
- **The auditor was ineffective.** 61 of 67 challenges were rejected, 52 of them because they just
  repeated the tripwire of the claim they challenged.
- A telling example (CCL): "RSI is in the oversold range" at RSI 40.35. It's false by the usual
  definition (≤ 30), and it was protected by a tripwire pinned at 40.348574.

**Decision:** drop 8B and work with 14B.

### Run 2: qwen3:14b is clean but one-sided
- No pinned tripwires; the auditor worked (26 of 54 challenges valid, 13 claims vetoed).
- But it argued against stocks far less than Claude did (48 against, to Claude's 62), and it found
  **none of Claude's four AVOIDs** (MCX, LLOYDSME, LUPIN, CCL). On MCX it reported strong ROCE, ROE
  and profit growth, all true, and never mentioned a P/E of 53× against a sector 32×.

**Change → risk checklist** (run 3).

### Run 3: the risk-only checklist overcorrected
- Claims swung to 38 for : 64 against, with 7 AVOIDs. It caught 3 of Claude's 4, but invented 4 Claude didn't agree with.
- **The 6-claim cap** (`ClaimBatch.claims`, `max_length=6`) collided with six risk items per analyst, so the
  risks used every slot and strengths disappeared.
- The valuation item named six columns; Qwen wrote one tripwire over all six, which broke the
  4-column limit. The MCX valuation claim was rejected, along with nine others.

**Change → two-sided checklist** with one measure per item, plus reference levels (run 4). The RSI zones
agreed that day were:

| RSI-14 | Reading |
|---|---|
| ≤ 30 | Oversold (possible rebound) |
| 30–45 | Weak: a **pullback in an uptrend** if price is above its 200-day average, **bearish** if below |
| 45–55 | Neutral, no judgement |
| 55–70 | Bullish |
| ≥ 70 | Overbought |

MACD was spelled out: `macd_hist > 0` means the MACD line is above its signal line (bullish), and
`< 0` means the signal line is above the MACD line (bearish).

### Run 4: the checklist was recited, not checked
- Qwen wrote a P/E claim for all 15 stocks, and **14 said "cheap on earnings"**. **7 of the 8 stocks
  whose P/E was above the sector's were called cheap.** 13 of 15 fundamental answers filled all 6 slots
  from the top of the "for" list, in list order.
- **Some tripwires were backwards.** ACUTAAS: "P/E below the sector P/E", tripwire `pe <= sector_pe`,
  with P/E 70.7 against 6.0. Three such claims survived, and helped make it a BUY. This is the "inverted
  tripwire" hazard CLAUDE.md records as undetectable by the checker.
- **Wrong operators:** MCX "thin rally" survived with a 5-day return of −2.53%, because the tripwire
  joined its two conditions with AND where the negation needs OR.
- **Why:** `ClaimDraft` fields are generated in the order `predicate → direction → statement →
  justification → evidence → confidence → falsifier`. With thinking off, Qwen commits to a direction
  and a sentence **before writing any figure**. Claude's statements quote both numbers ("P/E of 53.07x
  … above sector 32.37x"); Qwen's quoted none.
- **Bad benchmarks from Upstox:** some sector values are negative (LLOYDSME sector P/E −142.9,
  KAJARIACER −35.9, CCL −11.1; LUPIN sector ROE −2.52%). "Above the sector" means nothing against these.

**Change → the signal table** (run 5).

### Run 5: the table fixed the sentences and tripwires, not the labels
- 142 of 184 claims copied a table tripwire exactly. **Nothing was discarded as invalid.**
- Statements now quote figures and reach the right conclusion. Every "cheap" claim on an expensive
  stock was correctly refuted by its tripwire. The negative-sector guard refuted comparisons against
  meaningless benchmarks.
- **8 claims had the right sentence and tripwire but the opposite label**, e.g. "P/B 7.36x against a
  sector 2.13x, so it is expensive on book value", labelled FOR. All 8 survived, and 5 **added** to the score.
- **Scores ran hot** (GESHIP +8.85 against Claude's +3.49): with 26 indicators, a strong company
  collects many correlated "for" claims (ROCE, ROE and ROA; 30-day, 90-day and 1-year returns; three
  moving averages).

**Change → table-assigned directions and grouped scoring** (run 6).

### Run 6: the current state
- 6 labels set from the table (noted in the dossier and the audit log). 145 of 179 claims used table
  tripwires, and 2 were discarded.
- Close to Claude on AEGISLOG, GESHIP, EMMVEE, ACUTAAS, IKS, GLENMARK, MCX (−0.48 against −0.54) and CCL
  (AVOID in both).
- **Still disagrees:** LLOYDSME (Qwen BUY +2.56, Claude AVOID −0.87), KAJARIACER (Qwen AVOID −0.94,
  Claude WATCH +1.62), LUPIN (Qwen +0.71, Claude AVOID −0.91), TITAN (Qwen +1.50, Claude 0.00).

---

## 4. Code changes

### 4.1 `sealed_window/claims/signals.py` (new): the signal table
- **26 signals, 56 rules, 11 groups, 105 worked examples.**
  - Fundamental (13): P/E, EV/EBITDA and P/B vs sector; ROCE, ROE and ROA vs sector; revenue and profit
    growth; margin trend; leverage trend; bank NIM, Net NPA and CASA.
  - Technical (13): RSI (five zones); 200-, 50- and 20-day trend; MACD vs signal; MACD vs zero; 30-day,
    90-day and 1-year return; drawdown from the 52-week high; volatility; ATR; volume behind the move.
- Each `Rule` carries its direction, a label, its condition, its **falsifier (tripwire)** and a
  statement template. Each `Signal` carries a meaning line and a **guard**: when the comparison means
  anything. For example, `sector_pe > 0`, `is_bank == 0` for ROCE and leverage, and `is_bank == 1` for
  the bank ratios.
- **Guards are folded into the falsifier**, so a claim against a meaningless benchmark is refuted
  rather than counted: "expensive on earnings" is wrong if `pe <= sector_pe OR pe <= 0 OR sector_pe <= 0`.
- **Thresholds** are round zone edges only: RSI 30/45/55/70; 0 for trends, MACD and returns; drawdown
  −10/−20; volatility 20/35; volume ratio 1; ATR 2/4.
  - *ATR changed from the first draft:* no Nifty 500 stock had ATR below 1.5% (median 2.8%), so the bands
    moved from 1.5/3 to 2/4.
- **Worked examples** are rendered by Python from real readings in the 17 Sep snapshot, never from
  that day's 15-stock shortlist. The picking rules: the stock must satisfy the rule clearly (not on its
  edge), its readings must be inside the universe's 1st–99th percentile, and no stock is used more than
  twice. Every example states the figures before the conclusion. The examples are frozen text in the
  module, so system prompts stay identical across runs.
  - *Why not Claude's or Qwen's claims:* Qwen's quoted no figures. Claude's were good in style, but a
    few had loose tripwires. Rendered examples are correct by construction.
- **Lookups used after the model answers:**
  - `rule_for_falsifier(text, dimension)` finds an exact copy of a table tripwire, ignoring spacing and case.
  - `group_for_falsifier(text, dimension)` returns the group of a copied tripwire. Otherwise it groups by
    columns: if every column (apart from `is_bank`) belongs to one group, so does the claim. So Claude's
    `roe_pct < sector_roe_pct OR roce_pct < sector_roce_pct` joins "returns on capital". A tripwire that
    spans groups, or uses columns outside the table, stays ungrouped.

### 4.2 `sealed_window/agents/prompts.py`: prompts rendered from the table
- The fundamental and technical system prompts now end with **Signal rules**, generated from the table.
  They contain six instructions, then each signal's meaning, use condition, FOR and AGAINST rules with
  tripwires, and examples. The instructions are:
  1. compare the figures yourself;
  2. write figures first, conclusion after;
  3. one rule per claim, copying its direction and tripwire exactly;
  4. at most 6 claims, the most material, with both sides whenever both hold;
  5. skip signals outside their use;
  6. claims outside the table are allowed, with the model's own tripwire.
- `MAX_CLAIMS_PER_CALL` is read from `ClaimBatch`, so the stated limit can't drift from the enforced one.
- The news analyst gets no table.
- Prompt size: about 4,400 tokens (fundamental) and 4,000 (technical), well inside the 24,000 / 20,000 input limits.
- This replaced two intermediate versions written the same day, the risk-only checklist and the
  two-sided checklist with reference levels. Their history is kept in the `_SIGNAL_RULES_INTRO` docstring.

### 4.3 `sealed_window/agents/claim_agents.py` and `claims/schema.py`: directions from the table
- `direction_from_table(draft, dimension)`: if a draft copies a table tripwire and its label contradicts
  that rule, the rule's direction is used. A draft with the model's own tripwire is never touched.
- `Claim.direction_source`: `"model"` or `"signal_table"`. It defaults to `"model"`, so older `claims.json` files still load.
- Each correction is written to the audit log as `claim.direction_from_table`, with the model's label
  and the table's. The orchestrator adds a dossier note with the count.

### 4.4 `sealed_window/adjudicate/adjudicator.py`: grouped scoring, `ranking-v2`
- **Score = Σ weight(dimension) × Σ over units of sign × confidence.** A *unit* is one side of one
  signal group and counts its **highest** surviving confidence. An ungrouped claim is its own unit.
- The for and against sides of a group both count, so a real disagreement (cheap on earnings, expensive
  on book) is kept.
- Dossier tallies still count every surviving claim; only the score arithmetic changed.
- **Scores under v2 are not comparable with v1 scores.** Re-score stored runs before comparing them;
  the dossier records the version used.

### 4.5 `sealed_window/orchestrator.py`
- Passes the audit log to the claim agent, and adds the direction-correction note.

### 4.6 Tests: 209 → 226
- `tests/test_signal_table.py` (8 tests):
  - every tripwire is **proven** to be the exact negation of its guarded rule over 3,000 sampled rows per
    rule, clustered on the edges;
  - every example satisfies its rule and quotes only its own figures;
  - columns are real and in the right dimension; tripwires are within language limits; thresholds are round;
  - every signal has both sides and at least 4 examples;
  - the prompts carry exactly their own dimension's rules; the RSI zones and MACD lines are spelled out.
  - A mutation check confirmed the negation test catches a backwards operator, AND in place of OR, and a
    wrong edge (`>= 30` for `> 30`).
- `tests/test_direction_and_grouping.py` (9 tests): the rule lookup inverts the table; the direction is
  corrected only for a contradiction with a copied rule; old ledgers load; related claims count once at
  the highest confidence; both sides of a group count; ungrouped claims count individually; the model's
  own tripwires are grouped by column; the ranking version is recorded.
- `tests/test_signal_checklist.py`, written for the intermediate checklist, was removed with it.

---

## 5. Design decisions

- **Is setting the direction in Python "de-agenting" the system?** No. The model still decides:
  - which of 26 signals matter (only 6 claims are allowed);
  - reading the figures and deciding which rule holds;
  - the sentence, the argument and the confidence;
  - claims outside the table, with its own tripwires;
  - all news analysis;
  - every audit challenge.

  Once the model has chosen "expensive on book value", "against" follows by definition. The harness
  already assigns each claim's company and ID for the same reason, and has always computed the score.
- **Checklist, not quota.** An item that doesn't hold produces no claim, so a clean company isn't
  forced into an invented negative (prompt rule 6, "never pad").
- **Correct the label, don't reject the claim.** Rejecting would have discarded correct negatives, the
  very claims that were missing.
- **Highest confidence per group side, not the sum.** Three statements of one strength are one strength.
- **No Jinja.** The project restricts what the sealed analysis process may import, and adding a
  template engine would be a policy change. Plain Python renders the same single template from one table.
- **Examples rendered, not hand-picked.** They're correct by construction, and re-checked by tests.

---

## 6. Open items

1. **LLOYDSME, KAJARIACER, LUPIN and TITAN** still differ from Claude. They need a claim-by-claim look,
   as done for MCX in `data/analysis/`.
2. **Negative sector benchmarks** come from Upstox itself. The guards neutralise them in tripwires, but
   the screen and the other models (including Claude) still see them.
3. **Near-value tripwires** (e.g. `drawdown >= -5.0` with a reading of −5.13) still escape the pinned
   check, which only catches exact copies. A per-indicator band check was proposed and not built. Table
   tripwires avoid the problem, but a model's own tripwires can still have it.
4. **The analyst tools are never used.** Four research tools (deeper history, full statements, peer
   comparison, more news) are budgeted in every plan, and no model, Claude included, has called one.
   This is the most obvious place to add real agency.
5. **Not yet tried:**
   - larger or newer Qwen models (`qwen3:32b`, `qwen3.6:27b`, `qwen3.8:27b`, `qwen3.6:35b-a3b`);
   - thinking mode (`REASONING_EFFORT = "none"` in `local_client.py`; turning it on needs larger output limits);
   - a fresh Claude run with the signal table, needed to know whether Claude also changes under it;
   - fresh data (the snapshot is from 17 Sep).
6. **OpenAI GPT support** was discussed and parked. It would need a policy change (new host, key and
   base-URL guard), a client adapted from `local_client.py`, prices, and tests.
7. **Claude baseline caveats:** two of Claude's fundamental answers (ANANDRATHI, CCL) and one auditor
   batch (KAJARIACER, CCL, APOLLOHOSP) were lost to length errors on 18 Sep. The prompt fix for that
   (commit `b998a45`) hasn't been exercised by a real Claude run.
8. **Housekeeping:** README still says 127 tests, and its `uv pip install -e` line fails on this layout.
   Install with `uv pip install --python .venv/bin/python -r pyproject.toml --extra dev`.
