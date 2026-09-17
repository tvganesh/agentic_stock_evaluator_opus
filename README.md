# Sealed Window stock evaluator

Implementation of `ARCHITECTURE_OPUS.md`: NSE equity research using fundamental, technical and
news analysis, with governance as the spine. Research and recommendation only; there is no
code path that places orders or reads a portfolio.

**Invariant:** the Upstox network and a running model are never live at the same time.

## Quick start

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q                                   # 127 tests, no network, no keys

# 1. ACQUIRE (network live, no model). Offline fixture, or live Upstox (Nifty 500 by default):
.venv/bin/python -m sealed_window acquire --source fixture --as-of 2026-09-11
UPSTOX_ANALYTICS_TOKEN=... .venv/bin/python -m sealed_window acquire --source upstox --as-of 2026-09-14
UPSTOX_ANALYTICS_TOKEN=... .venv/bin/python -m sealed_window acquire --source upstox --universe config/universe.txt  # 48 large caps

# 2. PLAN (sealed, no model): prints the committed spend and the plan hash to approve
.venv/bin/python -m sealed_window plan --snapshot <root_hash>

# 3-6. EVALUATE (sealed): offline stand-in model, or the Anthropic API
.venv/bin/python -m sealed_window evaluate --snapshot <root_hash> --approve-plan <plan_hash> --model offline
ANTHROPIC_API_KEY=... .venv/bin/python -m sealed_window evaluate --snapshot <root_hash> --approve-plan <plan_hash> --model anthropic

# UI: sliders, seal indicator, spend approval, claim ledger
.venv/bin/python -m sealed_window serve        # http://127.0.0.1:8000

# P8: price-only walk-forward over the snapshot's candles (no model, no network, free)
.venv/bin/python -m sealed_window backtest --snapshot <root_hash> --ranking momentum_90d
```

## Walk-forward backtest (P8)

`backtest` rebuilds the technical screen at past dates from the snapshot's own candles, ranks the
survivors by a named rule (`momentum_90d`, `risk_adjusted_momentum`, `trend_quality`), and measures
the next 20 and 60 sessions for the top 5 and top 10: precision, mean net and excess return, worst
drawdown, annualised volatility, turnover and coverage, against acceptance gates.

What it does and does not establish:

- **Benchmark = the screen.** Selections are compared with an equal-weight basket of every stock that
  passed the same screen, so it answers "does ranking beat the raw screen?", not "does it beat the Nifty".
- **Price-only.** A screen config containing fundamental or news filters is refused
  (`LookaheadError`): Upstox ratios carry no as-of date, so using them in a past window leaks the future.
- **Survivorship bias.** The universe is today's index membership; delisted and demoted names are absent.
- **The claim and veto layers are not measured.** That would need a model call per stock per window.

Snapshots request four years of daily candles (`CANDLE_LOOKBACK_DAYS = 1460`, ~35 MB), in the same one
request per instrument that two years used, so the backtest gets ~34 windows instead of ~10 — past the
20-window gate. Upstox allows a decade, but the backtest uses *today's* index membership, so deeper
history compounds survivorship bias faster than it adds evidence; collect more only alongside
point-in-time membership. Candle bytes are
hash-verified when a snapshot loads but parsed only when read, and each window hands the indicator
engine at most 300 sessions, so work per window stays bounded.

First measured result (15 Sep 2026 snapshot, 498 stocks, 21 overlapping windows, two years of history,
technical screen, 20bps round trip). None of the three ranking rules passed its gates:

| ranking | h20 k5 precision | h20 k5 excess % | h60 k5 precision | h60 k5 excess % |
|---|---|---|---|---|
| momentum_90d | 0.533 | +1.30 | 0.419 | −1.02 |
| risk_adjusted_momentum | 0.448 | +0.21 | 0.467 | −1.45 |
| trend_quality | 0.476 | +0.75 | 0.457 | +2.80 |

Short-horizon momentum and long-horizon trend quality look mildly useful; every rule sits near or
below coin-flip precision, turnover is 0.66–0.85 per window, and drawdowns reach −26%.

Acquire and evaluate are separate processes by design. The analysis process scrubs every
`UPSTOX_*` variable. The ETL refuses to start if any Upstox credential other than the
analytics token is present.

## Universe and acquisition time

`config/nifty500.csv` is NSE's official constituents file (`ind_nifty500list.csv` from
niftyindices.com, 501 rows when downloaded on 15 Sep 2026). Stocks are matched to Upstox by
ISIN, and the file's hash is recorded in the snapshot manifest. Index membership changes at
each rebalance, so download the file again to refresh it.

A Nifty 500 run makes about 2,550 requests. Upstox allows 2,000 per 30 minutes, and the egress
gate holds to 1,900, so a run takes roughly **35 minutes**. Progress prints every 25 stocks.
Completed reads are checkpointed under `data/acquire_checkpoints/`. If Upstox returns HTTP 429,
or the run is interrupted, the command exits (code 4 for a 429). Re-run the same command to
resume without re-fetching. Checkpoints are deleted once the snapshot is sealed.

Safeguards added after the first live run:

- **Dates:** `acquire --source upstox` refuses an `--as-of` date whose session isn't final
  (16:00 IST that day), and exits 2. Without `--as-of` it picks the latest final session.
  Checkpoints saved before that moment are discarded.
- **Rate limits:** requests sent by earlier runs in the last 30 minutes count against the
  rate limits, because Upstox limits the account, not the process.
- **One run at a time:** a second concurrent acquisition exits 5.
- **News:** every news page is recorded in the audit log (`news.page`), with its article count
  and Upstox's pagination metadata.
- **Today's prices:** Upstox's historical candle API leaves out the current day. When `--as-of`
  is today, the day's candle comes from the intraday endpoint, one extra request per stock.
  Every snapshot records `prices_as_of` (its newest candle). If that is earlier than `as_of`,
  the CLI warns, the audit log records `snapshot.price_lag`, and the dossier and UI show both
  dates. A gap is normal for weekends and holidays; otherwise a session is missing.

## Data coverage notes (Upstox, Nifty 500, 14 Sep 2026)

- **Banks** report NIM, Net NPA and CASA instead of ROCE, and are flagged `is_bank`. The ROCE
  and leverage sliders skip banks, and the Net NPA slider applies only to banks.
- **Quarterly year-on-year growth** was removed (`derived-v2`): Upstox returns only four quarters.
- **About 50 stocks** have empty or years-old statements (e.g. COLPAL at 2010). The 400-day
  freshness rule excludes them.
- **News** is limited by Upstox, not by pagination. The 15 Sep run returned one page per batch
  (4–43 articles per 30 stocks), covering only about a week (8–15 Sep), and 366 of 498 stocks had
  no headlines. The column is therefore named `news_count_window` (`derived-v3`), not "30d".
- **Sector benchmarks** for P/B, ROA and EV/EBITDA are columns as of `derived-v3`.

## Column versions

Snapshots record the derived-column version they were built with. `plan`, `evaluate` and the
dashboard refuse a snapshot from a different version (CLI exit 3, dashboard HTTP 409), because
prompts, falsifiers and evidence would disagree about column names. Re-run `acquire` after upgrading.

## Calibration from the first Claude run (15 Sep 2026, 2 candidates)

7 calls in 66 seconds, $0.14 actual against $0.41 committed. Claim calls used 740–1,311 output
tokens; the veto used 3,996, so its limit is now 16,000. Expect a 20-candidate run to cost
roughly $1.00–1.30 actual against about $3.69 committed.

## Layout

| Path | Phase | What it holds |
|---|---|---|
| `sealed_window/governance/` | all | policy, egress gate, credentials, seal, process roles, spend plan, LLM gateway, audit log |
| `sealed_window/snapshot/` | P0 | content hashing, column catalogue, sealed snapshot store |
| `sealed_window/acquire/` | 1 | Upstox adapter, synthetic fixture market, indicator engine, ETL |
| `sealed_window/screen/` | 2 | slider config and deterministic screen |
| `sealed_window/agents/` | 3, 5 | prompts, claim agents, veto auditor, offline stand-in model |
| `sealed_window/claims/` | 3-4 | claim schema, falsifier DSL, validator |
| `sealed_window/adjudicate/` | 4 | verdicts and ranking |
| `sealed_window/publish/` | 6 | dossier and Markdown rendering |
| `sealed_window/orchestrator.py` | 2-6 | deterministic control flow |
| `sealed_window/app/` | UI | FastAPI server and static front end |
| `tests/` | gates | denied-path suite, seal, roles, P0-P6 gates, pipeline, app |

Outputs go to `data/snapshots/<root_hash>/` (read-only) and to `data/runs/<run_id>/`. Each run
directory holds the audit log, plan, transcript, claims and dossier.
