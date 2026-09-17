"""Build-order phase P8: does the ranking beat the raw screen?

Everything here is deterministic arithmetic over a sealed snapshot's candles. No model runs, no
network is touched, and no claim is involved -- a backtest can only measure a *frozen rule*, and a
model's answers are not frozen. (The codex project reaches the same conclusion: its walk-forward
imports only its deterministic scorer.)

Modules
-------
walk_forward  Rebuilds the technical screen at past dates, measures what happened next, and reports
              precision, excess return, drawdown, volatility, turnover and coverage against
              acceptance gates.

Scope and honesty limits, all reported in every backtest:

* **Price-only.** Upstox returns undated current ratios, so using them in a past window would leak
  the future; the backtest refuses any screen config with a fundamental or news filter.
* **Survivorship bias.** The universe is today's index membership, so names that were delisted or
  demoted never appear. This inflates results and cannot be fixed without point-in-time membership.
* **The benchmark is the screen itself** -- an equal-weight basket of every stock that passed -- so
  the question answered is "does ranking add value over the screen?", not "does it beat the Nifty".
* **The claim layer is not tested.** That would need a model call per stock per window.
"""
