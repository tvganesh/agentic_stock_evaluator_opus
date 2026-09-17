"""Phase 4 -- ADJUDICATE. Model: none. Network: sealed.

Deterministic evaluation of every claim's falsifier against the snapshot, and of every veto
refutation against the claims it attacks, followed by ranking as fixed arithmetic over the
survivors. The model's job is to generate hypotheses; this package's job is to kill the bad
ones and count what is left.

Modules
-------
adjudicator  Verdicts (survived / refuted / unevaluable / rejected / vacuous / vetoed) and the
             versioned scoring formula.
"""
