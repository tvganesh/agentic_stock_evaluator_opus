"""Phase 1 -- ACQUIRE. Model: none. Network: live (Upstox only).

A scheduled, deterministic ETL that reads the Upstox Analytics API through the egress gate
and materialises a sealed snapshot. No LLM is importable in this process role.

Modules
-------
upstox_adapter  Typed wrappers over the six allowlisted capabilities (via the egress gate).
fixture_source  A deterministic synthetic market with the same interface, for tests and
                offline demos; its snapshots are labelled ``synthetic-fixture``.
etl             Orchestrates acquisition, normalisation, evidence IDs and sealing. Derived numbers
                come from ``snapshot.indicators``, which lives outside this package because it is
                pure arithmetic that the sealed process (and its backtest) must also be able to use.

This package is import-forbidden in the SEALED process role.
"""
