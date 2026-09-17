"""Sealed snapshots: the frozen, content-addressed record of everything the system knows.

Modules
-------
hashing   Canonical JSON, BLAKE2b content hashes and evidence IDs (P0 provenance layer).
columns   The catalogue of derived columns: the vocabulary of screens and falsifiers.
indicators The engine computing those columns (RSI, MACD, ATR, MAs, returns, growth, leverage).
          Pure arithmetic with no network or credentials, so both the ETL (phase 1) and the sealed
          walk-forward backtest (P8) may use it; the ETL package itself stays forbidden in the
          sealed process role.
store     Writing a snapshot atomically as read-only files, and loading one with full
          integrity verification.

Phase 1 (ACQUIRE) writes snapshots; every later phase only reads them. A snapshot is the
sole input the sealed process has about the market.
"""
