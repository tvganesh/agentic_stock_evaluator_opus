"""Sealed Window stock evaluator.

An NSE equity research system built around one invariant from ARCHITECTURE_OPUS.md:
``NETWORK_LIVE`` (market-data network) and ``MODEL_RUNNING`` are never true at the
same time. A run is split into six phases:

    1 ACQUIRE     sealed_window.acquire      ETL, Upstox network live, no model
    2 SCREEN      sealed_window.screen       deterministic filter + spend plan
    3 CLAIM       sealed_window.agents       models emit falsifiable claims
    4 ADJUDICATE  sealed_window.adjudicate   deterministic falsifier evaluation
    5 VETO        sealed_window.agents.veto  auditor that can only refute
    6 PUBLISH     sealed_window.publish      deterministic dossier

Governance (``sealed_window.governance``) is the spine that every phase goes through:
the access policy, the egress gate, the network seal, process-role import guards,
the spend plan, the LLM gateway and the hash-chained audit log.

Scope: research and recommendation only. There is no code path that places,
modifies or cancels an order, or reads a portfolio, holdings, positions or funds.
"""

__version__ = "0.1.0"
