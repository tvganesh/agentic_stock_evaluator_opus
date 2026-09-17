"""Governance: the backbone every phase of a Sealed Window run passes through.

Modules
-------
policy          Pure data: the least-privilege access policy (credentials, hosts,
                path templates, query parameters, import boundaries, limits).
errors          The fail-closed exception hierarchy raised by every control.
credentials     Loads the one permitted Upstox credential and refuses to start if
                any other Upstox credential is present in the environment.
egress          The only code allowed to issue an HTTP request to Upstox. It turns a
                capability name + typed parameters into a URL and authorises it.
seal            The invariant: socket-level network seal and the phase/model state.
process_roles   Import boundaries: the ETL process cannot load an LLM client and the
                analysis process cannot load the Upstox adapter.
audit           Append-only, hash-chained JSONL audit log.
spend           Spend-plan compiler and the slot ledger (a counter that only goes down).
llm_gateway     The only code allowed to call a model; every call must redeem a slot.

The ``acquire`` side uses policy/credentials/egress/seal/audit. The sealed side uses
policy/seal/process_roles/spend/llm_gateway/audit. ``egress`` and ``llm_gateway`` are
never loaded in the same process role.
"""
