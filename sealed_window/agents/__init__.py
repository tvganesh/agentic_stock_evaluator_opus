"""Phases 3 and 5 -- the only places a model runs. Model: running. Network: sealed.

Agents are deliberately *not* autonomous loops. Each agent is one governed model call over a
snapshot slice passed in memory, returning a typed object. Agents hold no tools, no
filesystem, no network and no shell; they cannot call other agents; control flow belongs to
the orchestrator. That is the "workflow, not free-roaming agent" pattern from Anthropic's
*Building effective agents*, chosen because every step here can be specified in advance.

Modules
-------
prompts        Frozen system prompts, the column catalogue and DSL guide, and the data
               envelope that marks snapshot content as data, never instruction.
claim_agents   Fundamental, technical and news claim agents (phase 3).
veto           The auditor, which can only refute (phase 5).
offline_model  A deterministic, rule-based stand-in for the model client, for dry runs and
               tests. It is not an LLM and its output is labelled as such.

This package is import-forbidden in the ACQUIRE process role.
"""
