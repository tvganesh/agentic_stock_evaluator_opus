"""Phase 2 -- SCREEN. Model: none. Network: sealed.

The snapshot plus the user's slider settings produce a candidate set, purely by filtering.
Slider values are arguments to a deterministic filter and are hashed into the screen
config; they never reach a prompt, because no model exists in this phase to reach.

Modules
-------
config  :class:`ScreenConfig` (the sliders), its hash, and slider metadata for the UI.
screen  :func:`run_screen` -- deterministic, fail-closed filtering over derived columns.

The spend plan is compiled right after this phase (``governance.spend``), because the
candidate count now fixes the exact shape and cost of the run.
"""
