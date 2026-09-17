"""Governance exception hierarchy.

Every control in the Sealed Window design fails *closed* by raising one of these
exceptions. They are deliberately distinct from ordinary runtime errors so that a
governance denial is visible by name in code review and in the audit log, and so a
broad ``except Exception`` that swallows one stands out as a defect.

Rule of use: a ``GovernanceViolation`` stops the operation that raised it. Nothing in
the codebase may catch one in order to retry around it, relax a limit, or fall back to
a less-governed path. The orchestrator catches them only to record them in the audit
log and abort (or skip) the affected unit of work.
"""


class GovernanceViolation(Exception):
    """Base class for every denial raised by a governance control.

    Catching this anywhere except an audit boundary is a review red flag.
    """


class EgressDenied(GovernanceViolation):
    """Raised by the egress gate when a request falls outside the Upstox allowlist."""


class CredentialViolation(GovernanceViolation):
    """Raised when a forbidden credential is present or the permitted one is missing/misused."""


class SealViolation(GovernanceViolation):
    """Raised when a network connection is attempted that the current seal state forbids."""


class ProcessRoleViolation(GovernanceViolation):
    """Raised when a process imports a module its role forbids (e.g. an LLM client in the ETL)."""


class PhaseViolation(GovernanceViolation):
    """Raised when the run state machine is asked to make an illegal phase transition."""


class SpendViolation(GovernanceViolation):
    """Base class for spend-plan denials: calls without slots, exhausted pools, oversize prompts."""


class NoSlotAvailable(SpendViolation):
    """Raised when a slot class in the compiled spend plan has no calls left to issue."""


class InvalidSlotTicket(SpendViolation):
    """Raised when a model call presents a ticket the ledger never issued or already redeemed."""


class PromptOverBudget(SpendViolation):
    """Raised when an assembled prompt exceeds its slot's ``max_in``; this is an assembler bug."""


class PlanNotApproved(SpendViolation):
    """Raised when a run starts without an approval hash equal to the freshly compiled plan's hash."""


class SnapshotIntegrityError(GovernanceViolation):
    """Raised when a sealed snapshot's bytes no longer match the hashes in its manifest."""


class SnapshotIncompatible(GovernanceViolation):
    """Raised when a snapshot was built with a different derived-column version than this code.

    Column names and meanings differ between versions, so analysing such a snapshot would let prompts,
    falsifiers and evidence disagree; the fix is a fresh acquisition, never a silent partial run.
    """


class LookaheadError(GovernanceViolation):
    """Raised when a backtest would use information that did not exist at the window's as-of date.

    Covers a screen touching undated fundamentals, a candle dated after the as-of, and a forward
    path that starts on or before it. A leaked backtest is worse than none: it manufactures confidence.
    """


class AuditChainBroken(GovernanceViolation):
    """Raised when an audit log's hash chain fails verification (edited, reordered or truncated)."""
