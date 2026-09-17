"""Credential handling for the ACQUIRE process: one permitted token, nothing else.

The Upstox Analytics token is the only Upstox credential this system will touch. This
module enforces three things before the ETL may open a connection:

1. No other Upstox credential exists in the environment. A trading/OAuth access token or
   API secret sitting next to the analytics token is a latent privilege escalation, so
   its mere presence aborts the run (fail closed) -- we never "just ignore" it.
2. The token is removed from ``os.environ`` as soon as it is read, so it cannot leak into
   child processes, crash dumps of the environment, or a later ``os.environ`` log line.
3. The token is held in an opaque object whose ``repr``/``str`` are redacted, which cannot
   be pickled, and which the egress gate drops when the acquisition window closes.

Note on token identity: an access token and an analytics token look alike on the wire, so
this code cannot prove *which* kind the operator supplied. That is why the egress gate
(GET-only, exact path allowlist, no account paths) is the real control, and the token
being read-only by issue is defence in depth rather than the only barrier.

This module is forbidden in the SEALED process role (see ``policy``).
"""

from __future__ import annotations

import os
from typing import MutableMapping

from . import policy
from .errors import CredentialViolation


def forbidden_credential_names(environ: MutableMapping[str, str]) -> list[str]:
    """Return the *names* (never values) of environment variables the policy forbids.

    Used by the ETL at startup and by tests that prove a trading token blocks the run.
    """
    found = []
    for name in environ:
        upper = name.upper()
        if upper == policy.UPSTOX_TOKEN_ENV:
            continue
        if upper.startswith(policy.FORBIDDEN_ENV_PREFIXES):
            found.append(name)
    return sorted(found)


def assert_no_forbidden_credentials(environ: MutableMapping[str, str] | None = None) -> None:
    """Raise :class:`CredentialViolation` if any forbidden Upstox credential is present.

    This is the least-privilege gate: the ETL refuses to run in an environment that could
    hand it more authority than the analytics token.
    """
    env = os.environ if environ is None else environ
    names = forbidden_credential_names(env)
    if names:
        raise CredentialViolation(
            "Forbidden Upstox credentials present in environment: "
            + ", ".join(names)
            + f". Only {policy.UPSTOX_TOKEN_ENV} may be set; unset the others and retry."
        )


class AnalyticsToken:
    """Opaque, redacting holder for the Upstox Analytics token.

    Only the egress gate reads the value (via :meth:`bearer_header`); everything else sees
    ``AnalyticsToken(<redacted>)``. Dropping it makes every later use fail closed.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        """Wrap a raw token string; callers should obtain instances via :func:`load_analytics_token`."""
        self._value: str | None = value

    def bearer_header(self) -> dict[str, str]:
        """Return the ``Authorization`` header for an allowlisted Upstox API request.

        Raises :class:`CredentialViolation` once the token has been dropped (after the seal).
        """
        if self._value is None:
            raise CredentialViolation("Analytics token has been dropped; the acquisition window is closed.")
        return {"Authorization": f"Bearer {self._value}"}

    @property
    def dropped(self) -> bool:
        """True once :meth:`drop` has been called; the seal indicator reports this."""
        return self._value is None

    def drop(self) -> None:
        """Forget the token value. Called by the egress gate when the ETL seals its snapshot."""
        self._value = None

    def __repr__(self) -> str:
        """Redacted representation so the token never appears in logs or tracebacks."""
        return "AnalyticsToken(<dropped>)" if self._value is None else "AnalyticsToken(<redacted>)"

    __str__ = __repr__

    def __reduce__(self):  # noqa: D401 - documented below
        """Refuse pickling so the token cannot be serialised into a file or another process."""
        raise TypeError("AnalyticsToken cannot be pickled")


def load_analytics_token(environ: MutableMapping[str, str] | None = None) -> AnalyticsToken:
    """Read ``UPSTOX_ANALYTICS_TOKEN`` after the forbidden-credential check, then scrub it.

    This is the ETL's single entry point for credentials; the returned object is handed
    straight to the egress gate and to nothing else.
    """
    env = os.environ if environ is None else environ
    assert_no_forbidden_credentials(env)
    raw = env.get(policy.UPSTOX_TOKEN_ENV, "")
    value = raw.strip()
    if not value:
        raise CredentialViolation(f"{policy.UPSTOX_TOKEN_ENV} is not set.")
    if len(value) < 20 or any(ch.isspace() for ch in value):
        raise CredentialViolation(f"{policy.UPSTOX_TOKEN_ENV} is malformed.")
    env.pop(policy.UPSTOX_TOKEN_ENV, None)
    return AnalyticsToken(value)
