"""Process roles: the ACQUIRE process and the SEALED analysis process are different programs.

A process takes exactly one role, once, before it does any work:

* ``ACQUIRE`` (``python -m sealed_window acquire``): may hold the Upstox analytics token and
  reach Upstox. It may never import an LLM client, an agent or the orchestrator -- so no
  model can exist while the network is live.
* ``SEALED`` (``plan``, ``evaluate``, ``serve``): may call the model provider inside model
  windows. It may never import the Upstox adapter, the egress gate, the credential loader
  or a generic HTTP client, and every ``UPSTOX_*`` variable is scrubbed from its environment.

Enforcement is structural rather than advisory: a ``sys.meta_path`` finder raises
:class:`ProcessRoleViolation` on any forbidden import, the role refuses to start if a
forbidden module is already loaded, and the socket guard from ``seal`` is installed with the
role's network mode. The rules themselves live in ``policy``.
"""

from __future__ import annotations

import importlib.abc
import os
import sys
from enum import Enum
from typing import MutableMapping, Sequence

from . import policy
from .errors import CredentialViolation, ProcessRoleViolation
from .seal import SEAL, install_socket_guard


class ProcessRole(str, Enum):
    """The two process roles of the Sealed Window design."""

    ACQUIRE = "acquire"
    SEALED = "sealed"


def _module_matches(name: str, prefixes: Sequence[str]) -> bool:
    """True if ``name`` is one of ``prefixes`` or a submodule of one."""
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


class _DenyImportFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that refuses to locate modules forbidden for the process role."""

    def __init__(self, role: ProcessRole, prefixes: Sequence[str]) -> None:
        """Remember the role (for error messages) and the forbidden module prefixes."""
        self.role = role
        self.prefixes = tuple(prefixes)

    def find_spec(self, fullname, path, target=None):
        """Raise for forbidden modules; return ``None`` to let normal finders handle the rest."""
        if _module_matches(fullname, self.prefixes):
            raise ProcessRoleViolation(
                f"Module {fullname!r} may not be imported in the {self.role.value} process role."
            )
        return None


_ACTIVE_ROLE: ProcessRole | None = None


def current_role() -> ProcessRole | None:
    """Return the role this process has taken, or ``None`` (library use / tests)."""
    return _ACTIVE_ROLE


def _claim_role(role: ProcessRole, forbidden: Sequence[str]) -> None:
    """Shared role setup: single assignment, no forbidden module loaded, install the import finder."""
    global _ACTIVE_ROLE
    if _ACTIVE_ROLE is not None:
        raise ProcessRoleViolation(f"Process already has role {_ACTIVE_ROLE.value}; roles are taken once.")
    loaded = sorted(name for name in sys.modules if _module_matches(name, forbidden))
    if loaded:
        raise ProcessRoleViolation(
            f"Cannot enter {role.value} role: forbidden modules already loaded: {', '.join(loaded)}"
        )
    sys.meta_path.insert(0, _DenyImportFinder(role, forbidden))
    _ACTIVE_ROLE = role


def enter_acquire_role() -> None:
    """Configure this process as the ETL: no model imports, socket guard on, data network live.

    Called by the ``acquire`` CLI command before it imports the Upstox adapter.
    """
    _claim_role(ProcessRole.ACQUIRE, policy.ACQUIRE_ROLE_FORBIDDEN_MODULES)
    install_socket_guard(SEAL)
    SEAL.open_data_network()


def enter_sealed_role(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Configure this process for analysis: scrub Upstox credentials, forbid ETL imports, seal.

    Returns the names of scrubbed variables so the caller can audit them. Raises
    :class:`CredentialViolation` if the model endpoint has been redirected.
    """
    env = os.environ if environ is None else environ
    for name in policy.FORBIDDEN_MODEL_ENV_NAMES:
        if env.get(name) and env[name].rstrip("/") != policy.MODEL_PROVIDER_BASE_URL:
            raise CredentialViolation(f"{name} redirects model traffic away from {policy.MODEL_PROVIDER_BASE_URL}.")
    scrubbed = sorted(name for name in env if name.upper().startswith(policy.FORBIDDEN_ENV_PREFIXES))
    for name in scrubbed:
        env.pop(name, None)
    _claim_role(ProcessRole.SEALED, policy.SEALED_ROLE_FORBIDDEN_MODULES)
    install_socket_guard(SEAL)
    SEAL.seal()
    return scrubbed


def _reset_for_tests() -> None:
    """Remove role finders and clear the role. Test-only; production processes never leave a role."""
    global _ACTIVE_ROLE
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _DenyImportFinder)]
    _ACTIVE_ROLE = None
