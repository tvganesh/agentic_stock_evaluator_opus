"""The invariant: ``NETWORK_LIVE`` and ``MODEL_RUNNING`` are never simultaneously true.

This module turns clause 01 of ARCHITECTURE_OPUS.md into running code, at two levels.

1. **Seal state** (:class:`SealState`, process singleton :data:`SEAL`). Tracks the network
   mode, how many model calls are in flight, and whether the process has been sealed. Its
   transitions refuse any combination that would break the invariant: the data network
   cannot open while a model runs, a model window cannot open while the data network is
   live, and once sealed the data network never reopens.

2. **Socket guard** (:func:`install_socket_guard`). Patches ``socket.getaddrinfo`` and
   ``socket.socket.connect``/``connect_ex`` so the invariant is enforced where bytes
   actually leave the process, regardless of which HTTP library tries. Hostnames are
   resolved only if permitted in the current mode; connections are allowed only to IPs
   that a permitted hostname resolved to during the current window.

What "no network while a model runs" means precisely
----------------------------------------------------
Calling Claude is itself a network request. The seal therefore distinguishes the *data*
network (Upstox -- the only source of facts and the only place a credential could act)
from the *model provider* endpoint. During a model window exactly one host is reachable,
``api.anthropic.com``, and only the LLM gateway talks to it. The model holds no tools, so
it cannot direct that channel anywhere. Outside model windows the sealed process can
reach nothing at all.

Limits of an in-process guard
-----------------------------
Monkeypatching is a strong tripwire, not a kernel boundary: native extensions that open
sockets without Python's ``socket`` module would bypass it. Production deployments should
additionally run the sealed process in a network namespace (or container) whose only
route is the model provider. The guard makes the rule testable today (build-order gate P1:
"a test proves a socket attempt raises").
"""

from __future__ import annotations

import contextlib
import ipaddress
import socket
import threading
from enum import Enum
from typing import Iterator

from . import policy
from .errors import PhaseViolation, SealViolation


class NetworkMode(str, Enum):
    """Which network, if any, the process may currently reach."""

    UNGUARDED = "unguarded"  # guard not installed yet (tests, library import)
    DATA_LIVE = "data_live"  # ACQUIRE: Upstox hosts reachable, no model may exist
    SEALED = "sealed"  # nothing reachable
    MODEL_WINDOW = "model_window"  # only the model provider reachable, during a model call


class RunPhase(str, Enum):
    """The six phases of a run (plus bookkeeping states), each with its network/model permissions."""

    ACQUIRE = "1_acquire"
    SCREEN = "2_screen"
    CLAIM = "3_claim"
    ADJUDICATE = "4_adjudicate"
    VETO = "5_veto"
    ADJUDICATE_VETO = "4b_adjudicate_veto"
    PUBLISH = "6_publish"
    DONE = "done"
    ABORTED = "aborted"


MODEL_PHASES: frozenset[RunPhase] = frozenset({RunPhase.CLAIM, RunPhase.VETO})
"""The only phases in which a model window may open."""

_LEGAL_TRANSITIONS: dict[RunPhase | None, frozenset[RunPhase]] = {
    None: frozenset({RunPhase.SCREEN}),
    RunPhase.SCREEN: frozenset({RunPhase.CLAIM}),
    RunPhase.CLAIM: frozenset({RunPhase.ADJUDICATE}),
    RunPhase.ADJUDICATE: frozenset({RunPhase.VETO, RunPhase.PUBLISH}),
    RunPhase.VETO: frozenset({RunPhase.ADJUDICATE_VETO}),
    RunPhase.ADJUDICATE_VETO: frozenset({RunPhase.PUBLISH}),
    RunPhase.PUBLISH: frozenset({RunPhase.DONE}),
}
"""Phase order for the sealed analysis process. ACQUIRE lives in a different process."""


def _is_loopback_or_unspecified(host: str) -> bool:
    """True for literal loopback/unspecified addresses, which servers resolve in order to bind."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost"
    return ip.is_loopback or ip.is_unspecified


class SealState:
    """Process-wide seal state; the single source of truth rendered by the UI seal indicator."""

    def __init__(self) -> None:
        """Start unguarded with no model in flight; a process role sets the real mode."""
        self._lock = threading.RLock()
        self._mode = NetworkMode.UNGUARDED
        self._sealed_permanently = False
        self._model_calls_in_flight = 0
        self._permitted_hosts: frozenset[str] = frozenset()
        self._permitted_ips: set[str] = set()

    # ---- observable state -------------------------------------------------------------

    @property
    def mode(self) -> NetworkMode:
        """Current network mode."""
        return self._mode

    @property
    def network_live(self) -> bool:
        """``NETWORK_LIVE`` from the invariant: the data (Upstox) network is reachable."""
        return self._mode is NetworkMode.DATA_LIVE

    @property
    def model_running(self) -> bool:
        """``MODEL_RUNNING`` from the invariant: at least one model call is in flight."""
        return self._model_calls_in_flight > 0

    def status(self) -> dict[str, object]:
        """Return a JSON-safe view of the seal for the UI indicator and the audit log."""
        with self._lock:
            return {
                "mode": self._mode.value,
                "network_live": self.network_live,
                "model_running": self.model_running,
                "sealed_permanently": self._sealed_permanently,
                "permitted_hosts": sorted(self._permitted_hosts),
            }

    # ---- transitions ------------------------------------------------------------------

    def open_data_network(self) -> None:
        """Enter ``DATA_LIVE`` (ACQUIRE only). Refused if sealed or if a model is running."""
        with self._lock:
            if self._sealed_permanently:
                raise SealViolation("The seal never lifts: this process is already sealed.")
            if self.model_running:
                raise SealViolation("Invariant: cannot open the data network while a model is running.")
            self._mode = NetworkMode.DATA_LIVE
            self._permitted_hosts = policy.UPSTOX_HOSTS
            self._permitted_ips = set()

    def seal(self) -> None:
        """Drop the seal: nothing reachable, permanently. Called at the end of ACQUIRE and at start of analysis."""
        with self._lock:
            if self.model_running:
                raise SealViolation("Cannot seal while a model call is in flight.")
            self._sealed_permanently = True
            self._mode = NetworkMode.SEALED
            self._permitted_hosts = frozenset()
            self._permitted_ips = set()

    def assert_data_network_allowed(self) -> None:
        """Raise unless the process is in ``DATA_LIVE``; the egress gate calls this before every request."""
        if self._mode is not NetworkMode.DATA_LIVE or self.model_running:
            raise SealViolation(f"Data network is not live (mode={self._mode.value}).")

    @contextlib.contextmanager
    def model_window(self) -> Iterator[None]:
        """Context in which exactly the model provider host is reachable and ``MODEL_RUNNING`` is true.

        Only the LLM gateway opens this. Refused while the data network is live or before sealing.
        """
        with self._lock:
            if self.network_live:
                raise SealViolation("Invariant: cannot run a model while the data network is live.")
            if not self._sealed_permanently:
                raise SealViolation("A model window requires a sealed process.")
            self._model_calls_in_flight += 1
            self._mode = NetworkMode.MODEL_WINDOW
            self._permitted_hosts = policy.MODEL_PROVIDER_HOSTS
        try:
            yield
        finally:
            with self._lock:
                self._model_calls_in_flight -= 1
                if self._model_calls_in_flight == 0:
                    self._mode = NetworkMode.SEALED
                    self._permitted_hosts = frozenset()
                    self._permitted_ips = set()

    # ---- socket guard hooks -----------------------------------------------------------

    def check_resolve(self, host: str) -> bool:
        """Decide whether ``host`` may be resolved now. Returns True if resolved IPs should be remembered."""
        if self._mode is NetworkMode.UNGUARDED:
            return False
        if _is_loopback_or_unspecified(host):
            return False  # resolving for bind() is harmless; connect() is checked separately
        if host.lower() in self._permitted_hosts:
            return True
        raise SealViolation(f"Seal: resolving {host!r} is not permitted in mode {self._mode.value}.")

    def remember_ips(self, ips: list[str]) -> None:
        """Record IPs returned for a permitted hostname so :meth:`check_connect` can allow them."""
        with self._lock:
            self._permitted_ips.update(ips)

    def check_connect(self, family: int, address: object) -> None:
        """Raise unless an outbound connection to ``address`` is permitted in the current window."""
        if self._mode is NetworkMode.UNGUARDED:
            return
        if family in (socket.AF_INET, socket.AF_INET6) and isinstance(address, tuple):
            ip = str(address[0])
            if ip in self._permitted_ips and self._permitted_hosts:
                return
            raise SealViolation(f"Seal: connection to {ip} is not permitted in mode {self._mode.value}.")
        raise SealViolation(f"Seal: socket family {family} connections are not permitted.")

    def _reset_for_tests(self) -> None:
        """Return to the unguarded initial state. Test-only; never called by application code."""
        with self._lock:
            self.__init__()


SEAL = SealState()
"""The process singleton. Process roles configure it; the gateway and egress gate consult it."""


# --------------------------------------------------------------------------------------
# Socket guard
# --------------------------------------------------------------------------------------

_ORIGINALS: dict[str, object] = {}


def install_socket_guard(state: SealState = SEAL) -> None:
    """Patch ``socket`` so every resolve/connect is checked against ``state``. Idempotent."""
    if _ORIGINALS:
        return
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    _ORIGINALS.update(
        getaddrinfo=original_getaddrinfo, connect=original_connect, connect_ex=original_connect_ex
    )

    def guarded_getaddrinfo(host, port, *args, **kwargs):
        """Resolve only hosts permitted by the seal; remember their IPs for connect checks."""
        host_str = host.decode() if isinstance(host, bytes) else host
        remember = False if host_str is None else state.check_resolve(host_str)
        results = original_getaddrinfo(host, port, *args, **kwargs)
        if remember:
            state.remember_ips([str(res[4][0]) for res in results])
        return results

    def guarded_connect(sock, address):
        """Connect only to addresses the seal currently permits."""
        state.check_connect(sock.family, address)
        return original_connect(sock, address)

    def guarded_connect_ex(sock, address):
        """``connect_ex`` variant of :func:`guarded_connect`."""
        state.check_connect(sock.family, address)
        return original_connect_ex(sock, address)

    socket.getaddrinfo = guarded_getaddrinfo
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex


def uninstall_socket_guard() -> None:
    """Restore the original socket functions. Test-only; production processes never uninstall."""
    if not _ORIGINALS:
        return
    socket.getaddrinfo = _ORIGINALS["getaddrinfo"]
    socket.socket.connect = _ORIGINALS["connect"]
    socket.socket.connect_ex = _ORIGINALS["connect_ex"]
    _ORIGINALS.clear()


# --------------------------------------------------------------------------------------
# Per-run phase machine
# --------------------------------------------------------------------------------------


class PhaseMachine:
    """Enforces the phase order of one analysis run and gates model windows to CLAIM and VETO.

    The orchestrator advances it; the LLM gateway asks it for a model window, so a model
    call attempted during SCREEN, ADJUDICATE or PUBLISH fails closed.
    """

    def __init__(self, state: SealState = SEAL, on_transition=None) -> None:
        """Bind to the process seal; ``on_transition(old, new)`` lets the orchestrator audit changes."""
        self._state = state
        self._phase: RunPhase | None = None
        self._on_transition = on_transition

    @property
    def phase(self) -> RunPhase | None:
        """The current phase (``None`` before SCREEN starts)."""
        return self._phase

    def advance(self, new_phase: RunPhase) -> None:
        """Move to ``new_phase`` if the transition is legal, else raise :class:`PhaseViolation`."""
        if new_phase is RunPhase.ABORTED:
            old, self._phase = self._phase, RunPhase.ABORTED
        else:
            if new_phase not in _LEGAL_TRANSITIONS.get(self._phase, frozenset()):
                raise PhaseViolation(f"Illegal phase transition {self._phase} -> {new_phase}")
            if self._state.model_running:
                raise PhaseViolation("Cannot change phase while a model call is in flight.")
            old, self._phase = self._phase, new_phase
        if self._on_transition:
            self._on_transition(old, self._phase)

    @contextlib.contextmanager
    def model_window(self) -> Iterator[None]:
        """Open a seal model window, but only if the current phase permits models."""
        if self._phase not in MODEL_PHASES:
            raise PhaseViolation(f"Models may not run in phase {self._phase}.")
        with self._state.model_window():
            yield
