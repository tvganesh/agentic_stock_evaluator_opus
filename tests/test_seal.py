"""Tests for the invariant: NETWORK_LIVE and MODEL_RUNNING are never simultaneously true.

Covers the seal state transitions, the socket guard (a real socket attempt raises), and the
phase machine that restricts model windows to CLAIM and VETO.
"""

from __future__ import annotations

import socket

import pytest

from sealed_window.governance.errors import PhaseViolation, SealViolation
from sealed_window.governance.seal import SEAL, PhaseMachine, RunPhase, install_socket_guard


def test_model_cannot_run_while_data_network_is_live():
    """Opening a model window during acquisition is refused."""
    SEAL.open_data_network()
    with pytest.raises(SealViolation):
        with SEAL.model_window():
            pass


def test_data_network_cannot_open_while_model_runs():
    """Opening the data network inside a model window is refused."""
    SEAL.seal()
    with SEAL.model_window():
        assert SEAL.model_running and not SEAL.network_live
        with pytest.raises(SealViolation):
            SEAL.open_data_network()


def test_seal_never_lifts():
    """Once sealed, the data network cannot be reopened."""
    SEAL.open_data_network()
    SEAL.seal()
    with pytest.raises(SealViolation):
        SEAL.open_data_network()


def test_model_window_requires_seal():
    """A model window cannot open in a process that was never sealed."""
    with pytest.raises(SealViolation):
        with SEAL.model_window():
            pass


def test_socket_attempt_raises_when_sealed():
    """Build-order gate P1: after the seal, a real outbound socket attempt raises before connecting."""
    install_socket_guard()
    SEAL.seal()
    with pytest.raises(SealViolation):
        socket.create_connection(("93.184.216.34", 443), timeout=1)
    with pytest.raises(SealViolation):
        socket.getaddrinfo("api.upstox.com", 443)


def test_model_window_reaches_only_the_model_provider(monkeypatch):
    """Inside a model window only api.anthropic.com resolves and connects; Upstox stays sealed."""
    fake_ip = "203.0.113.10"
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (fake_ip, port))])
    install_socket_guard()
    SEAL.seal()
    with SEAL.model_window():
        with pytest.raises(SealViolation):
            socket.getaddrinfo("api.upstox.com", 443)
        socket.getaddrinfo("api.anthropic.com", 443)
        SEAL.check_connect(socket.AF_INET, (fake_ip, 443))
        with pytest.raises(SealViolation):
            SEAL.check_connect(socket.AF_INET, ("198.51.100.7", 443))
    with pytest.raises(SealViolation):
        SEAL.check_connect(socket.AF_INET, (fake_ip, 443))


def test_loopback_resolution_is_allowed_for_binding():
    """Resolving localhost (needed to bind the UI server) is not blocked; connecting out still is."""
    install_socket_guard()
    SEAL.seal()
    socket.getaddrinfo("127.0.0.1", 8000)
    with pytest.raises(SealViolation):
        SEAL.check_connect(socket.AF_INET, ("127.0.0.1", 8000))


def test_phase_machine_gates_models_and_order():
    """Models only run in CLAIM/VETO, and phases only advance in the documented order."""
    SEAL.seal()
    phases = PhaseMachine()
    phases.advance(RunPhase.SCREEN)
    with pytest.raises(PhaseViolation):
        with phases.model_window():
            pass
    with pytest.raises(PhaseViolation):
        phases.advance(RunPhase.PUBLISH)
    phases.advance(RunPhase.CLAIM)
    with phases.model_window():
        assert SEAL.model_running
    phases.advance(RunPhase.ADJUDICATE)
    with pytest.raises(PhaseViolation):
        with phases.model_window():
            pass
