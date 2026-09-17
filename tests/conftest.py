"""Shared pytest fixtures for the Sealed Window test suite.

* ``clean_seal`` (autouse) resets the process seal and removes the socket guard around every
  test, because the seal is a process singleton and tests exercise it in different modes.
* ``sealed_snapshot`` builds one synthetic snapshot per session (the fixture market) and
  returns ``(snapshot_root, root_hash)``; ``snapshot`` loads it with full verification.

Nothing here touches the network or needs an Upstox token or Anthropic API key.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from sealed_window.acquire.etl import build_snapshot_content
from sealed_window.acquire.fixture_source import SYNTHETIC_SYMBOLS, SyntheticMarket
from sealed_window.governance.seal import SEAL, uninstall_socket_guard
from sealed_window.snapshot.store import SealedSnapshot, write_snapshot

AS_OF = date(2026, 9, 11)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_seal():
    """Reset the process-wide seal and socket guard before and after each test."""
    uninstall_socket_guard()
    SEAL._reset_for_tests()
    yield
    uninstall_socket_guard()
    SEAL._reset_for_tests()


@pytest.fixture(scope="session")
def sealed_snapshot(tmp_path_factory) -> tuple[Path, str]:
    """Build and seal the synthetic fixture snapshot once per session."""
    root = tmp_path_factory.mktemp("snapshots")
    content = build_snapshot_content(SyntheticMarket(AS_OF), list(SYNTHETIC_SYMBOLS), AS_OF, "synthetic-fixture")
    return root, write_snapshot(content, root)


@pytest.fixture
def snapshot(sealed_snapshot) -> SealedSnapshot:
    """Load the session snapshot with integrity verification."""
    root, root_hash = sealed_snapshot
    return SealedSnapshot.load(root, root_hash)
