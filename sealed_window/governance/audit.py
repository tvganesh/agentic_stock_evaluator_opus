"""Append-only, hash-chained audit log.

Every governance-relevant event is written here: phase transitions, seal changes, every
egress allow/deny, credential scrubs, plan compilation and approval, every model call with
its slot and token usage, schema rejections, adjudication outcomes and publication.

Each JSONL line carries ``prev`` (the previous entry's hash) and ``hash`` (the BLAKE2b of
the entry without its own hash). Editing, deleting, inserting or reordering any line breaks
the chain, which :meth:`AuditLog.verify` detects. Given a dossier, the audit log plus the
reproducibility triple reconstructs exactly what produced it.

The log refuses detail keys that look like secrets, so a credential can never be recorded
by accident (the values themselves never reach this module by design).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..snapshot.hashing import canonical_json, content_hash
from .errors import AuditChainBroken

GENESIS_HASH = "0" * 64
_SECRET_KEY_MARKERS = ("authorization", "token", "api_key", "apikey", "secret", "password")


def _assert_no_secret_keys(detail: Any) -> None:
    """Recursively reject mappings whose keys look like credentials."""
    if isinstance(detail, Mapping):
        for key, value in detail.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in _SECRET_KEY_MARKERS) and not lowered.endswith("_tokens"):
                raise ValueError(f"Audit detail key {key!r} looks like a secret and is refused.")
            _assert_no_secret_keys(value)
    elif isinstance(detail, (list, tuple)):
        for item in detail:
            _assert_no_secret_keys(item)


def _entry_hash(entry: Mapping[str, Any]) -> str:
    """Hash an entry's canonical encoding excluding its own ``hash`` field."""
    body = {k: v for k, v in entry.items() if k != "hash"}
    return content_hash(canonical_json(body))


class AuditLog:
    """Thread-safe append-only audit log, backed by a JSONL file or held in memory."""

    def __init__(self, path: Path | None = None) -> None:
        """Open (and verify) an existing log at ``path`` to continue its chain, or start a new one."""
        self._lock = threading.Lock()
        self._path = path
        self._entries: list[dict[str, Any]] = []
        self._prev = GENESIS_HASH
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                AuditLog.verify(path)
                for line in path.read_text(encoding="utf-8").splitlines():
                    self._entries.append(json.loads(line))
                if self._entries:
                    self._prev = self._entries[-1]["hash"]

    @property
    def path(self) -> Path | None:
        """Location of the backing file, if any."""
        return self._path

    def record(self, event: str, detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Append one event and return the stored entry (including its chained hash)."""
        detail = dict(detail or {})
        _assert_no_secret_keys(detail)
        with self._lock:
            entry: dict[str, Any] = {
                "seq": len(self._entries),
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "event": event,
                "detail": detail,
                "prev": self._prev,
            }
            entry["hash"] = _entry_hash(entry)
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(canonical_json(entry).decode("utf-8") + "\n")
            self._entries.append(entry)
            self._prev = entry["hash"]
            return entry

    def entries(self, event_prefix: str | None = None) -> list[dict[str, Any]]:
        """Return a copy of recorded entries, optionally filtered by event-name prefix (for UI and tests)."""
        with self._lock:
            items = list(self._entries)
        if event_prefix is None:
            return items
        return [e for e in items if e["event"].startswith(event_prefix)]

    @staticmethod
    def verify(path: Path) -> int:
        """Verify the hash chain of a log file and return the number of entries.

        Raises :class:`AuditChainBroken` on the first inconsistency.
        """
        prev = GENESIS_HASH
        count = 0
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditChainBroken(f"line {lineno}: not valid JSON") from exc
            if entry.get("seq") != lineno or entry.get("prev") != prev:
                raise AuditChainBroken(f"line {lineno}: sequence or chain link mismatch")
            if entry.get("hash") != _entry_hash(entry):
                raise AuditChainBroken(f"line {lineno}: entry hash mismatch (edited)")
            prev = entry["hash"]
            count += 1
        return count
