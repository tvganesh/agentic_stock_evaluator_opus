"""Content addressing: canonical serialisation, content hashes and evidence IDs.

This is the provenance layer of the whole system (build-order phase P0). A sealed
snapshot is identified by the hash of its manifest; every fact inside it is referred
to downstream only by an ``evidence_id``; the screen config and spend plan are
identified by their hashes. Together those form the reproducibility triple stamped on
every published recommendation.

Hash function: BLAKE2b from the Python standard library. The architecture document
names BLAKE3; BLAKE2b gives the same property we need (a collision-resistant content
address) without adding a native third-party dependency to the trusted base.

Determinism rules
-----------------
* Objects are serialised as canonical JSON: sorted keys, no whitespace, UTF-8, and
  NaN/Infinity rejected (callers convert missing numbers to ``None``).
* Floats that enter a snapshot are rounded by :func:`stable_float` so that two builds
  over identical inputs produce byte-identical files and therefore identical hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

FLOAT_DECIMALS = 6
"""Decimal places kept for every float written into a snapshot or derived table."""


def canonical_json(obj: Any) -> bytes:
    """Serialise ``obj`` to canonical JSON bytes (sorted keys, compact, UTF-8, no NaN).

    Every hash in the system is computed over this encoding, so it must never change
    without a snapshot format version bump.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def content_hash(data: bytes) -> str:
    """Return the 64-hex-character BLAKE2b-256 digest of ``data``.

    Used for snapshot files, the snapshot root, screen configs, spend plans and the
    audit-log hash chain.
    """
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def hash_object(obj: Any) -> str:
    """Hash a JSON-compatible object via its canonical encoding.

    This is how configs and plans get the stable identities shown in the UI and dossier.
    """
    return content_hash(canonical_json(obj))


def evidence_id(query: str, params: dict[str, Any], as_of: str, row_key: str) -> str:
    """Compute ``evidence_id = hash(query, params, as_of, row_key)`` as ``ev:<16 hex>``.

    Evidence IDs are the only way claims may refer to facts; the claim validator rejects
    any ID that is not present in the snapshot's evidence index.
    """
    digest = hashlib.blake2b(
        canonical_json([query, params, as_of, row_key]), digest_size=8
    ).hexdigest()
    return f"ev:{digest}"


def stable_float(value: float | int | None) -> float | None:
    """Round a number for storage, mapping NaN/inf/None to ``None``.

    Keeps derived indicators byte-stable across builds so snapshot hashes are reproducible.
    """
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    rounded = round(value, FLOAT_DECIMALS)
    return 0.0 if rounded == 0 else rounded
