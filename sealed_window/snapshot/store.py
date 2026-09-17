"""Writing and loading sealed snapshots (build-order phase P0).

A snapshot is a directory named by its root hash under the snapshot root::

    data/snapshots/<root_hash>/
      manifest.json      format version, as_of, source, query-set hash, universe,
                         acquisition notes, and the hash + size of every file below
      instruments.json   the resolved NSE equity instruments in the universe
      candles.json       daily OHLCV per instrument (ascending by date)
      fundamentals.json  key ratios, yearly/quarterly income statement, balance sheet
      news.json          30-day headline window per instrument
      derived.json       every derived column per instrument (see ``columns``)
      evidence.json      the evidence index: evidence_id -> record with lineage

``root_hash = hash(canonical manifest)``, and the manifest contains every file's hash, so
the directory name commits to every byte. Files are written as canonical JSON (not Parquet)
so that identical inputs yield byte-identical files regardless of library versions.

Writes are atomic (temporary directory then rename) and the result is made read-only.
Loading re-hashes every file and raises :class:`SnapshotIntegrityError` on any mismatch, so
the sealed process can never reason over a tampered snapshot.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..governance.errors import SnapshotIntegrityError
from .columns import DERIVED_COLUMNS_VERSION, Dimension
from .hashing import canonical_json, content_hash, hash_object

SNAPSHOT_FORMAT_VERSION = "sealed-window-snapshot/1"
_ROOT_HASH_RE = re.compile(r"[0-9a-f]{64}")


@dataclass
class SnapshotContent:
    """In-memory contents of a snapshot as assembled by the ETL, before it is sealed to disk."""

    as_of: str
    source: str
    query_set_hash: str
    universe: list[str]
    instruments: list[dict[str, Any]]
    candles: dict[str, list[list[Any]]]
    fundamentals: dict[str, dict[str, Any]]
    news: dict[str, list[dict[str, Any]]]
    derived: dict[str, dict[str, float | None]]
    evidence: dict[str, dict[str, Any]]
    acquisition_notes: list[dict[str, Any]] = field(default_factory=list)
    request_count: int = 0
    universe_source: dict[str, Any] | None = None
    prices_as_of: str | None = None
    instruments_behind_as_of: int | None = None


def _write_readonly(path: Path, data: bytes) -> None:
    """Write bytes to ``path`` and mark the file read-only."""
    path.write_bytes(data)
    os.chmod(path, 0o444)


def write_snapshot(content: SnapshotContent, root_dir: Path) -> str:
    """Seal ``content`` to ``root_dir/<root_hash>`` atomically and return the root hash.

    Idempotent: writing identical content again verifies the existing directory and
    returns the same hash (the P0 gate: identical inputs produce identical hashes).
    """
    payloads = {
        "instruments.json": content.instruments,
        "candles.json": content.candles,
        "fundamentals.json": content.fundamentals,
        "news.json": content.news,
        "derived.json": content.derived,
        "evidence.json": content.evidence,
    }
    encoded = {name: canonical_json(obj) for name, obj in payloads.items()}
    manifest = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "as_of": content.as_of,
        "prices_as_of": content.prices_as_of,
        "instruments_behind_as_of": content.instruments_behind_as_of,
        "source": content.source,
        "query_set_hash": content.query_set_hash,
        "derived_columns_version": DERIVED_COLUMNS_VERSION,
        "universe": sorted(content.universe),
        "universe_source": content.universe_source,
        "request_count": content.request_count,
        "acquisition_notes": content.acquisition_notes,
        "files": {name: {"hash": content_hash(data), "bytes": len(data)} for name, data in encoded.items()},
    }
    root_hash = hash_object(manifest)
    root_dir.mkdir(parents=True, exist_ok=True)
    target = root_dir / root_hash
    if target.exists():
        SealedSnapshot.load(root_dir, root_hash)
        return root_hash

    tmp = root_dir / f".tmp-{root_hash}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    for name, data in encoded.items():
        _write_readonly(tmp / name, data)
    _write_readonly(tmp / "manifest.json", canonical_json(manifest))
    os.chmod(tmp, 0o555)
    os.replace(tmp, target)
    return root_hash


class SealedSnapshot:
    """A verified, read-only view of a sealed snapshot; the sealed process's only market input."""

    def __init__(
        self, root_hash: str, manifest: dict[str, Any], files: dict[str, Any], directory: Path
    ) -> None:
        """Hold verified contents; construct via :meth:`load` so verification cannot be skipped."""
        self._root_hash = root_hash
        self._manifest = manifest
        self._directory = directory
        self._instruments: list[dict[str, Any]] = files["instruments.json"]
        # Candles are hash-verified at load but parsed on first use: a decade of daily candles for
        # ~500 instruments is large, and only the walk-forward backtest reads them.
        self._candles: dict[str, list] | None = None
        self._fundamentals: dict[str, dict] = files["fundamentals.json"]
        self._news: dict[str, list] = files["news.json"]
        self._derived: dict[str, dict] = files["derived.json"]
        self._evidence: dict[str, dict] = files["evidence.json"]
        self._by_key = {inst["instrument_key"]: inst for inst in self._instruments}

    @classmethod
    def load(cls, root_dir: Path, root_hash: str) -> "SealedSnapshot":
        """Load ``root_dir/root_hash``, re-verifying the manifest hash and every file hash."""
        if not _ROOT_HASH_RE.fullmatch(root_hash):
            raise SnapshotIntegrityError(f"{root_hash!r} is not a snapshot root hash")
        directory = root_dir / root_hash
        try:
            manifest = json.loads((directory / "manifest.json").read_bytes())
        except FileNotFoundError as exc:
            raise SnapshotIntegrityError(f"snapshot {root_hash} not found under {root_dir}") from exc
        if hash_object(manifest) != root_hash:
            raise SnapshotIntegrityError(f"snapshot {root_hash}: manifest does not match its hash")
        files: dict[str, Any] = {}
        for name, meta in manifest["files"].items():
            raw = (directory / name).read_bytes()
            if content_hash(raw) != meta["hash"]:
                raise SnapshotIntegrityError(f"snapshot {root_hash}: {name} has been modified")
            if name != "candles.json":  # verified here, parsed lazily by :meth:`candles_for`
                files[name] = json.loads(raw)
        return cls(root_hash, manifest, files, directory)

    # ---- identity ---------------------------------------------------------------------

    @property
    def root_hash(self) -> str:
        """The snapshot's content address; first element of the reproducibility triple."""
        return self._root_hash

    @property
    def as_of(self) -> str:
        """The moment the snapshot describes (ISO date); shown prominently on every recommendation."""
        return self._manifest["as_of"]

    @property
    def prices_as_of(self) -> str | None:
        """Newest price candle in the snapshot (``None`` for snapshots sealed before this was recorded)."""
        return self._manifest.get("prices_as_of")

    @property
    def source(self) -> str:
        """Where the data came from: ``upstox-analytics`` or ``synthetic-fixture``."""
        return self._manifest["source"]

    @property
    def is_synthetic(self) -> bool:
        """True for fixture snapshots; the dossier labels such results as not real market data."""
        return self.source == "synthetic-fixture"

    @property
    def manifest(self) -> dict[str, Any]:
        """A copy of the manifest (for the UI and the dossier)."""
        return json.loads(canonical_json(self._manifest))

    # ---- contents ---------------------------------------------------------------------

    @property
    def instruments(self) -> list[dict[str, Any]]:
        """Instrument records for the universe, in snapshot order."""
        return list(self._instruments)

    def instrument(self, instrument_key: str) -> dict[str, Any]:
        """Return one instrument record; raises ``KeyError`` if not in the snapshot."""
        return self._by_key[instrument_key]

    def derived_row(self, instrument_key: str) -> dict[str, float | None]:
        """Return the derived columns for one instrument (empty dict if absent)."""
        return dict(self._derived.get(instrument_key, {}))

    def derived_table(self) -> dict[str, dict[str, float | None]]:
        """Return all derived rows keyed by instrument (used by the screen and adjudicator stats)."""
        return {key: dict(row) for key, row in self._derived.items()}

    def candles_for(self, instrument_key: str) -> list[list[Any]]:
        """Return the ascending daily candles for one instrument (used by the walk-forward backtest).

        Parses ``candles.json`` on first call; its bytes were already hash-verified by :meth:`load`.
        """
        if self._candles is None:
            self._candles = json.loads((self._directory / "candles.json").read_bytes())
        return [list(row) for row in self._candles.get(instrument_key, [])]

    def news_for(self, instrument_key: str) -> list[dict[str, Any]]:
        """Return the normalised headlines for one instrument."""
        return list(self._news.get(instrument_key, []))

    def has_evidence(self, evidence_id: str) -> bool:
        """True if ``evidence_id`` exists in this snapshot's evidence index."""
        return evidence_id in self._evidence

    def evidence_record(self, evidence_id: str) -> dict[str, Any]:
        """Return one evidence record; raises ``KeyError`` for unknown IDs."""
        return dict(self._evidence[evidence_id])

    def evidence_for(self, instrument_key: str, dimension: Dimension | None = None) -> list[dict[str, Any]]:
        """Return evidence records for an instrument, optionally one dimension, in stable order.

        This is the "snapshot slice" handed to agents and to the veto auditor.
        """
        records = [
            dict(rec)
            for rec in self._evidence.values()
            if rec["instrument_key"] == instrument_key
            and (dimension is None or rec["dimension"] == dimension.value)
        ]
        return sorted(records, key=lambda rec: (rec["kind"], rec["row_key"], rec["evidence_id"]))


def list_snapshots(root_dir: Path) -> list[dict[str, Any]]:
    """List snapshots under ``root_dir`` (newest ``as_of`` first) without full verification.

    For UI pickers only; anything that reasons over a snapshot must use :meth:`SealedSnapshot.load`.
    """
    if not root_dir.exists():
        return []
    found = []
    for child in root_dir.iterdir():
        if not (child.is_dir() and _ROOT_HASH_RE.fullmatch(child.name)):
            continue
        try:
            manifest = json.loads((child / "manifest.json").read_bytes())
        except (OSError, json.JSONDecodeError):
            continue
        found.append(
            {
                "root_hash": child.name,
                "as_of": manifest.get("as_of"),
                "source": manifest.get("source"),
                "universe_size": len(manifest.get("universe", [])),
            }
        )
    return sorted(found, key=lambda item: (item["as_of"] or "", item["root_hash"]), reverse=True)
