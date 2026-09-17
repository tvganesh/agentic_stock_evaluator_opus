"""Resumable acquisition: a disk checkpoint of successful source reads for one (source, as_of).

A Nifty 500 acquisition takes over 30 minutes because of Upstox's 30-minute rate limit. If it
is interrupted -- a 429, a laptop lid, a network drop -- starting again from zero would waste
the rate budget and risk a suspension. :class:`CheckpointedSource` wraps any market source
(the live adapter or the fixture) and:

* serves a read from disk if the same method and arguments were already fetched for this
  (source, as_of) -- no request, no rate-limit cost;
* otherwise performs the read and writes the result atomically before returning it;
* never stores failures, so a vendor error is retried on the next run.

Stale checkpoints: a directory records when it was created (``.created``). Given
``not_before`` -- the moment the ``as_of`` session's data became final -- checkpoints created
earlier are discarded before use. Without this, reads captured during market hours would be
silently reused by a run after the close, and the snapshot would miss that day's session.

Determinism is preserved: results are keyed by the method and its arguments (which already
include ``as_of``-derived dates), so a resumed build produces the same snapshot as an
uninterrupted one. The checkpoint directory is deleted once the snapshot is sealed.

This is intermediate ETL state inside the ACQUIRE process. It holds vendor market data only,
never credentials, and nothing in the sealed process reads it.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from ..snapshot.hashing import canonical_json, hash_object

_MARKER = ".created"


def _jsonable(value: Any) -> Any:
    """Convert call arguments (dates, tuples) into a JSON-safe form for the cache key."""
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _created_at(directory: Path) -> datetime | None:
    """Read a checkpoint directory's creation marker; ``None`` if absent or unreadable."""
    try:
        return datetime.fromisoformat((directory / _MARKER).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


class CheckpointedSource:
    """Market source wrapper that reuses reads already saved for this acquisition."""

    def __init__(self, inner: Any, directory: Path, *, not_before: datetime | None = None) -> None:
        """Wrap ``inner``; discard checkpoints created before ``not_before``; create the directory."""
        self._inner = inner
        self._dir = directory
        self.discarded_stale = False
        if not_before is not None and directory.exists():
            created = _created_at(directory)
            if created is None or created < not_before:
                shutil.rmtree(directory, ignore_errors=True)
                self.discarded_stale = True
        self._dir.mkdir(parents=True, exist_ok=True)
        if not (self._dir / _MARKER).exists():
            (self._dir / _MARKER).write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        self.hits = 0
        self.misses = 0

    @property
    def directory(self) -> Path:
        """Where this acquisition's checkpoints live."""
        return self._dir

    def _call(self, method: str, *args: Any) -> Any:
        """Return a saved result for ``method(*args)`` or fetch, save and return it."""
        key = hash_object([method, _jsonable(list(args))])[:32]
        path = self._dir / f"{method}-{key}.json"
        if path.exists():
            try:
                value = json.loads(path.read_bytes())
                self.hits += 1
                return value
            except json.JSONDecodeError:
                path.unlink()  # torn write from an earlier crash: fetch again
        value = getattr(self._inner, method)(*args)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(canonical_json(value))
        os.replace(tmp, path)
        self.misses += 1
        return value

    def instrument_master(self) -> list[dict[str, Any]]:
        """Checkpointed instrument master."""
        return self._call("instrument_master")

    def daily_candles(self, instrument_key: str, from_date: date, to_date: date) -> list[list[Any]]:
        """Checkpointed daily candles."""
        return self._call("daily_candles", instrument_key, from_date, to_date)

    def intraday_daily_candle(self, instrument_key: str) -> list[list[Any]]:
        """Checkpointed current-session candle (the directory is per as_of, so the key needs no date)."""
        return self._call("intraday_daily_candle", instrument_key)

    def key_ratios(self, isin: str) -> list[dict[str, Any]]:
        """Checkpointed key ratios."""
        return self._call("key_ratios", isin)

    def income_statement(self, isin: str, time_period: str) -> dict[str, Any]:
        """Checkpointed income statement."""
        return self._call("income_statement", isin, time_period)

    def balance_sheet(self, isin: str) -> dict[str, Any]:
        """Checkpointed balance sheet."""
        return self._call("balance_sheet", isin)

    def news(self, instrument_keys: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Checkpointed headlines for the whole universe."""
        return self._call("news", list(instrument_keys))

    def discard(self) -> None:
        """Delete this acquisition's checkpoints (called after the snapshot is sealed)."""
        shutil.rmtree(self._dir, ignore_errors=True)
