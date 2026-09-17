"""Client-side rolling-window rate limiter for the egress gate.

Upstox publishes limits for standard APIs of 50 requests per second, 500 per minute and 2,000
per 30 minutes, and warns that exceeding them "might result in temporary suspension of
access". A simple fixed delay between requests only protects the first two; a Nifty 500
acquisition (~2,500 requests) would blow through the 30-minute limit in minutes.

:class:`RollingRateLimiter` enforces any number of ``(window seconds, max requests)`` limits
simultaneously. Before each request it waits until *every* window has room, then records the
request in all of them. The limits it is given come from ``policy.UPSTOX_RATE_LIMITS`` and sit
deliberately below the vendor's numbers, so clock skew or retries never tip us over.

Upstox counts requests per account, not per process. On 15 Sep 2026 three acquisitions started
within 15 minutes of each other sent 2,456 requests in one 30-minute window, because each
process's limiter started empty. :func:`recent_request_ages` therefore reads the ``egress.allow``
events from recent ACQUIRE audit logs, and :meth:`RollingRateLimiter.seed` counts them against
the new run's windows.

Only the ACQUIRE process uses this; the clock and sleep functions are injectable for tests.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


class RollingRateLimiter:
    """Blocks until a request fits within every configured rolling window, then records it."""

    def __init__(
        self,
        limits: Iterable[tuple[float, int]],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Configure ``(window_seconds, max_requests)`` limits and the time source."""
        self._limits = tuple((float(seconds), int(count)) for seconds, count in limits)
        if not self._limits or any(seconds <= 0 or count <= 0 for seconds, count in self._limits):
            raise ValueError("rate limits must be positive (seconds, count) pairs")
        self._history: list[deque[float]] = [deque() for _ in self._limits]
        self._clock = clock
        self._sleep = sleep

    def seed(self, ages_seconds: Iterable[float]) -> int:
        """Count requests made ``age`` seconds ago (e.g. by an earlier run) against every window.

        Returns how many fell inside the longest window and were therefore counted.
        """
        now = self._clock()
        longest = max(seconds for seconds, _ in self._limits)
        stamps = sorted(now - age for age in ages_seconds if 0 <= age < longest)
        for (seconds, _), history in zip(self._limits, self._history):
            merged = sorted([*history, *(t for t in stamps if t > now - seconds)])
            history.clear()
            history.extend(merged)
        return len(stamps)

    def acquire(self) -> float:
        """Wait until one more request is allowed by all windows; return the seconds spent waiting."""
        waited = 0.0
        while True:
            now = self._clock()
            wait = 0.0
            for (seconds, count), history in zip(self._limits, self._history):
                while history and history[0] <= now - seconds:
                    history.popleft()
                if len(history) >= count:
                    wait = max(wait, history[0] + seconds - now)
            if wait <= 0:
                break
            self._sleep(wait)
            waited += wait
        for history in self._history:
            history.append(now)
        return waited

    def in_window(self) -> list[int]:
        """Requests currently counted in each window, in configuration order (for progress/audit)."""
        now = self._clock()
        return [sum(1 for t in history if t > now - seconds)
                for (seconds, _), history in zip(self._limits, self._history)]


def recent_request_ages(
    audit_dir: Path, window_seconds: float = 1800.0, now: datetime | None = None
) -> list[float]:
    """Ages in seconds of Upstox requests (``egress.allow``) recorded in ACQUIRE audit logs within the window.

    Files last modified before the window are skipped without being read.
    """
    now = now or datetime.now(timezone.utc)
    if not audit_dir.exists():
        return []
    ages: list[float] = []
    cutoff = now.timestamp() - window_seconds
    for log in audit_dir.glob("*.jsonl"):
        if log.stat().st_mtime < cutoff:
            continue
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") != "egress.allow":
                continue
            age = (now - datetime.fromisoformat(entry["ts"])).total_seconds()
            if 0 <= age < window_seconds:
                ages.append(age)
    return sorted(ages)
