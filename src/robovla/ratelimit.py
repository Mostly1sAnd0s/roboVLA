"""Fixed-window rate limiter (design §9.8: login 5/min/IP · uploads 2
concurrent/student · submissions 10/min/student · admin 60/min · robot
~5 req/s per student).

In-memory, keyed by (kind, subject); time is injected so tests advance the
window deterministically. LAN-only console, one process — no distributed
state needed (§9.8).
"""

from __future__ import annotations


class RateLimiter:
    def __init__(self, now_ms) -> None:
        self._now_ms = now_ms
        self._hits: dict[tuple[str, str], list[int]] = {}
        self._inflight: dict[tuple[str, str], int] = {}

    def allow(self, kind: str, subject: str, limit: int, window_ms: int) -> bool:
        """Sliding-window count: True if under ``limit`` in ``window_ms``."""
        now = self._now_ms()
        key = (kind, subject)
        hits = [t for t in self._hits.get(key, []) if now - t < window_ms]
        ok = len(hits) < limit
        hits.append(now)
        self._hits[key] = hits
        return ok

    def acquire(self, kind: str, subject: str, limit: int) -> bool:
        """Concurrent-in-flight guard (uploads): True while under the cap."""
        key = (kind, subject)
        n = self._inflight.get(key, 0)
        if n >= limit:
            return False
        self._inflight[key] = n + 1
        return True

    def release(self, kind: str, subject: str) -> None:
        key = (kind, subject)
        n = self._inflight.get(key, 0)
        if n <= 1:
            self._inflight.pop(key, None)
        else:
            self._inflight[key] = n - 1
