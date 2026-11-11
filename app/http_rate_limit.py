"""Per-client-IP rate limit for the REST messages endpoint.

WHY a separate limiter from app/llm/rate_limiter.py: that one keeps *our* calls to
Gemini under the provider's limit. This one stops a single client from using up the
shared quota (and the other users' share of it) by flooding the API.

Sliding window: an IP may make `limit` requests in any `window_seconds` span. In memory
and per process; see "Known limitations" in docs/DESIGN.md.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable


class IpRateLimiter:
    """Thread-safe sliding-window counter keyed by client IP."""

    def __init__(
        self,
        limit: int,
        *,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """WHY inject `clock`: tests move time forward instantly instead of sleeping."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def allow(self, ip: str) -> bool:
        """Record one request from `ip` and say whether it is within the limit.

        A refused request is not recorded, so a client that keeps retrying is let
        back in as soon as its old requests leave the window.
        """
        now = self._clock()
        with self._lock:
            self._drop_old(now)
            hits = self._hits.setdefault(ip, deque())
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True

    def _drop_old(self, now: float) -> None:
        """Forget requests older than the window, and IPs with none left.

        WHY sweep every IP: removing empty entries keeps memory bounded by the number
        of recently active clients, not by every IP ever seen. Caller holds the lock.
        """
        cutoff = now - self._window
        for ip in list(self._hits):
            hits = self._hits[ip]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if not hits:
                del self._hits[ip]
