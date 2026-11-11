"""Client-side token bucket that keeps us under the free-tier requests-per-minute limit.

WHY on our side: Gemini answers 429 once we go over a model's RPM, and each 429 still
costs a round trip plus retries. Pacing calls ourselves avoids most of them. When we
would have to wait too long, we fail fast with `LLMRateLimited` and the pipeline shows
the "unavailable" message: no answer and no strike (fail closed).

How the bucket works:
- Each model has its own bucket, because Google sets quota per model (the classifier's
  lighter model gets more RPM than the answer model).
- A bucket holds at most `burst` tokens and refills at `rpm / 60` tokens per second.
  Every real call takes one token.
- WHY a small burst: a token bucket allows up to `burst + rpm` calls in any rolling
  60 s window (a full bucket, plus one minute of refill). With burst=2 and the default
  RPMs (8 answer, 12 classifier) that is at most 10 and 14, within the ~10 and ~15 RPM
  free tiers.

Thread safety: a caller *reserves* a token under a lock (the count may go below zero,
which books a future slot), then sleeps **outside** the lock. So concurrent callers
queue in order, the limit always holds, and no thread blocks others while it sleeps.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Mapping

from app.config import Settings
from app.llm.errors import LLMRateLimited

logger = logging.getLogger(__name__)

DEFAULT_BURST = 2


class TokenBucket:
    """One model's bucket. Not locked itself: `RateLimiter` holds the lock around it."""

    def __init__(self, rpm: float, burst: int, now: float) -> None:
        if rpm <= 0 or burst < 1:
            raise ValueError("rpm must be > 0 and burst >= 1")
        self._rate = rpm / 60.0  # tokens per second
        self._capacity = float(burst)
        self._tokens = float(burst)  # start full: the first calls never wait
        self._updated = now

    def reserve(self, now: float, max_wait: float) -> float | None:
        """Take one token and return how long to wait for it, or None if too long.

        WHY return None rather than raise: the caller logs and raises with no lock
        held. If the wait is too long, nothing changes, so a refused call costs no
        quota.
        """
        self._refill(now)
        wait = max(0.0, (1.0 - self._tokens) / self._rate)
        if wait > max_wait:
            return None
        self._tokens -= 1.0
        return wait

    def wait_needed(self, now: float) -> float:
        """Seconds until the next token is free (used for `retry_after`)."""
        self._refill(now)
        return max(0.0, (1.0 - self._tokens) / self._rate)

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated)  # max(): never trust a clock to go back
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._updated = now


class RateLimiter:
    """Per-model token buckets shared by every thread that makes real Gemini calls."""

    def __init__(
        self,
        limits: Mapping[str, float],
        *,
        default_rpm: float,
        max_wait: float,
        burst: int = DEFAULT_BURST,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """`limits` maps a model name to its RPM. Unlisted models use `default_rpm`.

        WHY inject `clock` and `sleep`: tests pass a fake clock and a recording sleep, so
        "wait 7.5 seconds" runs instantly and gives exact, predictable numbers.
        """
        self._limits = dict(limits)
        self._default_rpm = default_rpm
        self._max_wait = max_wait
        self._burst = burst
        self._clock = clock
        self._sleep = sleep
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> RateLimiter:
        """Build the limiter from config.

        Fallback models share their role's RPM. If one model name serves both roles,
        it gets the lower RPM, since Google counts the quota per model, not per role.
        Unknown models get the lower of the two as a safe default.
        """
        limits: dict[str, float] = {}
        roles = (
            (settings.classifier_rpm, (settings.classifier_model, settings.classifier_fallback_model)),
            (settings.answer_rpm, (settings.answer_model, settings.answer_fallback_model)),
        )
        for rpm, models in roles:
            for model in models:
                if model:
                    limits[model] = min(rpm, limits.get(model, rpm))
        return cls(
            limits,
            default_rpm=min(settings.classifier_rpm, settings.answer_rpm),
            max_wait=settings.rate_limit_max_wait_seconds,
        )

    def acquire(self, model: str) -> None:
        """Block until `model` may make one call, or raise `LLMRateLimited`.

        WHY no model name in the log or error: model names are treated as secrets
        (DESIGN.md 2.4), the same as everywhere else in the app.
        """
        with self._lock:
            now = self._clock()
            bucket = self._bucket(model, now)
            wait = bucket.reserve(now, self._max_wait)
            if wait is None:
                retry_after = bucket.wait_needed(now)
        if wait is None:
            logger.warning("Client-side rate limit reached: retry_after=%.1fs", retry_after)
            raise LLMRateLimited(status_code=None, retry_after=retry_after)
        if wait > 0:
            self._sleep(wait)

    def _bucket(self, model: str, now: float) -> TokenBucket:
        """Get or lazily create the bucket for `model`. Caller holds the lock."""
        bucket = self._buckets.get(model)
        if bucket is None:
            rpm = self._limits.get(model, self._default_rpm)
            bucket = self._buckets[model] = TokenBucket(rpm, self._burst, now)
        return bucket
