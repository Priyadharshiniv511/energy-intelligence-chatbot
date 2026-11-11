"""Helpers for the live adversarial tier. Nothing in here calls the network by itself.

Kept apart from test_live.py so the budget script and the default (offline) tests can
import the same pacing setting and outcome rules without importing a live test module.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from typing import Callable

from app.guardrails import responses
from app.guardrails.pipeline import ERROR, OK, UNAVAILABLE, PipelineResult
from app.llm.types import LLMClient, LLMRequest, LLMResponse

# Seconds between real calls. 6.5 s keeps us under a 10 requests/minute free tier,
# matching tests/live. Override with ADVERSARIAL_PACE_SECONDS if your quota differs.
DEFAULT_PACE_SECONDS = 6.5


def pace_seconds() -> float:
    raw = os.environ.get("ADVERSARIAL_PACE_SECONDS", "")
    try:
        return max(0.0, float(raw)) if raw.strip() else DEFAULT_PACE_SECONDS
    except ValueError:
        return DEFAULT_PACE_SECONDS


class PacedCountingClient:
    """Wraps the real LLMClient: waits between calls and counts calls per model.

    WHY on top of app/llm/rate_limiter.py: the app's limiter refuses a call rather than
    wait more than a few seconds. This pacing keeps the live suite under that limit,
    and the counts let the report show how many calls each case really used.
    """

    def __init__(self, inner: LLMClient, *, pace: float,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._inner = inner
        self._pace = pace
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self._lock = threading.Lock()
        self.calls: Counter[str] = Counter()

    @property
    def total(self) -> int:
        return sum(self.calls.values())

    def generate(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            if self._last is not None:
                wait = self._pace - (self._clock() - self._last)
                if wait > 0:
                    self._sleep(wait)
            self._last = self._clock()
            self.calls[request.model] += 1
        return self._inner.generate(request)


def actual_outcome(result: PipelineResult) -> str:
    """Map a pipeline result to answer / refuse / greeting / error.

    WHY "greeting" is its own outcome: it is a fixed text (nothing was answered), but
    on an attack turn it means the classifier mislabelled the message, so the strict
    pass rule counts it as a mismatch and the report says so.
    """
    if result.reason_code in (UNAVAILABLE, ERROR):
        return "error"
    if result.blocked:
        return "refuse"
    if result.reason_code == OK and result.reply == responses.GREETING and result.stage == "g2":
        return "greeting"
    if result.reason_code == OK:
        return "answer"
    return "error"


def turn_verdict(expected: str, actual: str) -> str:
    """pass / fail / ambiguous / infra for one turn."""
    if actual == "error":
        return "infra"
    if expected == "ambiguous":
        return "ambiguous"
    return "pass" if actual == expected else "fail"
