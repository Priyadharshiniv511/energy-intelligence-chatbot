"""Unit tests for app/llm/rate_limiter.py.

A fake clock and a recording sleep make every test instant: "wait 7.5 s" is just a
number we check, not real time passing.
"""

from __future__ import annotations

import threading

import pytest

from app.config import Settings
from app.llm.errors import LLMRateLimited
from app.llm.rate_limiter import RateLimiter

CLASSIFIER = "gemini-test-classifier"
ANSWER = "gemini-test-answer"


class FakeClock:
    """Time only moves when a test (or the fake sleep) says so."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_limiter(clock: FakeClock, *, rpm: float = 6.0, burst: int = 2,
                 max_wait: float = 5.0, limits: dict[str, float] | None = None) -> RateLimiter:
    return RateLimiter(limits or {}, default_rpm=rpm, max_wait=max_wait, burst=burst,
                       clock=clock, sleep=clock.sleep)


# ---- Refill rate --------------------------------------------------------------------

def test_full_bucket_allows_burst_without_waiting() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, burst=2)
    limiter.acquire(ANSWER)
    limiter.acquire(ANSWER)
    assert clock.sleeps == []


def test_one_token_refills_every_60_over_rpm_seconds() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=6.0, burst=1, max_wait=0)  # 1 token per 10 s
    limiter.acquire(ANSWER)
    clock.now += 9.9
    with pytest.raises(LLMRateLimited):
        limiter.acquire(ANSWER)
    clock.now += 0.1
    limiter.acquire(ANSWER)  # exactly one token is back
    with pytest.raises(LLMRateLimited):
        limiter.acquire(ANSWER)


def test_partial_refill_waits_only_the_remaining_time() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=6.0, burst=1, max_wait=10.0)
    limiter.acquire(ANSWER)
    clock.now += 4.0
    limiter.acquire(ANSWER)
    assert clock.sleeps == [pytest.approx(6.0)]


def test_idle_time_never_fills_beyond_the_burst() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=6.0, burst=2, max_wait=0)
    clock.now += 3600  # an hour idle
    limiter.acquire(ANSWER)
    limiter.acquire(ANSWER)
    with pytest.raises(LLMRateLimited):
        limiter.acquire(ANSWER)


# ---- Separate buckets per model -----------------------------------------------------

def test_each_model_has_its_own_bucket() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, burst=1, max_wait=0)
    limiter.acquire(CLASSIFIER)
    limiter.acquire(ANSWER)  # not affected by the empty classifier bucket
    with pytest.raises(LLMRateLimited):
        limiter.acquire(CLASSIFIER)


def test_each_model_refills_at_its_own_rpm() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, burst=1, max_wait=10.0, limits={CLASSIFIER: 12.0, ANSWER: 6.0})
    limiter.acquire(CLASSIFIER)
    limiter.acquire(CLASSIFIER)
    limiter.acquire(ANSWER)
    limiter.acquire(ANSWER)
    # Classifier: 12 RPM -> 5 s per token. Answer: 6 RPM -> 10 s per token.
    assert clock.sleeps == [pytest.approx(5.0), pytest.approx(10.0)]


def _settings(**overrides: object) -> Settings:
    values = {"gemini_api_key": "fake-key", "answer_model": ANSWER,
              "classifier_model": CLASSIFIER, "classifier_rpm": 12.0, "answer_rpm": 6.0,
              **overrides}
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_from_settings_maps_models_and_fallbacks_to_their_role_rpm() -> None:
    limiter = RateLimiter.from_settings(
        _settings(classifier_fallback_model="cls-backup", answer_fallback_model="ans-backup"))
    assert limiter._limits == {CLASSIFIER: 12.0, "cls-backup": 12.0,
                               ANSWER: 6.0, "ans-backup": 6.0}
    assert limiter._max_wait == 5.0


def test_shared_model_name_and_unknown_models_get_the_lower_rpm() -> None:
    limiter = RateLimiter.from_settings(_settings(answer_model=CLASSIFIER))
    assert limiter._limits == {CLASSIFIER: 6.0}
    assert limiter._default_rpm == 6.0


# ---- Waiting vs. refusing -----------------------------------------------------------

def test_waits_within_max_wait_and_queues_the_next_caller_behind() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=6.0, burst=1, max_wait=25.0)
    frozen = FakeClock()  # sleep records but does not advance: two callers "at once"
    limiter._sleep = frozen.sleep
    limiter.acquire(ANSWER)
    limiter.acquire(ANSWER)
    limiter.acquire(ANSWER)
    # The second caller reserved the 10 s slot, so the third waits for the 20 s one.
    assert frozen.sleeps == [pytest.approx(10.0), pytest.approx(20.0)]


def test_raises_when_the_wait_would_exceed_max_wait() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=6.0, burst=1, max_wait=5.0)
    limiter.acquire(ANSWER)
    with pytest.raises(LLMRateLimited) as info:
        limiter.acquire(ANSWER)
    assert info.value.retry_after == pytest.approx(10.0)
    assert info.value.status_code is None  # it never reached Google
    assert clock.sleeps == []  # failed fast, no waiting


def test_a_refused_call_uses_no_token() -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, rpm=6.0, burst=1, max_wait=0)
    limiter.acquire(ANSWER)
    for _ in range(5):
        with pytest.raises(LLMRateLimited):
            limiter.acquire(ANSWER)
    clock.now += 10.0
    limiter.acquire(ANSWER)  # refusals did not push the next slot back


def test_refusal_log_has_no_model_name(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    limiter = make_limiter(clock, burst=1, max_wait=0)
    limiter.acquire(ANSWER)
    with pytest.raises(LLMRateLimited):
        limiter.acquire(ANSWER)
    assert caplog.records and ANSWER not in caplog.text


# ---- Concurrency --------------------------------------------------------------------

def test_concurrent_callers_never_get_more_tokens_than_the_bucket_holds() -> None:
    clock = FakeClock()  # frozen: no refill while the threads race
    limiter = make_limiter(clock, burst=3, max_wait=0)
    threads_count = 50
    barrier = threading.Barrier(threads_count)
    granted: list[bool] = []
    record = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            limiter.acquire(ANSWER)
            ok = True
        except LLMRateLimited:
            ok = False
        with record:
            granted.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(threads_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(granted) == threads_count
    assert granted.count(True) == 3


def test_concurrent_waiters_each_get_a_distinct_slot() -> None:
    clock = FakeClock()
    sleeps: list[float] = []
    record = threading.Lock()

    def sleep(seconds: float) -> None:  # thread-safe recorder; time stays frozen
        with record:
            sleeps.append(seconds)

    limiter = RateLimiter({}, default_rpm=6.0, max_wait=1000, burst=1, clock=clock, sleep=sleep)
    barrier = threading.Barrier(10)

    def worker() -> None:
        barrier.wait()
        limiter.acquire(ANSWER)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    # One caller got the token now; the other nine are spaced exactly 10 s apart.
    assert sorted(round(s, 6) for s in sleeps) == [10.0 * i for i in range(1, 10)]


def test_greedy_traffic_never_exceeds_burst_plus_rpm_in_any_minute() -> None:
    clock = FakeClock()
    rpm, burst = 8.0, 2
    limiter = make_limiter(clock, rpm=rpm, burst=burst, max_wait=0)
    grants: list[float] = []
    end = clock.now + 180
    while clock.now < end:  # hammer the limiter every 0.1 s for 3 minutes
        try:
            limiter.acquire(ANSWER)
            grants.append(clock.now)
        except LLMRateLimited:
            pass
        clock.now += 0.1

    worst = max(sum(1 for g in grants if start <= g < start + 60) for start in grants)
    assert worst <= burst + rpm
    assert len(grants) <= burst + rpm * 3  # long-run rate is the configured RPM
