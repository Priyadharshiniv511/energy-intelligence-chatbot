"""Unit tests for app/conversation/store.py. Pure in-memory: no network, no sleeping."""

from __future__ import annotations

import dataclasses
import threading

import pytest

from app.conversation.models import Conversation, Turn
from app.conversation.store import (
    InMemorySessionStore,
    SessionLimitError,
    SessionNotFoundError,
    SessionStore,
)


class FakeClock:
    """Manually advanced clock, so TTL tests are instant and deterministic."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(clock: FakeClock) -> InMemorySessionStore:
    return InMemorySessionStore(ttl_seconds=60, max_turns=10, clock=clock)


def test_in_memory_store_satisfies_interface(store: InMemorySessionStore) -> None:
    as_interface: SessionStore = store  # checked by type checkers; must not error at runtime
    assert as_interface.create()


# ---- isolation (DESIGN.md 6.3) ---------------------------------------------------------

def test_two_sessions_never_see_each_others_history(store: InMemorySessionStore) -> None:
    a, b = store.create(), store.create()
    assert a != b

    store.append_turn(a, "What is a kWh?", "A unit of energy.")
    store.record_strike(a)
    store.append_turn(b, "How do heat pumps work?", "They move heat.")

    assert store.get(a).turns == (Turn("What is a kWh?", "A unit of energy."),)
    assert store.get(b).turns == (Turn("How do heat pumps work?", "They move heat."),)
    assert store.get(a).strikes == 1
    assert store.get(b).strikes == 0


def test_separate_store_instances_share_nothing(clock: FakeClock) -> None:
    first = InMemorySessionStore(clock=clock)
    second = InMemorySessionStore(clock=clock)
    session_id = first.create()

    with pytest.raises(SessionNotFoundError):
        second.get(session_id)


def test_snapshot_is_read_only_and_detached(store: InMemorySessionStore) -> None:
    session_id = store.create()
    snapshot = store.get(session_id)

    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.strikes = 99  # type: ignore[misc]
    store.append_turn(session_id, "q", "a")

    assert snapshot.turns == ()  # an older snapshot doesn't change after later writes
    assert len(store.get(session_id).turns) == 1


# ---- history cap -----------------------------------------------------------------------

def test_history_keeps_only_last_10_turns_in_order(store: InMemorySessionStore) -> None:
    session_id = store.create()
    for i in range(15):
        store.append_turn(session_id, f"q{i}", f"a{i}")

    turns = store.get(session_id).turns
    assert len(turns) == 10
    assert [t.user_text for t in turns] == [f"q{i}" for i in range(5, 15)]


# ---- refusals and strikes --------------------------------------------------------------

def test_strike_counter_increments_and_returns_new_total(store: InMemorySessionStore) -> None:
    session_id = store.create()
    assert store.get(session_id).strikes == 0

    assert [store.record_strike(session_id) for _ in range(3)] == [1, 2, 3]
    assert store.get(session_id).strikes == 3


def test_strikes_never_add_turns_to_history(store: InMemorySessionStore) -> None:
    session_id = store.create()
    store.append_turn(session_id, "accepted", "answer")
    store.record_strike(session_id)
    store.record_strike(session_id)

    assert store.get(session_id).turns == (Turn("accepted", "answer"),)


# ---- TTL -------------------------------------------------------------------------------

def test_session_expires_after_ttl_and_is_removed(
    store: InMemorySessionStore, clock: FakeClock
) -> None:
    session_id = store.create()
    clock.advance(60)

    with pytest.raises(SessionNotFoundError):
        store.get(session_id)
    assert len(store) == 0


def test_purge_removes_expired_sessions_without_being_touched(
    store: InMemorySessionStore, clock: FakeClock
) -> None:
    old = store.create()
    clock.advance(59)
    fresh = store.create()
    clock.advance(1)  # `old` is now exactly 60 s idle, `fresh` only 1 s

    assert store.purge_expired() == 1
    assert len(store) == 1
    assert store.get(fresh).session_id == fresh
    with pytest.raises(SessionNotFoundError):
        store.get(old)


def test_activity_slides_the_ttl(store: InMemorySessionStore, clock: FakeClock) -> None:
    session_id = store.create()
    for _ in range(5):
        clock.advance(50)
        store.append_turn(session_id, "q", "a")  # each write resets the 60 s window

    assert len(store.get(session_id).turns) == 5


def test_creating_a_session_purges_expired_ones(
    store: InMemorySessionStore, clock: FakeClock
) -> None:
    for _ in range(3):
        store.create()
    clock.advance(61)
    store.create()

    assert len(store) == 1


# ---- unknown ids -----------------------------------------------------------------------

@pytest.mark.parametrize(
    "operation",
    [
        lambda s: s.get("nope"),
        lambda s: s.append_turn("nope", "q", "a"),
        lambda s: s.record_strike("nope"),
        lambda s: s.delete("nope"),
    ],
    ids=["get", "append_turn", "record_strike", "delete"],
)
def test_unknown_session_raises_clear_error(store: InMemorySessionStore, operation) -> None:
    with pytest.raises(SessionNotFoundError) as excinfo:
        operation(store)
    assert excinfo.value.session_id == "nope"
    assert str(excinfo.value) == "Unknown or expired session."


def test_deleted_session_is_gone(store: InMemorySessionStore) -> None:
    session_id = store.create()
    store.delete(session_id)

    with pytest.raises(SessionNotFoundError):
        store.append_turn(session_id, "q", "a")


@pytest.mark.parametrize("kwargs", [{"ttl_seconds": 0}, {"max_turns": 0}])
def test_rejects_invalid_limits(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        InMemorySessionStore(**kwargs)


# ---- concurrency -----------------------------------------------------------------------

def _run_together(workers: list) -> None:
    """Start all workers at the same instant (Barrier) to maximise contention."""
    barrier = threading.Barrier(len(workers))
    errors: list[BaseException] = []

    def wrap(fn):
        def run() -> None:
            try:
                barrier.wait()
                fn()
            except BaseException as exc:  # surfaced below; threads swallow exceptions
                errors.append(exc)
        return run

    threads = [threading.Thread(target=wrap(w)) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors, errors


def test_concurrent_appends_to_one_session_are_not_lost_or_mixed(clock: FakeClock) -> None:
    store = InMemorySessionStore(max_turns=10_000, clock=clock)
    session_id = store.create()
    threads, per_thread = 8, 200

    def worker(n: int):
        return lambda: [store.append_turn(session_id, f"t{n}-q{i}", f"t{n}-a{i}")
                        for i in range(per_thread)]

    _run_together([worker(n) for n in range(threads)])

    turns = store.get(session_id).turns
    assert len(turns) == threads * per_thread
    for n in range(threads):
        mine = [t for t in turns if t.user_text.startswith(f"t{n}-")]
        # Each thread's turns are all present, in the order it wrote them, and intact.
        assert [t.user_text for t in mine] == [f"t{n}-q{i}" for i in range(per_thread)]
        assert all(t.answer_text == t.user_text.replace("-q", "-a") for t in mine)


def test_concurrent_appends_still_respect_the_cap(store: InMemorySessionStore) -> None:
    session_id = store.create()
    _run_together([lambda: [store.append_turn(session_id, "q", "a") for _ in range(100)]
                   for _ in range(8)])

    assert len(store.get(session_id).turns) == 10


def test_concurrent_strikes_are_all_counted(store: InMemorySessionStore) -> None:
    session_id = store.create()
    _run_together([lambda: [store.record_strike(session_id) for _ in range(250)]
                   for _ in range(8)])

    assert store.get(session_id).strikes == 2000


def test_snapshot_type(store: InMemorySessionStore) -> None:
    assert isinstance(store.get(store.create()), Conversation)


# ---- Capacity --------------------------------------------------------------------------

def test_store_refuses_new_sessions_when_full(clock: FakeClock) -> None:
    store = InMemorySessionStore(ttl_seconds=60, max_sessions=2, clock=clock)
    store.create()
    store.create()
    with pytest.raises(SessionLimitError):
        store.create()


def test_expired_sessions_free_capacity(clock: FakeClock) -> None:
    store = InMemorySessionStore(ttl_seconds=60, max_sessions=1, clock=clock)
    store.create()
    clock.advance(61)
    store.create()  # the expired one is purged first, so this fits
    assert len(store) == 1


def test_max_sessions_must_be_positive() -> None:
    with pytest.raises(ValueError):
        InMemorySessionStore(max_sessions=0)
