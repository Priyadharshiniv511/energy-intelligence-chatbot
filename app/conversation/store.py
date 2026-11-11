"""SessionStore: per-session conversations with a lock, a TTL and a turn cap.

Design (DESIGN.md 2.1, 2.2, 3):
- `SessionStore` is an interface (Protocol). `InMemorySessionStore` is the only
  implementation for now. A Redis version could replace it later without changes
  to callers.
- Every operation is one atomic method call (append a turn, record a strike).
  Callers never get a live object to modify. That is what makes a Redis swap
  realistic: Redis can do atomic operations, but it can't share Python objects.
- Unknown or expired session ids raise `SessionNotFoundError`. The API maps it to 404.
- Each store instance owns all its state. There are no module-level mutable
  globals, so two stores (or two tests) can never share sessions.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from typing import Callable, Protocol

from app.conversation.models import Conversation, Turn

DEFAULT_MAX_TURNS = 10
DEFAULT_TTL_SECONDS = 30 * 60.0
# WHY a cap: every live session holds memory for up to the TTL, so without one a client
# looping on "create session" could exhaust memory and take every other session down.
DEFAULT_MAX_SESSIONS = 10_000


class SessionNotFoundError(KeyError):
    """The session id is unknown, expired or deleted.

    WHY one error for all three cases: the caller reacts the same way each time
    (404, start a new session). Not saying which case it was also avoids telling a
    guesser whether an id once existed.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.session_id = session_id

    def __str__(self) -> str:
        # Fixed message: the id is kept as an attribute, not echoed into logs or users.
        return "Unknown or expired session."


class SessionLimitError(RuntimeError):
    """The store already holds its maximum number of live sessions.

    WHY a separate error: it is a capacity problem, not a bad id, so the API answers
    503 ("try again later") rather than 404.
    """

    def __str__(self) -> str:
        return "Too many active sessions."


class SessionStore(Protocol):
    """What the rest of the app may do with sessions. Nothing more."""

    def create(self) -> str:
        """Start an empty session and return its new, unguessable id.

        Raises SessionLimitError when the store is full.
        """
        ...

    def get(self, session_id: str) -> Conversation:
        """Return a read-only snapshot of the session (refreshes its TTL)."""
        ...

    def append_turn(self, session_id: str, user_text: str, answer_text: str) -> None:
        """Store one turn that passed ALL guardrails. Never call this for a refusal."""
        ...

    def record_strike(self, session_id: str) -> int:
        """Count one refused message and return the session's new strike total."""
        ...

    def delete(self, session_id: str) -> None:
        """Remove the session. Later use of its id raises SessionNotFoundError."""
        ...


class _SessionState:
    """Mutable state for one session. Only InMemorySessionStore touches it, and only
    while holding `lock`."""

    def __init__(self, session_id: str, max_turns: int, now: float) -> None:
        self.session_id = session_id
        self.lock = threading.Lock()
        # deque(maxlen=N) drops the oldest turn by itself when an N+1th is added.
        self.turns: deque[Turn] = deque(maxlen=max_turns)
        self.strikes = 0
        self.last_active = now
        # Set when the store removes this session. A thread that looked the session
        # up just before removal will see the flag and fail cleanly, so its write
        # can't land on an orphaned object.
        self.removed = False

    def snapshot(self) -> Conversation:
        return Conversation(
            session_id=self.session_id, turns=tuple(self.turns), strikes=self.strikes
        )


class InMemorySessionStore:
    """Thread-safe in-memory SessionStore with a sliding TTL.

    Locking:
    - `self._lock` guards the dict of sessions (create, look up, remove).
    - Each session has its own lock that guards its turns and strike count, so busy
      sessions don't block each other.
    - Lock order is always store lock, then session lock, never the other way
      round. That fixed order is what prevents deadlocks.

    WHY threading.Lock: FastAPI runs sync endpoints in a thread pool, so two requests
    for the same session can really run at the same time.

    Expiry is "sliding": every successful access resets the clock. Expired sessions
    are removed when touched, and `purge_expired()` sweeps the rest (it runs on every
    `create()` so memory can't grow forever).
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self._ttl = ttl_seconds
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        # WHY monotonic: wall-clock time can jump (NTP, DST) and expire everything at
        # once. Tests inject a fake clock to "advance time" instantly.
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionState] = {}

    # ---- public API ---------------------------------------------------------------

    def create(self) -> str:
        self.purge_expired()
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                raise SessionLimitError()
            # WHY secrets, not a counter: the id is the only thing that grants access
            # to a conversation, so it must be unguessable (DESIGN.md 6.3 isolation).
            session_id = secrets.token_urlsafe(16)
            while session_id in self._sessions:  # practically never, but cheap
                session_id = secrets.token_urlsafe(16)
            self._sessions[session_id] = _SessionState(
                session_id, self._max_turns, self._clock()
            )
            return session_id

    def get(self, session_id: str) -> Conversation:
        state = self._lookup(session_id)
        with state.lock:
            self._ensure_present(state)
            state.last_active = self._clock()
            return state.snapshot()

    def append_turn(self, session_id: str, user_text: str, answer_text: str) -> None:
        state = self._lookup(session_id)
        with state.lock:
            self._ensure_present(state)
            state.turns.append(Turn(user_text=user_text, answer_text=answer_text))
            state.last_active = self._clock()

    def record_strike(self, session_id: str) -> int:
        state = self._lookup(session_id)
        with state.lock:
            self._ensure_present(state)
            state.strikes += 1
            state.last_active = self._clock()
            return state.strikes

    def delete(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError(session_id)
            self._remove_locked(state)

    def purge_expired(self) -> int:
        """Remove every expired session and return how many were removed."""
        with self._lock:
            expired = [s for s in self._sessions.values() if self._is_expired(s)]
            for state in expired:
                self._remove_locked(state)
            return len(expired)

    def __len__(self) -> int:
        """Number of sessions currently held (expired ones may remain until purged)."""
        with self._lock:
            return len(self._sessions)

    # ---- internals ----------------------------------------------------------------

    def _lookup(self, session_id: str) -> _SessionState:
        """Find a live session, removing it right away if it has expired."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError(session_id)
            if self._is_expired(state):
                self._remove_locked(state)
                raise SessionNotFoundError(session_id)
            return state

    def _is_expired(self, state: _SessionState) -> bool:
        # Caller holds self._lock. Taking the session lock here keeps the order
        # store lock, then session lock.
        with state.lock:
            return self._clock() - state.last_active >= self._ttl

    def _remove_locked(self, state: _SessionState) -> None:
        # Caller holds self._lock.
        with state.lock:
            state.removed = True
        self._sessions.pop(state.session_id, None)

    @staticmethod
    def _ensure_present(state: _SessionState) -> None:
        # Caller holds state.lock. Covers the case where another thread deleted or
        # purged the session between our lookup and taking its lock.
        if state.removed:
            raise SessionNotFoundError(state.session_id)
