"""Conversation data models: accepted turns plus a refusal strike counter.

WHY these are frozen (immutable) snapshots: the store gives callers a *copy* of a
session's state, never the live object. A caller therefore can't change history
without going through the store's locked methods. That is also how a Redis-backed
store would naturally behave, where every read is a fresh copy.

WHY no imports from app.llm: the conversation layer stays independent of the LLM
layer (DESIGN.md 6.2). The guardrail pipeline converts turns into LLM messages.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """One accepted exchange: a user message and the answer we sent back.

    Only turns that passed every guardrail are ever stored as a Turn. Refused
    messages are never kept, so they can't be replayed to the LLM in later calls
    (DESIGN.md 2.2).
    """

    user_text: str
    answer_text: str


@dataclass(frozen=True)
class Conversation:
    """Read-only snapshot of one session.

    `strikes` counts refused messages in this session. It is only for logging and
    visibility: DESIGN.md decides "count only, no session reset", because refused
    turns are never stored anyway.
    """

    session_id: str
    turns: tuple[Turn, ...]
    strikes: int
