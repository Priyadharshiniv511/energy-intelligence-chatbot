"""Tier 1: what G1 alone must decide, through the full ChatService, with zero LLM calls.

WHY this tier exists: G1 is the only guardrail that is fully deterministic. Anything it
is designed to catch must be caught every time, for free, before the classifier ever
sees it. And it must never block a real energy question that happens to use words like
"ignore", "kill switch" or "developer mode".
"""

from __future__ import annotations

import pytest

from app.guardrails import responses
from app.guardrails.input_filter import check_input
from tests.adversarial.cases import CASES, Case, Category, Turn, case_param_id
from tests.unit.pipeline_helpers import make_service, make_settings

G1_TURNS: list[tuple[Case, int, Turn]] = [
    (case, i, turn) for case in CASES for i, turn in enumerate(case.turns) if turn.g1_block
]
LEGIT_TURNS: list[tuple[Case, Turn]] = [
    (case, turn) for case in CASES if case.category is Category.LEGITIMATE for turn in case.turns
]


def test_g1_tier_is_not_empty() -> None:
    """Guards against a refactor that silently drops every g1_block flag."""
    assert len(G1_TURNS) >= 20


@pytest.mark.parametrize(("case", "index", "turn"), G1_TURNS,
                         ids=[f"{case_param_id(c)}-t{i}" for c, i, _ in G1_TURNS])
def test_g1_blocks_with_zero_llm_calls(case: Case, index: int, turn: Turn) -> None:
    service, fake = make_service()  # empty script: any LLM call would raise
    session = service.create_session()

    result = service.send(session, turn.text)

    assert result.blocked, case.note
    assert result.stage == "g1"
    assert result.reply == responses.REFUSAL
    assert fake.requests == []
    conversation = service.conversation(session)
    assert conversation.strikes == 1
    assert conversation.turns == ()


@pytest.mark.parametrize(("case", "turn"), LEGIT_TURNS,
                         ids=[case_param_id(c) for c, _ in LEGIT_TURNS])
def test_g1_lets_legitimate_questions_through(case: Case, turn: Turn) -> None:
    """A G1 false positive would refuse a real question before G2 could judge it."""
    result = check_input(turn.text, max_chars=make_settings().max_input_chars)
    assert result.allowed, f"{case.id}: {result.reason} ({case.note})"
