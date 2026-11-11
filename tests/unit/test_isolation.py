"""Two sessions through the full pipeline must not influence each other (DESIGN.md 6.3, 5.9)."""

from __future__ import annotations

import pytest

from app.conversation.store import SessionNotFoundError
from app.guardrails import responses
from tests.unit.pipeline_helpers import (
    G1_ATTACK,
    answer_json,
    make_service,
    verdict_json,
)

A_QUESTION = "How does a heat pump work in winter?"
A_ANSWER = "It moves heat from outside air even when it is cold."
B_QUESTION = "What does a smart meter measure?"
B_ANSWER = "It records your electricity use in short intervals."


def all_text(request: object) -> str:
    return request.system_instruction + "\n".join(m.text for m in request.messages)  # type: ignore[attr-defined]


def test_two_sessions_stay_isolated() -> None:
    service, fake = make_service(
        verdict_json(), answer_json(A_ANSWER),   # A turn 1
        verdict_json(), answer_json(B_ANSWER),   # B turn 1
    )
    a, b = service.create_session(), service.create_session()

    service.send(a, A_QUESTION)
    service.send(a, G1_ATTACK)          # strike in A only, no LLM call
    service.send(b, B_QUESTION)

    b_requests = fake.requests[2:]
    assert len(b_requests) == 2
    for request in b_requests:
        text = all_text(request)
        assert A_QUESTION not in text and A_ANSWER not in text

    conv_a, conv_b = service.conversation(a), service.conversation(b)
    assert [t.user_text for t in conv_a.turns] == [A_QUESTION]
    assert [t.user_text for t in conv_b.turns] == [B_QUESTION]
    assert (conv_a.strikes, conv_b.strikes) == (1, 0)


def test_refusals_in_one_session_do_not_change_the_other() -> None:
    service, _ = make_service(verdict_json(), answer_json(B_ANSWER))
    a, b = service.create_session(), service.create_session()
    for _ in range(3):
        assert service.send(a, G1_ATTACK).reply == responses.REFUSAL
    assert service.send(b, B_QUESTION).reply == B_ANSWER


def test_deleting_one_session_leaves_the_other_intact() -> None:
    service, _ = make_service(verdict_json(), answer_json(A_ANSWER),
                              verdict_json(), answer_json(B_ANSWER))
    a, b = service.create_session(), service.create_session()
    service.send(a, A_QUESTION)
    service.send(b, B_QUESTION)

    service.delete_session(a)

    with pytest.raises(SessionNotFoundError):
        service.conversation(a)
    assert [t.answer_text for t in service.conversation(b).turns] == [B_ANSWER]
