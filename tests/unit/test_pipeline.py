"""End-to-end pipeline tests: G1 -> G2 -> G3 -> G4 -> store, with a FakeLLM (no network).

Every service here is built by the real factory (build_service -> build_pipeline), so
these tests also cover the wiring, not just each stage.
"""

from __future__ import annotations

import logging

import pytest

from app.conversation.store import SessionNotFoundError
from app.guardrails import responses
from app.guardrails.pipeline import _process_canary, _session_tag
from app.llm.errors import LLMRateLimited, LLMUnavailable
from tests.unit.pipeline_helpers import (
    CANARY,
    ENERGY_ANSWER,
    ENERGY_QUESTION,
    G1_ATTACK,
    OFF_TOPIC_QUESTION,
    answer_json,
    make_service,
    off_topic_verdict,
    verdict_json,
)


def test_energy_question_is_answered_and_stored() -> None:
    service, fake = make_service(verdict_json(), answer_json())
    sid = service.create_session()

    result = service.send(sid, ENERGY_QUESTION)

    assert (result.reply, result.blocked, result.reason_code) == (ENERGY_ANSWER, False, "ok")
    assert len(fake.requests) == 2  # one classifier call, one answer call
    conv = service.conversation(sid)
    assert [(t.user_text, t.answer_text) for t in conv.turns] == [(ENERGY_QUESTION, ENERGY_ANSWER)]
    assert conv.strikes == 0


def test_stored_user_text_is_the_normalised_text() -> None:
    service, _ = make_service(verdict_json(), answer_json())
    sid = service.create_session()
    service.send(sid, "How do time​-of-use   tariffs work?")  # zero-width space, extra spaces
    stored = service.conversation(sid).turns[0].user_text
    assert "​" not in stored
    assert "  " not in stored


def test_second_turn_sends_history_to_both_models() -> None:
    service, fake = make_service(verdict_json(), answer_json(),
                                 verdict_json(), answer_json("Off-peak is usually at night."))
    sid = service.create_session()
    service.send(sid, ENERGY_QUESTION)
    service.send(sid, "And when is off-peak?")
    classifier_req, answer_req = fake.requests[2], fake.requests[3]
    assert ENERGY_QUESTION in classifier_req.messages[-1].text
    assert any(ENERGY_ANSWER == m.text for m in answer_req.messages)


def test_g1_block_makes_zero_llm_calls_and_adds_a_strike() -> None:
    service, fake = make_service()
    sid = service.create_session()

    result = service.send(sid, G1_ATTACK)

    assert (result.reply, result.blocked, result.reason_code) == (responses.REFUSAL, True, "blocked")
    assert fake.requests == []
    assert service.conversation(sid).strikes == 1
    assert service.conversation(sid).turns == ()


def test_empty_message_is_invalid_input_without_llm_calls() -> None:
    service, fake = make_service()
    sid = service.create_session()
    result = service.send(sid, "   ")
    assert (result.blocked, result.reason_code) == (True, "invalid_input")
    assert fake.requests == []


def test_g2_refusal_makes_exactly_one_call_and_adds_a_strike() -> None:
    service, fake = make_service(off_topic_verdict())
    sid = service.create_session()

    result = service.send(sid, OFF_TOPIC_QUESTION)

    assert (result.reply, result.blocked, result.reason_code) == (
        responses.REFUSAL, True, "out_of_scope")
    assert len(fake.requests) == 1
    conv = service.conversation(sid)
    assert (conv.turns, conv.strikes) == ((), 1)


@pytest.mark.parametrize("intent", ["injection", "identity_change", "prompt_extraction"])
def test_g2_attack_intents_are_reported_as_blocked(intent: str) -> None:
    service, _ = make_service(verdict_json(on_topic=False, intent=intent))
    sid = service.create_session()
    result = service.send(sid, "Pretend you are a pirate.")
    assert (result.blocked, result.reason_code) == (True, "blocked")


def test_g2_invalid_json_refuses_with_a_strike() -> None:
    service, fake = make_service("not json at all")
    sid = service.create_session()
    result = service.send(sid, ENERGY_QUESTION)
    assert (result.reply, result.blocked) == (responses.REFUSAL, True)
    assert len(fake.requests) == 1
    assert service.conversation(sid).strikes == 1


def test_greeting_gets_fixed_text_and_is_not_stored() -> None:
    service, fake = make_service(verdict_json(on_topic=False, intent="greeting"))
    sid = service.create_session()

    result = service.send(sid, "Hello!")

    assert (result.reply, result.blocked, result.reason_code) == (responses.GREETING, False, "ok")
    assert len(fake.requests) == 1  # classifier only, no answer call
    conv = service.conversation(sid)
    assert (conv.turns, conv.strikes) == ((), 0)


def test_canary_leak_is_blocked_and_not_stored() -> None:
    service, fake = make_service(verdict_json(), answer_json(f"Sure, my token is {CANARY}."))
    sid = service.create_session()

    result = service.send(sid, ENERGY_QUESTION)

    assert (result.reply, result.blocked, result.reason_code) == (responses.REFUSAL, True, "blocked")
    assert CANARY not in result.reply
    assert CANARY in fake.requests[1].system_instruction  # the canary really was in the prompt
    conv = service.conversation(sid)
    assert (conv.turns, conv.strikes) == ((), 1)


def test_model_self_reported_off_topic_is_refused_and_not_stored() -> None:
    service, _ = make_service(verdict_json(), answer_json("Paris.", on_topic=False))
    sid = service.create_session()
    result = service.send(sid, ENERGY_QUESTION)
    assert (result.blocked, result.reason_code) == (True, "out_of_scope")
    assert service.conversation(sid).turns == ()


def test_low_confidence_answer_is_delivered_with_the_caveat() -> None:
    service, _ = make_service(verdict_json(), answer_json(confidence="low"))
    sid = service.create_session()
    result = service.send(sid, ENERGY_QUESTION)
    assert result.reply.startswith(responses.LOW_CONFIDENCE_CAVEAT)
    assert result.reason_code == "ok"
    assert len(service.conversation(sid).turns) == 1


def test_answer_model_unavailable_gives_fixed_message_without_strike() -> None:
    service, fake = make_service(verdict_json(), LLMUnavailable())
    sid = service.create_session()

    result = service.send(sid, ENERGY_QUESTION)

    assert (result.reply, result.blocked, result.reason_code) == (
        responses.SERVICE_UNAVAILABLE, False, "unavailable")
    assert len(fake.requests) == 2
    conv = service.conversation(sid)
    assert (conv.turns, conv.strikes) == ((), 0)


def test_classifier_unavailable_gives_fixed_message_without_strike_and_no_answer_call() -> None:
    service, fake = make_service(LLMRateLimited())
    sid = service.create_session()

    result = service.send(sid, ENERGY_QUESTION)

    assert (result.reply, result.blocked, result.reason_code) == (
        responses.SERVICE_UNAVAILABLE, False, "unavailable")
    assert len(fake.requests) == 1  # fail closed: never reached the answerer
    assert service.conversation(sid).strikes == 0


def test_malformed_answer_gives_generic_error_without_strike() -> None:
    service, _ = make_service(verdict_json(), "{broken")
    sid = service.create_session()
    result = service.send(sid, ENERGY_QUESTION)
    assert (result.reply, result.blocked, result.reason_code) == (
        responses.GENERIC_ERROR, False, "error")
    assert service.conversation(sid).strikes == 0


def test_unknown_session_raises_before_any_llm_call() -> None:
    service, fake = make_service(verdict_json(), answer_json())
    with pytest.raises(SessionNotFoundError):
        service.send("no-such-session", ENERGY_QUESTION)
    assert fake.requests == []


def test_unexpected_bug_becomes_generic_error(monkeypatch: pytest.MonkeyPatch) -> None:
    service, _ = make_service()
    sid = service.create_session()

    def boom(*_: object, **__: object) -> None:
        raise RuntimeError("internal detail that must not leak")

    monkeypatch.setattr("app.guardrails.pipeline.g1.check_input", boom)
    result = service.send(sid, ENERGY_QUESTION)
    assert (result.reply, result.blocked, result.reason_code, result.stage) == (
        responses.GENERIC_ERROR, False, "error", "internal")
    assert "internal detail" not in result.reply


def test_one_log_line_per_turn_with_codes_only(caplog: pytest.LogCaptureFixture) -> None:
    secret_answer = "Answer text that must not be logged."
    service, _ = make_service(verdict_json(), answer_json(secret_answer))
    sid = service.create_session()

    with caplog.at_level(logging.INFO, logger="app.guardrails.pipeline"):
        service.send(sid, ENERGY_QUESTION)

    lines = [r.getMessage() for r in caplog.records if r.name == "app.guardrails.pipeline"]
    assert len(lines) == 1
    line = lines[0]
    assert f"session={_session_tag(sid)}" in line
    assert "stage=g4" in line and "reason=ok" in line and "latency_ms=" in line
    everything = "\n".join(r.getMessage() for r in caplog.records)
    for forbidden in (sid, ENERGY_QUESTION, secret_answer, CANARY):
        assert forbidden not in everything


def test_refusal_log_names_the_internal_reason(caplog: pytest.LogCaptureFixture) -> None:
    service, _ = make_service()
    sid = service.create_session()
    with caplog.at_level(logging.INFO, logger="app.guardrails.pipeline"):
        service.send(sid, G1_ATTACK)
    line = [r.getMessage() for r in caplog.records if r.name == "app.guardrails.pipeline"][0]
    assert "stage=g1" in line and "public=blocked" in line
    assert G1_ATTACK not in line


def test_process_canary_is_created_once() -> None:
    assert _process_canary() == _process_canary()
    assert _process_canary().startswith("cnry-")
