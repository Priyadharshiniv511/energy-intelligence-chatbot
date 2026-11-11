"""Fallback model tests: the wrapper on its own, config, and the full pipeline. FakeLLM only."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.guardrails import responses
from app.guardrails.output_filter import secret_values_from_settings
from app.llm.errors import (
    LLMBlocked,
    LLMEmpty,
    LLMError,
    LLMRateLimited,
    LLMRequestError,
    LLMTimeout,
    LLMUnavailable,
)
from app.llm.fallback import FallbackLLMClient
from app.llm.types import ChatMessage, LLMRequest
from tests.fakes import FakeLLMClient
from tests.unit.pipeline_helpers import (
    ANSWER_MODEL,
    CLASSIFIER_MODEL,
    ENERGY_ANSWER,
    ENERGY_QUESTION,
    answer_json,
    make_service_with_fallbacks,
    make_settings,
    verdict_json,
)

MAIN, BACKUP = "gemini-main", "gemini-backup"


def request(model: str = MAIN) -> LLMRequest:
    return LLMRequest(model=model, system_instruction="sys", temperature=0.3,
                      messages=[ChatMessage("user", "hi")], response_schema={"type": "object"})


# ---- The wrapper ---------------------------------------------------------------------------

@pytest.mark.parametrize("error", [LLMUnavailable(), LLMTimeout(), LLMRateLimited()])
def test_overload_errors_switch_to_the_backup(error: LLMError) -> None:
    fake = FakeLLMClient(error, "backup reply")
    reply = FallbackLLMClient(fake, {MAIN: BACKUP}).generate(request())
    assert reply.text == "backup reply"
    assert [r.model for r in fake.requests] == [MAIN, BACKUP]


def test_backup_request_is_identical_apart_from_the_model() -> None:
    fake = FakeLLMClient(LLMUnavailable(), "ok")
    FallbackLLMClient(fake, {MAIN: BACKUP}).generate(request())
    first, second = fake.requests
    assert second.model == BACKUP
    assert (second.system_instruction, second.messages, second.temperature, second.response_schema) == (
        first.system_instruction, first.messages, first.temperature, first.response_schema)


@pytest.mark.parametrize("error", [LLMRequestError(), LLMBlocked(), LLMEmpty(), LLMError()])
def test_other_errors_do_not_switch_models(error: LLMError) -> None:
    fake = FakeLLMClient(error)
    with pytest.raises(type(error)):
        FallbackLLMClient(fake, {MAIN: BACKUP}).generate(request())
    assert len(fake.requests) == 1


def test_backup_failure_is_raised_and_not_retried_again() -> None:
    fake = FakeLLMClient(LLMUnavailable(), LLMUnavailable())
    with pytest.raises(LLMUnavailable):
        FallbackLLMClient(fake, {MAIN: BACKUP}).generate(request())
    assert len(fake.requests) == 2


def test_model_without_a_backup_just_fails() -> None:
    fake = FakeLLMClient(LLMUnavailable())
    with pytest.raises(LLMUnavailable):
        FallbackLLMClient(fake, {MAIN: BACKUP}).generate(request("gemini-other"))
    assert len(fake.requests) == 1


def test_backup_equal_to_main_is_ignored() -> None:
    fake = FakeLLMClient(LLMUnavailable())
    with pytest.raises(LLMUnavailable):
        FallbackLLMClient(fake, {MAIN: MAIN}).generate(request())
    assert len(fake.requests) == 1


# ---- Config --------------------------------------------------------------------------------

def test_fallbacks_default_to_none_and_empty_means_none() -> None:
    assert make_settings().answer_fallback_model is None
    settings = Settings(_env_file=None, gemini_api_key="k", answer_model="a", classifier_model="c",
                        answer_fallback_model="", classifier_fallback_model="  ")  # type: ignore[call-arg]
    assert (settings.answer_fallback_model, settings.classifier_fallback_model) == (None, None)


def test_fallback_model_typo_fails_at_startup() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, gemini_api_key="k", answer_model="a", classifier_model="c",
                 answer_fallback_model="Gemini Flash")  # type: ignore[call-arg]


def test_fallback_model_names_are_treated_as_secrets_by_g4() -> None:
    settings = make_settings().model_copy(update={"answer_fallback_model": "gemini-backup-x"})
    assert "gemini-backup-x" in secret_values_from_settings(settings)


# ---- Full pipeline ---------------------------------------------------------------------------

def test_answer_model_overload_is_rescued_by_the_fallback() -> None:
    service, fake = make_service_with_fallbacks(
        verdict_json(), LLMUnavailable(), answer_json(), answer_fallback=BACKUP)
    sid = service.create_session()

    result = service.send(sid, ENERGY_QUESTION)

    assert (result.reply, result.reason_code) == (ENERGY_ANSWER, "ok")
    assert [r.model for r in fake.requests] == [CLASSIFIER_MODEL, ANSWER_MODEL, BACKUP]
    assert len(service.conversation(sid).turns) == 1


def test_classifier_overload_is_rescued_by_its_own_fallback() -> None:
    service, fake = make_service_with_fallbacks(
        LLMUnavailable(), verdict_json(), answer_json(),
        answer_fallback="gemini-answer-backup", classifier_fallback=BACKUP)
    sid = service.create_session()

    result = service.send(sid, ENERGY_QUESTION)

    assert result.reason_code == "ok"
    assert [r.model for r in fake.requests] == [CLASSIFIER_MODEL, BACKUP, ANSWER_MODEL]


def test_both_models_down_still_gives_the_fixed_unavailable_message() -> None:
    service, _ = make_service_with_fallbacks(
        verdict_json(), LLMUnavailable(), LLMUnavailable(), answer_fallback=BACKUP)
    sid = service.create_session()
    result = service.send(sid, ENERGY_QUESTION)
    assert (result.reply, result.reason_code) == (responses.SERVICE_UNAVAILABLE, "unavailable")
    assert service.conversation(sid).strikes == 0


def test_fallback_answers_still_go_through_g4() -> None:
    # The backup model leaking its own name must still be blocked.
    service, _ = make_service_with_fallbacks(
        verdict_json(), LLMUnavailable(), answer_json(f"I am {BACKUP}."), answer_fallback=BACKUP)
    sid = service.create_session()
    result = service.send(sid, ENERGY_QUESTION)
    assert (result.blocked, result.reason_code) == (True, "blocked")
    assert BACKUP not in result.reply
