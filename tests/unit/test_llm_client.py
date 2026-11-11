"""Unit tests for app/llm/client.py. A FakeSDK simulates every failure: no network calls."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.llm.client import ChatMessage, GeminiClient, LLMClient, LLMRequest, RetryPolicy
from app.llm.errors import (
    LLMBlocked,
    LLMEmpty,
    LLMError,
    LLMRateLimited,
    LLMRequestError,
    LLMTimeout,
    LLMUnavailable,
)
from tests import fakes
from tests.fakes import FakeLLMClient, FakeSDK

REQUEST = LLMRequest(
    model="test-model",
    system_instruction="You answer energy questions.",
    messages=[ChatMessage(role="user", text="How do solar panels work?")],
)


def make_client(sdk: FakeSDK, *, max_attempts: int = 3, max_delay: float = 8.0,
                rng_value: float = 0.5) -> tuple[GeminiClient, list[float]]:
    """GeminiClient with a recording sleep (tests run instantly) and a fixed rng."""
    sleeps: list[float] = []
    client = GeminiClient(
        sdk,
        timeout_seconds=15,
        retry=RetryPolicy(max_attempts=max_attempts, base_delay=1.0, max_delay=max_delay),
        sleep=sleeps.append,
        rng=lambda: rng_value,
    )
    return client, sleeps


# ---- Happy path and request shape ---------------------------------------------------

def test_success_returns_text_and_finish_reason() -> None:
    client, sleeps = make_client(FakeSDK(fakes.ok_response("Hello grid")))
    result = client.generate(REQUEST)
    assert result.text == "Hello grid"
    assert result.finish_reason == "STOP"
    assert sleeps == []


def test_every_call_carries_a_timeout_and_the_request_settings() -> None:
    sdk = FakeSDK(fakes.ok_response())
    client, _ = make_client(sdk)
    client.generate(REQUEST)
    call = sdk.calls[0]
    assert call["model"] == "test-model"
    assert call["config"].http_options.timeout == 15_000  # milliseconds
    assert call["config"].system_instruction == "You answer energy questions."
    assert call["contents"][0].role == "user"
    assert call["contents"][0].parts[0].text == "How do solar panels work?"


def test_response_schema_switches_on_json_mode() -> None:
    sdk = FakeSDK(fakes.ok_response('{"on_topic": true}'))
    client, _ = make_client(sdk)
    schema = {"type": "object", "properties": {"on_topic": {"type": "boolean"}}}
    client.generate(LLMRequest(model="m", system_instruction="s",
                               messages=REQUEST.messages, response_schema=schema))
    config = sdk.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema


# ---- Retryable failures -------------------------------------------------------------

@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(fakes.timeout_error(), id="network-timeout"),
        pytest.param(fakes.api_error(408), id="408"),
        pytest.param(fakes.api_error(429), id="429"),
        pytest.param(fakes.api_error(500), id="500"),
        pytest.param(fakes.api_error(503), id="503"),
    ],
)
def test_transient_failure_is_retried_then_succeeds(failure: Exception) -> None:
    sdk = FakeSDK(failure, fakes.ok_response("recovered"))
    client, sleeps = make_client(sdk)
    assert client.generate(REQUEST).text == "recovered"
    assert len(sdk.calls) == 2
    assert len(sleeps) == 1


@pytest.mark.parametrize(
    ("failure_factory", "expected"),
    [
        (fakes.timeout_error, LLMTimeout),
        (lambda: fakes.api_error(429), LLMRateLimited),
        (lambda: fakes.api_error(503), LLMUnavailable),
    ],
)
def test_gives_up_after_max_attempts_with_typed_error(failure_factory, expected) -> None:
    sdk = FakeSDK(*(failure_factory() for _ in range(3)))
    client, sleeps = make_client(sdk, max_attempts=3)
    with pytest.raises(expected):
        client.generate(REQUEST)
    assert len(sdk.calls) == 3
    assert len(sleeps) == 2  # no sleep after the final attempt


def test_max_attempts_is_configurable() -> None:
    sdk = FakeSDK(*(fakes.api_error(503) for _ in range(5)))
    client, _ = make_client(sdk, max_attempts=5)
    with pytest.raises(LLMUnavailable):
        client.generate(REQUEST)
    assert len(sdk.calls) == 5


def test_max_attempts_one_means_no_retry() -> None:
    sdk = FakeSDK(fakes.api_error(503))
    client, sleeps = make_client(sdk, max_attempts=1)
    with pytest.raises(LLMUnavailable):
        client.generate(REQUEST)
    assert sleeps == []


@pytest.mark.parametrize(
    ("code", "expected"),
    [(408, LLMTimeout), (504, LLMTimeout), (500, LLMUnavailable), (502, LLMUnavailable),
     (503, LLMUnavailable)],
)
def test_status_codes_map_to_the_right_error(code: int, expected: type) -> None:
    sdk = FakeSDK(fakes.api_error(code))
    client, _ = make_client(sdk, max_attempts=1)
    with pytest.raises(expected) as info:
        client.generate(REQUEST)
    assert info.value.status_code == code


def test_network_error_maps_to_unavailable() -> None:
    import httpx
    sdk = FakeSDK(httpx.ConnectError("dns failure"))
    client, _ = make_client(sdk, max_attempts=1)
    with pytest.raises(LLMUnavailable):
        client.generate(REQUEST)


# ---- Backoff ------------------------------------------------------------------------

def test_backoff_grows_exponentially_with_jitter() -> None:
    sdk = FakeSDK(*(fakes.api_error(503) for _ in range(4)), fakes.ok_response())
    client, sleeps = make_client(sdk, max_attempts=5, max_delay=100, rng_value=0.5)
    client.generate(REQUEST)
    # step = 1, 2, 4, 8; delay = step/2 + 0.5 * step/2 = 0.75 * step
    assert sleeps == [0.75, 1.5, 3.0, 6.0]


def test_backoff_is_capped_at_max_delay() -> None:
    policy = RetryPolicy(max_attempts=10, base_delay=1.0, max_delay=4.0)
    assert policy.backoff(attempt=8, rng=lambda: 1.0) == 4.0


def test_jitter_spreads_delays() -> None:
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0)
    low, high = policy.backoff(3, lambda: 0.0), policy.backoff(3, lambda: 1.0)
    assert (low, high) == (2.0, 4.0)


def test_rate_limit_honours_retry_after_header() -> None:
    sdk = FakeSDK(fakes.api_error(429, headers={"Retry-After": "2"}), fakes.ok_response())
    client, sleeps = make_client(sdk, rng_value=0.0)
    client.generate(REQUEST)
    assert sleeps == [2.0]


def test_rate_limit_honours_google_retry_info() -> None:
    info = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3s"}]
    sdk = FakeSDK(fakes.api_error(429, details=info), fakes.ok_response())
    client, sleeps = make_client(sdk, rng_value=0.0)
    client.generate(REQUEST)
    assert sleeps == [3.0]


def test_rate_limit_with_long_retry_after_fails_fast() -> None:
    """If the server says 'wait 60 s', don't make the user hang: fail now."""
    sdk = FakeSDK(fakes.api_error(429, headers={"Retry-After": "60"}))
    client, sleeps = make_client(sdk, max_delay=8.0)
    with pytest.raises(LLMRateLimited) as info:
        client.generate(REQUEST)
    assert info.value.retry_after == 60.0
    assert sleeps == []
    assert len(sdk.calls) == 1


# ---- Non-retryable failures ---------------------------------------------------------

@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_are_not_retried(code: int) -> None:
    sdk = FakeSDK(fakes.api_error(code))
    client, sleeps = make_client(sdk)
    with pytest.raises(LLMRequestError) as info:
        client.generate(REQUEST)
    assert info.value.status_code == code
    assert len(sdk.calls) == 1
    assert sleeps == []


def test_unexpected_exception_is_wrapped_and_not_retried() -> None:
    sdk = FakeSDK(ValueError(fakes.RAW_SECRET_DETAIL))
    client, _ = make_client(sdk)
    with pytest.raises(LLMError) as info:
        client.generate(REQUEST)
    assert type(info.value) is LLMError
    assert len(sdk.calls) == 1


# ---- Empty and blocked responses ----------------------------------------------------

@pytest.mark.parametrize(
    ("response_factory", "expected"),
    [
        (fakes.blocked_prompt_response, LLMBlocked),
        (fakes.blocked_answer_response, LLMBlocked),
        (fakes.no_candidates_response, LLMEmpty),
        (fakes.whitespace_response, LLMEmpty),
        (fakes.thought_only_response, LLMEmpty),
    ],
)
def test_empty_or_blocked_responses_raise_typed_errors(response_factory, expected) -> None:
    sdk = FakeSDK(response_factory())
    client, _ = make_client(sdk)
    with pytest.raises(expected):
        client.generate(REQUEST)
    assert len(sdk.calls) == 1  # not retried: the same input would be blocked again


# ---- No leaks -----------------------------------------------------------------------

@pytest.mark.parametrize("code", [400, 429, 503])
def test_raw_api_error_text_never_leaks(code: int) -> None:
    sdk = FakeSDK(*(fakes.api_error(code) for _ in range(3)))
    client, _ = make_client(sdk)
    with pytest.raises(LLMError) as info:
        client.generate(REQUEST)
    assert fakes.RAW_SECRET_DETAIL not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True  # raw SDK error hidden from traceback


def test_settings_repr_hides_the_api_key() -> None:
    settings = Settings(_env_file=None, gemini_api_key="AIza-super-secret",
                        answer_model="a", classifier_model="c")
    assert "AIza-super-secret" not in repr(settings)
    assert "AIza-super-secret" not in str(settings.model_dump())


def test_from_settings_builds_client_without_network() -> None:
    settings = Settings(_env_file=None, gemini_api_key="fake-key", answer_model="a",
                        classifier_model="c", llm_timeout_seconds=7, llm_max_attempts=4)
    client = GeminiClient.from_settings(settings)
    assert client._timeout_ms == 7_000
    assert client._retry.max_attempts == 4


# ---- The interface ------------------------------------------------------------------

def test_fake_llm_client_satisfies_the_interface() -> None:
    fake: LLMClient = FakeLLMClient("an answer", LLMTimeout())
    assert fake.generate(REQUEST).text == "an answer"
    with pytest.raises(LLMTimeout):
        fake.generate(REQUEST)
    assert len(fake.requests) == 2
