"""Test doubles shared by all test suites. None of them touch the network.

- FakeSDK imitates the google-genai client (`sdk.models.generate_content`), so we can
  test GeminiClient's real retry and error-mapping code against scripted failures.
- FakeLLMClient implements our LLMClient interface directly. Later pipeline and
  adversarial tests use it instead of GeminiClient.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import httpx
from google.genai import errors as genai_errors
from google.genai import types

from app.llm.client import LLMRequest, LLMResponse
from app.llm.errors import LLMError


# ---- Builders for realistic SDK responses and errors ---------------------------------

def ok_response(text: str = "Solar panels convert sunlight into electricity.") -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )


def blocked_prompt_response() -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[],
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.SAFETY
        ),
    )


def blocked_answer_response() -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=None, finish_reason=types.FinishReason.SAFETY)]
    )


def no_candidates_response() -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[])


def whitespace_response() -> types.GenerateContentResponse:
    return ok_response(text="   \n ")


def thought_only_response() -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model", parts=[types.Part(text="secret reasoning", thought=True)]
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )


RAW_SECRET_DETAIL = "INTERNAL-DETAIL-projects/123/locations/x should never reach users"


def api_error(code: int, details: list[dict[str, Any]] | None = None,
              headers: dict[str, str] | None = None) -> genai_errors.APIError:
    """A real SDK error with a payload that includes text we must never leak."""
    body = {"error": {"code": code, "message": RAW_SECRET_DETAIL, "status": "X",
                      "details": details or []}}
    response = httpx.Response(code, headers=headers or {})
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, body, response)


def timeout_error() -> httpx.TimeoutException:
    return httpx.ReadTimeout("timed out")


# ---- Fakes ---------------------------------------------------------------------------

class _FakeModels:
    def __init__(self, outcomes: Iterable[Any]) -> None:
        self._outcomes = deque(outcomes)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self._outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeSDK:
    """Plays back a script of outcomes: each item is a response to return or an
    exception to raise, one per call."""

    def __init__(self, *outcomes: Any) -> None:
        self.models = _FakeModels(outcomes)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.models.calls


class FakeLLMClient:
    """LLMClient for higher-level tests: returns scripted texts or raises LLMErrors,
    and records every request so tests can inspect what would have been sent."""

    def __init__(self, *outcomes: str | LLMError) -> None:
        self._outcomes = deque(outcomes)
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        outcome = self._outcomes.popleft()
        if isinstance(outcome, LLMError):
            raise outcome
        return LLMResponse(text=outcome, finish_reason="STOP")
