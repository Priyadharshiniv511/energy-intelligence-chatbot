"""LLM client: the only code that talks to the Gemini API.

Layout:
- `LLMClient` is a small interface (a Protocol). The rest of the app depends only on
  it, so tests can swap in a fake and never touch the network or the quota.
- `GeminiClient` implements it with the google-genai SDK and adds what the SDK call
  alone lacks for us: a timeout on every call, our own retry policy, handling of
  empty or safety-blocked replies, and mapping SDK errors to our typed errors.

Per docs/DESIGN.md section 6.4, only the guardrail pipeline may build a GeminiClient.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import Settings
from app.llm.errors import (
    LLMBlocked,
    LLMEmpty,
    LLMError,
    LLMRateLimited,
    LLMRequestError,
    LLMTimeout,
    LLMUnavailable,
)
from app.llm.rate_limiter import RateLimiter
# The interface lives in app.llm.types (so other modules can use it without importing
# this file, DESIGN.md 6.4). Re-exported here so existing imports keep working.
from app.llm.types import ChatMessage, LLMClient, LLMRequest, LLMResponse, Role

try:  # google-genai >= 2 prefers httpx2 (a fork of httpx) when it is installed
    import httpx2
except ImportError:  # pragma: no cover - depends on the installed SDK
    httpx2 = None  # type: ignore[assignment]

__all__ = [
    "ChatMessage", "GeminiClient", "LLMClient", "LLMRequest", "LLMResponse", "RetryPolicy",
    "Role", "map_exception", "parse_response",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """How many times to try, and how long to wait between tries."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 8.0

    @classmethod
    def from_settings(cls, settings: Settings) -> RetryPolicy:
        return cls(
            max_attempts=settings.llm_max_attempts,
            base_delay=settings.llm_backoff_base_seconds,
            max_delay=settings.llm_backoff_max_seconds,
        )

    def backoff(self, attempt: int, rng: Callable[[], float]) -> float:
        """Exponential backoff with 'equal jitter': half fixed, half random.

        WHY jitter: if many requests fail at once and all retry after exactly 1 s,
        2 s and 4 s, they collide again. Random spread avoids that. WHY 'equal' rather
        than fully random: we always wait at least half the step, so a retry never
        fires instantly into the same overload.
        """
        step = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
        return step / 2 + rng() * step / 2


# Shortest attempt worth starting. WHY: an attempt with a few milliseconds left would
# only time out, spending quota for nothing.
_MIN_ATTEMPT_MS = 1000

# Finish reasons meaning "the provider blocked the content", not "the model finished".
_BLOCKED_FINISH_REASONS = frozenset(
    {
        types.FinishReason.SAFETY,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.BLOCKLIST,
        types.FinishReason.SPII,
        types.FinishReason.RECITATION,
    }
)


class GeminiClient:
    """`LLMClient` backed by Google Gemini, with timeouts, retries and error mapping."""

    def __init__(
        self,
        sdk_client: Any,
        *,
        timeout_seconds: float,
        retry: RetryPolicy,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
        rate_limiter: RateLimiter | None = None,
        call_budget_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wrap an SDK client.

        WHY inject `sdk_client`, `sleep` and `rng`: tests pass a fake SDK that raises
        each kind of failure, a `sleep` that only records delays (so tests run
        instantly), and a fixed `rng` (so delays are predictable).
        `rate_limiter` is None by default, so tests that build a client directly are
        never slowed down. `from_settings` (the real app) always attaches one.
        `call_budget_seconds` caps the total time of one `generate` call (None = no
        cap); `clock` is injectable so tests can move time forward.
        """
        self._sdk = sdk_client
        self._timeout_ms = int(timeout_seconds * 1000)  # the SDK takes milliseconds
        self._retry = retry
        self._sleep = sleep
        self._rng = rng
        self._limiter = rate_limiter
        self._budget = call_budget_seconds
        self._clock = clock

    @classmethod
    def from_settings(cls, settings: Settings) -> GeminiClient:
        """Build the real client. The key is read here and stored nowhere else."""
        sdk = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value(),
            http_options=types.HttpOptions(
                timeout=int(settings.llm_timeout_seconds * 1000),
                # Turn off the SDK's own retries: we retry ourselves, and two retry
                # layers stacked would multiply the attempts (3 x 5 = 15 calls).
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        return cls(
            sdk,
            timeout_seconds=settings.llm_timeout_seconds,
            retry=RetryPolicy.from_settings(settings),
            rate_limiter=RateLimiter.from_settings(settings),
            call_budget_seconds=settings.llm_call_budget_seconds,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Call the model, retrying transient failures, and return clean text.

        The whole call (limiter waits, attempts, backoff) stays within the time budget.
        """
        deadline = None if self._budget is None else self._clock() + self._budget
        contents = [
            types.Content(role=m.role, parts=[types.Part(text=m.text)]) for m in request.messages
        ]
        for attempt in range(1, self._retry.max_attempts + 1):
            # One token per attempt: each retry is a real request that uses quota.
            # Outside the `try` on purpose: if our own limiter says "too busy", retrying
            # here would only wait more, so LLMRateLimited goes straight to the caller.
            if self._limiter is not None:
                self._limiter.acquire(request.model)
            config = self._build_config(request, self._attempt_timeout_ms(deadline))
            try:
                raw = self._sdk.models.generate_content(
                    model=request.model, contents=contents, config=config
                )
            except Exception as exc:  # noqa: BLE001 - every failure must be mapped
                error = map_exception(exc)
                self._wait_or_raise(error, attempt, deadline)
                continue
            return parse_response(raw)
        raise AssertionError("unreachable: the loop always returns or raises")

    def _attempt_timeout_ms(self, deadline: float | None) -> int:
        """This attempt's timeout: the normal one, shortened to what is left of the budget.

        Raises LLMTimeout if the budget is already used up (e.g. by a limiter wait).
        """
        if deadline is None:
            return self._timeout_ms
        remaining_ms = int((deadline - self._clock()) * 1000)
        if remaining_ms < _MIN_ATTEMPT_MS:
            logger.warning("LLM call budget used up before an attempt")
            raise LLMTimeout()
        return min(self._timeout_ms, remaining_ms)

    def _build_config(self, request: LLMRequest, timeout_ms: int) -> types.GenerateContentConfig:
        """Per-call config. The timeout is set on every call, not only on the client."""
        return types.GenerateContentConfig(
            system_instruction=request.system_instruction,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            response_mime_type="application/json" if request.response_schema else None,
            response_json_schema=request.response_schema,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

    def _wait_or_raise(self, error: LLMError, attempt: int, deadline: float | None) -> None:
        """Sleep before the next try, or raise if we should not try again.

        `raise ... from None` drops the SDK exception from the traceback, so its raw
        payload cannot surface by accident. We log only the error type and status.
        """
        delay = self._retry_delay(error, attempt)
        if delay is not None and deadline is not None:
            # Don't sleep into a retry that would have (almost) no time left to run.
            if self._clock() + delay + _MIN_ATTEMPT_MS / 1000 > deadline:
                delay = None
        if delay is None:
            logger.warning(
                "LLM call failed, giving up: %s status=%s attempt=%d",
                type(error).__name__, error.status_code, attempt,
            )
            raise error from None
        logger.info(
            "LLM call failed, retrying: %s status=%s attempt=%d delay=%.2fs",
            type(error).__name__, error.status_code, attempt, delay,
        )
        self._sleep(delay)

    def _retry_delay(self, error: LLMError, attempt: int) -> float | None:
        """Seconds to wait before retrying, or None to give up now."""
        if not error.retryable or attempt >= self._retry.max_attempts:
            return None
        retry_after = getattr(error, "retry_after", None)
        if retry_after is not None:
            # The server told us how long to wait. If that is longer than we are
            # willing to keep a user waiting, fail now rather than hang.
            if retry_after > self._retry.max_delay:
                return None
            return retry_after + self._rng() * self._retry.base_delay
        return self._retry.backoff(attempt, self._rng)


def _http_classes(name: str) -> tuple[type[BaseException], ...]:
    """The exception class `name` from httpx and, if installed, httpx2.

    WHY both: newer google-genai versions send requests through httpx2 when it is
    installed. httpx2's exceptions are separate classes (not subclasses of httpx's),
    so checking httpx alone would treat every timeout as an unknown, non-retryable
    error: no retries and no fallback model.
    """
    modules = (httpx, httpx2) if httpx2 is not None else (httpx,)
    return tuple(getattr(m, name) for m in modules if hasattr(m, name))


_TIMEOUT_ERRORS = (*_http_classes("TimeoutException"), TimeoutError)
_TRANSPORT_ERRORS = _http_classes("TransportError")


def map_exception(exc: Exception) -> LLMError:
    """Translate any exception from the SDK or network into our typed errors."""
    if isinstance(exc, _TIMEOUT_ERRORS):
        return LLMTimeout()
    if isinstance(exc, _TRANSPORT_ERRORS):  # DNS failure, connection reset, ...
        return LLMUnavailable()
    if isinstance(exc, genai_errors.APIError):
        return _map_api_error(exc)
    # Unknown: not retried (it may be a bug), and its text is still never exposed.
    logger.error("Unexpected LLM client error: %s", type(exc).__name__)
    return LLMError()


def _map_api_error(exc: genai_errors.APIError) -> LLMError:
    """Map an HTTP error from Gemini by status code."""
    code = exc.code
    if code == 429:
        return LLMRateLimited(retry_after=_retry_after_seconds(exc))
    # 504 counts as a timeout too: the SDK sends our timeout to Google as an
    # X-Server-Timeout header, so a 504 usually means "your own deadline passed".
    if code in (408, 504):
        return LLMTimeout(status_code=code)
    if code is not None and code >= 500:
        return LLMUnavailable(status_code=code)
    return LLMRequestError(status_code=code)


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)s?\s*$")


def _retry_after_seconds(exc: genai_errors.APIError) -> float | None:
    """Find how long the server asked us to wait, if it said.

    Two places to look: the standard HTTP `Retry-After` header, and Google's
    `RetryInfo` detail in the error body (e.g. {"retryDelay": "12s"}).
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        value = _parse_seconds(headers.get("retry-after"))
        if value is not None:
            return value
    details = exc.details if isinstance(exc.details, dict) else {}
    inner = details.get("error", details)
    for item in inner.get("details", []) if isinstance(inner, dict) else []:
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
            return _parse_seconds(item.get("retryDelay"))
    return None


def _parse_seconds(value: Any) -> float | None:
    """Parse '12', '12s' or '1.5s' into seconds. Anything else (e.g. an HTTP date) -> None."""
    if value is None:
        return None
    match = _DURATION_RE.match(str(value))
    return float(match.group(1)) if match else None


def parse_response(raw: types.GenerateContentResponse) -> LLMResponse:
    """Extract usable text, or raise LLMBlocked / LLMEmpty.

    WHY not just `raw.text`: it returns None both for 'blocked' and for 'empty', and
    we treat those two differently. We also skip 'thought' parts (internal reasoning
    from thinking models), which must never be shown to the user.
    """
    feedback = raw.prompt_feedback
    if feedback is not None and feedback.block_reason is not None:
        raise LLMBlocked()  # the prompt itself was blocked; there are no candidates
    if not raw.candidates:
        raise LLMEmpty()
    candidate = raw.candidates[0]
    reason = candidate.finish_reason
    if reason in _BLOCKED_FINISH_REASONS:
        raise LLMBlocked()
    parts = candidate.content.parts if candidate.content and candidate.content.parts else []
    text = "".join(p.text for p in parts if p.text and not p.thought)
    if not text.strip():
        raise LLMEmpty()
    return LLMResponse(text=text, finish_reason=reason.name if reason else None)
