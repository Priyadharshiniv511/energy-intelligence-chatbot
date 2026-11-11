"""Our own typed LLM errors.

WHY: raw SDK exceptions carry Google's full error payload (request details, internal
status text). If one reached a user, it would leak implementation details. Every SDK
failure is mapped to one of these classes. Their messages are fixed, safe strings, and
the guardrail pipeline turns them into friendly replies.

`retryable` tells the client's retry loop whether trying again could help.
"""

from __future__ import annotations

from typing import ClassVar


class LLMError(Exception):
    """Base class. Catch this to handle any LLM failure."""

    retryable: ClassVar[bool] = False
    default_message: ClassVar[str] = "The language model request failed."

    def __init__(self, message: str | None = None, *, status_code: int | None = None) -> None:
        super().__init__(message or self.default_message)
        # The HTTP status alone is safe to keep: it helps logging without leaking payloads.
        self.status_code = status_code


class LLMTimeout(LLMError):
    """The call took longer than the configured timeout."""

    retryable = True
    default_message = "The language model did not respond in time."


class LLMRateLimited(LLMError):
    """HTTP 429: we sent too many requests or ran out of quota."""

    retryable = True
    default_message = "The language model is rate limited."

    def __init__(
        self,
        message: str | None = None,
        *,
        status_code: int | None = 429,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code)
        # Seconds the server asked us to wait, if it said so.
        self.retry_after = retry_after


class LLMUnavailable(LLMError):
    """HTTP 5xx or a network failure: the service is down or overloaded."""

    retryable = True
    default_message = "The language model service is unavailable."


class LLMRequestError(LLMError):
    """HTTP 4xx other than 408/429: our request is wrong (bad key, bad model, bad input).

    Not retryable: sending the same bad request again gives the same answer.
    """

    default_message = "The language model rejected the request."


class LLMBlocked(LLMError):
    """The provider's safety filters blocked the prompt or the answer."""

    default_message = "The language model blocked this content."


class LLMEmpty(LLMError):
    """The call succeeded but returned no usable text."""

    default_message = "The language model returned an empty response."
