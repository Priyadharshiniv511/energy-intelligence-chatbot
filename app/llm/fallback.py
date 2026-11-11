"""FallbackLLMClient: try a backup model when the main one is overloaded.

WHY: on the free tier, Gemini often answers 503 ("model overloaded") for one model
while another model is fine. The main client already retries with backoff; if every
retry still fails with an error a *different* model could avoid, we send the same
request once more to the configured backup model (which gets its own retries).

Which errors switch models:
- LLMUnavailable (503 and other 5xx), LLMTimeout, LLMRateLimited (429). Overload and
  quota are per model, so another model may well succeed.
Which errors don't:
- LLMRequestError (400/403: bad key or bad request), LLMBlocked (safety filter),
  LLMEmpty. The backup would most likely fail the same way, and for a blocked prompt
  we must not go shopping for a model that will answer it.

The client-side rate limiter lives in the inner GeminiClient, not here, because this
wrapper makes no real calls itself. If the main model's bucket is empty (LLMRateLimited
from our own limiter), the backup model is tried once with its own bucket.

This wrapper only changes `request.model`. Prompt, history, schema and temperature are
identical, and every reply still goes through G4, so guardrails don't change.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Mapping

from app.llm.errors import LLMRateLimited, LLMTimeout, LLMUnavailable
from app.llm.types import LLMClient, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

FALLBACK_ERRORS = (LLMUnavailable, LLMTimeout, LLMRateLimited)


class FallbackLLMClient:
    """`LLMClient` that retries a failed request on a backup model."""

    def __init__(self, inner: LLMClient, fallbacks: Mapping[str, str]) -> None:
        """`fallbacks` maps a main model name to its backup, e.g. {answer: backup}.

        Pairs where the backup equals the main model are dropped: retrying the same
        overloaded model is what the inner client's retries already did.
        """
        self._inner = inner
        self._fallbacks = {main: backup for main, backup in fallbacks.items() if backup != main}

    def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            return self._inner.generate(request)
        except FALLBACK_ERRORS as exc:
            backup = self._fallbacks.get(request.model)
            if backup is None:
                raise
            # Codes only: model names stay out of logs as well as replies.
            logger.warning("LLM primary failed (%s), trying fallback model", type(exc).__name__)
        return self._inner.generate(dataclasses.replace(request, model=backup))
