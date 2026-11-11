"""G3: the answer-generation call (DESIGN.md 2.2).

One call to ANSWER_MODEL, with the energy-only system prompt (with a canary),
this session's accepted history only, and structured output. The answer is NOT shown
to the user from here: G4 (output_filter.py) checks it first.

WHY never raise: like G2, a failure here must not escape as a raw exception (it could
carry provider details). Every failure becomes an `AnswerFailure` code, and G4 maps it
to a fixed message.
"""

from __future__ import annotations

import logging
import re
import secrets
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

from app.config import Settings
from app.conversation.models import Turn
from app.domain.prompts import (
    DEFAULT_LANGUAGE_LINE,
    build_answerer_system_prompt,
    build_answerer_user_content,
    quote_previous_user_turn,
)
from app.llm.errors import LLMBlocked, LLMEmpty, LLMError
from app.llm.types import ChatMessage, LLMClient, LLMRequest

logger = logging.getLogger(__name__)

# WHY 10 (DESIGN.md 3): bounds tokens and cost, and limits how much a slowly drifting
# conversation can steer the model.
HISTORY_TURNS = 10

# WHY 0.2: low randomness makes invented numbers less likely (§4.7), while keeping
# answers natural.
ANSWER_TEMPERATURE = 0.2

# ISO 639-1 code, optionally with a region/script subtag ("en", "pt-BR").
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})?$", re.IGNORECASE)


def new_canary() -> str:
    """Make a fresh canary. The pipeline calls this once per process.

    WHY random and per process: a fixed canary in source could be learned and
    filtered around; a random one only exists in memory.
    """
    return "cnry-" + secrets.token_hex(16)


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnswerPayload(BaseModel):
    """The model's JSON reply, validated strictly (same reasoning as ClassifierVerdict)."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    answer: str
    on_topic: bool
    confidence: Confidence


# Flat, hand-written schema like G2's. A unit test keeps it in sync with AnswerPayload.
ANSWER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "on_topic": {"type": "boolean"},
        "confidence": {"type": "string", "enum": [c.value for c in Confidence]},
    },
    "required": ["answer", "on_topic", "confidence"],
}


class AnswerFailure(StrEnum):
    """Why G3 produced no usable payload."""

    LLM_UNAVAILABLE = "llm_unavailable"  # timeout, rate limit, 5xx, rejected request
    BLOCKED = "blocked"                  # provider safety filter
    EMPTY = "empty"                      # no text at all
    MALFORMED = "malformed"              # text that doesn't match our schema
    INTERNAL = "internal"                # anything unexpected


@dataclass(frozen=True)
class AnswerAttempt:
    """Exactly one of `payload` and `failure` is set."""

    payload: AnswerPayload | None = None
    failure: AnswerFailure | None = None


def language_line(language: str | None) -> str:
    """Turn G2's language field into the line sent to the answerer.

    WHY validate: the value was written by an LLM that read user text, so it could
    carry an injected instruction. Only a short language code gets through.
    """
    if language and _LANGUAGE_RE.fullmatch(language.strip()):
        return f"Reply language: {language.strip().lower()}"
    return DEFAULT_LANGUAGE_LINE


class Answerer:
    """Runs the G3 call. Holds the system prompt (and so the canary) for this process."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        model: str,
        canary: str,
        nonce_factory: Callable[[], str] = lambda: secrets.token_hex(8),
    ) -> None:
        """WHY inject `llm`, `canary` and `nonce_factory`: tests use a fake client, a
        known canary and a fixed nonce; the pipeline passes one canary per process."""
        self._llm = llm
        self._model = model
        self._system_prompt = build_answerer_system_prompt(canary)
        self._nonce_factory = nonce_factory

    @classmethod
    def from_settings(cls, llm: LLMClient, settings: Settings, canary: str) -> Answerer:
        return cls(llm, model=settings.answer_model, canary=canary)

    def __repr__(self) -> str:
        # WHY: the default repr is harmless, but this makes sure a debug print of the
        # object can never show the prompt or canary.
        return f"Answerer(model={self._model!r})"

    @property
    def system_prompt(self) -> str:
        """The exact prompt sent to the model. G4 needs it for the echo check."""
        return self._system_prompt

    def answer(self, message: str, history: Sequence[Turn], language: str | None) -> AnswerAttempt:
        """Generate an answer. `history` must be the session's accepted turns only."""
        try:
            response = self._llm.generate(self._build_request(message, history, language))
            payload = AnswerPayload.model_validate_json(response.text)
        except LLMBlocked:
            return self._fail(AnswerFailure.BLOCKED)
        except LLMEmpty:
            return self._fail(AnswerFailure.EMPTY)
        except LLMError as exc:
            logger.warning("G3 answer call failed: %s", type(exc).__name__)
            return self._fail(AnswerFailure.LLM_UNAVAILABLE)
        except ValidationError:
            # The raw reply is not logged: it could contain the canary or user text.
            return self._fail(AnswerFailure.MALFORMED)
        except Exception as exc:  # noqa: BLE001 - fail closed on anything unexpected
            logger.error("G3 unexpected error: %s", type(exc).__name__)
            return self._fail(AnswerFailure.INTERNAL)
        logger.info("G3 answered on_topic=%s confidence=%s", payload.on_topic, payload.confidence)
        return AnswerAttempt(payload=payload)

    def _build_request(self, message: str, history: Sequence[Turn],
                       language: str | None) -> LLMRequest:
        nonce = self._nonce_factory()
        messages: list[ChatMessage] = []
        for turn in list(history)[-HISTORY_TURNS:]:
            messages.append(ChatMessage("user", quote_previous_user_turn(turn.user_text, nonce=nonce)))
            messages.append(ChatMessage("model", turn.answer_text))
        messages.append(ChatMessage("user", build_answerer_user_content(
            message, language_line=language_line(language), nonce=nonce)))
        return LLMRequest(
            model=self._model,
            system_instruction=self._system_prompt,
            messages=messages,
            temperature=ANSWER_TEMPERATURE,
            response_schema=ANSWER_RESPONSE_SCHEMA,
        )

    @staticmethod
    def _fail(failure: AnswerFailure) -> AnswerAttempt:
        logger.info("G3 failure=%s", failure)
        return AnswerAttempt(failure=failure)
