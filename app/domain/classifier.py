"""G2: the LLM topic and intent classifier (DESIGN.md 2.2).

WHY a separate LLM call before the answer: G1's regexes match wording, but attacks are
about meaning: disguised off-topic questions, paraphrased or non-English injections,
multi-turn drift. A second model with its own small prompt judges each message. It has
nothing to leak and can't be hijacked by the same text that targets the answerer.

WHY fail closed: the classifier is a gate. If it times out, errors, or replies with
anything that isn't exactly our schema, we can't know the message is safe, so we refuse.
A wrong refusal costs a retry; a wrong answer can leak or go off-scope.

The decision rule is a pure function (`decide`) so it can be tested exhaustively
without any model.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings
from app.domain.prompts import CLASSIFIER_SYSTEM_PROMPT, build_classifier_user_content
from app.guardrails.input_filter import InputFilterResult
from app.llm.errors import LLMError
from app.llm.types import ChatMessage, LLMClient, LLMRequest

logger = logging.getLogger(__name__)

# How many previous accepted user turns the classifier sees. WHY 3: enough to resolve
# follow-ups ("and in winter?") and notice drift, small enough that a long, slowly
# drifting conversation can't drown out the latest message.
HISTORY_TURNS = 3

# WHY 0: classification should be as repeatable as possible. Same input, same verdict.
CLASSIFIER_TEMPERATURE = 0.0


class Intent(StrEnum):
    ANSWER = "answer"
    INJECTION = "injection"
    IDENTITY_CHANGE = "identity_change"
    PROMPT_EXTRACTION = "prompt_extraction"
    GREETING = "greeting"
    OFF_TOPIC = "off_topic"


class ClassifierVerdict(BaseModel):
    """The model's JSON reply, validated strictly.

    WHY strict + extra="forbid": a reply with a wrong type ("true" as a string), an
    unknown field or a missing field means the model didn't follow the format, so we
    shouldn't trust its content either. Validation fails and the message is refused.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    on_topic: bool
    intent: Intent
    disguised: bool
    confidence: float = Field(ge=0.0, le=1.0)
    language: str


# Sent to Gemini as `response_json_schema`. Hand-written and flat (no $defs/$ref/
# additionalProperties) because pydantic's generated schema uses $defs for the enum,
# and a flat schema is the most widely supported form. A unit test checks that it
# matches ClassifierVerdict, so the two can't drift apart.
CLASSIFIER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "on_topic": {"type": "boolean"},
        "intent": {"type": "string", "enum": [i.value for i in Intent]},
        "disguised": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "language": {"type": "string"},
    },
    "required": ["on_topic", "intent", "disguised", "confidence", "language"],
}


class Decision(StrEnum):
    ALLOW = "allow"        # go on to the answerer (G3)
    GREETING = "greeting"  # reply with the canned capability message, no answer call
    REFUSE = "refuse"      # reply with the fixed refusal


class Reason(StrEnum):
    """Stable codes for why a decision was made (for logs, tests and the strike counter)."""

    OK = "ok"
    GREETING = "greeting"
    G1_BLOCKED = "g1_blocked"
    LOW_CONFIDENCE = "low_confidence"
    DISGUISED = "disguised"
    NOT_ON_TOPIC = "not_on_topic"
    INJECTION = "injection"
    IDENTITY_CHANGE = "identity_change"
    PROMPT_EXTRACTION = "prompt_extraction"
    OFF_TOPIC = "off_topic"
    LLM_ERROR = "llm_error"
    INVALID_OUTPUT = "invalid_output"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class Classification:
    """G2's verdict. `verdict` is None when we refused without a usable model reply."""

    decision: Decision
    reason: Reason
    verdict: ClassifierVerdict | None = None


def _refuse(reason: Reason, verdict: ClassifierVerdict | None = None) -> Classification:
    return Classification(Decision.REFUSE, reason, verdict)


def decide(verdict: ClassifierVerdict, *, threshold: float, g1_blocked: bool = False) -> Classification:
    """Turn a model verdict into a decision. Anything not explicitly allowed is refused.

    Order matters: hostile intents are checked before confidence, so the logged reason
    names the attack rather than just "low confidence".
    """
    if g1_blocked:
        return _refuse(Reason.G1_BLOCKED, verdict)
    if verdict.intent is Intent.GREETING:
        if verdict.confidence >= threshold:
            return Classification(Decision.GREETING, Reason.GREETING, verdict)
        return _refuse(Reason.LOW_CONFIDENCE, verdict)
    if verdict.intent is not Intent.ANSWER:
        return _refuse(Reason(verdict.intent.value), verdict)
    if not verdict.on_topic:
        return _refuse(Reason.NOT_ON_TOPIC, verdict)
    # WHY refuse on disguised even when "on_topic": the model is contradicting itself,
    # and an unsure gate must refuse.
    if verdict.disguised:
        return _refuse(Reason.DISGUISED, verdict)
    if verdict.confidence < threshold:
        return _refuse(Reason.LOW_CONFIDENCE, verdict)
    return Classification(Decision.ALLOW, Reason.OK, verdict)


class TopicClassifier:
    """Runs one classifier call per message and applies `decide`."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        model: str,
        threshold: float,
        nonce_factory: Callable[[], str] = lambda: secrets.token_hex(8),
    ) -> None:
        """WHY inject `llm` and `nonce_factory`: tests pass a FakeLLMClient (no network)
        and a fixed nonce, so they can find the exact data-block boundaries in the prompt.
        """
        self._llm = llm
        self._model = model
        self._threshold = threshold
        self._nonce_factory = nonce_factory

    @classmethod
    def from_settings(cls, llm: LLMClient, settings: Settings) -> TopicClassifier:
        return cls(llm, model=settings.classifier_model,
                   threshold=settings.classifier_confidence_threshold)

    def classify(self, g1: InputFilterResult, history: Sequence[str]) -> Classification:
        """Classify the latest message, given G1's result and earlier accepted user turns.

        `history` holds plain user texts, oldest first. WHY strings rather than Turn
        objects: the domain layer stays independent of the conversation layer.
        """
        if not g1.allowed:
            # G1 already found a strong attack. Don't spend quota on it. The pipeline
            # normally stops before this, so this is a second lock on the same door.
            return self._log(_refuse(Reason.G1_BLOCKED))
        try:
            response = self._llm.generate(self._build_request(g1, history))
            verdict = ClassifierVerdict.model_validate_json(response.text)
        except LLMError as exc:
            logger.warning("G2 classifier call failed: %s", type(exc).__name__)
            return self._log(_refuse(Reason.LLM_ERROR))
        except ValidationError:
            # Not even the raw reply is logged: it may echo the user's text.
            return self._log(_refuse(Reason.INVALID_OUTPUT))
        except Exception as exc:  # noqa: BLE001 - fail closed on anything unexpected
            logger.error("G2 classifier unexpected error: %s", type(exc).__name__)
            return self._log(_refuse(Reason.INTERNAL_ERROR))
        return self._log(decide(verdict, threshold=self._threshold))

    def _build_request(self, g1: InputFilterResult, history: Sequence[str]) -> LLMRequest:
        content = build_classifier_user_content(
            g1.normalized_text,
            history=list(history)[-HISTORY_TURNS:],
            flags=g1.flags,
            decoded_segments=g1.decoded_segments,
            nonce=self._nonce_factory(),
        )
        return LLMRequest(
            model=self._model,
            system_instruction=CLASSIFIER_SYSTEM_PROMPT,
            messages=[ChatMessage(role="user", text=content)],
            temperature=CLASSIFIER_TEMPERATURE,
            response_schema=CLASSIFIER_RESPONSE_SCHEMA,
        )

    @staticmethod
    def _log(result: Classification) -> Classification:
        """Log the outcome with codes only, never the message text (it may hold secrets)."""
        logger.info("G2 decision=%s reason=%s", result.decision, result.reason)
        return result
