"""G4: deterministic checks on every reply before the user sees it (DESIGN.md 2.2).

WHY deterministic code after the model: G1-G3 all depend on either patterns or a model
following instructions. If a novel jailbreak gets through them, this is the last line,
and plain code can't be talked out of its checks (§4.9).

Order of checks: provider failures, then leaks (canary, secrets, prompt echo), then the
model's own topic report, then the confidence caveat. Leaks come first so the logged
reason names the worst problem.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from pydantic import SecretStr

from app.config import Settings
from app.guardrails import responses
from app.guardrails.answerer import AnswerAttempt, AnswerFailure, Confidence

logger = logging.getLogger(__name__)

# Echo check: word n-grams of this length shared with the system prompt.
# WHY 8 words and at least 2 shared: a shared 8-gram run of 2 means 9+ consecutive prompt
# words were copied. The scope phrases a normal energy answer might reuse ("greenhouse-gas
# emissions from energy generation and use") are at most about 8 words, while any real
# echo of a rule sentence is much longer.
ECHO_NGRAM = 8
ECHO_MIN_SHARED_NGRAMS = 2

# Google API keys: "AIza" followed by 35 URL-safe characters.
_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
_ENV_NAME_RE = re.compile(r"GEMINI_API_KEY", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")

# Settings that are not SecretStr but still reveal implementation details (§4.5).
# WHY an explicit list: guessing "any long string setting" would also catch harmless
# values (e.g. "production") and block normal answers that use those words.
_SENSITIVE_PLAIN_SETTINGS = (
    "answer_model", "classifier_model", "answer_fallback_model", "classifier_fallback_model",
)


class Action(StrEnum):
    DELIVER = "deliver"  # show the (possibly caveated) model answer
    REFUSE = "refuse"    # show the fixed refusal
    BLOCK = "block"      # a leak was caught; show the fixed refusal
    ERROR = "error"      # no usable answer; show a fixed error


class Reason(StrEnum):
    OK = "ok"
    LOW_CONFIDENCE = "low_confidence"
    CANARY_LEAK = "canary_leak"
    PROMPT_ECHO = "prompt_echo"
    SECRET_LEAK = "secret_leak"
    OFF_TOPIC_SELF_REPORT = "off_topic_self_report"
    EMPTY = "empty"
    PROVIDER_BLOCKED = "provider_blocked"
    MALFORMED = "malformed"
    LLM_UNAVAILABLE = "llm_unavailable"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class OutputResult:
    """`text` is always exactly what the user will see."""

    action: Action
    reason: Reason
    text: str


_FAILURE_MAP: dict[AnswerFailure, tuple[Reason, str]] = {
    AnswerFailure.LLM_UNAVAILABLE: (Reason.LLM_UNAVAILABLE, responses.SERVICE_UNAVAILABLE),
    AnswerFailure.BLOCKED: (Reason.PROVIDER_BLOCKED, responses.GENERIC_ERROR),
    AnswerFailure.EMPTY: (Reason.EMPTY, responses.GENERIC_ERROR),
    AnswerFailure.MALFORMED: (Reason.MALFORMED, responses.GENERIC_ERROR),
    AnswerFailure.INTERNAL: (Reason.INTERNAL_ERROR, responses.GENERIC_ERROR),
}


def secret_values_from_settings(settings: Settings) -> tuple[str, ...]:
    """Values that must never appear in a reply: every SecretStr setting plus the
    model names. Nothing else (see _SENSITIVE_PLAIN_SETTINGS)."""
    values: list[str] = []
    for name in type(settings).model_fields:
        value = getattr(settings, name)
        if isinstance(value, SecretStr):
            values.append(value.get_secret_value())
        elif name in _SENSITIVE_PLAIN_SETTINGS and isinstance(value, str):
            values.append(value)
    return tuple(v for v in values if v)


def _squash(text: str) -> str:
    """Lowercase and drop everything but letters and digits.

    WHY: catches leaks written with spaces, dashes or line breaks in between
    ("c n r y - 1 2 ...").
    """
    return _NON_ALNUM_RE.sub("", text.lower())


def _contains_canary(answer: str, canary: str) -> bool:
    """Look for the random part only (after the last "-"), so a leak that drops the
    "cnry-" prefix is still caught. 32 random hex chars never appear by chance."""
    core = _squash(canary.rsplit("-", 1)[-1])
    return bool(core) and core in _squash(answer)


def _contains_secret(answer: str, secret_values: Sequence[str]) -> bool:
    if _GOOGLE_KEY_RE.search(answer) or _ENV_NAME_RE.search(answer):
        return True
    squashed = _squash(answer)
    for value in secret_values:
        if value in answer:
            return True
        # WHY a length floor for the squashed form: very short values squash to a few
        # letters that could appear by chance.
        squashed_value = _squash(value)
        if len(squashed_value) >= 8 and squashed_value in squashed:
            return True
    return False


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _echoes_prompt(answer: str, system_prompt: str) -> bool:
    shared = _ngrams(answer, ECHO_NGRAM) & _ngrams(system_prompt, ECHO_NGRAM)
    return len(shared) >= ECHO_MIN_SHARED_NGRAMS


def check_output(
    attempt: AnswerAttempt,
    *,
    canary: str,
    system_prompt: str,
    secret_values: Sequence[str],
) -> OutputResult:
    """Decide what the user sees for one G3 attempt."""
    return _log(_decide(attempt, canary=canary, system_prompt=system_prompt,
                        secret_values=secret_values))


def _decide(attempt: AnswerAttempt, *, canary: str, system_prompt: str,
            secret_values: Sequence[str]) -> OutputResult:
    payload = attempt.payload
    if payload is None:
        reason, text = _FAILURE_MAP.get(
            attempt.failure or AnswerFailure.INTERNAL, (Reason.INTERNAL_ERROR, responses.GENERIC_ERROR))
        return OutputResult(Action.ERROR, reason, text)
    answer = payload.answer
    if not answer.strip():
        return OutputResult(Action.ERROR, Reason.EMPTY, responses.GENERIC_ERROR)
    if _contains_canary(answer, canary):
        return OutputResult(Action.BLOCK, Reason.CANARY_LEAK, responses.REFUSAL)
    if _contains_secret(answer, secret_values):
        return OutputResult(Action.BLOCK, Reason.SECRET_LEAK, responses.REFUSAL)
    if _echoes_prompt(answer, system_prompt):
        return OutputResult(Action.BLOCK, Reason.PROMPT_ECHO, responses.REFUSAL)
    if not payload.on_topic:
        return OutputResult(Action.REFUSE, Reason.OFF_TOPIC_SELF_REPORT, responses.REFUSAL)
    if payload.confidence is Confidence.LOW:
        return OutputResult(Action.DELIVER, Reason.LOW_CONFIDENCE,
                            f"{responses.LOW_CONFIDENCE_CAVEAT}\n\n{answer.strip()}")
    return OutputResult(Action.DELIVER, Reason.OK, answer.strip())


def _log(result: OutputResult) -> OutputResult:
    """Codes only: never the answer text (it might be the very leak we just caught)."""
    logger.info("G4 action=%s reason=%s", result.action, result.reason)
    return result
