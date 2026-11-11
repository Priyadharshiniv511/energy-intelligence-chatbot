"""THE only path to the LLM: runs G1 input filter, G2 classifier, G3 answer, G4 output filter.

Flow for one user turn (DESIGN.md 2.2):

    store.get -> G1 -> G2 -> G3 -> G4 -> store.append_turn (only on success)

Rules this module enforces:
- `build_pipeline` is the only place in app/ that builds a GeminiClient (DESIGN.md 6.4).
  A test checks that no other app module imports `app.llm.client`.
- Refusals (G1, G2, G4 block/refuse) add a strike and return a fixed text. The turn is
  never stored, so a manipulation attempt can't build up in the history.
- Outages and errors (the model unavailable, malformed output, a bug) return a fixed
  text with no strike: the user did nothing wrong. Nothing is answered, so this still
  fails closed.
- One structured log line per turn, made of codes only: never the message, the reply,
  the canary or the prompt.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from functools import cache
from typing import Callable, Sequence

from app.config import Settings
from app.conversation.store import SessionNotFoundError, SessionStore
from app.domain import classifier as g2
from app.domain.classifier import Decision, TopicClassifier
from app.guardrails import input_filter as g1
from app.guardrails import output_filter as g4
from app.guardrails import responses
from app.guardrails.answerer import Answerer, new_canary
from app.llm.client import GeminiClient
from app.llm.fallback import FallbackLLMClient
from app.llm.types import LLMClient

logger = logging.getLogger(__name__)


# ---- Public result ---------------------------------------------------------------------

# Coarse, public-safe reason codes. WHY coarse: the fine-grained internal reasons
# ("canary_leak", "prompt_extraction"...) would tell an attacker exactly which check
# caught them, which helps them tune the next attempt. They stay in the logs only.
OK = "ok"
OUT_OF_SCOPE = "out_of_scope"
BLOCKED = "blocked"
INVALID_INPUT = "invalid_input"
UNAVAILABLE = "unavailable"
ERROR = "error"


@dataclass(frozen=True)
class PipelineResult:
    """What one turn produced.

    `reply`, `blocked` and `reason_code` may be shown to users. `stage` is for logs
    only: the API and CLI never output it.
    """

    reply: str
    blocked: bool
    reason_code: str
    stage: str


# ---- Internal reason -> public reason code ---------------------------------------------

_G1_PUBLIC: dict[g1.Reason, str] = {
    g1.Reason.EMPTY: INVALID_INPUT,
    g1.Reason.TOO_LONG: INVALID_INPUT,
    g1.Reason.INJECTION: BLOCKED,
    g1.Reason.PROMPT_LEAK: BLOCKED,
    g1.Reason.SECRET_REQUEST: BLOCKED,
    g1.Reason.ENCODED_INJECTION: BLOCKED,
}

_G2_PUBLIC: dict[g2.Reason, str] = {
    g2.Reason.NOT_ON_TOPIC: OUT_OF_SCOPE,
    g2.Reason.OFF_TOPIC: OUT_OF_SCOPE,
    g2.Reason.DISGUISED: OUT_OF_SCOPE,
    g2.Reason.LOW_CONFIDENCE: OUT_OF_SCOPE,
    g2.Reason.INVALID_OUTPUT: OUT_OF_SCOPE,
    g2.Reason.INJECTION: BLOCKED,
    g2.Reason.IDENTITY_CHANGE: BLOCKED,
    g2.Reason.PROMPT_EXTRACTION: BLOCKED,
    g2.Reason.G1_BLOCKED: BLOCKED,
}

# G2 refusals that are outages or bugs, not the user's fault: a fixed error text, no
# strike (the user's choice). The message is still not answered, so this fails closed.
_G2_NO_STRIKE: dict[g2.Reason, tuple[str, str]] = {
    g2.Reason.LLM_ERROR: (UNAVAILABLE, responses.SERVICE_UNAVAILABLE),
    g2.Reason.INTERNAL_ERROR: (ERROR, responses.GENERIC_ERROR),
}


def _g4_public(result: g4.OutputResult) -> str:
    if result.action is g4.Action.DELIVER:
        return OK
    if result.action is g4.Action.BLOCK:
        return BLOCKED
    if result.action is g4.Action.REFUSE:
        return OUT_OF_SCOPE
    return UNAVAILABLE if result.reason is g4.Reason.LLM_UNAVAILABLE else ERROR


def _session_tag(session_id: str) -> str:
    """Short hash of the session id for logs.

    WHY hash: the id is the only credential for a conversation, so a raw id in a log
    file would let anyone who reads the logs join that conversation. The hash still
    lets you follow one session across log lines.
    """
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]


# ---- The pipeline ----------------------------------------------------------------------

class GuardrailPipeline:
    """Runs every guardrail stage for one turn and records the outcome in the store."""

    def __init__(
        self,
        store: SessionStore,
        classifier: TopicClassifier,
        answerer: Answerer,
        *,
        canary: str,
        secret_values: Sequence[str],
        max_input_chars: int,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """WHY everything is injected: tests build the pipeline with a FakeLLM-backed
        classifier and answerer, and a known canary, without touching the network."""
        self._store = store
        self._classifier = classifier
        self._answerer = answerer
        self._canary = canary
        self._secret_values = tuple(secret_values)
        self._max_input_chars = max_input_chars
        self._clock = clock

    def __repr__(self) -> str:
        # Never show the canary, prompt or secret values in a debug print.
        return "GuardrailPipeline()"

    def handle(self, session_id: str, text: str) -> PipelineResult:
        """Run one user turn through G1 -> G4.

        Raises SessionNotFoundError for an unknown session, before any LLM call, so the
        API can answer 404. Any other exception becomes a fixed error reply: a bug must
        never let an unchecked answer (or a stack trace) through.
        """
        started = self._clock()
        try:
            result, reason = self._run(session_id, text)
        except SessionNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 - fail closed on anything unexpected
            logger.error("pipeline unexpected error: %s", type(exc).__name__)
            result = PipelineResult(responses.GENERIC_ERROR, False, ERROR, "internal")
            reason = "internal_error"
        self._log(session_id, result, reason, started)
        return result

    # ---- stages -------------------------------------------------------------------
    # Each stage returns (result, internal_reason). WHY a return value and not an
    # attribute: one pipeline serves concurrent requests, so per-turn state must not
    # live on `self`.

    def _run(self, session_id: str, text: str) -> tuple[PipelineResult, str]:
        conversation = self._store.get(session_id)

        checked = g1.check_input(text, max_chars=self._max_input_chars)
        if not checked.allowed:
            code = _G1_PUBLIC.get(checked.reason, BLOCKED)
            return self._refuse(session_id, "g1", code), checked.reason.value

        history = [turn.user_text for turn in conversation.turns]
        classification = self._classifier.classify(checked, history)
        reason = classification.reason.value
        if classification.decision is Decision.GREETING:
            # Canned text, no answer call, not stored: nothing to remember.
            return PipelineResult(responses.GREETING, False, OK, "g2"), reason
        if classification.decision is not Decision.ALLOW:
            if classification.reason in _G2_NO_STRIKE:
                code, reply = _G2_NO_STRIKE[classification.reason]
                return PipelineResult(reply, False, code, "g2"), reason
            # Unknown reasons default to BLOCKED: an unmapped refusal must still refuse.
            code = _G2_PUBLIC.get(classification.reason, BLOCKED)
            return self._refuse(session_id, "g2", code), reason

        language = classification.verdict.language if classification.verdict else None
        attempt = self._answerer.answer(checked.normalized_text, conversation.turns, language)
        output = g4.check_output(
            attempt,
            canary=self._canary,
            system_prompt=self._answerer.system_prompt,
            secret_values=self._secret_values,
        )
        return self._finish(session_id, checked.normalized_text, output), output.reason.value

    def _finish(self, session_id: str, user_text: str, output: g4.OutputResult) -> PipelineResult:
        """Act on G4's verdict: store a delivered turn, strike a block or refusal."""
        code = _g4_public(output)
        if output.action is g4.Action.DELIVER:
            # Store the *normalised* text: the history replayed to G2/G3 later must not
            # bring back invisible characters or homoglyphs that G1 removed.
            self._store.append_turn(session_id, user_text, output.text)
            return PipelineResult(output.text, False, code, "g4")
        if output.action in (g4.Action.BLOCK, g4.Action.REFUSE):
            self._store.record_strike(session_id)
            return PipelineResult(output.text, True, code, "g4")
        return PipelineResult(output.text, False, code, "g4")  # ERROR: no strike

    def _refuse(self, session_id: str, stage: str, code: str) -> PipelineResult:
        self._store.record_strike(session_id)
        return PipelineResult(responses.REFUSAL, True, code, stage)

    def _log(self, session_id: str, result: PipelineResult, reason: str, started: float) -> None:
        """The one log line per turn: codes and timing only."""
        latency_ms = int((self._clock() - started) * 1000)
        logger.info(
            "turn session=%s stage=%s reason=%s public=%s latency_ms=%d",
            _session_tag(session_id), result.stage, reason, result.reason_code, latency_ms,
        )


# ---- Factory: the only place a real LLM client is built ------------------------------

@cache
def _process_canary() -> str:
    """One canary per process (DESIGN.md 2.2).

    WHY cached: every pipeline in this process must check for the same canary that is
    in the answerer's prompt, and a fresh one per request would add nothing.
    """
    return new_canary()


def build_pipeline(
    settings: Settings,
    store: SessionStore,
    *,
    llm: LLMClient | None = None,
    canary: str | None = None,
) -> GuardrailPipeline:
    """Build the full pipeline from settings.

    `llm` and `canary` exist for tests only: they drive this real factory with a
    FakeLLM and a known canary. In the app both are None, so the real GeminiClient
    and the process canary are used.
    """
    client: LLMClient = llm if llm is not None else GeminiClient.from_settings(settings)
    canary = canary or _process_canary()
    return GuardrailPipeline(
        store,
        TopicClassifier.from_settings(
            _with_fallback(client, settings.classifier_model, settings.classifier_fallback_model),
            settings),
        Answerer.from_settings(
            _with_fallback(client, settings.answer_model, settings.answer_fallback_model),
            settings, canary),
        canary=canary,
        secret_values=g4.secret_values_from_settings(settings),
        max_input_chars=settings.max_input_chars,
    )


def _with_fallback(client: LLMClient, model: str, fallback: str | None) -> LLMClient:
    """Wrap `client` so `model` falls back to `fallback` when overloaded.

    WHY one wrapper per role: the classifier and answerer may share a main model but
    have different backups, so each gets its own mapping.
    """
    if not fallback or fallback == model:
        return client
    return FallbackLLMClient(client, {model: fallback})
