"""ChatService: the single entry point that both the CLI and the REST API call.

WHY a service layer with no logic of its own: the API and the CLI must behave the same
way. If each called the store and pipeline itself, a guardrail fix in one shell could
be missed in the other. Every message goes through `GuardrailPipeline.handle`; this
class has no other route to the LLM (DESIGN.md 6.4).
"""

from __future__ import annotations

from app.config import Settings
from app.conversation.models import Conversation
from app.conversation.store import InMemorySessionStore, SessionStore
from app.guardrails.pipeline import GuardrailPipeline, PipelineResult, build_pipeline
from app.llm.types import LLMClient


class ChatService:
    """Creates, uses and deletes chat sessions."""

    def __init__(self, store: SessionStore, pipeline: GuardrailPipeline) -> None:
        self._store = store
        self._pipeline = pipeline

    def create_session(self) -> str:
        """Start a new, empty conversation and return its id."""
        return self._store.create()

    def send(self, session_id: str, text: str) -> PipelineResult:
        """Run one message through the guardrails. Unknown ids raise SessionNotFoundError."""
        return self._pipeline.handle(session_id, text)

    def conversation(self, session_id: str) -> Conversation:
        """Read-only snapshot of a session (accepted turns and strike count).

        WHY expose it: tests and future admin tooling can check what was stored
        without reaching into the store. It is a copy, so callers can't edit history.
        """
        return self._store.get(session_id)

    def delete_session(self, session_id: str) -> None:
        """Forget a conversation. Unknown ids raise SessionNotFoundError."""
        self._store.delete(session_id)


def build_service(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    canary: str | None = None,
) -> ChatService:
    """Wire one store and one pipeline together.

    `llm` and `canary` are for tests (a FakeLLM and a known canary); the app passes
    neither. WHY the store is built here and passed to the
    pipeline: both must share the *same* store, or answers would be saved somewhere
    the service never reads.
    """
    store = InMemorySessionStore()
    return ChatService(store, build_pipeline(settings, store, llm=llm, canary=canary))
