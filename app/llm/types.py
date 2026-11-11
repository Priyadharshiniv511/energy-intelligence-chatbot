"""The LLM interface and its request/response shapes, with no SDK imports.

WHY a separate module: DESIGN.md 6.4 says only the guardrail pipeline may import
`app.llm.client` (the module that builds the real GeminiClient). Callers such as the
G2 classifier still need the *interface* to send a request. Keeping it here lets them
depend on the shape of a call without being able to construct a real client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

Role = Literal["user", "model"]


@dataclass(frozen=True)
class ChatMessage:
    """One turn of a conversation, in the shape Gemini expects ('user' or 'model')."""

    role: Role
    text: str


@dataclass(frozen=True)
class LLMRequest:
    """Everything needed for one model call.

    WHY a request object rather than many arguments: the classifier and the answerer
    each build one, and a fake client can record and inspect them in tests.
    """

    model: str
    system_instruction: str
    messages: Sequence[ChatMessage]
    temperature: float = 0.2
    # If set, Gemini must answer with JSON that matches this JSON Schema.
    response_schema: dict[str, Any] | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    """The usable text of a reply, plus the reason the model stopped (e.g. 'STOP')."""

    text: str
    finish_reason: str | None = None


class LLMClient(Protocol):
    """What the rest of the app needs from an LLM. Fakes implement this too."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Return the model's reply, or raise an `LLMError` subclass."""
        ...
