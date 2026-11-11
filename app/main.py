"""FastAPI shell: POST /sessions, POST /sessions/{id}/messages, DELETE /sessions/{id}, GET /health.

A thin layer over ChatService (DESIGN.md 2.1). It adds only HTTP concerns: request
validation, status codes, a per-IP rate limit, and error bodies that never expose
internals (no stack traces, no exception text, no echoed input, no guardrail names).

Run with: uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import describe_config_error, get_settings
from app.conversation.store import SessionLimitError, SessionNotFoundError
from app.http_rate_limit import IpRateLimiter
from app.service import ChatService, build_service

logger = logging.getLogger(__name__)

# Transport-level cap. WHY larger than G1's 2,000-char limit: G1 owns the real limit
# and answers with a proper reply; this cap only stops absurd payloads before they are
# parsed and normalised.
MAX_MESSAGE_CHARS = 4000

# Used when a service is injected without a limiter (tests). Matches the setting's default.
DEFAULT_RATE_LIMIT_PER_MINUTE = 20


# ---- Request / response models ---------------------------------------------------------

class MessageIn(BaseModel):
    """Body of POST /sessions/{id}/messages."""

    # WHY extra="forbid": unexpected fields are a sign of a confused or probing client.
    model_config = ConfigDict(extra="forbid")

    message: str = Field(max_length=MAX_MESSAGE_CHARS)


class SessionOut(BaseModel):
    """Body returned by POST /sessions."""

    session_id: str


class ReplyOut(BaseModel):
    """Only public-safe fields. `stage` from PipelineResult is deliberately absent."""

    reply: str
    blocked: bool
    reason_code: str


def _detail(code: int, text: str) -> JSONResponse:
    """A fixed {"detail": ...} error body, never built from exception or user text."""
    return JSONResponse(status_code=code, content={"detail": text})


# ---- App factory -----------------------------------------------------------------------

def create_app(
    service: ChatService | None = None,
    rate_limiter: IpRateLimiter | None = None,
    session_rate_limiter: IpRateLimiter | None = None,
) -> FastAPI:
    """Build the API.

    With no `service`, the lifespan handler builds one from settings at startup.
    WHY at startup and not on the first request: two first requests arriving together
    could each build their own store, splitting sessions between them. WHY not at
    import time: importing this module (in tests, or for tooling) must not need a .env.
    Tests pass their own service, so settings are never read.

    `rate_limiter` limits messages per IP; `session_rate_limiter` limits session
    creation per IP. WHY two: creating sessions must not use up a client's message
    budget, but it must be limited, or a loop of POST /sessions could fill memory.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            try:
                settings = get_settings()
            except ValidationError as exc:
                # WHY: uvicorn would print pydantic's full error, which includes the
                # values read from .env. Name the bad settings only, then stop.
                message = describe_config_error(exc)
                logger.error(message)
                raise RuntimeError(message) from None
            app.state.service = build_service(settings)
            limit = settings.api_rate_limit_per_minute
        else:
            app.state.service = service
            limit = DEFAULT_RATE_LIMIT_PER_MINUTE
        app.state.rate_limiter = rate_limiter or IpRateLimiter(limit)
        app.state.session_rate_limiter = session_rate_limiter or IpRateLimiter(limit)
        yield

    app = FastAPI(title="Energy Intelligence Chatbot", lifespan=lifespan)
    _add_error_handlers(app)
    _add_routes(app)
    return app


def _add_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(SessionNotFoundError)
    async def not_found(_: Request, __: SessionNotFoundError) -> JSONResponse:
        # Same text for unknown, expired and deleted ids (see SessionNotFoundError).
        return _detail(status.HTTP_404_NOT_FOUND, "Session not found.")

    @app.exception_handler(SessionLimitError)
    async def full(_: Request, __: SessionLimitError) -> JSONResponse:
        # Capacity, not the client's fault: "try again later", nothing internal.
        return _detail(status.HTTP_503_SERVICE_UNAVAILABLE, "Service busy, try again later.")

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, __: RequestValidationError) -> JSONResponse:
        # WHY replace FastAPI's default 422 body: it echoes the submitted input back,
        # which could reflect an attacker's payload or a user's private text.
        return _detail(422, "Invalid request.")  # literal: the constant's name differs across Starlette versions

    @app.exception_handler(Exception)
    async def crashed(_: Request, exc: Exception) -> JSONResponse:
        # Type name only: the exception text could hold user input or internals.
        logger.error("unhandled API error: %s", type(exc).__name__)
        return _detail(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error.")


def _add_routes(app: FastAPI) -> None:
    # Sync `def` handlers: FastAPI runs them in a thread pool, which suits the store's
    # thread locks and the synchronous Gemini client.

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness only. No LLM call and no config details."""
        return {"status": "ok"}

    @app.post("/sessions", status_code=status.HTTP_201_CREATED, response_model=SessionOut)
    def create_session(request: Request) -> SessionOut | JSONResponse:
        """Start a conversation. Rate limited per IP so sessions can't flood memory."""
        if not request.app.state.session_rate_limiter.allow(_client_ip(request)):
            return _detail(status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests.")
        return SessionOut(session_id=_service(request).create_session())

    @app.post("/sessions/{session_id}/messages", response_model=ReplyOut)
    def send_message(session_id: str, body: MessageIn, request: Request) -> ReplyOut | JSONResponse:
        """Send one message through the guardrail pipeline (via ChatService)."""
        if not request.app.state.rate_limiter.allow(_client_ip(request)):
            return _detail(status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests.")
        result = _service(request).send(session_id, body.message)
        return ReplyOut(reply=result.reply, blocked=result.blocked, reason_code=result.reason_code)

    @app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_session(session_id: str, request: Request) -> Response:
        """Forget a conversation. Unknown ids get the same 404 as everywhere else."""
        _service(request).delete_session(session_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)


def _service(request: Request) -> ChatService:
    """The ChatService built at startup (or injected by a test)."""
    return request.app.state.service


def _client_ip(request: Request) -> str:
    """Socket IP of the caller. Behind a proxy this is the proxy (see Known limitations)."""
    return request.client.host if request.client else "unknown"


app = create_app()
