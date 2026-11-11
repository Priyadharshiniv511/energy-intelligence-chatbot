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
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.conversation.store import SessionNotFoundError
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
    # WHY extra="forbid": unexpected fields are a sign of a confused or probing client.
    model_config = ConfigDict(extra="forbid")

    message: str = Field(max_length=MAX_MESSAGE_CHARS)


class SessionOut(BaseModel):
    session_id: str


class ReplyOut(BaseModel):
    """Only public-safe fields. `stage` from PipelineResult is deliberately absent."""

    reply: str
    blocked: bool
    reason_code: str


def _detail(code: int, text: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={"detail": text})


# ---- App factory -----------------------------------------------------------------------

def create_app(
    service: ChatService | None = None,
    rate_limiter: IpRateLimiter | None = None,
) -> FastAPI:
    """Build the API.

    With no `service`, the lifespan handler builds one from settings at startup.
    WHY at startup and not on the first request: two first requests arriving together
    could each build their own store, splitting sessions between them. WHY not at
    import time: importing this module (in tests, or for tooling) must not need a .env.
    Tests pass their own service, so settings are never read.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            settings = get_settings()
            app.state.service = build_service(settings)
            app.state.rate_limiter = rate_limiter or IpRateLimiter(settings.api_rate_limit_per_minute)
        else:
            app.state.service = service
            app.state.rate_limiter = rate_limiter or IpRateLimiter(DEFAULT_RATE_LIMIT_PER_MINUTE)
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
    def create_session(request: Request) -> SessionOut:
        return SessionOut(session_id=_service(request).create_session())

    @app.post("/sessions/{session_id}/messages", response_model=ReplyOut)
    def send_message(session_id: str, body: MessageIn, request: Request) -> ReplyOut | JSONResponse:
        limiter: IpRateLimiter = request.app.state.rate_limiter
        if not limiter.allow(request.client.host if request.client else "unknown"):
            return _detail(status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests.")
        result = _service(request).send(session_id, body.message)
        return ReplyOut(reply=result.reply, blocked=result.blocked, reason_code=result.reason_code)

    @app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_session(session_id: str, request: Request) -> Response:
        _service(request).delete_session(session_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)


def _service(request: Request) -> ChatService:
    return request.app.state.service


app = create_app()
