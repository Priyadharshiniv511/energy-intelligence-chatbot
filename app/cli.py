"""CLI REPL: creates one session and calls the same ChatService as the API.

Run with: python -m app.cli
Commands: /reset (start a new conversation), /quit (exit).

Only replies are printed: no reason codes, guardrail stages or stack traces, for the
same reason the API hides them (they'd help an attacker tune their next attempt).
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from pydantic import ValidationError

from app.config import describe_config_error, get_settings
from app.conversation.store import SessionNotFoundError
from app.guardrails import responses
from app.service import ChatService, build_service

logger = logging.getLogger(__name__)

PROMPT = "you> "
WELCOME = "Energy assistant. Ask about electricity, renewables, efficiency, grids or bills. /reset, /quit"
RESET_NOTICE = "Started a new conversation."
EXPIRED_NOTICE = "Your session expired, so I started a new conversation."


def configure_logging() -> None:
    """Keep log lines off the chat screen.

    WHY: with no logging setup, Python prints warnings straight to the terminal, so a
    Gemini outage showed lines like "LLM call failed ... status=503" between replies.
    Users should see replies only. Set CLI_LOG_FILE to keep the logs in a file instead
    (they hold codes only, never message text).
    """
    root = logging.getLogger()
    if root.handlers:
        return  # someone (e.g. a test runner) already decided where logs go
    log_file = os.environ.get("CLI_LOG_FILE")
    if log_file:
        logging.basicConfig(
            filename=log_file, level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    else:
        root.addHandler(logging.NullHandler())


def main(
    service: ChatService | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Run the REPL until /quit or end of input. Returns a process exit code.

    WHY injectable input/output: tests script a whole session without a terminal.
    """
    if service is None:
        try:
            service = build_service(get_settings())
        except ValidationError as exc:
            # Setting names only: the full error can include the values that were read.
            output_fn(describe_config_error(exc))
            return 1
    output_fn(WELCOME)
    session_id = service.create_session()
    while True:
        try:
            line = input_fn(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not line:
            continue
        if line == "/quit":
            return 0
        if line == "/reset":
            session_id = _reset(service, session_id)
            output_fn(RESET_NOTICE)
            continue
        session_id = _send(service, session_id, line, output_fn)


def _send(service: ChatService, session_id: str, line: str,
          output_fn: Callable[[str], None]) -> str:
    """Send one message and print the reply. Returns the (possibly new) session id."""
    try:
        try:
            output_fn(service.send(session_id, line).reply)
        except SessionNotFoundError:
            # The session timed out while the user was idle; carry on in a fresh one.
            session_id = service.create_session()
            output_fn(EXPIRED_NOTICE)
            output_fn(service.send(session_id, line).reply)
    except Exception as exc:  # noqa: BLE001 - never show a stack trace to the user
        logger.error("CLI error: %s", type(exc).__name__)
        output_fn(responses.GENERIC_ERROR)
    return session_id


def _reset(service: ChatService, session_id: str) -> str:
    """Drop the old conversation and start an empty one."""
    try:
        service.delete_session(session_id)
    except SessionNotFoundError:
        pass  # already expired: nothing to delete
    return service.create_session()


if __name__ == "__main__":
    configure_logging()
    raise SystemExit(main())
