"""Manual smoke test: one real Gemini call, to check that the key and model work.

Run from the project root:  python scripts/smoke_test.py

WHY this exists: unit tests use fakes and never touch the network, so they can't tell
you whether your real .env works. This spends exactly one request of quota.

WHY it doesn't use the guardrail pipeline: this checks the LLM wiring only, and the
question is fixed and on-topic. It is a developer tool, never a user-facing path.

Safety: it never prints the API key or a raw traceback. On failure it prints only the
error type and our own fixed message.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# `python scripts/smoke_test.py` puts scripts/ on sys.path, not the project root,
# so `import app` would fail without this line.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from app.config import Settings  # noqa: E402
from app.llm.client import ChatMessage, GeminiClient, LLMRequest  # noqa: E402
from app.llm.errors import LLMError  # noqa: E402

QUESTION = "What is a kilowatt-hour?"


def load_settings() -> Settings | None:
    """Load settings, reporting only the names of bad fields.

    WHY not print the ValidationError itself: pydantic includes the input values in
    its message, and those can contain the API key.
    """
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, e["loc"])).upper() for e in exc.errors())
        print(f"Config error: missing or invalid setting(s) in .env: {fields}")
        return None


def main() -> int:
    # WHY: Windows consoles default to a legacy code page, so characters like "×"
    # in model replies print as "�". UTF-8 output fixes that.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    # Show the client's retry log lines (type, status, attempt, delay only; no secrets).
    # WHY scope INFO to "app" only: third-party libraries (httpx, google_genai) log
    # every request at INFO, which buries our lines. They still show warnings.
    logging.basicConfig(level=logging.WARNING, format="  [log] %(message)s")
    logging.getLogger("app").setLevel(logging.INFO)
    settings = load_settings()
    if settings is None:
        return 2
    client = GeminiClient.from_settings(settings)
    request = LLMRequest(
        model=settings.answer_model,
        system_instruction="You are a concise assistant for energy questions.",
        messages=[ChatMessage(role="user", text=QUESTION)],
    )
    print(f"Model: {settings.answer_model}")
    print(f"Q: {QUESTION}")
    try:
        reply = client.generate(request)
    except LLMError as exc:
        status = f" (HTTP {exc.status_code})" if exc.status_code else ""
        print(f"FAILED: {type(exc).__name__}{status}: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - never show a raw traceback
        print(f"FAILED: unexpected {type(exc).__name__}")
        return 1
    print(f"A: {reply.text}")
    print(f"(finish reason: {reply.finish_reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
