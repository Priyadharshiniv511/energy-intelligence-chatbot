"""Architecture rule (DESIGN.md 6.4): only the guardrail pipeline may reach the real LLM client.

WHY a test: "every LLM call goes through the guardrails" is easy to break by accident
(a quick helper that imports GeminiClient directly). This test fails the build if that
happens.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[2] / "app"
ALLOWED_IMPORTERS = {APP_DIR / "guardrails" / "pipeline.py"}
# client.py defines GeminiClient; pipeline.py is the one place that builds it.
ALLOWED_GEMINI_MENTIONS = ALLOWED_IMPORTERS | {APP_DIR / "llm" / "client.py"}


def _app_modules() -> list[Path]:
    return sorted(APP_DIR.rglob("*.py"))


def _imports_llm_client(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "app.llm.client" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "app.llm.client":
                return True
            names = {alias.name for alias in node.names}
            if node.module == "app.llm" and "client" in names:
                return True  # from app.llm import client
            if node.module == "app" and "llm" in names:
                return True  # from app import llm  (then llm.client)
            if node.level > 0 and node.module in ("client", "llm.client"):
                return True  # relative imports of the same module
    return False


def test_app_modules_are_found() -> None:
    assert len(_app_modules()) > 10  # guards against a wrong path silently passing


def test_only_pipeline_imports_the_llm_client() -> None:
    offenders = [
        str(path.relative_to(APP_DIR))
        for path in _app_modules()
        if path not in ALLOWED_IMPORTERS
        and _imports_llm_client(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert offenders == []


def test_pipeline_does_import_the_llm_client() -> None:
    pipeline = APP_DIR / "guardrails" / "pipeline.py"
    assert _imports_llm_client(ast.parse(pipeline.read_text(encoding="utf-8")))


def _uses_gemini_client(tree: ast.AST) -> bool:
    """True if code (not comments or docstrings) refers to the name GeminiClient."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "GeminiClient":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "GeminiClient":
            return True
        if isinstance(node, ast.alias) and node.name == "GeminiClient":
            return True
    return False


def test_gemini_client_is_only_used_in_pipeline_and_client() -> None:
    offenders = [
        str(path.relative_to(APP_DIR))
        for path in _app_modules()
        if path not in ALLOWED_GEMINI_MENTIONS
        and _uses_gemini_client(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert offenders == []
