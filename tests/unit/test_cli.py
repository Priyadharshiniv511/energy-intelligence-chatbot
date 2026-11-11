"""CLI REPL tests with scripted input and captured output. FakeLLM only."""

from __future__ import annotations

import logging
from typing import Iterator

import pytest

from app import cli
from app.guardrails import responses
from tests.unit.pipeline_helpers import (
    ENERGY_ANSWER,
    ENERGY_QUESTION,
    OFF_TOPIC_QUESTION,
    answer_json,
    make_service,
    off_topic_verdict,
    verdict_json,
)

INTERNAL_WORDS = ("out_of_scope", "blocked", "reason", "stage", "g1", "g2", "Traceback")


def scripted(lines: list[str]) -> tuple[object, list[str]]:
    """An input function that plays back `lines`, then raises EOFError like a closed stdin."""
    feed: Iterator[str] = iter(lines)

    def input_fn(_: str) -> str:
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    return input_fn, []


def run(service: object, lines: list[str]) -> tuple[int, list[str]]:
    input_fn, out = scripted(lines)
    code = cli.main(service, input_fn=input_fn, output_fn=out.append)  # type: ignore[arg-type]
    return code, out


def test_energy_question_is_answered() -> None:
    service, _ = make_service(verdict_json(), answer_json())
    code, out = run(service, [ENERGY_QUESTION, "/quit"])
    assert code == 0
    assert out[-1] == ENERGY_ANSWER


def test_off_topic_is_refused_without_internal_details() -> None:
    service, _ = make_service(off_topic_verdict())
    _, out = run(service, [OFF_TOPIC_QUESTION, "/quit"])
    assert out[-1] == responses.REFUSAL
    for line in out:
        for word in INTERNAL_WORDS:
            assert word not in line


def test_reset_clears_history() -> None:
    service, fake = make_service(verdict_json(), answer_json(),
                                 verdict_json(), answer_json("Second answer."))
    _, out = run(service, [ENERGY_QUESTION, "/reset", "What is a smart meter?", "/quit"])
    assert cli.RESET_NOTICE in out
    second_answer_request = fake.requests[3]
    # Only the new question: no earlier user/model turns were replayed.
    assert len(second_answer_request.messages) == 1
    assert ENERGY_QUESTION not in second_answer_request.messages[0].text


def test_quit_exits_immediately() -> None:
    service, fake = make_service()
    code, out = run(service, ["/quit", ENERGY_QUESTION])
    assert code == 0
    assert out == [cli.WELCOME]
    assert fake.requests == []


def test_end_of_input_exits_cleanly() -> None:
    service, _ = make_service()
    code, _ = run(service, [])
    assert code == 0


def test_unexpected_error_shows_generic_text(monkeypatch: pytest.MonkeyPatch) -> None:
    service, _ = make_service()

    def boom(*_: object) -> None:
        raise RuntimeError("internal detail")

    monkeypatch.setattr(service, "send", boom)
    _, out = run(service, [ENERGY_QUESTION, "/quit"])
    assert out[-1] == responses.GENERIC_ERROR
    assert all("internal detail" not in line for line in out)


def test_missing_config_prints_one_fixed_line(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GEMINI_API_KEY", "ANSWER_MODEL", "CLASSIFIER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir("tests")  # no .env here
    cli.get_settings.cache_clear()
    try:
        code, out = run(None, ["/quit"])
    finally:
        cli.get_settings.cache_clear()
    assert code == 1
    assert out == [cli.CONFIG_ERROR]


def test_cli_logging_keeps_log_lines_off_the_screen(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(logging.getLogger(), "handlers", [])  # restored after the test
    monkeypatch.delenv("CLI_LOG_FILE", raising=False)
    cli.configure_logging()
    logging.getLogger("app.llm.client").warning("LLM call failed, giving up: status=503")
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_cli_logging_can_go_to_a_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    log_file = tmp_path / "cli.log"
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    monkeypatch.setenv("CLI_LOG_FILE", str(log_file))
    cli.configure_logging()
    logging.getLogger("app.llm.client").warning("status=503")
    for handler in logging.getLogger().handlers:
        handler.flush()
        handler.close()
    assert "status=503" in log_file.read_text(encoding="utf-8")
