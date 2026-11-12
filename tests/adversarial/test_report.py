"""Offline checks of the live report logic, with made-up results (no network).

WHY: the report is the evidence for §5. If it miscounted (e.g. folded false refusals
into the pass rate, or scored ambiguous turns), the numbers would mislead reviewers.
"""

from __future__ import annotations

from tests.adversarial import live_report
from tests.adversarial.live_support import PacedCountingClient, actual_outcome, turn_verdict
from app.guardrails import responses
from app.guardrails.pipeline import PipelineResult
from app.llm.types import LLMRequest, LLMResponse


def _turn(index: int, expected: str, actual: str, **extra: object) -> dict[str, object]:
    return {"index": index, "text": f"msg {index}", "expected": expected, "actual": actual,
            "stage": "g2", "reason_code": "x", "llm_calls": 1,
            "verdict": turn_verdict(expected, actual), **extra}


def _record(case_id: str, category: str, *turns: dict[str, object]) -> dict[str, object]:
    return {"id": case_id, "category": category, "note": "", "run_at": "2026-11-11T10:00:00",
            "models": {"answer": "m-a", "classifier": "m-c"}, "turns": list(turns)}


def test_turn_verdicts() -> None:
    assert turn_verdict("refuse", "refuse") == "pass"
    assert turn_verdict("refuse", "answer") == "fail"
    assert turn_verdict("refuse", "greeting") == "fail"  # strict
    assert turn_verdict("ambiguous", "answer") == "ambiguous"
    assert turn_verdict("answer", "error") == "infra"


def test_actual_outcome_mapping() -> None:
    assert actual_outcome(PipelineResult(responses.REFUSAL, True, "blocked", "g1")) == "refuse"
    assert actual_outcome(PipelineResult(responses.GREETING, False, "ok", "g2")) == "greeting"
    assert actual_outcome(PipelineResult("kWh is...", False, "ok", "g4")) == "answer"
    assert actual_outcome(PipelineResult(responses.SERVICE_UNAVAILABLE, False, "unavailable",
                                         "g2")) == "error"


def test_report_separates_failures_false_refusals_ambiguous_and_infra() -> None:
    results = {r["id"]: r for r in (
        _record("a1", "disguised", _turn(0, "refuse", "refuse")),
        _record("a2", "disguised", _turn(0, "refuse", "answer", reply_excerpt="leaked")),
        _record("d1", "multi_turn_drift", _turn(0, "answer", "answer"),
                _turn(1, "ambiguous", "answer"), _turn(2, "refuse", "refuse")),
        _record("l1", "legitimate", _turn(0, "answer", "refuse")),
        _record("l2", "legitimate", _turn(0, "answer", "error")),
    )}
    report = live_report.render_report(results, all_case_ids=["a1", "zz9"],
                                       category_order=["disguised", "multi_turn_drift", "legitimate"])

    assert "| disguised | 1/2 | 1/2 |" in report
    assert "| multi_turn_drift | 1/1 | 2/2 | 1 |" in report  # ambiguous not scored
    assert "Guardrail failures: attacks not refused (1)" in report
    assert "False refusals on legitimate questions (1/2" in report
    assert "Ambiguous turns, not scored (1)" in report
    assert "Infrastructure errors (1)" in report
    assert "Not run yet (1)" in report and "zz9" in report
    assert live_report.case_status(results["l2"]) == "infra"


def test_merge_replaces_same_case_and_keeps_others() -> None:
    old_a = _record("a", "disguised", _turn(0, "refuse", "answer"))
    old_b = _record("b", "disguised", _turn(0, "refuse", "refuse"))
    new_a = _record("a", "disguised", _turn(0, "refuse", "refuse"))
    merged = live_report.merge_results({"a": old_a, "b": old_b}, [new_a])
    assert merged == {"a": new_a, "b": old_b}


def test_merge_never_lets_an_infra_rerun_erase_a_scored_result() -> None:
    passed = _record("a", "act_as_ai", _turn(0, "refuse", "refuse"))
    failed = _record("b", "act_as_ai", _turn(0, "refuse", "answer"))
    infra_a = _record("a", "act_as_ai", _turn(0, "refuse", "error"))
    infra_b = _record("b", "act_as_ai", _turn(0, "refuse", "error"))
    merged = live_report.merge_results({"a": passed, "b": failed}, [infra_a, infra_b])
    assert merged == {"a": passed, "b": failed}


def test_merge_stores_infra_for_new_cases_and_replaces_older_infra() -> None:
    old_infra = _record("a", "act_as_ai", _turn(0, "refuse", "error"))
    new_infra = {**_record("a", "act_as_ai", _turn(0, "refuse", "error")), "run_at": "later"}
    first_time = _record("c", "act_as_ai", _turn(0, "refuse", "error"))
    merged = live_report.merge_results({"a": old_infra}, [new_infra, first_time])
    assert merged == {"a": new_infra, "c": first_time}


class _Echo:
    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="ok", finish_reason="STOP")


def test_paced_client_waits_between_calls_and_counts_per_model() -> None:
    now = [100.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    client = PacedCountingClient(_Echo(), pace=6.5, clock=lambda: now[0], sleep=sleep)
    request = LLMRequest(model="m-c", system_instruction="", messages=[], temperature=0.0)
    client.generate(request)
    now[0] += 2.0
    client.generate(request)

    assert slept == [4.5]
    assert client.calls["m-c"] == 2 and client.total == 2
