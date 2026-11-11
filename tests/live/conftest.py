"""Live-suite helpers: collect each case's outcome and print a Markdown table at the end.

WHY a terminal-summary hook rather than print(): pytest hides print output unless you
pass -s, and the table is meant to be pasted into the README.
"""

from __future__ import annotations

from typing import Any

import pytest

LIVE_RESULTS: list[dict[str, str]] = []


def record(case: str, expected: str, got: str, reason: str) -> None:
    LIVE_RESULTS.append({"case": case, "expected": expected, "got": got, "reason": reason})


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    if not LIVE_RESULTS:
        return
    passed = sum(r["expected"] == r["got"] for r in LIVE_RESULTS)
    tr = terminalreporter
    tr.section("Live results")
    tr.write_line("| Case | Expected | Got | Reason |")
    tr.write_line("|---|---|---|---|")
    for r in LIVE_RESULTS:
        mark = "" if r["expected"] == r["got"] else " ✗"
        tr.write_line(f"| {_cell(r['case'])} | {r['expected']} | {r['got']}{mark} | {r['reason']} |")
    tr.write_line("")
    tr.write_line(f"{passed}/{len(LIVE_RESULTS)} cases as expected.")
