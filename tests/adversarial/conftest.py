"""Collects live adversarial results and writes the report when the session ends.

WHY pytest_sessionfinish: it runs even when the live run stops early (pytest.exit after
repeated quota errors), so whatever was measured is still saved. Offline runs record
nothing, so they never touch the docs/ files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.adversarial import live_report
from tests.adversarial.cases import CASES, Category

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "docs" / "adversarial_results.json"
REPORT_PATH = ROOT / "docs" / "adversarial_report.md"
CROSS_SESSION_IDS = ("xs01",)

LIVE_RECORDS: list[dict[str, Any]] = []


def record_case(record: dict[str, Any]) -> None:
    LIVE_RECORDS.append(record)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not LIVE_RECORDS:
        return
    merged = live_report.merge_results(live_report.load_results(RESULTS_PATH), LIVE_RECORDS)
    live_report.save_results(RESULTS_PATH, merged)
    REPORT_PATH.write_text(
        live_report.render_report(
            merged,
            all_case_ids=[c.id for c in CASES] + list(CROSS_SESSION_IDS),
            category_order=[c.value for c in Category] + ["cross_session"],
        ),
        encoding="utf-8",
    )
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"Adversarial report written to {REPORT_PATH.relative_to(ROOT)} "
                            f"({len(LIVE_RECORDS)} cases this run, {len(merged)} stored).")
