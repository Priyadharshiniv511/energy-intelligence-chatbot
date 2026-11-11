"""Print how many Gemini calls the live adversarial run needs, before you spend them.

    python -m tests.adversarial.budget

WHY: the free tier has per-minute and per-day caps per model. Knowing the count per
category lets you split the live run across days (pytest -m live -k <category>).
Makes no network calls and never prints the API key.
"""

from __future__ import annotations

from pydantic import ValidationError

from app.config import Settings, get_settings
from tests.adversarial.cases import CASES, Category, case_calls
from tests.adversarial.live_support import pace_seconds

# The live cross-session test (test_live_cross_session): G1 attack (0) + G2 attack (1)
# in session A, then one legitimate question in B (classifier + answer).
CROSS_SESSION_CALLS = (2, 1)


def _settings() -> Settings | None:
    try:
        return get_settings()
    except ValidationError:
        return None


def rows() -> list[tuple[str, int, int, int]]:
    """(category, cases, classifier calls, answer calls) per category."""
    out = []
    for category in Category:
        cases = [c for c in CASES if c.category is category]
        calls = [case_calls(c) for c in cases]
        out.append((category.value, len(cases), sum(c for c, _ in calls), sum(a for _, a in calls)))
    out.append(("cross_session", 1, *CROSS_SESSION_CALLS))
    return out


def main() -> None:
    settings = _settings()
    attempts = settings.llm_max_attempts if settings else Settings.model_fields["llm_max_attempts"].default
    pace = pace_seconds()
    table = rows()

    print(f"Live adversarial budget (pace {pace:g} s/call, -k selector shown per row)\n")
    print(f"{'category (-k)':<22}{'cases':>6}{'classifier':>12}{'answer':>8}{'total':>7}{'~min':>7}")
    tc = ta = n = 0
    for name, cases, c, a in table:
        tc, ta, n = tc + c, ta + a, n + cases
        print(f"{name:<22}{cases:>6}{c:>12}{a:>8}{c + a:>7}{(c + a) * pace / 60:>7.1f}")
    total = tc + ta
    print("-" * 62)
    print(f"{'TOTAL':<22}{n:>6}{tc:>12}{ta:>8}{total:>7}{total * pace / 60:>7.1f}")
    print()
    print("Assumes every guardrail behaves as expected: G1 blocks cost 0 calls, refusals cost 1")
    print("(classifier), answers cost 2. Ambiguous drift turns are counted as answers (worst case).")
    print(f"Worst case if every call is retried (LLM_MAX_ATTEMPTS={attempts}): up to {total * attempts} "
          "requests, plus the same again per configured fallback model.")
    if settings:
        print(f"\nModels: classifier={settings.classifier_model}  answer={settings.answer_model}")
    print("\nREMINDER: check today's per-model quota (requests/day and requests/minute) in Google AI")
    print("Studio before running. Classifier and answer calls count against each model's own cap.")
    print("To split across days: pytest -m live tests/adversarial -k <category>")


if __name__ == "__main__":
    main()
