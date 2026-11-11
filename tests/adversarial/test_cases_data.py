"""The case table itself is complete and follows its own rules.

WHY test the data: a suite that silently loses a category, or quietly marks an attack
"ambiguous" to dodge a failure, would still be green. These checks keep it honest.
"""

from __future__ import annotations

from app.domain.scope import IN_SCOPE, OUT_OF_SCOPE
from tests.adversarial.cases import (
    ATTACK_CATEGORIES,
    CASES,
    Category,
    Expect,
    by_category,
)


def test_ids_are_unique() -> None:
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))


def test_every_attack_category_has_at_least_five_cases() -> None:
    for category in ATTACK_CATEGORIES:
        assert len(by_category(category)) >= 5, category


def test_at_least_twenty_legitimate_questions_all_expected_answered() -> None:
    legit = by_category(Category.LEGITIMATE)
    assert len(legit) >= 20
    assert all(t.expect is Expect.ANSWER and not t.g1_block for c in legit for t in c.turns)


def test_single_turn_attacks_expect_refusal() -> None:
    for category in ATTACK_CATEGORIES:
        if category is Category.MULTI_TURN_DRIFT:
            continue
        for case in by_category(category):
            assert [t.expect for t in case.turns] == [Expect.REFUSE], case.id


def test_drift_cases_have_three_or_four_turns_start_on_energy_and_end_refused() -> None:
    for case in by_category(Category.MULTI_TURN_DRIFT):
        assert 3 <= len(case.turns) <= 4, case.id
        assert case.turns[0].expect is Expect.ANSWER, case.id
        assert case.turns[-1].expect is Expect.REFUSE, case.id


def test_ambiguous_only_on_middle_turns_of_drift_cases() -> None:
    """The rule agreed for Expect.AMBIGUOUS: drift only, never first or last turn."""
    for case in CASES:
        for index, turn in enumerate(case.turns):
            if turn.expect is Expect.AMBIGUOUS:
                assert case.category is Category.MULTI_TURN_DRIFT, case.id
                assert 0 < index < len(case.turns) - 1, case.id


def test_every_scope_example_is_a_case() -> None:
    """DESIGN.md 2.5: every IN and OUT scope item has at least one case."""
    texts = {t.text for c in CASES for t in c.turns}
    for area in IN_SCOPE + OUT_OF_SCOPE:
        assert area.example in texts, area.key


def test_non_english_cases_exist_for_attacks_and_legitimate() -> None:
    def has_non_ascii(category: Category) -> bool:
        return any(not t.text.isascii() for c in by_category(category) for t in c.turns)

    assert has_non_ascii(Category.ENCODED)
    assert has_non_ascii(Category.LEGITIMATE)
    assert has_non_ascii(Category.ACT_AS_AI)


def test_every_case_has_a_note() -> None:
    assert all(c.note.strip() for c in CASES)
