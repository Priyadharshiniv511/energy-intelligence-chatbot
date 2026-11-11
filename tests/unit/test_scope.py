"""Sanity checks on the scope lists that prompts and tests are built from."""

from __future__ import annotations

from app.domain.scope import IN_SCOPE, OUT_OF_SCOPE


def test_both_lists_are_non_empty() -> None:
    assert IN_SCOPE and OUT_OF_SCOPE


def test_keys_are_unique_across_both_lists() -> None:
    keys = [a.key for a in IN_SCOPE + OUT_OF_SCOPE]
    assert len(keys) == len(set(keys))


def test_every_area_has_a_description_and_an_example() -> None:
    for area in IN_SCOPE + OUT_OF_SCOPE:
        assert area.description.strip() and area.example.strip(), area.key


def test_design_md_decisions_are_present() -> None:
    """The IN/OUT items DESIGN.md 2.5 calls out explicitly."""
    in_keys = {a.key for a in IN_SCOPE}
    out_keys = {a.key for a in OUT_OF_SCOPE}
    assert {"ev_charging", "prices_tariffs", "emissions", "mining_datacentres",
            "appliance_efficiency", "billing"} <= in_keys
    assert {"investment_advice", "food_energy", "non_power_physics", "shopping_advice",
            "coding_help"} <= out_keys
