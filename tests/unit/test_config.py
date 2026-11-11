"""Unit tests for app/config.py. They never read the real .env (_env_file=None)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def make(**overrides: object) -> Settings:
    values = {"gemini_api_key": "fake-key", "answer_model": "gemini-3.5-flash",
              "classifier_model": "gemini-3.5-flash-lite", **overrides}
    return Settings(_env_file=None, **values)


@pytest.mark.parametrize("name", ["gemini-3.5-flash", "gemini-3.1-pro-preview", "gemma-3"])
def test_valid_model_names_are_accepted(name: str) -> None:
    assert make(answer_model=name).answer_model == name


@pytest.mark.parametrize(
    "name",
    ["=gemini-3.5-flash", " gemini-3.5-flash", "gemini 3.5", "Gemini-3.5", "", "models/x"],
)
def test_typo_in_model_name_fails_at_startup(name: str) -> None:
    with pytest.raises(ValidationError) as info:
        make(answer_model=name)
    assert info.value.errors()[0]["loc"] == ("answer_model",)


def test_api_key_is_hidden_when_printed() -> None:
    settings = make(gemini_api_key="AIza-super-secret")
    assert "AIza-super-secret" not in repr(settings)
    assert "AIza-super-secret" not in str(settings.model_dump())


def test_input_cap_defaults_to_2000_and_rejects_nonsense() -> None:
    assert make().max_input_chars == 2000
    with pytest.raises(ValidationError):
        make(max_input_chars=0)


def test_classifier_threshold_defaults_to_0_7_and_must_be_a_probability() -> None:
    assert make().classifier_confidence_threshold == 0.7
    with pytest.raises(ValidationError):
        make(classifier_confidence_threshold=1.5)
