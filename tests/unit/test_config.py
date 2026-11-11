"""Unit tests for app/config.py. They never read the real .env (_env_file=None)."""

from __future__ import annotations

from pathlib import Path

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


def test_rate_limit_defaults_sit_below_the_free_tier_and_reject_nonsense() -> None:
    settings = make()
    assert (settings.classifier_rpm, settings.answer_rpm) == (12.0, 8.0)
    assert settings.rate_limit_max_wait_seconds == 5.0
    with pytest.raises(ValidationError):
        make(answer_rpm=0)
    with pytest.raises(ValidationError):
        make(rate_limit_max_wait_seconds=-1)


# ---- .env.example stays in sync with Settings --------------------------------------

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def _env_example_keys() -> list[str]:
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    return [line.split("=", 1)[0].strip() for line in lines
            if line.strip() and not line.lstrip().startswith("#")]


def test_env_example_lists_every_setting_and_nothing_else() -> None:
    """WHY both directions: a new setting must be documented, and a removed one must not
    linger in the example and confuse whoever copies it."""
    keys = _env_example_keys()
    expected = {name.upper() for name in Settings.model_fields}
    assert sorted(keys) == sorted(set(keys)), "duplicate key in .env.example"
    assert set(keys) == expected


def test_env_example_values_are_valid_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The placeholders must pass validation too, so `cp .env.example .env` starts."""
    for name in Settings.model_fields:  # real env vars would override the file
        monkeypatch.delenv(name.upper(), raising=False)
    settings = Settings(_env_file=ENV_EXAMPLE)  # type: ignore[call-arg]
    assert settings.gemini_api_key.get_secret_value() == "your-gemini-api-key-here"


def test_config_error_names_settings_but_never_shows_values() -> None:
    from pydantic import ValidationError

    from app.config import describe_config_error
    with pytest.raises(ValidationError) as info:
        Settings(_env_file=None, gemini_api_key="AIza-very-secret",  # type: ignore[call-arg]
                 answer_model="Bad Model secret-value", classifier_model="ok-model")
    message = describe_config_error(info.value)
    assert "ANSWER_MODEL" in message
    assert "secret" not in message.lower()


def test_empty_api_key_fails_at_startup() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Settings(_env_file=None, gemini_api_key="", answer_model="a",  # type: ignore[call-arg]
                 classifier_model="b")
