"""The live suites must skip, not run, when the key is the .env.example placeholder."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from tests import live_config

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def _settings(key: str) -> Settings:
    return Settings(_env_file=None, gemini_api_key=key,  # type: ignore[call-arg]
                    answer_model="gemini-3.5-flash", classifier_model="gemini-3.5-flash-lite")


def test_placeholder_constant_matches_env_example() -> None:
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    value = next(line.split("=", 1)[1] for line in lines if line.startswith("GEMINI_API_KEY="))
    assert value == live_config.PLACEHOLDER_API_KEY


def test_placeholder_key_is_not_a_real_key() -> None:
    assert not live_config.has_real_key(_settings(live_config.PLACEHOLDER_API_KEY))
    assert live_config.has_real_key(_settings("AIzaSyFAKEfakeFAKEfakeFAKEfakeFAKEfake12"))


def test_placeholder_key_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_config, "get_settings",
                        lambda: _settings(live_config.PLACEHOLDER_API_KEY))
    with pytest.raises(pytest.skip.Exception) as info:
        live_config.live_settings_or_skip()
    assert "placeholder" in str(info.value)
    assert live_config.PLACEHOLDER_API_KEY not in str(info.value)  # names, never values


def test_real_key_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings("AIzaSyFAKEfakeFAKEfakeFAKEfakeFAKEfake12")
    monkeypatch.setattr(live_config, "get_settings", lambda: settings)
    assert live_config.live_settings_or_skip() is settings
