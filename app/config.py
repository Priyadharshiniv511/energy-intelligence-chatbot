"""Application settings, loaded from environment variables (and a local .env file).

WHY a settings module: one typed place for all configuration means secrets come only
from the environment (never hardcoded), and bad config fails fast at startup instead
of halfway through a user's request.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Gemini model ids look like "gemini-3.5-flash" or "gemini-3.1-pro-preview":
# lowercase letters, digits, dots and dashes, starting with a letter or digit.
_MODEL_NAME_PATTERN = r"^[a-z0-9][a-z0-9.\-]*$"


class Settings(BaseSettings):
    """All runtime configuration.

    The API key is a SecretStr so that printing or logging the settings object shows
    '**********' instead of the real key.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: SecretStr
    # Model names are required config, so no model version is hardcoded in the code.
    # WHY the pattern: a typo such as `ANSWER_MODEL==gemini-x` would otherwise only
    # show up as a vague HTTP 400 on the first real call. Failing at startup with the
    # setting's name is much easier to debug.
    answer_model: str = Field(pattern=_MODEL_NAME_PATTERN)
    classifier_model: str = Field(pattern=_MODEL_NAME_PATTERN)

    llm_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    llm_max_attempts: int = Field(default=3, ge=1, le=10)
    llm_backoff_base_seconds: float = Field(default=1.0, ge=0)
    llm_backoff_max_seconds: float = Field(default=8.0, ge=0)

    # G1 input cap. WHY: long inputs cost quota, hide injections in bulk text, and
    # give regexes more to chew on. 2,000 chars is plenty for an energy question.
    max_input_chars: int = Field(default=2000, ge=1, le=20000)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read once.

    WHY cached: reading .env on every request is wasteful. Tests build their own
    Settings(...) directly instead of calling this, so they never need a real .env.
    """
    return Settings()  # type: ignore[call-arg]  # values come from env
