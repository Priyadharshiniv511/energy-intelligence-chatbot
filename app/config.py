"""Application settings, loaded from environment variables (and a local .env file).

WHY a settings module: one typed place for all configuration means secrets come only
from the environment (never hardcoded), and bad config fails fast at startup instead
of halfway through a user's request.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
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

    # Optional backup models, tried when the main model is overloaded (503), rate
    # limited (429) or times out, after its own retries. WHY: free-tier overloads are
    # usually per model, so a different model often answers when the main one can't.
    # Leave unset (or empty) for no fallback.
    answer_fallback_model: str | None = Field(default=None, pattern=_MODEL_NAME_PATTERN)
    classifier_fallback_model: str | None = Field(default=None, pattern=_MODEL_NAME_PATTERN)

    llm_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    llm_max_attempts: int = Field(default=3, ge=1, le=10)
    llm_backoff_base_seconds: float = Field(default=1.0, ge=0)
    llm_backoff_max_seconds: float = Field(default=8.0, ge=0)

    # G1 input cap. WHY: long inputs cost quota, hide injections in bulk text, and
    # give regexes more to chew on. 2,000 chars is plenty for an energy question.
    max_input_chars: int = Field(default=2000, ge=1, le=20000)

    # G2: the classifier's self-reported confidence must reach this for a message to
    # be answered. WHY configurable: it is tuned against the live suite, and raising it
    # trades more false refusals for fewer leaks without a code change.
    classifier_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # REST API: messages allowed per client IP per minute. WHY: every message can cost
    # up to two Gemini calls, so one noisy client could burn the whole free-tier quota.
    api_rate_limit_per_minute: int = Field(default=20, ge=1, le=1000)

    # Client-side Gemini rate limit, per model (app/llm/rate_limiter.py). WHY slightly
    # below the free tier (~15 RPM for lite models, ~10 for flash): the bucket also allows
    # a small burst on top, and staying under the limit avoids 429s and wasted retries.
    classifier_rpm: float = Field(default=12.0, gt=0, le=1000)
    answer_rpm: float = Field(default=8.0, gt=0, le=1000)
    # Longest we hold a user's request while waiting for a free slot. If the wait would be
    # longer, the user gets the "unavailable" message right away instead of a hang.
    rate_limit_max_wait_seconds: float = Field(default=5.0, ge=0, le=60)

    @field_validator("answer_fallback_model", "classifier_fallback_model", mode="before")
    @classmethod
    def _empty_means_none(cls, value: object) -> object:
        """`ANSWER_FALLBACK_MODEL=` (empty) in .env means "no fallback", not a typo."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read once.

    WHY cached: reading .env on every request is wasteful. Tests build their own
    Settings(...) directly instead of calling this, so they never need a real .env.
    """
    return Settings()  # type: ignore[call-arg]  # values come from env
