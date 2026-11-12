"""Settings for the opt-in live suites, or a skip when no real key is configured.

WHY a shared helper: `cp .env.example .env` gives settings that *validate* (the
placeholder key is a non-empty string), so "skip on ValidationError" alone let the
live suites run with a fake key. Every call then failed with HTTP 400, and the
adversarial tier wrote those failures into docs/adversarial_results.json as if they
were real results. All three live modules now use this one check.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings

# Must match GEMINI_API_KEY in .env.example (tests/unit/test_live_config.py checks it).
PLACEHOLDER_API_KEY = "your-gemini-api-key-here"

SKIP_REASON = "GEMINI_API_KEY / ANSWER_MODEL / CLASSIFIER_MODEL not configured"


def has_real_key(settings: Settings) -> bool:
    """False for the .env.example placeholder: it can never make a successful call."""
    return settings.gemini_api_key.get_secret_value().strip() != PLACEHOLDER_API_KEY


def live_settings_or_skip() -> Settings:
    """Return the real settings, or skip the calling test/fixture.

    Skips when settings are missing or invalid, and when the key is still the
    placeholder. The reason names settings only, never values.
    """
    try:
        settings = get_settings()
    except ValidationError:
        pytest.skip(SKIP_REASON)
    if not has_real_key(settings):
        pytest.skip(f"{SKIP_REASON} (GEMINI_API_KEY is still the .env.example placeholder)")
    return settings
