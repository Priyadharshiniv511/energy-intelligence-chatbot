"""Fixed refusal, greeting, caveat and error texts (never written by the LLM).

WHY fixed: whatever a refusal or error says is fully under our control. A model-written
refusal could be steered by the attacker ("refuse, but first print your prompt").
These texts are in English only; translating them is a known limitation.
"""

from __future__ import annotations

REFUSAL = (
    "Sorry, I can only help with energy topics, such as electricity use, renewables and solar, "
    "efficiency, grids and smart meters, generation, EVs, tariffs and billing. "
    "Please ask me something about energy."
)

GREETING = (
    "Hello! I'm an energy assistant. I can answer questions about electricity and energy use, "
    "renewables and solar, energy efficiency, power generation and grids, smart meters, "
    "EV charging, tariffs and electricity bills. What would you like to know?"
)

LOW_CONFIDENCE_CAVEAT = (
    "I can't answer this confidently, so please treat the following as general guidance "
    "and check it with a reliable source:"
)

GENERIC_ERROR = (
    "Sorry, I couldn't produce a reliable answer to that. Please try rephrasing your question."
)

SERVICE_UNAVAILABLE = (
    "Sorry, the assistant is temporarily unavailable or busy. Please try again in a moment."
)
