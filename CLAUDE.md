# Energy Intelligence Chatbot
Take-home assignment. Brief: docs/ass.pdf. Read it before big changes.

## Stack
Python 3.11+, FastAPI, google-genai SDK, pydantic, pytest, python-dotenv.

## Commands
- Install: pip install -r requirements.txt
- Run API: uvicorn app.main:app --reload
- Run CLI: python -m app.cli
- Tests: pytest -q (unit tests must NOT call the real API; mock it)

## Rules
- Secrets only from environment variables. Never print, log or hardcode keys.
- Every user message goes through app/guardrails/pipeline.py. No LLM call bypasses it.
- Guardrails fail closed: if the classifier errors or is unsure, refuse.
- Keep modules separate: llm/, conversation/, domain/, guardrails/.
- Small functions, type hints, docstrings explaining WHY.
- I am learning: after each change, summarise what you did and why in plain words.