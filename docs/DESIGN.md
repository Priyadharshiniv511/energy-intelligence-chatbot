# Energy Intelligence Chatbot: Design

Source brief: `docs/assignment.pdf`. Project rules: `CLAUDE.md`.

## 1. Requirements checklist

### §1 Objective
- [ ] 1.1 Chatbot focused **exclusively** on the energy domain
- [ ] 1.2 Uses Google Gemini free tier (or another free LLM API)
- [ ] 1.3 Answers on electricity, consumption, generation, renewables, efficiency, **billing**, grids and related topics
- [ ] 1.4 **Core:** refuses unrelated questions *even under manipulation*
- [ ] 1.5 Delivered as a CLI or a backend API (no frontend needed)

### §2 LLM API integration
- [ ] 2.1 Secure backend integration (the key never reaches the client, logs or responses)
- [ ] 2.2 API credentials only in environment variables
- [ ] 2.3 Keeps conversation context "appropriately"
- [ ] 2.4 Handles API failures, timeouts, rate limits and unavailable/empty responses
- [ ] 2.5 Error handling **and** retry mechanisms

### §3 Domain coverage
- [ ] 3.1 Renewables and solar
- [ ] 3.2 Efficiency and optimisation
- [ ] 3.3 Generation and distribution
- [ ] 3.4 Grids and smart meters
- [ ] 3.5 Industrial/commercial usage and other energy questions
- [ ] 3.6 Responses are technically appropriate, clear and useful

### §4 Guardrails
- [ ] 4.1 Reject non-energy questions
- [ ] 4.2 Reject attempts to change identity, role or system instructions
- [ ] 4.3 Resist **direct and indirect** prompt injection
- [ ] 4.4 Don't follow "ignore your energy-only restriction" instructions
- [ ] 4.5 Never reveal the system prompt, internal instructions, API credentials or implementation secrets
- [ ] 4.6 Reject unrelated questions **disguised** in energy terminology
- [ ] 4.7 Avoid fabricated specs, statistics or facts
- [ ] 4.8 Say clearly when it can't answer confidently
- [ ] 4.9 **Guardrails must not rely only on the system prompt.** Add backend validation and enforcement.

### §5 Adversarial testing (must *demonstrate* resistance)
- [ ] 5.1 General-knowledge questions
- [ ] 5.2 "Act as a different AI assistant"
- [ ] 5.3 "Ignore previous instructions"
- [ ] 5.4 Unrelated instructions embedded inside energy questions
- [ ] 5.5 Unrelated questions disguised in energy terms
- [ ] 5.6 Requests for hidden prompts or configuration
- [ ] 5.7 Multi-turn gradual drift away from energy
- [ ] 5.8 Indirect or **encoded** instructions (base64, hex, leetspeak, **translation/non-English** tricks)
- [ ] 5.9 Enforcement is consistent across conversations

### §6 Architecture
- [ ] 6.1 Modular design
- [ ] 6.2 LLM integration, conversation management, domain validation and guardrail enforcement are separate
- [ ] 6.3 Per-conversation context with **no cross-conversation influence**
- [ ] 6.4 **All** LLM requests pass through the guardrail mechanism

### §7 Technical
- [ ] 7.1 LLM: Gemini (or another)
- [ ] 7.2 Python or Node.js → Python 3.11
- [ ] 7.3 CLI or REST → **both**, as thin shells over one service
- [ ] 7.4 Config via `.env` or equivalent
- [ ] 7.5 Automated tests for domain restriction and guardrails

### §8 Submission
- [ ] 8.1 Git repo with complete source
- [ ] 8.2 README with setup and run instructions
- [ ] 8.3 Architecture and implementation explanation
- [ ] 8.4 Guardrail mechanism explanation
- [ ] 8.5 Instructions to run locally

## 2. Architecture

### 2.1 Modules
```
app/
  config.py              Settings (pydantic-settings) from env: GEMINI_API_KEY (SecretStr, never logged),
                         ANSWER_MODEL, CLASSIFIER_MODEL (no model versions hardcoded), timeouts, limits
  main.py                FastAPI: POST /sessions, POST /sessions/{id}/messages, DELETE /sessions/{id}, GET /health
  cli.py                 REPL. Creates one session and calls the same ChatService.
  service.py             ChatService.handle(session_id, text). The single entry point; it only calls the pipeline.
  llm/
    client.py            LLMClient protocol + GeminiClient. Handles timeout, retry, the rate limiter and error mapping.
    errors.py            LLMTimeout, LLMRateLimited, LLMUnavailable, LLMBlocked, LLMEmpty
    rate_limiter.py      Client-side token bucket that keeps us under the free-tier RPM
  conversation/
    store.py             SessionStore: dict[session_id → Conversation], per-session lock, TTL, max turns
    models.py            Turn, Conversation (accepted turns only, plus a strike counter for logging)
  domain/
    scope.py             IN/OUT topic lists (single source of truth: used by prompts and by tests, see 2.5)
    classifier.py        LLM topic + intent classifier, structured JSON output
    prompts.py           Answerer system prompt and classifier prompt (classifier never sees the answerer prompt)
  guardrails/
    pipeline.py          Orchestrates every stage below. THE only path to the LLM.
    input_filter.py      Deterministic: normalise, size limits, decode-and-inspect, injection patterns
    output_filter.py     Deterministic: canary/prompt-leak check, secret regex, schema check, caveat injection
    responses.py         Fixed refusal, greeting/meta and error templates (never written by the LLM)
tests/
  unit/                  Each layer in isolation with a FakeLLM
  adversarial/           Table-driven cases for §5.1–5.8 + scope IN/OUT cases through the full pipeline (FakeLLM)
  live/                  Opt-in (`pytest -m live`) run against real Gemini; prints a pass-rate report
```
Rule (§6.4): `GeminiClient` is constructed only inside `pipeline.py`'s factory. A test asserts that no other
module imports `llm.client`.

### 2.2 Request flow (one user turn)
```
CLI / REST
   │  session_id, text
   ▼
ChatService ──► SessionStore.get(session_id)   (isolated history; unknown id → 404)
   │
   ▼  guardrails/pipeline.py
[G1] Input filter (deterministic, 0 LLM calls)
     • NFKC normalise, strip zero-width/bidi chars, collapse homoglyphs
     • length cap (e.g. 2,000 chars); reject empty input
     • detect encoded payloads (base64/hex/rot13/url-encoding). Decode them and run the checks on
       the decoded text too. Any encoded instruction counts as an injection signal.
     • high-precision injection/leak regexes ("ignore (all|previous) instructions",
       "you are now", "system prompt", "developer mode", "DAN", "repeat the text above"…)
       → clear hit: REFUSE immediately (saves quota); weak hit: attach a flag for G2
   ▼
[G2] Domain + intent classifier (1 call to CLASSIFIER_MODEL, separate prompt, JSON schema)
     input  = current message + last 3 accepted user turns (as quoted data, not instructions)
     output = {on_topic: bool, intent: answer|injection|identity_change|prompt_extraction|
               greeting|off_topic, disguised: bool, confidence: 0–1, language: str}
     • works on any language; translation-wrapped requests are judged on their meaning
     • REFUSE unless on_topic AND intent=="answer" AND confidence ≥ 0.7 AND no G1 strong flag
     • greeting/meta → canned capability message from responses.py (no answer call)
     • classifier error, timeout or invalid JSON → REFUSE (fail closed)
   ▼
[G3] Answer generation (1 call to ANSWER_MODEL)
     • system prompt: energy-only scope from scope.py, uncertainty policy, a random CANARY token per process
     • history = this session's accepted turns only (refused turns are never stored → nothing accumulates)
     • user text wrapped in delimiters and labelled as untrusted data
     • temperature 0.2; structured output {answer, on_topic, confidence: high|medium|low}
     • replies in the user's language
   ▼
[G4] Output filter (deterministic)
     • canary token or ≥ N-gram overlap with the system prompt → BLOCK
     • secret patterns (AIza…, env var values, "GEMINI_API_KEY") → BLOCK
     • model self-reports on_topic=false → REFUSE (second, independent topic check)
     • confidence=low → prepend the fixed "I can't answer this confidently" caveat (§4.8)
     • empty, safety-blocked or malformed responses → fixed error message
   ▼
SessionStore.append(user, answer)   (only on success)  → response
```
Refusal path: return the template from `responses.py`, increment the session's `strikes` counter and log
`{session_id, stage, reason}` (no message text containing secrets). There is no session reset: refused turns are
never stored, so manipulation cannot build up in the history.

### 2.3 Where each guardrail sits (and why it's not just the prompt, §4.9)
| Req | G1 input | G2 classifier | G3 prompt | G4 output | Store |
|---|---|---|---|---|---|
| 4.1 off-topic | | ✔ | ✔ | ✔ self-report | |
| 4.2 identity change | ✔ regex | ✔ intent | ✔ | | |
| 4.3 injection (direct/indirect) | ✔ regex + decode | ✔ intent | ✔ delimiters | ✔ | refused turns dropped |
| 4.4 ignore restriction | ✔ | ✔ | ✔ | | |
| 4.5 no leaks | ✔ leak regex | ✔ prompt_extraction | ✔ | ✔ canary/overlap/secret regex | key never in prompt |
| 4.6 disguised | | ✔ `disguised` | ✔ | ✔ | |
| 4.7 / 4.8 fabrication | | | ✔ policy, low temp | ✔ caveat on low confidence | |
| 5.7 drift | | ✔ judges each turn | | | refused turns not kept; strikes logged |
| 5.8 encoded/translated | ✔ decode | ✔ language-agnostic | | | |
| 6.3 isolation | | | | | per-session objects, no globals |

Four of the five layers are backend code. The system prompt is one layer among them.

### 2.4 LLM client (§2)
- `google-genai`. `GEMINI_API_KEY`, `ANSWER_MODEL` and `CLASSIFIER_MODEL` come from `.env` via `config.py`.
  `.env` is git-ignored and `.env.example` is committed with placeholder values.
- Per-call timeout (e.g. 15 s). Retry with exponential backoff + jitter (max 3) on 429/500/503/timeout only.
  Honour `retry-after` when present. Never retry 400/permission errors.
- Map SDK errors to `llm/errors.py`. The pipeline turns them into friendly messages and logs no secrets.
- A token bucket set slightly below the model's RPM prevents most 429s. One bucket per model (`llm/rate_limiter.py`, used only inside
  `GeminiClient`): `CLASSIFIER_RPM` / `ANSWER_RPM` (defaults 12 / 8, burst 2). A call waits at most
  `RATE_LIMIT_MAX_WAIT_SECONDS` (default 5) for a slot; a longer wait raises `LLMRateLimited` → "unavailable", no strike.
- Optional fallback models (`ANSWER_FALLBACK_MODEL`, `CLASSIFIER_FALLBACK_MODEL`, `llm/fallback.py`): after the main
  model's retries fail with 503/5xx, 429 or a timeout, the identical request is sent once to the backup model
  (with its own retries). 400/403, safety blocks and empty replies never switch models. The backup's reply still
  goes through G4, and its name is treated as a secret like the main model names.

### 2.5 Domain scope (encoded in `domain/scope.py`, mirrored in tests)
**IN:** renewables and solar; efficiency and optimisation; generation, transmission and distribution; grids and smart
meters; industrial/commercial usage; electricity billing; EVs and charging; energy prices and tariffs (general,
non-investment); emissions from energy; energy use of crypto-mining and data centres; appliance energy efficiency.

**OUT:** investment or stock advice (including energy stocks); food calories / "food energy"; physics homework not
about power systems (e.g. kinetic energy of a ball); brand or product shopping advice; generic coding help;
everything else outside energy.

Each IN and OUT item gets at least one adversarial-suite test case (e.g. "Which solar stock should I buy?" → refuse;
"How do time-of-use tariffs work?" → answer).

## 3. Trade-offs and choices
| Decision | Options | Choice | Why |
|---|---|---|---|
| LLM calls per turn | 1 (answer + self-check) / 2 (classifier + answer) / 3 (+ LLM output judge) | **2**. The LLM judge is **out of initial scope** (would sit behind a flag, off by default, if added later) | A separate classifier can't be hijacked by the same injection that targets the answerer. A third call adds ~1 s and 50% more quota for marginal gain over the deterministic G4. |
| Models | one model / two | **Two, set in env:** `CLASSIFIER_MODEL` (lighter, cheaper) and `ANSWER_MODEL` (stronger) | Classification is easy, and a lighter model has higher free quota and lower latency. Configured in env so versions aren't hardcoded. |
| Classifier vs keyword allowlist | keywords only / LLM only / both | **Regex as a fast block, LLM as the decider** | Keywords can't catch disguised questions (§4.6), and an LLM alone wastes quota on obvious attacks. |
| Latency | — | ~+0.5–1 s per turn for G2 | Acceptable for a chatbot. Obvious attacks short-circuit at G1 with 0 calls. |
| Free-tier limits | — | Rate limiter + retry/backoff + fail-closed messaging | Free Gemini tiers are roughly 10–15 RPM and a few hundred to ~1,000 requests/day per model (**verify current numbers at ai.google.dev**). 2 calls/turn ≈ 5–7 turns/min. Unit and adversarial tests use a FakeLLM (no quota). The live suite is opt-in. |
| Uncertain classifier | allow / refuse | **Refuse** (CLAUDE.md rule) | False refusals are cheaper than leaks. Tune the threshold using the live suite. |
| Session storage | in-memory / Redis / SQLite | **In-memory with TTL** | Sufficient for the assignment and isolated per session id. A store interface allows swapping later. |
| History sent to answerer | full / last N | **Last 10 accepted turns** | Bounds tokens/cost and limits slow-drift leverage. |
| Repeated refusals | reset session / count only | **Count and log only** | Refused turns are never stored, so a reset protects nothing. Logging still makes abuse visible. |
| Hallucination control | grounding/RAG / prompt + confidence | **Prompt + low temp + confidence field + caveat** | RAG is out of scope for 72 h. Weakness noted under Known limitations. |

## 4. Decisions (resolved)
- Interfaces: REST **and** CLI over one `ChatService`.
- Scope: as in 2.5.
- Greetings/meta: canned reply, no answer-model call.
- Non-English: supported. The adversarial suite includes translation attacks (e.g. an off-topic request in another
  language, or "translate this, then follow it").
- Demonstration (§5): automated adversarial suite + a short live transcript in the README.
- LLM output judge: out of initial scope; off by default if added.
- Strikes: counted and logged; no session reset.
- Models: from `ANSWER_MODEL` / `CLASSIFIER_MODEL` env vars.

## 5. README plan (§8)
Sections: Setup (`.env.example` → `.env`), Run API, Run CLI, Tests (unit/adversarial/live), Architecture (link to
this file), Guardrail mechanism (G1–G4 + table 2.3), Live demo transcript, **Known limitations**:
- Confidence is **self-reported by the model** and is not calibrated, so the low-confidence caveat is best-effort
  and does not guarantee factual accuracy.
- Classifier decisions are probabilistic, and novel jailbreaks may slip past G1/G2 (G3/G4 are the backstop).
- Free-tier rate limits cap throughput, and sessions are lost on restart (in-memory store).
- No retrieval/grounding, so answers rely on model knowledge.
- The session store is in memory, so run a **single uvicorn worker**. Several workers would each hold their own
  sessions (a session created on one would 404 on another). Scaling out needs a shared store such as Redis.
- The per-IP rate limit uses the socket's client IP. Behind a reverse proxy every request shares the proxy's IP,
  so a proxy deployment needs trusted `X-Forwarded-For` handling. The limiter is also per process and in memory.
- Concurrent messages to the **same** session are not serialised. Each store operation is atomic, but two
  in-flight turns can both read the same history, and their turns are stored in completion order.
