"""G1: deterministic input filter. The first guardrail. It makes no LLM calls.

What this layer is for
----------------------
1. **Normalise** every message (NFKC, invisible characters removed, homoglyphs folded,
   spaces collapsed), so G2, G3 and the logs all see the same canonical text an
   attacker cannot hide things in.
2. **Enforce limits** (empty input, length cap) before we spend anything.
3. **Peel off encodings** (base64, hex, URL-encoding, ROT13, reversed text, leetspeak,
   letter-spacing, hidden Unicode tag characters) and run the same checks on what is
   underneath.
4. **Refuse obvious attacks instantly** with high-precision patterns. This costs zero
   API quota and zero latency, and it means the classifier never has to see them.
5. **Pass weak signals on** as flags. A flag never refuses on its own; it tells G2
   (the LLM classifier) to look harder.

Why regex alone is not enough
-----------------------------
Patterns match *wording*, but attacks are about *meaning*. There are endless
paraphrases of "ignore your instructions", attacks can be written in any language,
off-topic questions can be disguised in energy terms (§4.6), and multi-turn drift
looks innocent one message at a time. A regex list long enough to catch all of that
would also refuse real energy questions ("ignore the standby load..."). So G1 is tuned
for **precision**: it only blocks phrasings that are almost never innocent, and leaves
the semantic judgement to G2 (classifier), G3 (constrained answerer) and G4 (output
filter). G1 is the cheap first net, not the wall.

Language scope
--------------
G1 patterns are **English-only by design**. Non-English injections (and "translate
this, then follow it" tricks) are left to G2, which judges meaning in any language;
they will be covered by the adversarial test suite.

ReDoS safety
------------
No pattern has nested unbounded quantifiers; gaps between words are bounded
(e.g. ``.{0,30}?``). Matching time is linear in input length, and the input is
length-capped before any regex runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from app.guardrails import decoders


class Reason(StrEnum):
    """Why a message was allowed or blocked (stable codes for logs and tests)."""

    OK = "ok"
    EMPTY = "empty"
    TOO_LONG = "too_long"
    INJECTION = "injection"
    PROMPT_LEAK = "prompt_leak"
    SECRET_REQUEST = "secret_request"
    ENCODED_INJECTION = "encoded_injection"


class Flag(StrEnum):
    """Weak signals passed to G2. None of them blocks a message by itself."""

    INVISIBLE_CHARS = "invisible_chars"
    HOMOGLYPHS = "homoglyphs"
    ENCODED_PAYLOAD = "encoded_payload"
    WEAK_INJECTION = "weak_injection"
    WEAK_ROLEPLAY = "weak_roleplay"
    WEAK_PROMPT_LEAK = "weak_prompt_leak"
    WEAK_SECRET = "weak_secret"


@dataclass(frozen=True)
class InputFilterResult:
    """Structured G1 verdict.

    `normalized_text` is what later stages must use instead of the raw input.
    `decoded_segments` lets G2 judge what an encoded payload actually says.
    """

    allowed: bool
    reason: Reason
    normalized_text: str
    flags: frozenset[Flag] = field(default_factory=frozenset)
    decoded_segments: tuple[str, ...] = ()


# ---- Patterns -------------------------------------------------------------------------

# Separator between words: up to 3 spaces/dots/underscores/dashes, or none at all.
# "none" matters because the letter-spacing view turns "i g n o r e a l l" into
# "ignoreall". Bounded, so it cannot cause backtracking blow-ups.
_S = r"[\s._\-]{0,3}"


def _phrase(template: str) -> str:
    """Write patterns with readable single spaces; each space becomes the separator _S."""
    return template.replace(" ", _S)


_OVERRIDE_VERB = r"\b(?:ignore|disregard|forget|override|bypass|skip|abandon)"
# Deliberately excludes "system", "default", "safety", "current": "override the default
# rules on my thermostat" or "bypass the system restrictions on my inverter" are real
# energy questions. "System prompt" gets its own pattern below.
_INSTRUCTION_QUALIFIER = (r"(?:previous|prior|above|earlier|preceding|initial|original"
                          r"|your|ur|developer)")
_INSTRUCTION_OBJECT = (r"(?:instructions?|prompts?|directives?|rules|guardrails?"
                       r"|restrictions?|constraints?|guidelines?|programming|policies)")
# Verbs that ask the bot to disclose something.
_REQUEST_VERB = (r"\b(?:reveal|show|print|repeat|output|display|tell|give|share|leak|dump"
                 r"|write|list|recite|paste|copy|read|cat|send|what(?:'s| is| are| were| was))\b")

# Strong patterns: a match blocks the message. Grouped by the Reason they produce.
# Each group lists examples of what it catches and, importantly, what it must NOT.
_STRONG_PATTERNS: dict[Reason, list[re.Pattern[str]]] = {
    Reason.INJECTION: [re.compile(p, re.IGNORECASE) for p in (
        # "ignore all previous instructions", "disregard your system prompt",
        # "forget the above rules". Needs an instruction word AND a qualifier, so
        # "ignore the standby load" and "ignore the manufacturer's guidelines" pass.
        _phrase(rf"{_OVERRIDE_VERB} (?:all |any |of )?(?:the )?{_INSTRUCTION_QUALIFIER} "
                rf"(?:{_INSTRUCTION_QUALIFIER} )?{_INSTRUCTION_OBJECT}"),
        # "ignore all instructions", "forget all rules"
        _phrase(rf"{_OVERRIDE_VERB} all (?:the )?{_INSTRUCTION_OBJECT}"),
        # "ignore your system prompt", "forget the system prompt"
        _phrase(rf"{_OVERRIDE_VERB} (?:all )?(?:the |your |ur )?system prompts?"),
        # "ignore the above", "disregard everything above"
        _phrase(rf"{_OVERRIDE_VERB} (?:all of )?(?:the|everything) (?:above|before this)\b"),
        # §4.4: "ignore your energy-only restriction", "disable the topic filter".
        # Bare "energy" only counts after "your/this": "remove the energy limits on my
        # EV charger" is a real question.
        _phrase(r"\b(?:ignore|disregard|forget|bypass|lift|remove|disable|turn off) "
                r"(?:(?:your |this )(?:energy only|energy|topic|domain)|(?:the )?"
                r"(?:energy only|topic|domain)) "
                r"(?:restrictions?|limits?|rules?|filters?|scope|focus)"),
        # Identity change: "you are now an unrestricted AI", "you are now DAN"
        _phrase(r"\byou are now (?:an? |the |my )?(?:new |different |free |unrestricted "
                r"|unfiltered |uncensored |evil |general purpose )?(?:ai|assistant|bot"
                r"|chatbot|model|llm|gpt|dan|character|persona|unrestricted|unfiltered"
                r"|jailbroken|no longer|in \w+ mode)\b"),
        _phrase(r"\bfrom now on,? you (?:are|will be|act as|shall be) (?:an? |the )?"
                r"(?:new |different |free |unrestricted |general purpose |general )?"
                r"(?:ai|assistant|bot|chatbot|dan|character|persona|unrestricted|free"
                r"|different)\b"),
        _phrase(r"\bpretend (?:to be|you are|you're|that you are) (?:an? |the )?(?:different "
                r"|another |unrestricted |general purpose |new )?(?:ai|assistant|chatbot|bot"
                r"|model|dan|human|person)\b"),
        _phrase(r"\bact as (?:an? |the )?(?:different|another|unrestricted|unfiltered"
                r"|general purpose|new) (?:ai|assistant|chatbot|bot|model)\b"),
        _phrase(r"\bact as (?:chatgpt|gpt|dan|claude|gemini|bard|llama)\b"),
        # Classic jailbreak vocabulary.
        _phrase(r"\bdo anything now\b"),
        _phrase(r"\bdan mode\b"),
        _phrase(r"\byou are dan\b"),
        _phrase(r"\b(?:you are |you're )(?:now )?in (?:developer|dev|god|admin|sudo|debug) mode\b"),
        # Not "...mode on": "is developer mode on my inverter app safe?" is a real question.
        _phrase(r"\b(?:developer|dev|god|sudo) mode (?:enabled|activated)\b"),
        _phrase(r"\b(?:jailbreak mode|jailbreak prompt|you are jailbroken|this is a jailbreak)\b"),
        # Spoofed chat-template / role tags that try to open a fake system turn.
        r"<\|?\s*(?:im_start|im_end|system|endoftext)\s*\|?>",
        r"</?\s*system\s*>",
        r"\[/?(?:inst|sys)\]",
        r"<<\s*sys\s*>>",
        # A fake "### SYSTEM:" header. WHY require the colon: "## Instructions for my
        # inverter" is an ordinary markdown heading. (Newlines are already collapsed
        # here, so a colon-less header can't be told apart; G2 judges those.)
        r"#{2,}\s*(?:system|instructions?)\s*:",
        _phrase(r"\bnew (?:system )?instructions?\s*:"),
        _phrase(r"\b(?:system|admin|developer) (?:override|message|notice)\s*:"),
    )],
    Reason.PROMPT_LEAK: [re.compile(p, re.IGNORECASE) for p in (
        # "show me your system prompt", "what is the initial prompt". The object must
        # be "prompt": "show me the system configuration for a 5 kW install" passes.
        rf"{_REQUEST_VERB}.{{0,30}}?\b(?:system|initial|hidden|secret|internal|developer"
        r"|pre-?)[\s_\-]?prompt\b",
        # "print your hidden instructions", "what are the secret instructions"
        rf"{_REQUEST_VERB}.{{0,30}}?\b(?:hidden|secret|internal) instructions\b",
        # "what are your instructions", "show me your configuration".
        # Not "what are the rules for net metering" (needs "your").
        rf"{_REQUEST_VERB}.{{0,30}}?\b(?:your|ur) (?:system |initial |hidden |secret "
        r"|internal |original )?(?:instructions|prompt|rules|configuration|config"
        r"|guidelines|directives)\b",
        # "repeat everything above", "print the text before this verbatim"
        r"\b(?:repeat|print|output|recite|echo|copy)\b.{0,25}?\b(?:everything|all "
        r"(?:the )?(?:text|words)|the (?:text|words)|verbatim)\b.{0,15}?"
        r"\b(?:above|before|preceding)\b",
        r"\bwhat (?:was|is) written (?:above|before)\b",
    )],
    Reason.SECRET_REQUEST: [re.compile(p, re.IGNORECASE) for p in (
        # The literal env var name is never part of a real energy question.
        r"gemini_api_key",
        # "what is your API key", "print the server's credentials". "your/the bot's"
        # is required: "how do I get an API key for my utility" must only be flagged.
        rf"{_REQUEST_VERB}.{{0,30}}?\b(?:your|ur|the bot'?s|the server'?s|this bot'?s"
        r"|gemini|google)\s+(?:own\s+|secret\s+|backend\s+)?(?:api[\s_\-]?keys?"
        r"|credentials|secrets?|passwords?|env(?:ironment)?\s+variables?|\.env)",
    )],
}

# Weak patterns: a match only adds a flag for G2. These words appear in genuine
# questions too ("I need an API key for my utility's portal"), so they never block.
_WEAK_PATTERNS: dict[Flag, list[re.Pattern[str]]] = {
    Flag.WEAK_INJECTION: [re.compile(p, re.IGNORECASE) for p in (
        r"\byou are now\b",
        r"\bfrom now on,? you\b",
        r"\bdeveloper mode\b",
        r"\bjailbr(?:eak|oken)\b",
        r"\btranslate\b.{0,60}?\b(?:then|and)\s+(?:follow|execute|obey|do)\b",
    )],
    Flag.WEAK_ROLEPLAY: [re.compile(p, re.IGNORECASE) for p in (
        r"\bpretend\b",
        r"\bimagine (?:you(?:'re| are)|that you)\b",
        r"\brole[\s\-]?play",
        r"\blet'?s play a game\b",
        r"\byou\b.{0,20}?\bact as\b",
        r"^act as\b",
    )] + [re.compile(r"\bDAN\b")],  # case-sensitive: "Dan" the name is fine
    Flag.WEAK_PROMPT_LEAK: [re.compile(p, re.IGNORECASE) for p in (
        r"\bsystem[\s_\-]?prompt\b",
        r"\b(?:initial|hidden|internal) (?:prompt|instructions)\b",
    )],
    Flag.WEAK_SECRET: [re.compile(p, re.IGNORECASE) for p in (
        r"\bapi[\s_\-]?keys?\b",
        r"\bcredentials\b",
        r"\benv(?:ironment)? variables?\b",
        r"\.env\b",
    )],
}


# ---- Checks ---------------------------------------------------------------------------

def _first_strong_hit(text: str) -> Reason | None:
    """Return the Reason of the first strong pattern found in text, or None."""
    for reason, patterns in _STRONG_PATTERNS.items():
        if any(p.search(text) for p in patterns):
            return reason
    return None


def _weak_flags(text: str) -> set[Flag]:
    """All weak flags whose patterns appear in text."""
    return {flag for flag, patterns in _WEAK_PATTERNS.items()
            if any(p.search(text) for p in patterns)}


def _transformed_views(text: str) -> list[str]:
    """Readings of the text an attacker might have intended us to miss.

    ROT13 and reversal can't be detected, so we always try them. That is safe because
    only *strong* patterns run on these views, and those essentially never appear by
    accident in the ROT13 or reverse of a real question. The fully folded view catches
    words written entirely in Cyrillic/Greek look-alikes without changing what G2 sees.
    """
    return [decoders.despace(text), decoders.unleet(text),
            decoders.rot13(text), decoders.reverse(text),
            decoders.fold_all_confusables(text)]


def _block(reason: Reason, text: str, flags: set[Flag],
           decoded: tuple[str, ...] = ()) -> InputFilterResult:
    return InputFilterResult(allowed=False, reason=reason, normalized_text=text,
                             flags=frozenset(flags), decoded_segments=decoded)


def check_input(text: str, *, max_chars: int) -> InputFilterResult:
    """Run G1 on one user message.

    `max_chars` is passed in (from Settings.max_input_chars) rather than read here, so
    this function stays pure and trivially testable.
    """
    # Length is checked on the raw text first so a huge payload is rejected before we
    # spend any time normalising or decoding it.
    if len(text) > max_chars:
        return _block(Reason.TOO_LONG, "", set())

    norm = decoders.normalise(text)
    clean = norm.text
    flags: set[Flag] = set()
    if norm.had_invisible:
        flags.add(Flag.INVISIBLE_CHARS)
    if norm.had_homoglyphs:
        flags.add(Flag.HOMOGLYPHS)

    if not clean:
        return _block(Reason.EMPTY, clean, flags)
    if len(clean) > max_chars:  # NFKC can expand some characters
        return _block(Reason.TOO_LONG, clean, flags)

    # Plain text: a hit here is a direct attack (possibly after Unicode tricks, which
    # normalisation already undid and the flags record).
    reason = _first_strong_hit(clean)
    if reason is not None:
        return _block(reason, clean, flags)

    # Decoded payloads (base64/hex/URL and hidden tag text) and transformed views.
    decoded = tuple(([norm.hidden_text] if norm.hidden_text else [])
                    + decoders.decode_payloads(clean))
    if decoded:
        flags.add(Flag.ENCODED_PAYLOAD)
    hidden_views = list(decoded) + _transformed_views(clean)
    if any(_first_strong_hit(view) for view in hidden_views):
        return _block(Reason.ENCODED_INJECTION, clean, flags, decoded)

    # Weak signals on the plain text and on anything we decoded.
    for view in (clean, *decoded):
        flags |= _weak_flags(view)

    return InputFilterResult(allowed=True, reason=Reason.OK, normalized_text=clean,
                             flags=frozenset(flags), decoded_segments=decoded)
