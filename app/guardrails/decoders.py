"""Text normalisation and decoding helpers for the G1 input filter.

WHY a separate module: input_filter.py should read as policy ("what do we block and
why"). The mechanics of undoing Unicode tricks and encodings live here. Every function
is pure (string in, string out), so each one can be tested on its own.

ReDoS safety: every regex here has a single quantifier with no nested repetition, so
matching time grows linearly with input length.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import unquote

# ---- Invisible characters -------------------------------------------------------------

# Zero-width and bidi controls. They are invisible on screen but split words
# ("ig​nore") or reorder text, so a regex would miss "ignore".
_ZERO_WIDTH = {"​", "‌", "‍", "⁠", "﻿", "­", "᠎"}
_BIDI = {"‎", "‏", "؜", *(chr(c) for c in range(0x202A, 0x202F)),
         *(chr(c) for c in range(0x2066, 0x206A))}
_KEEP_CONTROLS = {"\n", "\t", "\r"}

# Unicode "tag" characters (U+E0000 block) mirror ASCII but render as nothing.
# Attackers use them to smuggle a hidden instruction inside a normal-looking message.
_TAG_START, _TAG_END = 0xE0000, 0xE007F
_TAG_ASCII_OFFSET = 0xE0000


def extract_tag_text(text: str) -> tuple[str, str]:
    """Split text into (visible text, hidden ASCII carried by tag characters).

    WHY decode instead of just deleting: the hidden part *is* the attack. We remove it
    from what the user sees, but still inspect it.
    """
    visible: list[str] = []
    hidden: list[str] = []
    for ch in text:
        cp = ord(ch)
        if _TAG_START <= cp <= _TAG_END:
            if 0x20 <= cp - _TAG_ASCII_OFFSET <= 0x7E:
                hidden.append(chr(cp - _TAG_ASCII_OFFSET))
        else:
            visible.append(ch)
    return "".join(visible), "".join(hidden)


def strip_invisible(text: str) -> tuple[str, bool]:
    """Remove zero-width, bidi and other control/format characters.

    Returns (clean text, whether anything was removed). Newlines, tabs and carriage
    returns are kept because they are ordinary whitespace, not tricks.
    """
    kept: list[str] = []
    removed = False
    for ch in text:
        if ch in _KEEP_CONTROLS:
            kept.append(ch)
        elif ch in _ZERO_WIDTH or ch in _BIDI or unicodedata.category(ch) in ("Cf", "Cc"):
            removed = True
        else:
            kept.append(ch)
    return "".join(kept), removed


# ---- Homoglyphs -----------------------------------------------------------------------

# Cyrillic and Greek letters that look like Latin ones. This is a deliberately small,
# high-value subset of Unicode's confusables list.
_CONFUSABLES = str.maketrans({
    # Cyrillic
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p",
    "с": "c", "т": "t", "у": "y", "х": "x", "ѕ": "s", "і": "i", "ј": "j", "ԁ": "d",
    "ԛ": "q", "ԝ": "w", "ӏ": "l", "ɡ": "g",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X", "Ѕ": "S", "І": "I", "Ј": "J",
    # Greek
    "α": "a", "ο": "o", "ρ": "p", "ν": "v", "ι": "i", "κ": "k", "τ": "t", "υ": "u",
    "Α": "A", "Β": "B", "Ε": "E", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y", "Ζ": "Z",
})


def _script_of(ch: str) -> str:
    """Rough script of a letter: 'latin', 'other' (Cyrillic/Greek) or '' (not a letter)."""
    if not ch.isalpha():
        return ""
    name = unicodedata.name(ch, "")
    if name.startswith("LATIN"):
        return "latin"
    if name.startswith(("CYRILLIC", "GREEK")):
        return "other"
    return ""


def fold_homoglyphs(text: str) -> tuple[str, bool]:
    """Replace look-alike letters, but only inside mixed-script words.

    WHY only mixed-script words: "ignоre" with a Cyrillic "о" is an attack, but a
    genuine Russian or Greek sentence must reach G2 unchanged, because the bot supports
    non-English questions.
    """
    changed = False
    words = text.split(" ")
    for i, word in enumerate(words):
        scripts = {_script_of(ch) for ch in word}
        if {"latin", "other"} <= scripts:
            folded = word.translate(_CONFUSABLES)
            if folded != word:
                words[i] = folded
                changed = True
    return " ".join(words), changed


_WHITESPACE = re.compile(r"\s+")


def collapse_spaces(text: str) -> str:
    """Turn every run of whitespace into one space and trim the ends."""
    return _WHITESPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class Normalised:
    """Output of normalise(): the clean text plus what we had to undo to get it."""

    text: str
    hidden_text: str
    had_invisible: bool
    had_homoglyphs: bool


def normalise(raw: str) -> Normalised:
    """Canonical form every later check (and G2/G3) sees.

    Order matters: tags are pulled out before NFKC (which leaves them alone anyway),
    NFKC turns fullwidth/compat letters into plain ones, then invisibles go, then
    homoglyphs are folded, and finally spaces are collapsed.
    """
    visible, hidden = extract_tag_text(raw)
    text = unicodedata.normalize("NFKC", visible)
    text, had_invisible = strip_invisible(text)
    text = collapse_spaces(text)
    text, had_homoglyphs = fold_homoglyphs(text)
    return Normalised(text=text, hidden_text=hidden,
                      had_invisible=had_invisible or bool(hidden),
                      had_homoglyphs=had_homoglyphs)


# ---- Cheap transformations (always applied, no detection possible) --------------------

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
                       "@": "a", "$": "s", "!": "i", "|": "l"})


def unleet(text: str) -> str:
    """Map common leetspeak digits/symbols back to letters ("1gn0r3" -> "ignore")."""
    return text.translate(_LEET)


def rot13(text: str) -> str:
    """ROT13 is its own inverse, so applying it decodes ROT13 text."""
    return codecs.encode(text, "rot13")


def reverse(text: str) -> str:
    """Reversed text ("snoitcurtsni suoiverp erongi") read forwards."""
    return text[::-1]


_SPLIT_LETTER_SEPARATORS = (".", "-", "_")


def _is_split_letters(token: str) -> str | None:
    """Return joined letters if token looks like "i.g.n.o.r.e", else None."""
    for sep in _SPLIT_LETTER_SEPARATORS:
        parts = token.split(sep)
        if len(parts) >= 4 and all(len(p) == 1 and p.isalpha() for p in parts):
            return "".join(parts)
    return None


def despace(text: str) -> str:
    """Join letters written one at a time ("i g n o r e" or "i.g.n.o.r.e").

    WHY a token loop instead of a regex: the obvious regex needs a repeated group,
    which is the shape that can cause catastrophic backtracking. This is linear.
    Only runs of 4+ single letters are joined, so "a 5 kW PV array" is untouched.
    """
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 4:
            out.append("".join(run))
        else:
            out.extend(run)
        run.clear()

    for token in text.split(" "):
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        flush()
        joined = _is_split_letters(token)
        out.append(joined if joined is not None else token)
    flush()
    return " ".join(out)


# ---- Encodings that can be detected and decoded ---------------------------------------

# Long runs of base64 alphabet (standard and URL-safe). 16+ chars so ordinary words
# and short IDs are not candidates.
_BASE64_TOKEN = re.compile(r"[A-Za-z0-9+/_\-]{16,}={0,2}")
# Runs of hex digits, optionally with spaces between byte pairs ("69 67 6e ...").
_HEX_RUN = re.compile(r"[0-9a-fA-F][0-9a-fA-F ]{14,}[0-9a-fA-F]")
_HEX_PREFIX = re.compile(r"\\x|0x", re.IGNORECASE)
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")

_MAX_SEGMENTS = 20
_MAX_DEPTH = 2


def looks_like_text(candidate: str) -> bool:
    """True if a decoded blob is readable language rather than binary noise.

    WHY this gate matters: meter IDs, serial numbers and URL tokens are often valid
    base64 or hex by accident. Decoded, they become random bytes. Requiring printable,
    mostly-letter output keeps those from being flagged (no false positives on IDs).
    """
    if len(candidate) < 4 or not candidate.isprintable():
        return False
    letters = sum(ch.isalpha() for ch in candidate)
    letters_and_spaces = letters + candidate.count(" ")
    return letters >= 4 and letters_and_spaces / len(candidate) >= 0.8


def _decode_base64(token: str) -> str | None:
    """Decode one base64/base64url token to UTF-8, or None if it isn't text."""
    body = token.rstrip("=").replace("-", "+").replace("_", "/")
    if len(body) % 4 == 1:  # impossible length; drop a trailing stray char
        body = body[:-1]
    body += "=" * (-len(body) % 4)
    try:
        decoded = base64.b64decode(body, validate=True).decode("utf-8")
    except (binascii.Error, ValueError):
        return None
    return decoded if looks_like_text(decoded) else None


def _decode_hex(run: str) -> str | None:
    """Decode a hex run (spaces allowed between pairs) to UTF-8, or None."""
    digits = run.replace(" ", "")
    if len(digits) < 16 or len(digits) % 2:
        return None
    try:
        decoded = bytes.fromhex(digits).decode("utf-8")
    except ValueError:
        return None
    return decoded if looks_like_text(decoded) else None


def _decode_url(text: str) -> str | None:
    """Undo (possibly double) percent-encoding when there are at least two escapes."""
    if len(_PERCENT_ESCAPE.findall(text)) < 2:
        return None
    decoded = text
    for _ in range(3):  # handles double/triple encoding like %2569
        nxt = unquote(decoded)
        if nxt == decoded:
            break
        decoded = nxt
    return decoded if decoded != text else None


def _decode_once(text: str) -> list[str]:
    """All payloads we can decode from text in one pass."""
    found: list[str] = []
    url = _decode_url(text)
    if url is not None:
        found.append(url)
    for match in _BASE64_TOKEN.finditer(text):
        decoded = _decode_base64(match.group())
        if decoded is not None:
            found.append(decoded)
    unprefixed = _HEX_PREFIX.sub("", text)
    for match in _HEX_RUN.finditer(unprefixed):
        decoded = _decode_hex(match.group().strip())
        if decoded is not None:
            found.append(decoded)
    return found


def decode_payloads(text: str) -> list[str]:
    """Decode base64, hex and URL-encoding, recursively up to two levels deep.

    WHY recursive: "base64 inside URL-encoding" is a common way to slip past a filter
    that only peels off one layer. The depth and segment caps bound the work.
    """
    results: list[str] = []
    frontier = [text]
    for _ in range(_MAX_DEPTH):
        next_frontier: list[str] = []
        for item in frontier:
            for decoded in _decode_once(item):
                if decoded not in results and len(results) < _MAX_SEGMENTS:
                    results.append(decoded)
                    next_frontier.append(decoded)
        frontier = next_frontier
    return results
