"""Deterministic heuristic metrics for NRAM evaluation harness.

All functions are pure, offline, and rely only on the Python standard library
(no network calls). They are intentionally conservative heuristics meant to
characterize *stylistic* differences between a baseline model and an
NRAM-steered model: corporate-jargon density, repetition, sentence shape,
product-noun grounding, and forbidden cliches.

Every function is robust to empty / whitespace-only input and returns a
sensible zero value in that case.
"""
from __future__ import annotations

import re
from typing import List

# A token boundary that tolerates punctuation, digits, and unicode.
_B = r"(?<![A-Za-z0-9])"
_E = r"(?![A-Za-z0-9])"

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

CORPORATE_JARGON = [
    "synergy",
    "stakeholder alignment",
    "best-in-class",
    "leveraging",
    "leverage",
    "solutioning",
    "value-added",
    "paradigm shift",
    "robust ecosystem",
    "holistic framework",
    "digital transformation journey",
    "seamless integration",
    "end-to-end solution",
    "mission-critical",
    "future-proof",
    "game-changer",
    "game changer",
    "low-hanging fruit",
    "move the needle",
    "circle back",
    "touch base",
    "boil the ocean",
    "synergize",
    "ideate",
    "disruptive innovation",
    "next-generation",
    "scalable solution",
    "holistic approach",
    "strategic roadmap",
    "core competency",
    "bandwidth",
    "actionable insights",
    "thought leadership",
    "ecosystem",
    "unlock value",
]

PRODUCT_NOUNS = [
    "device",
    "interface",
    "tool",
    "product",
    "feature",
    "sensor",
    "screen",
    "chip",
    "model",
    "platform",
    "hardware",
    "software",
    "app",
    "application",
    "assistant",
    "engine",
    "module",
    "component",
    "battery",
    "display",
    "camera",
    "speaker",
    "microphone",
    "keyboard",
    "processor",
    "system",
    "service",
    "network",
    "cloud",
    "algorithm",
    "dashboard",
    "widget",
    "plugin",
    "sdk",
    "api",
]

METAPHOR_MARKERS = [
    "as if",
    "as though",
    "such as",
    "like",
    "imagine",
    "feels",
    "feel",
    "resembles",
    "resemble",
    "echoes",
    "echo",
    "evokes",
    "evoke",
    "reminiscent",
    "metaphor",
]

FORBIDDEN_CLICHES = [
    "insanely great",
    "one more thing",
    "revolutionary",
    "magical",
    "think different",
    "game changer",
    "game-changer",
    "paradigm shift",
    "cutting edge",
    "cutting-edge",
    "best in class",
    "best-in-class",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_phrase(text: str, phrase: str) -> int:
    """Count case-insensitive, token-bounded occurrences of ``phrase``."""
    return len(re.findall(_B + re.escape(phrase.lower()) + _E, text.lower()))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"[.!?]+", text)
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------


def corporate_jargon_count(text: str) -> int:
    """Number of corporate-jargon phrase occurrences in ``text``."""
    if not text or not text.strip():
        return 0
    return sum(_count_phrase(text, p) for p in CORPORATE_JARGON)


def repeated_phrase_ratio(text: str) -> float:
    """Fraction of word 3-grams that repeat (appear more than once).

    Returns 0.0 when there are fewer than 3 tokens (no 3-grams possible).
    """
    if not text or not text.strip():
        return 0.0
    words = _tokenize(text)
    if len(words) < 3:
        return 0.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    counts: dict = {}
    for tg in trigrams:
        counts[tg] = counts.get(tg, 0) + 1
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(trigrams)


def average_sentence_length(text: str) -> float:
    """Mean number of words per sentence. 0.0 for empty input."""
    if not text or not text.strip():
        return 0.0
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0
    total = sum(len(_tokenize(s)) for s in sentences)
    return total / len(sentences)


def short_declarative_ratio(text: str) -> float:
    """Fraction of declarative sentences that are short (<= 8 words).

    Question sentences (ending in ``?``) are excluded from both the numerator
    and the denominator. Returns 0.0 when there are no declarative sentences.
    """
    if not text or not text.strip():
        return 0.0
    declarative = [s for s in _split_sentences(text) if "?" not in s]
    if not declarative:
        return 0.0
    short = sum(1 for s in declarative if len(_tokenize(s)) <= 8)
    return short / len(declarative)


def product_noun_count(text: str) -> int:
    """Number of product-related noun occurrences in ``text``."""
    if not text or not text.strip():
        return 0
    return sum(_count_phrase(text, w) for w in PRODUCT_NOUNS)


def metaphor_marker_count(text: str) -> int:
    """Number of metaphor / simile marker occurrences in ``text``."""
    if not text or not text.strip():
        return 0
    return sum(_count_phrase(text, m) for m in METAPHOR_MARKERS)


def forbidden_cliche_count(text: str) -> int:
    """Number of forbidden marketing-cliche occurrences in ``text``."""
    if not text or not text.strip():
        return 0
    return sum(_count_phrase(text, c) for c in FORBIDDEN_CLICHES)


def output_length(text: str) -> int:
    """Word count of ``text``."""
    if not text or not text.strip():
        return 0
    return len(_tokenize(text))


def compute_all(text: str) -> dict:
    """Compute every heuristic metric for ``text`` and return them as a dict."""
    if text is None:
        text = ""
    return {
        "corporate_jargon_count": corporate_jargon_count(text),
        "repeated_phrase_ratio": round(repeated_phrase_ratio(text), 6),
        "average_sentence_length": round(average_sentence_length(text), 4),
        "short_declarative_ratio": round(short_declarative_ratio(text), 6),
        "product_noun_count": product_noun_count(text),
        "metaphor_marker_count": metaphor_marker_count(text),
        "forbidden_cliche_count": forbidden_cliche_count(text),
        "output_length": output_length(text),
    }
