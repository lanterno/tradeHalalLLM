"""Lexicon-based polarity classifier for financial-news headlines.

The Yahoo Finance search endpoint (and most free news feeds) ships
items without per-item polarity tags. Tagging each headline
``"neutral"`` defeats the LLM's ability to weight bullish vs bearish
context — a "beats earnings" headline reads identical to a "misses
guidance" one downstream.

This module is a small curated lexicon over the financial-news
register, scored the same way ``trading/fed_speak.py`` scores Fed
speeches. The shipping principle is the same: pure stdlib, no
HuggingFace install, deterministic + fast (we score every news item
on every cycle, so a transformer model would be overkill).

Output is the three :class:`NewsEvent.sentiment` labels the
CryptoPanic path already emits — ``"positive"``, ``"negative"``,
``"neutral"`` — so downstream consumers (the prompt formatter, the
news-event reactor) don't need to discriminate by source.
"""

from __future__ import annotations

import re
from typing import Literal

Polarity = Literal["positive", "negative", "neutral"]


# ── Lexicon ──────────────────────────────────────────────────────


# Curated for U.S. equities news register. Tokens are matched as
# whole words (case-insensitive); each occurrence adds its weight to
# the positive or negative bucket.
POSITIVE_TOKENS: dict[str, float] = {
    # Earnings beats / forward upside
    "beats": 1.5,
    "beat": 1.2,
    "beating": 1.2,
    "outperforms": 1.5,
    "tops": 1.0,
    "topped": 1.0,
    "exceeded": 1.2,
    "surges": 1.5,
    "soars": 1.5,
    "rallies": 1.2,
    "jumps": 1.0,
    "climbs": 0.7,
    "rises": 0.5,
    "gains": 0.5,
    "advances": 0.6,
    # Forward-looking positive
    "upgrade": 1.5,
    "upgraded": 1.5,
    "upgrades": 1.2,
    "raised": 0.8,
    "boosts": 1.0,
    "boosted": 1.0,
    "expands": 0.7,
    "expansion": 0.6,
    "growth": 0.5,
    "record": 0.8,
    # Deal flow
    "acquires": 0.6,
    "acquisition": 0.5,
    "merger": 0.4,
    "approves": 0.7,
    "approved": 0.7,
    "approval": 0.7,
    "launches": 0.5,
    "expand": 0.5,
    # Narrative / sentiment-of-coverage
    "strong": 0.6,
    "robust": 0.7,
    "bullish": 1.2,
    "buy": 0.6,  # analyst rating
    "outperform": 1.2,
    "overweight": 0.8,
}


NEGATIVE_TOKENS: dict[str, float] = {
    # Earnings misses / forward downside
    "misses": 1.5,
    "missed": 1.2,
    "miss": 1.0,
    "underperforms": 1.5,
    "disappoints": 1.5,
    "disappointed": 1.2,
    "plunges": 1.5,
    "tumbles": 1.5,
    "slides": 1.2,
    "drops": 1.0,
    "falls": 0.6,
    "declines": 0.6,
    "loses": 0.5,
    "slumps": 1.2,
    "sinks": 1.2,
    # Forward-looking negative
    "downgrade": 1.5,
    "downgraded": 1.5,
    "downgrades": 1.2,
    "cuts": 0.8,
    "warning": 1.2,
    "warns": 1.2,
    "warned": 1.0,
    "lowered": 0.7,
    "slashes": 1.5,
    "delays": 0.6,
    "delayed": 0.5,
    "halts": 1.0,
    # Regulatory / legal
    "investigation": 1.2,
    "investigates": 1.0,
    "subpoena": 1.5,
    "lawsuit": 1.0,
    "sues": 1.0,
    "fined": 1.0,
    "fine": 0.6,
    "penalty": 0.8,
    "recall": 1.2,
    "fraud": 1.5,
    "scandal": 1.5,
    "probe": 1.0,
    "settles": 0.5,  # mildly negative — implies wrongdoing
    "settlement": 0.5,
    # Operational pain
    "layoffs": 1.2,
    "layoff": 1.2,
    "cuts-jobs": 1.5,
    "bankruptcy": 2.0,
    "chapter-11": 2.0,
    "default": 1.5,
    "loss": 0.7,
    "losses": 0.7,
    # Narrative / sentiment-of-coverage
    "weak": 0.7,
    "weakness": 0.8,
    "bearish": 1.2,
    "sell": 0.6,  # analyst rating
    "underperform": 1.2,
    "underweight": 0.8,
    "concerns": 0.6,
    "concerning": 0.7,
    "risk": 0.4,
    "risks": 0.4,
}


# Net-score threshold to flip from neutral to a directional tag.
# Tuned so a single weak token (e.g. ``"rises"`` at +0.5) doesn't
# flip neutral → positive — needs at least two weak tokens or one
# strong one. Symmetric on the negative side.
_POLARITY_THRESHOLD = 0.6


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


# ── Scorer ───────────────────────────────────────────────────────


def score_headline(headline: str) -> tuple[float, float]:
    """Return ``(positive_score, negative_score)`` for one headline.

    Tokens are matched as whole words (case-insensitive). Each
    occurrence adds the lexicon weight to the respective bucket.
    Empty or whitespace-only headlines score ``(0, 0)``.
    """
    if not headline or not headline.strip():
        return 0.0, 0.0
    positive = 0.0
    negative = 0.0
    for tok in _TOKEN_RE.findall(headline.lower()):
        if tok in POSITIVE_TOKENS:
            positive += POSITIVE_TOKENS[tok]
        elif tok in NEGATIVE_TOKENS:
            negative += NEGATIVE_TOKENS[tok]
    return positive, negative


def classify_headline(headline: str) -> Polarity:
    """Classify a headline as positive / negative / neutral.

    Net score (``positive - negative``) above ``+_POLARITY_THRESHOLD``
    flips to positive; below ``-_POLARITY_THRESHOLD`` flips to
    negative; everything else stays neutral. This is a deliberately
    conservative threshold — a confused weak-signal headline lands
    on neutral, not on a hallucinated polarity.

    The output strings match :class:`NewsEvent.sentiment` literals
    that the CryptoPanic path already emits, so the downstream
    consumers (prompt formatter, event reactor) handle both sources
    identically.
    """
    positive, negative = score_headline(headline)
    net = positive - negative
    if net >= _POLARITY_THRESHOLD:
        return "positive"
    if net <= -_POLARITY_THRESHOLD:
        return "negative"
    return "neutral"


__all__ = [
    "Polarity",
    "POSITIVE_TOKENS",
    "NEGATIVE_TOKENS",
    "classify_headline",
    "score_headline",
]
