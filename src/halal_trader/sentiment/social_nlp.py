"""Lexicon-based social-media sentiment NLP — Round-5 Wave 11.A.

Reddit / StockTwits / Discord trader-chat is high-signal but noisy.
The bot's existing `sentiment/scoring.py` is keyword-density based;
this module adds a **trading-vocabulary-tuned lexicon** that scores
short messages on a [-1, +1] sentiment axis using:

- domain-specific lexicon (bull/bear words from trading vocabulary)
- emoji handling (🚀, 🌙, 💎, 📉)
- intensifiers ("very", "huge")
- negation handling ("not bullish")
- ALL CAPS amplification (heuristic for emphasis)
- ticker-mention extraction

This is **lexicon-based**, not LLM. The Round-4 filing-classifier
covers LLM-grade work; this module is the per-message fast-path
that runs in the cycle without an LLM call.

Pinned semantics:

- **Closed-set Sentiment ladder** (BEARISH / NEUTRAL / BULLISH).
- **Score is in [-1, +1]**, clipped on output.
- **Negation flips local sentiment for next 3 tokens** (matches
  natural-language scope of "not", "no").
- **Empty / whitespace-only message returns NEUTRAL with score 0.**
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Sentiment(str, Enum):
    """Closed-set sentiment ladder."""

    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"


# Trading vocabulary — signed weights.
_BULL_WORDS: dict[str, float] = {
    "moon": 0.9,
    "rally": 0.8,
    "breakout": 0.8,
    "buy": 0.6,
    "long": 0.5,
    "bullish": 1.0,
    "calls": 0.6,
    "rip": 0.7,
    "pump": 0.6,
    "ath": 0.7,
    "uptrend": 0.7,
    "support": 0.4,
    "diamond": 0.7,
    "hodl": 0.5,
    "to_the_moon": 1.0,
}

_BEAR_WORDS: dict[str, float] = {
    "dump": -0.7,
    "crash": -0.9,
    "tank": -0.8,
    "bearish": -1.0,
    "puts": -0.6,
    "short": -0.5,
    "sell": -0.5,
    "exit": -0.4,
    "downtrend": -0.7,
    "resistance": -0.4,
    "rugpull": -1.0,
    "rug": -0.9,
    "bagholder": -0.6,
    "rekt": -0.8,
    "blood": -0.7,
}

_INTENSIFIERS: dict[str, float] = {
    "very": 1.3,
    "extremely": 1.5,
    "huge": 1.4,
    "massive": 1.4,
    "super": 1.3,
}

_NEGATIONS: frozenset[str] = frozenset({"not", "no", "never", "n't"})

_EMOJI_WEIGHTS: dict[str, float] = {
    "🚀": 0.9,
    "🌙": 0.6,
    "💎": 0.6,
    "🐂": 0.7,
    "📈": 0.6,
    "🟢": 0.4,
    "📉": -0.6,
    "🐻": -0.7,
    "💀": -0.7,
    "🔴": -0.4,
    "💩": -0.6,
}

_TICKER_RE = re.compile(r"\$([A-Z]{1,6})\b")
_TOKEN_RE = re.compile(r"[A-Za-z']+|[\U0001F000-\U0001FAFF☀-➿]")


@dataclass(frozen=True)
class SentimentPolicy:
    """Operator-tunable thresholds."""

    bullish_threshold: float = 0.20
    bearish_threshold: float = -0.20
    caps_amplification: float = 1.2
    negation_window: int = 3

    def __post_init__(self) -> None:
        if self.bullish_threshold <= 0 or self.bearish_threshold >= 0:
            raise ValueError(
                "bullish_threshold > 0 and bearish_threshold < 0 required"
            )
        if not 1.0 <= self.caps_amplification <= 3.0:
            raise ValueError("caps_amplification must be in [1.0, 3.0]")
        if self.negation_window <= 0:
            raise ValueError("negation_window must be positive")


@dataclass(frozen=True)
class SentimentScore:
    """Result of scoring a single message."""

    score: float
    sentiment: Sentiment
    tickers: tuple[str, ...]
    tokens_evaluated: int

    def __post_init__(self) -> None:
        if not -1.0 <= self.score <= 1.0:
            raise ValueError("score must be in [-1, 1]")
        if self.tokens_evaluated < 0:
            raise ValueError("tokens_evaluated must be non-negative")


def extract_tickers(text: str) -> tuple[str, ...]:
    """Extract `$XYZ` style ticker mentions from text."""
    return tuple(_TICKER_RE.findall(text))


def _is_all_caps(token: str) -> bool:
    return len(token) >= 2 and token.isalpha() and token.isupper()


def score_message(
    text: str, *, policy: SentimentPolicy | None = None
) -> SentimentScore:
    """Score a single message on [-1, +1] sentiment axis."""
    pol = policy if policy is not None else SentimentPolicy()
    if not text or not text.strip():
        return SentimentScore(
            score=0.0,
            sentiment=Sentiment.NEUTRAL,
            tickers=(),
            tokens_evaluated=0,
        )

    tickers = extract_tickers(text)
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return SentimentScore(0.0, Sentiment.NEUTRAL, tickers, 0)

    total = 0.0
    counted = 0
    pending_intensifier = 1.0
    negation_remaining = 0

    for raw in tokens:
        is_caps = _is_all_caps(raw)
        token = raw.lower()

        # Update negation window
        if token in _NEGATIONS:
            negation_remaining = pol.negation_window
            continue

        # Intensifier modifies next sentiment token
        if token in _INTENSIFIERS:
            pending_intensifier = max(pending_intensifier, _INTENSIFIERS[token])
            continue

        weight = 0.0
        if token in _BULL_WORDS:
            weight = _BULL_WORDS[token]
        elif token in _BEAR_WORDS:
            weight = _BEAR_WORDS[token]
        elif raw in _EMOJI_WEIGHTS:
            weight = _EMOJI_WEIGHTS[raw]

        if weight != 0.0:
            if is_caps:
                weight *= pol.caps_amplification
            weight *= pending_intensifier
            if negation_remaining > 0:
                weight = -weight
            total += weight
            counted += 1
            pending_intensifier = 1.0

        if negation_remaining > 0:
            negation_remaining -= 1

    score = total / max(counted, 1) if counted > 0 else 0.0
    score = max(-1.0, min(1.0, score))

    if score >= pol.bullish_threshold:
        sentiment = Sentiment.BULLISH
    elif score <= pol.bearish_threshold:
        sentiment = Sentiment.BEARISH
    else:
        sentiment = Sentiment.NEUTRAL

    return SentimentScore(
        score=score,
        sentiment=sentiment,
        tickers=tickers,
        tokens_evaluated=counted,
    )


def aggregate_scores(scores: Iterable[SentimentScore]) -> SentimentScore:
    """Aggregate multiple message scores into a single composite."""
    scores_t = tuple(scores)
    if not scores_t:
        return SentimentScore(0.0, Sentiment.NEUTRAL, (), 0)
    total_score = sum(s.score for s in scores_t) / len(scores_t)
    tickers = tuple(sorted({t for s in scores_t for t in s.tickers}))
    total_evaluated = sum(s.tokens_evaluated for s in scores_t)
    pol = SentimentPolicy()
    if total_score >= pol.bullish_threshold:
        sentiment = Sentiment.BULLISH
    elif total_score <= pol.bearish_threshold:
        sentiment = Sentiment.BEARISH
    else:
        sentiment = Sentiment.NEUTRAL
    return SentimentScore(
        score=total_score,
        sentiment=sentiment,
        tickers=tickers,
        tokens_evaluated=total_evaluated,
    )


def render_score(score: SentimentScore) -> str:
    emoji = {
        Sentiment.BULLISH: "🟢",
        Sentiment.NEUTRAL: "🟡",
        Sentiment.BEARISH: "🔴",
    }[score.sentiment]
    tickers = ", ".join(f"${t}" for t in score.tickers) or "(none)"
    return (
        f"{emoji} {score.sentiment.value} "
        f"score={score.score:+.3f} tickers={tickers} "
        f"tokens={score.tokens_evaluated}"
    )
