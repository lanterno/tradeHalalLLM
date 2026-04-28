"""Sentiment scoring — combines Reddit and CryptoPanic signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SentimentSignal:
    """Composite sentiment signal for a single trading pair."""

    pair: str
    score: float = 0.0  # -1.0 (extremely bearish) to +1.0 (extremely bullish)
    buzz: float = 0.0  # Buzz multiplier vs average (1.0 = normal, 3.0 = 3x spike)
    confidence: float = 0.0  # 0-1 based on data volume
    top_narratives: list[str] = field(default_factory=list)
    news_headlines: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)


class SentimentScorer:
    """Combines sentiment signals from multiple sources into a composite score."""

    def __init__(self) -> None:
        self._buzz_history: dict[str, list[int]] = {}

    def compute_composite(
        self,
        pair: str,
        reddit_mentions: int = 0,
        reddit_top_posts: list[str] | None = None,
        reddit_avg_score: float = 0.0,
        news_sentiment: float = 0.0,
        news_headlines: list[str] | None = None,
        news_count: int = 0,
    ) -> SentimentSignal:
        """Compute a composite sentiment signal from all available sources."""
        signal = SentimentSignal(pair=pair)
        scores: list[float] = []
        weights: list[float] = []

        # Reddit component
        if reddit_mentions > 0:
            buzz = self._compute_buzz(pair, reddit_mentions)
            signal.buzz = buzz
            signal.data_sources.append("reddit")

            reddit_score = min(1.0, max(-1.0, (reddit_avg_score - 10) / 50))
            scores.append(reddit_score)
            weight = min(2.0, buzz) if buzz > 1.5 else 1.0
            weights.append(weight)

            if reddit_top_posts:
                signal.top_narratives = reddit_top_posts[:3]

        # CryptoPanic component
        if news_count > 0:
            signal.data_sources.append("cryptopanic")
            scores.append(news_sentiment)
            weights.append(1.2)

            if news_headlines:
                signal.news_headlines = news_headlines[:3]

        # Weighted average
        if scores and weights:
            total_weight = sum(weights)
            signal.score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            signal.confidence = min(1.0, (reddit_mentions + news_count) / 20)

        return signal

    def _compute_buzz(self, pair: str, current_mentions: int) -> float:
        """Compute buzz as ratio of current mentions to rolling average."""
        history = self._buzz_history.setdefault(pair, [])
        history.append(current_mentions)

        if len(history) > 168:  # 7 days of hourly data
            history[:] = history[-168:]

        if len(history) < 2:
            return 1.0

        avg = sum(history[:-1]) / len(history[:-1])
        if avg <= 0:
            return float(current_mentions) if current_mentions > 0 else 1.0
        return current_mentions / avg


def format_sentiment_for_prompt(signals: dict[str, SentimentSignal]) -> str:
    """Format sentiment signals into a text block for the LLM prompt."""
    if not signals:
        return "No sentiment data available."

    lines = []
    for pair, sig in sorted(signals.items()):
        if not sig.data_sources:
            continue

        direction = "BULLISH" if sig.score > 0.1 else ("BEARISH" if sig.score < -0.1 else "NEUTRAL")
        buzz_label = ""
        if sig.buzz >= 3.0:
            buzz_label = " [HIGH BUZZ]"
        elif sig.buzz >= 2.0:
            buzz_label = " [ELEVATED BUZZ]"

        lines.append(
            f"  {pair}: sentiment={sig.score:+.2f} ({direction}){buzz_label}, "
            f"confidence={sig.confidence:.0%}, sources={','.join(sig.data_sources)}"
        )

        if sig.top_narratives:
            for narrative in sig.top_narratives[:2]:
                lines.append(f'    - "{narrative}"')

        if sig.news_headlines:
            for headline in sig.news_headlines[:2]:
                lines.append(f"    - [{headline}]")

    return "\n".join(lines) if lines else "No sentiment data available."
