"""Sentiment scoring — combines Reddit and CryptoPanic signals with optional FinBERT."""

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

    def __init__(self, *, use_finbert: bool = False) -> None:
        self._use_finbert = use_finbert
        self._finbert_pipeline = None
        self._buzz_history: dict[str, list[int]] = {}

    def _load_finbert(self):
        """Lazily load FinBERT from HuggingFace."""
        if self._finbert_pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1,  # CPU
            )
            logger.info("FinBERT model loaded for sentiment scoring")
        except Exception as e:
            logger.warning("Failed to load FinBERT: %s", e)
            self._use_finbert = False

    def score_texts(self, texts: list[str]) -> list[tuple[str, float]]:
        """Score a batch of texts using FinBERT.

        Returns list of (label, score) where label is positive/negative/neutral.
        """
        if not self._use_finbert or not texts:
            return []

        self._load_finbert()
        if self._finbert_pipeline is None:
            return []

        try:
            truncated = [t[:512] for t in texts]
            results = self._finbert_pipeline(truncated, batch_size=16, truncation=True)
            scored = []
            for r in results:
                label = r["label"].lower()
                conf = r["score"]
                if label == "positive":
                    scored.append(("positive", conf))
                elif label == "negative":
                    scored.append(("negative", -conf))
                else:
                    scored.append(("neutral", 0.0))
            return scored
        except Exception as e:
            logger.warning("FinBERT scoring failed: %s", e)
            return []

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

            reddit_score = 0.0
            if self._use_finbert and reddit_top_posts:
                finbert_scores = self.score_texts(reddit_top_posts)
                if finbert_scores:
                    reddit_score = sum(s for _, s in finbert_scores) / len(finbert_scores)
            else:
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

                if self._use_finbert:
                    finbert_scores = self.score_texts(news_headlines[:5])
                    if finbert_scores:
                        finbert_avg = sum(s for _, s in finbert_scores) / len(finbert_scores)
                        scores.append(finbert_avg)
                        weights.append(1.5)
                        signal.data_sources.append("finbert")

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
                lines.append(f"    - \"{narrative}\"")

        if sig.news_headlines:
            for headline in sig.news_headlines[:2]:
                lines.append(f"    - [{headline}]")

    return "\n".join(lines) if lines else "No sentiment data available."
