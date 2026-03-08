"""FinGPT-based financial sentiment analysis.

Uses HuggingFace transformers + peft to run FinGPT sentiment models locally.
These provide a supplementary market-sentiment signal that is fed into the main
LLM's decision context — they are NOT used as the decision-maker directly.

Install the optional ``fingpt`` dependency group to enable:

    uv pip install halal-trader[fingpt]
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Predefined headline templates used when no live news feed is available.
# The model scores these against the target symbol.
_PLACEHOLDER_HEADLINES = [
    "{symbol} stock shows strong momentum in today's trading session",
    "Analysts upgrade {symbol} citing robust revenue growth",
    "Market volatility impacts {symbol} shares amid sector rotation",
]


@dataclass
class SentimentScore:
    """Sentiment result for a single symbol."""

    symbol: str
    score: float  # -1.0 (very bearish) to +1.0 (very bullish)
    label: str  # "positive", "negative", "neutral"
    confidence: float  # 0-1

    @property
    def signal(self) -> str:
        """Human-readable trading signal."""
        if self.score > 0.3:
            return "bullish"
        if self.score < -0.3:
            return "bearish"
        return "neutral"


@dataclass
class SentimentAnalyzer:
    """Financial sentiment analyzer using FinGPT / FinBERT models.

    Falls back to a no-op analyzer when the ``fingpt`` extra is not installed,
    returning neutral scores so the rest of the pipeline is unaffected.
    """

    model_name: str = "ProsusAI/finbert"
    _pipeline: object | None = field(default=None, init=False, repr=False)
    _available: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline  # noqa: F401

            self._available = True
            logger.info("FinGPT/sentiment dependencies available (model: %s)", self.model_name)
        except ImportError:
            self._available = False
            logger.info(
                "transformers not installed — sentiment analysis disabled. "
                "Install with: uv pip install halal-trader[fingpt]"
            )

    def _get_pipeline(self) -> object | None:
        """Lazy-load the HuggingFace sentiment pipeline."""
        if not self._available:
            return None
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline

            logger.info("Loading sentiment model: %s (this may take a moment)...", self.model_name)
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
            )
            logger.info("Sentiment model loaded successfully")
        return self._pipeline

    async def analyze(self, symbol: str, headlines: list[str] | None = None) -> SentimentScore:
        """Analyze sentiment for a symbol given a list of headlines.

        If no headlines are provided, uses placeholder headlines.
        If the sentiment model is unavailable, returns a neutral score.
        """
        pipe = self._get_pipeline()

        if pipe is None:
            return SentimentScore(symbol=symbol, score=0.0, label="neutral", confidence=0.0)

        texts = headlines or [h.format(symbol=symbol) for h in _PLACEHOLDER_HEADLINES]

        try:
            results = pipe(texts)  # type: ignore[operator]

            # Aggregate scores across all headlines
            total_score = 0.0
            for r in results:
                label = r["label"].lower()
                raw_score = r["score"]
                if label == "positive":
                    total_score += raw_score
                elif label == "negative":
                    total_score -= raw_score
                # neutral contributes 0

            avg_score = total_score / len(results) if results else 0.0
            avg_confidence = sum(r["score"] for r in results) / len(results) if results else 0.0

            if avg_score > 0.1:
                agg_label = "positive"
            elif avg_score < -0.1:
                agg_label = "negative"
            else:
                agg_label = "neutral"

            return SentimentScore(
                symbol=symbol,
                score=round(avg_score, 4),
                label=agg_label,
                confidence=round(avg_confidence, 4),
            )
        except Exception as e:
            logger.warning("Sentiment analysis failed for %s: %s", symbol, e)
            return SentimentScore(symbol=symbol, score=0.0, label="neutral", confidence=0.0)

    async def analyze_batch(self, symbols: list[str]) -> dict[str, SentimentScore]:
        """Analyze sentiment for multiple symbols.

        Returns a dict mapping symbol -> SentimentScore.
        """
        results: dict[str, SentimentScore] = {}
        for symbol in symbols:
            results[symbol] = await self.analyze(symbol)
        return results

    def format_for_prompt(self, scores: dict[str, SentimentScore]) -> str:
        """Format sentiment scores as text for inclusion in an LLM prompt."""
        if not scores or all(s.confidence == 0.0 for s in scores.values()):
            return "Sentiment data: not available"

        lines = ["Sentiment signals (FinGPT analysis):"]
        for sym, s in sorted(scores.items()):
            lines.append(
                f"  {sym}: {s.signal} (score={s.score:+.2f}, confidence={s.confidence:.0%})"
            )
        return "\n".join(lines)
