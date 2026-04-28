"""Sentiment manager — orchestrates collection and scoring on a schedule."""

from __future__ import annotations

import asyncio
import logging

from halal_trader.sentiment.cryptopanic import CryptoPanicCollector
from halal_trader.sentiment.reddit import RedditCollector
from halal_trader.sentiment.scoring import SentimentScorer, SentimentSignal

logger = logging.getLogger(__name__)


class SentimentManager:
    """Orchestrates sentiment collection from all sources and produces composite signals."""

    def __init__(
        self,
        trading_pairs: list[str],
        *,
        reddit_client_id: str = "",
        reddit_client_secret: str = "",
        cryptopanic_api_key: str = "",
        update_interval_seconds: int = 300,
    ) -> None:
        self._trading_pairs = trading_pairs
        self._update_interval = update_interval_seconds

        self._reddit: RedditCollector | None = None
        if reddit_client_id and reddit_client_secret:
            self._reddit = RedditCollector(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                trading_pairs=trading_pairs,
                cache_ttl_seconds=update_interval_seconds,
            )

        self._cryptopanic: CryptoPanicCollector | None = None
        if cryptopanic_api_key:
            self._cryptopanic = CryptoPanicCollector(
                api_key=cryptopanic_api_key,
                trading_pairs=trading_pairs,
                cache_ttl_seconds=update_interval_seconds,
            )

        self._scorer = SentimentScorer()
        self._latest_signals: dict[str, SentimentSignal] = {}
        self._running = False
        self._task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self._reddit is not None or self._cryptopanic is not None

    @property
    def latest_signals(self) -> dict[str, SentimentSignal]:
        return self._latest_signals

    async def start(self) -> None:
        """Start background sentiment collection."""
        if not self.enabled:
            logger.info("No sentiment sources configured — sentiment manager disabled")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="sentiment-manager")
        sources = []
        if self._reddit:
            sources.append("Reddit")
        if self._cryptopanic:
            sources.append("CryptoPanic")
        logger.info(
            "Sentiment manager started (sources: %s, interval: %ds)",
            ", ".join(sources),
            self._update_interval,
        )

    async def stop(self) -> None:
        """Stop background collection and close HTTP clients."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._cryptopanic:
            await self._cryptopanic.close()
        if self._reddit and hasattr(self._reddit, "close"):
            await self._reddit.close()
        logger.info("Sentiment manager stopped")

    async def _run_loop(self) -> None:
        """Background loop: collect and score sentiment periodically."""
        while self._running:
            try:
                await self.update()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Sentiment update failed: %s", e)

            await asyncio.sleep(self._update_interval)

    async def update(self) -> dict[str, SentimentSignal]:
        """Perform one collection + scoring cycle. Can also be called on-demand."""
        reddit_data = {}
        cryptopanic_data = {}

        tasks = []
        if self._reddit:
            tasks.append(("reddit", self._reddit.collect()))
        if self._cryptopanic:
            tasks.append(("cryptopanic", self._cryptopanic.collect()))

        for source, coro in tasks:
            try:
                data = await coro
                if source == "reddit":
                    reddit_data = data
                else:
                    cryptopanic_data = data
            except Exception as e:
                logger.warning("Failed to collect from %s: %s", source, e)

        signals: dict[str, SentimentSignal] = {}
        for pair in self._trading_pairs:
            reddit = reddit_data.get(pair)
            news = cryptopanic_data.get(pair)

            signal = self._scorer.compute_composite(
                pair=pair,
                reddit_mentions=reddit.mention_count if reddit else 0,
                reddit_top_posts=reddit.top_posts if reddit else None,
                reddit_avg_score=reddit.avg_score if reddit else 0.0,
                news_sentiment=news.sentiment_score if news else 0.0,
                news_headlines=[i.title for i in news.items[:5]] if news else None,
                news_count=len(news.items) if news else 0,
            )
            if signal.data_sources:
                signals[pair] = signal

        self._latest_signals = signals

        if signals:
            logger.info(
                "Sentiment updated for %d pairs: %s",
                len(signals),
                ", ".join(f"{p}={s.score:+.2f}" for p, s in signals.items()),
            )

        return signals
