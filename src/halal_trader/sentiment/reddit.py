"""Reddit sentiment collector — polls crypto subreddits for mentions and buzz."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_SUBREDDITS = [
    "CryptoCurrency",
    "Bitcoin",
    "ethereum",
    "solana",
    "CryptoMarkets",
]

_PAIR_TO_KEYWORDS: dict[str, list[str]] = {
    "BTCUSDT": ["bitcoin", "btc"],
    "ETHUSDT": ["ethereum", "eth"],
    "SOLUSDT": ["solana", "sol"],
    "ADAUSDT": ["cardano", "ada"],
    "BNBUSDT": ["bnb", "binance coin"],
    "XRPUSDT": ["xrp", "ripple"],
    "DOGEUSDT": ["doge", "dogecoin"],
    "DOTUSDT": ["polkadot", "dot"],
    "AVAXUSDT": ["avalanche", "avax"],
    "MATICUSDT": ["polygon", "matic"],
    "LINKUSDT": ["chainlink", "link"],
    "ATOMUSDT": ["cosmos", "atom"],
}


@dataclass
class RedditMention:
    """A single Reddit post or comment mentioning a crypto pair."""

    title: str
    body: str
    score: int
    subreddit: str
    created_utc: float
    url: str


@dataclass
class RedditSentimentData:
    """Aggregated Reddit data for a single pair."""

    pair: str
    mentions: list[RedditMention] = field(default_factory=list)
    mention_count: int = 0
    avg_score: float = 0.0
    top_posts: list[str] = field(default_factory=list)


class RedditCollector:
    """Collects crypto mentions from Reddit using PRAW."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        trading_pairs: list[str],
        *,
        user_agent: str = "halal-trader-bot/1.0",
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._trading_pairs = trading_pairs
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, RedditSentimentData] = {}
        self._cache_time: float = 0.0
        self._reddit = None

    def _ensure_reddit(self):
        """Lazily initialize the PRAW Reddit instance."""
        if self._reddit is not None:
            return
        try:
            import praw

            self._reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
            logger.info("Reddit client initialized")
        except ImportError:
            logger.warning("praw not installed — Reddit sentiment disabled")
        except Exception as e:
            logger.warning("Failed to initialize Reddit client: %s", e)

    async def collect(self) -> dict[str, RedditSentimentData]:
        """Collect mentions for all trading pairs from Reddit.

        Returns cached results if within TTL.
        """
        now = time.monotonic()
        if self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        self._ensure_reddit()
        if self._reddit is None:
            return {}

        result = await asyncio.to_thread(self._collect_sync)
        self._cache = result
        self._cache_time = now
        return result

    def _collect_sync(self) -> dict[str, RedditSentimentData]:
        """Synchronous Reddit collection (runs in thread pool)."""
        pair_data: dict[str, RedditSentimentData] = {
            pair: RedditSentimentData(pair=pair) for pair in self._trading_pairs
        }

        for sub_name in _SUBREDDITS:
            try:
                subreddit = self._reddit.subreddit(sub_name)
                for post in subreddit.hot(limit=50):
                    if post.score < 5 or post.removed_by_category:
                        continue
                    self._match_post(post, pair_data)

                for post in subreddit.new(limit=25):
                    if post.score < 2:
                        continue
                    self._match_post(post, pair_data)

            except Exception as e:
                logger.debug("Error fetching r/%s: %s", sub_name, e)

        for data in pair_data.values():
            if data.mentions:
                data.mention_count = len(data.mentions)
                data.avg_score = sum(m.score for m in data.mentions) / len(data.mentions)
                sorted_mentions = sorted(data.mentions, key=lambda m: m.score, reverse=True)
                data.top_posts = [m.title[:100] for m in sorted_mentions[:3]]

        return pair_data

    def _match_post(self, post, pair_data: dict[str, RedditSentimentData]) -> None:
        """Check if a post mentions any of our trading pairs."""
        text = f"{post.title} {post.selftext}".lower()
        for pair, keywords in _PAIR_TO_KEYWORDS.items():
            if pair not in pair_data:
                continue
            if any(kw in text for kw in keywords):
                mention = RedditMention(
                    title=post.title,
                    body=(post.selftext or "")[:500],
                    score=post.score,
                    subreddit=post.subreddit.display_name,
                    created_utc=post.created_utc,
                    url=f"https://reddit.com{post.permalink}",
                )
                pair_data[pair].mentions.append(mention)
