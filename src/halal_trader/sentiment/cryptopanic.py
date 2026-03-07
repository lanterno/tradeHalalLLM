"""CryptoPanic news sentiment collector — free API for crypto news with community votes."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://cryptopanic.com/api/free/v1/posts/"

_PAIR_TO_CURRENCY: dict[str, str] = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "ADAUSDT": "ADA",
    "BNBUSDT": "BNB",
    "XRPUSDT": "XRP",
    "DOGEUSDT": "DOGE",
    "DOTUSDT": "DOT",
    "AVAXUSDT": "AVAX",
    "MATICUSDT": "MATIC",
    "LINKUSDT": "LINK",
    "ATOMUSDT": "ATOM",
}


@dataclass
class NewsItem:
    """A single news item from CryptoPanic."""

    title: str
    source: str
    url: str
    published_at: str
    sentiment: str  # "positive", "negative", "neutral"
    votes: dict[str, int] = field(default_factory=dict)


@dataclass
class CryptoPanicData:
    """Aggregated CryptoPanic data for a single pair."""

    pair: str
    items: list[NewsItem] = field(default_factory=list)
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    sentiment_score: float = 0.0  # -1 to +1


class CryptoPanicCollector:
    """Collects crypto news sentiment from CryptoPanic's free API."""

    def __init__(
        self,
        api_key: str,
        trading_pairs: list[str],
        *,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._api_key = api_key
        self._trading_pairs = trading_pairs
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, CryptoPanicData] = {}
        self._cache_time: float = 0.0

    async def collect(self) -> dict[str, CryptoPanicData]:
        """Collect news for all trading pairs.

        Returns cached results if within TTL.
        """
        now = time.monotonic()
        if self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        if not self._api_key:
            return {}

        result: dict[str, CryptoPanicData] = {}
        currencies = set()
        for pair in self._trading_pairs:
            currency = _PAIR_TO_CURRENCY.get(pair)
            if currency:
                currencies.add(currency)

        if not currencies:
            return result

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    _BASE_URL,
                    params={
                        "auth_token": self._api_key,
                        "currencies": ",".join(currencies),
                        "filter": "hot",
                        "public": "true",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            for pair in self._trading_pairs:
                currency = _PAIR_TO_CURRENCY.get(pair)
                if not currency:
                    continue
                pair_data = CryptoPanicData(pair=pair)
                for item in data.get("results", []):
                    item_currencies = {
                        c.get("code", "") for c in item.get("currencies", [])
                    }
                    if currency not in item_currencies:
                        continue

                    votes = item.get("votes", {})
                    positive = votes.get("positive", 0)
                    negative = votes.get("negative", 0)

                    if positive > negative:
                        sentiment = "positive"
                        pair_data.bullish_count += 1
                    elif negative > positive:
                        sentiment = "negative"
                        pair_data.bearish_count += 1
                    else:
                        sentiment = "neutral"
                        pair_data.neutral_count += 1

                    pair_data.items.append(NewsItem(
                        title=item.get("title", ""),
                        source=item.get("source", {}).get("title", ""),
                        url=item.get("url", ""),
                        published_at=item.get("published_at", ""),
                        sentiment=sentiment,
                        votes=votes,
                    ))

                total = pair_data.bullish_count + pair_data.bearish_count + pair_data.neutral_count
                if total > 0:
                    pair_data.sentiment_score = (
                        (pair_data.bullish_count - pair_data.bearish_count) / total
                    )

                result[pair] = pair_data

        except Exception as e:
            logger.warning("CryptoPanic API error: %s", e)

        self._cache = result
        self._cache_time = now
        return result
