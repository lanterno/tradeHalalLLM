"""Event-driven news reactor — polls CryptoPanic for breaking news and triggers mini-cycles."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import httpx

logger = logging.getLogger(__name__)

_CRYPTOPANIC_URLS = [
    "https://cryptopanic.com/api/developer/v2/posts/",
    "https://cryptopanic.com/api/growth/v2/posts/",
    "https://cryptopanic.com/api/enterprise/v2/posts/",
]


@dataclass
class NewsEvent:
    """A high-impact news event that may require immediate trading action."""

    title: str
    source: str
    url: str
    published_at: str
    sentiment: str
    affected_pairs: list[str] = field(default_factory=list)
    importance: str = "normal"  # "normal", "hot", "breaking"


EventCallback = Callable[[NewsEvent], Coroutine[Any, Any, None]]


def _pair_to_currency(pair: str) -> str | None:
    for suffix in ("USDT", "BUSD"):
        if pair.upper().endswith(suffix):
            return pair.upper().removesuffix(suffix)
    return None


class NewsEventReactor:
    """Monitors CryptoPanic for breaking news and fires event callbacks.

    Polls every `poll_interval_seconds` (default 30s). When a new high-impact
    item appears that hasn't been seen before, it fires all registered callbacks.
    Callbacks can trigger emergency mini-cycles in the trading scheduler.
    """

    def __init__(
        self,
        api_key: str,
        trading_pairs: list[str],
        *,
        poll_interval_seconds: int = 30,
        importance_filter: str = "hot",
    ) -> None:
        self._api_key = api_key
        self._trading_pairs = trading_pairs
        self._poll_interval = poll_interval_seconds
        self._importance_filter = importance_filter
        self._seen_urls: set[str] = set()
        self._callbacks: list[EventCallback] = []
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        self._working_url: str | None = None

        self._currency_to_pairs: dict[str, list[str]] = {}
        for pair in trading_pairs:
            currency = _pair_to_currency(pair)
            if currency:
                self._currency_to_pairs.setdefault(currency, []).append(pair)

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def on_event(self, callback: EventCallback) -> None:
        """Register a callback to be invoked when a high-impact event is detected."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        if not self.enabled:
            logger.info("News reactor disabled — no CryptoPanic API key")
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="news-reactor")
        logger.info(
            "News reactor started (poll every %ds, filter: %s)",
            self._poll_interval, self._importance_filter,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        logger.info("News reactor stopped")

    async def _poll_loop(self) -> None:
        await asyncio.sleep(5)

        while self._running:
            try:
                events = await self._check_for_events()
                for event in events:
                    logger.info(
                        "Breaking news event: [%s] %s (affects %s)",
                        event.importance, event.title[:80],
                        ", ".join(event.affected_pairs) or "general",
                    )
                    for cb in self._callbacks:
                        try:
                            await cb(event)
                        except Exception as e:
                            logger.error("Event callback failed: %s", e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("News reactor poll error: %s", e)

            await asyncio.sleep(self._poll_interval)

    async def _check_for_events(self) -> list[NewsEvent]:
        """Fetch latest news and return unseen high-impact items."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15.0)

        currencies = set(self._currency_to_pairs.keys())
        if not currencies:
            return []

        urls = [self._working_url] if self._working_url else list(_CRYPTOPANIC_URLS)
        params = {
            "auth_token": self._api_key,
            "currencies": ",".join(currencies),
            "filter": self._importance_filter,
            "public": "true",
        }

        data = {}
        for url in urls:
            try:
                resp = await self._client.get(url, params=params)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                data = resp.json()
                self._working_url = url
                break
            except Exception as e:
                logger.debug("CryptoPanic URL failed (%s): %s", url, e)

        if not data:
            return []

        events: list[NewsEvent] = []
        for item in data.get("results", []):
            item_url = item.get("url", "")
            if not item_url or item_url in self._seen_urls:
                continue

            self._seen_urls.add(item_url)
            if len(self._seen_urls) > 1000:
                to_remove = list(self._seen_urls)[:500]
                for u in to_remove:
                    self._seen_urls.discard(u)

            item_currencies = {
                c.get("code", "") for c in item.get("currencies", [])
            }

            affected = []
            for currency in item_currencies:
                affected.extend(self._currency_to_pairs.get(currency, []))

            votes = item.get("votes", {})
            positive = votes.get("positive", 0)
            negative = votes.get("negative", 0)
            if positive > negative:
                sentiment = "positive"
            elif negative > positive:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            kind = item.get("kind", "news")
            importance = "breaking" if kind == "media" else "hot"

            events.append(NewsEvent(
                title=item.get("title", ""),
                source=item.get("source", {}).get("title", ""),
                url=item_url,
                published_at=item.get("published_at", ""),
                sentiment=sentiment,
                affected_pairs=affected,
                importance=importance,
            ))

        return events
