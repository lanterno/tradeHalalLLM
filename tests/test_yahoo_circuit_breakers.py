"""Pin the session-level circuit breaker for the Yahoo news source.

On 2026-05-21 every cycle was hitting ~9 news 429s — Yahoo had spent
the per-IP allowance. Hitting it on every cycle was burning HTTP
requests + log lines for no gain. The breaker stops calling after
5 consecutive failures for the rest of the process lifetime.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from halal_trader.sentiment.stocks_news import StockNewsCollector


@pytest.mark.asyncio
async def test_stocks_news_breaker_opens_after_five_failures():
    collector = StockNewsCollector(cache_ttl_seconds=0)
    collector._client = MagicMock()
    collector._client.get = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=MagicMock(),
            response=SimpleNamespace(status_code=429),
        )
    )

    for sym in ("AAPL", "MSFT", "GOOG", "AMZN", "META"):
        await collector._fetch_one(sym)
    assert collector._circuit_open is True
    assert collector._client.get.await_count == 5

    collector._client.get.reset_mock()
    out = await collector._fetch_one("NVDA")
    assert out == []
    collector._client.get.assert_not_called()


@pytest.mark.asyncio
async def test_stocks_news_breaker_open_short_circuits():
    """When already open, no HTTP call is made."""
    collector = StockNewsCollector(cache_ttl_seconds=0)
    collector._circuit_open = True
    collector._client = MagicMock()
    collector._client.get = AsyncMock()

    out = await collector._fetch_one("AAPL")
    assert out == []
    collector._client.get.assert_not_called()
