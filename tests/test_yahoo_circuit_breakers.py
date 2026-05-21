"""Pin the session-level circuit breakers for the two Yahoo sources.

On 2026-05-21 every cycle was hitting ~10 options-IV 401s and ~9
news 429s — Yahoo had rotated its anti-bot tokens / spent the
per-IP allowance. Hitting them on every cycle was burning HTTP
requests + log lines for no gain. The breakers stop calling after
5 consecutive failures for the rest of the process lifetime.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from halal_trader.sentiment.stocks_news import StockNewsCollector
from halal_trader.trading.options_iv import YahooOptionsIV


def _resp(status_code: int):
    """Minimal stand-in for an httpx response that the IV client only
    inspects via ``status_code`` + ``.json()``."""
    r = MagicMock()
    r.status_code = status_code
    r.json = MagicMock(return_value={})
    return r


@pytest.mark.asyncio
async def test_options_iv_breaker_opens_after_five_failures():
    iv = YahooOptionsIV()
    iv._client = MagicMock()
    iv._client.get = AsyncMock(return_value=_resp(401))

    # Five failures opens the breaker.
    for i in range(5):
        await iv._fetch_for(f"SYM{i}")
    assert iv._circuit_open is True
    # All five attempts actually hit the network.
    assert iv._client.get.await_count == 5

    # Subsequent calls short-circuit — no more HTTP.
    iv._client.get.reset_mock()
    await iv._fetch_for("SYM_AFTER")
    await iv._fetch_for("SYM_AGAIN")
    assert iv._client.get.await_count == 0


@pytest.mark.asyncio
async def test_options_iv_breaker_recovers_on_success():
    iv = YahooOptionsIV()
    iv._client = MagicMock()
    # Three failures (under the threshold), then a successful 200
    # carrying enough payload to reach the reset code path.
    good = _resp(200)
    good.json = MagicMock(
        return_value={
            "optionChain": {
                "result": [
                    {
                        "quote": {"regularMarketPrice": 100.0},
                        "options": [
                            {
                                "calls": [{"strike": 100, "impliedVolatility": 0.3}],
                                "puts": [{"strike": 100, "impliedVolatility": 0.3}],
                            }
                        ],
                    }
                ]
            }
        }
    )
    iv._client.get = AsyncMock(side_effect=[_resp(401), _resp(401), _resp(401), good])

    for sym in ("A", "B", "C"):
        await iv._fetch_for(sym)
    assert iv._consecutive_failures == 3
    assert iv._circuit_open is False
    await iv._fetch_for("D")
    assert iv._consecutive_failures == 0
    assert iv._circuit_open is False


@pytest.mark.asyncio
async def test_options_iv_breaker_open_returns_none_without_call():
    iv = YahooOptionsIV()
    iv._circuit_open = True
    iv._client = MagicMock()
    iv._client.get = AsyncMock()

    out = await iv._fetch_for("SYM")
    assert out is None
    iv._client.get.assert_not_called()


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
