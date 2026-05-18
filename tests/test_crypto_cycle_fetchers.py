"""Tests for `CryptoCycleService._fetch_klines` and `_fetch_orderbooks`.

These two helpers run the per-cycle market-data sweep with a 5-way
semaphore + REST/WS prefer logic + per-pair exception isolation.
A regression here would either over-pressure the Binance API
(rate-limit storm) or silently drop pairs from the cycle's view.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from binance import BinanceAPIException

from halal_trader.crypto.cycle import CryptoCycleService
from halal_trader.domain.models import Kline


def _service(*, ws_manager=None, broker=None) -> CryptoCycleService:
    """Construct a service with stubbed deps for fetcher tests."""
    return CryptoCycleService(
        broker=broker or AsyncMock(),
        screener=AsyncMock(),
        strategy=AsyncMock(),
        executor=AsyncMock(),
        portfolio=AsyncMock(),
        ws_manager=ws_manager,
        configured_pairs=["BTCUSDT", "ETHUSDT"],
    )


def _kline(open_time: int = 1, close: float = 100.0) -> Kline:
    return Kline(
        open_time=open_time,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1.0,
        close_time=open_time + 60_000,
    )


def _binance_rate_limit() -> BinanceAPIException:
    """Construct a -1003 (rate-limit) BinanceAPIException — the
    constructor takes (response, status_code, text). We forge a
    minimal shape to drive the exception's `.code` attribute."""
    fake_response = MagicMock(status_code=429)
    fake_response.json.return_value = {"code": -1003, "msg": "Too many requests"}
    return BinanceAPIException(fake_response, 429, '{"code":-1003,"msg":"Too many requests"}')


# ── _fetch_klines ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_klines_uses_ws_buffer_when_sufficient():
    """WS buffer with ≥20 bars → use it, skip the REST call."""
    ws = MagicMock()
    ws.get_klines.return_value = [_kline(i) for i in range(30)]  # 30 bars

    broker = AsyncMock()
    # If the test fails, this would still get called and we'd notice.
    broker.get_klines = AsyncMock(side_effect=AssertionError("REST should not be called"))

    svc = _service(ws_manager=ws, broker=broker)
    out = await svc._fetch_klines(["BTCUSDT"])

    assert "BTCUSDT" in out
    assert len(out["BTCUSDT"]) == 30
    broker.get_klines.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_klines_falls_back_to_rest_when_ws_buffer_short():
    """WS buffer with < 20 bars → fall back to REST. The threshold
    matters: indicators need ≥30 candles, but the WS prefers any
    buffer ≥20 since the bot will fetch more on the next tick."""
    ws = MagicMock()
    ws.get_klines.return_value = [_kline(i) for i in range(5)]  # only 5 bars

    rest_klines = [_kline(i, close=200.0) for i in range(50)]
    broker = AsyncMock()
    broker.get_klines = AsyncMock(return_value=rest_klines)

    svc = _service(ws_manager=ws, broker=broker)
    out = await svc._fetch_klines(["BTCUSDT"])

    assert out["BTCUSDT"][0].close == 200.0  # REST data, not WS
    broker.get_klines.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_klines_falls_back_to_rest_when_no_ws():
    """No WS manager wired → always REST."""
    rest_klines = [_kline(i) for i in range(50)]
    broker = AsyncMock()
    broker.get_klines = AsyncMock(return_value=rest_klines)

    svc = _service(ws_manager=None, broker=broker)
    out = await svc._fetch_klines(["BTCUSDT"])

    assert "BTCUSDT" in out
    broker.get_klines.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_klines_isolates_per_pair_failure():
    """One pair raising must NOT drop the others — the cycle continues
    with whatever data it could get."""
    ws = MagicMock()
    ws.get_klines.return_value = []  # always empty → falls to REST

    broker = AsyncMock()

    async def get_klines_side_effect(pair, **_):
        if pair == "BTCUSDT":
            raise RuntimeError("connection reset")
        return [_kline(i) for i in range(50)]

    broker.get_klines = AsyncMock(side_effect=get_klines_side_effect)

    svc = _service(ws_manager=ws, broker=broker)
    out = await svc._fetch_klines(["BTCUSDT", "ETHUSDT"])

    assert "BTCUSDT" not in out  # failed pair dropped
    assert "ETHUSDT" in out  # successful pair kept


@pytest.mark.asyncio
async def test_fetch_klines_empty_pairs_returns_empty_dict():
    """Empty input list → empty output, no broker calls."""
    broker = AsyncMock()
    broker.get_klines = AsyncMock()
    svc = _service(ws_manager=None, broker=broker)

    out = await svc._fetch_klines([])

    assert out == {}
    broker.get_klines.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_klines_handles_rate_limit_exception_quietly():
    """A `-1003` BinanceAPIException triggers a 30-s backoff log but
    must not propagate (the cycle continues with what it has). Patch
    asyncio.sleep so the test isn't slow."""
    import asyncio as _asyncio

    sleep_calls: list[float] = []

    async def _instant_sleep(seconds):
        sleep_calls.append(seconds)

    real_sleep = _asyncio.sleep
    _asyncio.sleep = _instant_sleep
    try:
        broker = AsyncMock()
        broker.get_klines = AsyncMock(side_effect=_binance_rate_limit())

        svc = _service(ws_manager=None, broker=broker)
        out = await svc._fetch_klines(["BTCUSDT"])

        # No data, but no exception either.
        assert out == {}
        # 30s backoff was triggered.
        assert 30 in sleep_calls
    finally:
        _asyncio.sleep = real_sleep


# ── _fetch_orderbooks ──────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_orderbooks_calls_broker_per_pair():
    broker = AsyncMock()
    broker.get_order_book = AsyncMock(return_value={"bids": [[100, 1]], "asks": [[101, 1]]})

    svc = _service(broker=broker)
    out = await svc._fetch_orderbooks(["BTCUSDT", "ETHUSDT"])

    assert "BTCUSDT" in out
    assert "ETHUSDT" in out
    assert broker.get_order_book.await_count == 2


@pytest.mark.asyncio
async def test_fetch_orderbooks_uses_limit_10():
    """The depth fetch always asks for limit=10 — pin so a refactor
    that widens it doesn't trigger rate-limit issues."""
    broker = AsyncMock()
    broker.get_order_book = AsyncMock(return_value={"bids": [], "asks": []})

    svc = _service(broker=broker)
    await svc._fetch_orderbooks(["BTCUSDT"])

    broker.get_order_book.assert_awaited_once()
    kwargs = broker.get_order_book.call_args.kwargs
    args = broker.get_order_book.call_args.args
    # limit could be positional or kwarg; check both.
    assert kwargs.get("limit") == 10 or 10 in args


@pytest.mark.asyncio
async def test_fetch_orderbooks_isolates_per_pair_failure():
    broker = AsyncMock()

    async def get_book_side_effect(pair, **_):
        if pair == "BTCUSDT":
            raise RuntimeError("transient")
        return {"bids": [], "asks": []}

    broker.get_order_book = AsyncMock(side_effect=get_book_side_effect)

    svc = _service(broker=broker)
    out = await svc._fetch_orderbooks(["BTCUSDT", "ETHUSDT"])

    assert "BTCUSDT" not in out
    assert "ETHUSDT" in out


@pytest.mark.asyncio
async def test_fetch_orderbooks_handles_rate_limit():
    """Same -1003 backoff path as klines."""
    import asyncio as _asyncio

    sleep_calls: list[float] = []

    async def _instant_sleep(seconds):
        sleep_calls.append(seconds)

    real_sleep = _asyncio.sleep
    _asyncio.sleep = _instant_sleep
    try:
        broker = AsyncMock()
        broker.get_order_book = AsyncMock(side_effect=_binance_rate_limit())

        svc = _service(broker=broker)
        out = await svc._fetch_orderbooks(["BTCUSDT"])

        assert out == {}
        assert 30 in sleep_calls
    finally:
        _asyncio.sleep = real_sleep


@pytest.mark.asyncio
async def test_fetch_orderbooks_empty_pairs_returns_empty_dict():
    broker = AsyncMock()
    broker.get_order_book = AsyncMock()

    svc = _service(broker=broker)
    out = await svc._fetch_orderbooks([])

    assert out == {}
    broker.get_order_book.assert_not_awaited()
