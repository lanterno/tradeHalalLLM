"""Tests for core/liquidate.py — panic-button auto-liquidation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.liquidate import (
    LiquidationResult,
    liquidate_crypto,
    liquidate_stocks,
)


def _balance(asset: str, free: float, locked: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(asset=asset, free=free, locked=locked)


def _crypto_broker(balances, prices, *, place_side_effect=None):
    broker = MagicMock()
    broker.get_balances = AsyncMock(return_value=balances)
    broker.get_ticker_price = AsyncMock(side_effect=lambda sym: prices.get(sym, 0.0))
    broker.round_quantity = MagicMock(side_effect=lambda sym, qty: qty)
    broker.place_order = AsyncMock(side_effect=place_side_effect)
    return broker


@pytest.mark.asyncio
async def test_liquidate_crypto_closes_tracked_assets():
    balances = [_balance("BTC", 0.5), _balance("ETH", 2.0)]
    prices = {"BTCUSDT": 70_000.0, "ETHUSDT": 3_500.0}
    broker = _crypto_broker(balances, prices)

    results = await liquidate_crypto(broker, ["BTCUSDT", "ETHUSDT"])
    closed = [r for r in results if r.status == "closed"]
    assert {r.symbol for r in closed} == {"BTCUSDT", "ETHUSDT"}
    assert broker.place_order.await_count == 2


@pytest.mark.asyncio
async def test_liquidate_crypto_skips_dust():
    balances = [_balance("BTC", 0.00001)]  # ~$0.70 at $70k
    prices = {"BTCUSDT": 70_000.0}
    broker = _crypto_broker(balances, prices)

    results = await liquidate_crypto(broker, ["BTCUSDT"])
    assert results[0].status == "skipped"
    assert "dust" in results[0].detail
    broker.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_liquidate_crypto_ignores_stablecoins():
    balances = [_balance("USDT", 1000.0), _balance("BUSD", 50.0), _balance("USDC", 25.0)]
    broker = _crypto_broker(balances, {})
    results = await liquidate_crypto(broker, ["BTCUSDT"])
    assert results == []  # nothing to do


@pytest.mark.asyncio
async def test_liquidate_crypto_ignores_untracked_assets():
    """Bot didn't open a BNB position, so don't sell user's BNB even if balance > 0."""
    balances = [_balance("BNB", 5.0)]
    prices = {"BNBUSDT": 600.0}
    broker = _crypto_broker(balances, prices)

    results = await liquidate_crypto(broker, ["BTCUSDT"])  # BNB not tracked
    assert results == []


@pytest.mark.asyncio
async def test_liquidate_crypto_handles_per_symbol_failure():
    balances = [_balance("BTC", 0.5), _balance("ETH", 1.0)]
    prices = {"BTCUSDT": 70_000.0, "ETHUSDT": 3_500.0}
    calls = {"n": 0}

    async def _flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("exchange down")
        return {"orderId": "ok"}

    broker = _crypto_broker(balances, prices, place_side_effect=_flaky)
    results = await liquidate_crypto(broker, ["BTCUSDT", "ETHUSDT"])
    statuses = {r.symbol: r.status for r in results}
    assert "error" in statuses.values()
    assert "closed" in statuses.values()


@pytest.mark.asyncio
async def test_liquidate_crypto_get_balances_failure():
    broker = MagicMock()
    broker.get_balances = AsyncMock(side_effect=RuntimeError("api 500"))
    results = await liquidate_crypto(broker, ["BTCUSDT"])
    assert len(results) == 1
    assert results[0].status == "error"
    assert "api 500" in results[0].detail


@pytest.mark.asyncio
async def test_liquidate_crypto_round_quantity_to_zero_skipped():
    balances = [_balance("BTC", 0.0001)]
    prices = {"BTCUSDT": 70_000.0}  # $7 — above dust
    broker = _crypto_broker(balances, prices)
    broker.round_quantity = MagicMock(return_value=0.0)  # lot size kills it

    results = await liquidate_crypto(broker, ["BTCUSDT"])
    assert results[0].status == "skipped"
    assert "rounded to zero" in results[0].detail


@pytest.mark.asyncio
async def test_liquidate_stocks_calls_broker_close_all():
    broker = MagicMock()
    broker.close_all_positions = AsyncMock(return_value={"ok": True})
    broker.get_all_positions = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="AAPL", qty=10),
            SimpleNamespace(symbol="MSFT", qty=5),
        ]
    )
    results = await liquidate_stocks(broker)
    broker.close_all_positions.assert_awaited_once()
    assert {r.symbol for r in results} == {"AAPL", "MSFT"}
    assert all(r.status == "closed" for r in results)


@pytest.mark.asyncio
async def test_liquidate_stocks_failure_surfaces():
    broker = MagicMock()
    broker.close_all_positions = AsyncMock(side_effect=RuntimeError("MCP down"))
    results = await liquidate_stocks(broker)
    assert results == [
        LiquidationResult(
            market="stocks", symbol="*", quantity=0.0, status="error", detail="MCP down"
        )
    ]
