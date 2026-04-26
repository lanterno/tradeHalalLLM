"""Graceful-shutdown order-cancel helper tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from halal_trader.core.shutdown import cancel_all_open_orders


def _broker(orders=None, cancel_side_effect=None):
    b = MagicMock()
    b.get_open_orders = AsyncMock(return_value=orders or [])
    b.cancel_order = AsyncMock(side_effect=cancel_side_effect)
    return b


async def test_no_open_orders_returns_empty():
    b = _broker(orders=[])
    result = await cancel_all_open_orders(b)
    assert result.cancelled == []
    assert result.failed == []


async def test_cancels_each_open_order():
    b = _broker(
        orders=[
            {"orderId": "1", "symbol": "BTCUSDT"},
            {"orderId": "2", "symbol": "ETHUSDT"},
        ]
    )
    result = await cancel_all_open_orders(b)
    assert sorted(result.cancelled) == ["1", "2"]
    assert b.cancel_order.await_count == 2


async def test_handles_alt_id_keys():
    """python-binance returns ``orderId`` but other clients use ``order_id``."""
    b = _broker(orders=[{"order_id": "X", "symbol": "BTCUSDT"}])
    result = await cancel_all_open_orders(b)
    assert result.cancelled == ["X"]


async def test_get_open_orders_failure_returns_failed_marker():
    b = MagicMock()
    b.get_open_orders = AsyncMock(side_effect=RuntimeError("api down"))
    b.cancel_order = AsyncMock()
    result = await cancel_all_open_orders(b)
    assert result.cancelled == []
    assert result.failed and "api down" in result.failed[0][1]
    b.cancel_order.assert_not_called()


async def test_per_order_cancel_failure_does_not_abort_pass():
    """A failure on one order shouldn't stop the rest from being cancelled."""
    side_effects = [RuntimeError("rejected"), None]
    b = _broker(
        orders=[
            {"orderId": "bad", "symbol": "BTCUSDT"},
            {"orderId": "good", "symbol": "ETHUSDT"},
        ],
        cancel_side_effect=side_effects,
    )
    result = await cancel_all_open_orders(b)
    assert "good" in result.cancelled
    assert any(oid == "bad" for oid, _ in result.failed)


async def test_get_open_orders_timeout_marked_in_failed():
    async def hang():
        await asyncio.sleep(10)
        return []

    b = MagicMock()
    b.get_open_orders = AsyncMock(side_effect=hang)
    b.cancel_order = AsyncMock()
    result = await cancel_all_open_orders(b, timeout=0.05)
    assert result.cancelled == []
    assert result.failed and "timed out" in result.failed[0][1]


async def test_missing_id_recorded_as_failed():
    b = _broker(orders=[{"symbol": "BTCUSDT"}])  # no id
    result = await cancel_all_open_orders(b)
    assert result.cancelled == []
    assert result.failed and "no order id" in result.failed[0][1]
