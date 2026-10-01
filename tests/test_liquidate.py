"""Tests for core/liquidate.py — panic-button auto-liquidation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.liquidate import (
    LiquidationResult,
    liquidate_stocks,
)


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
