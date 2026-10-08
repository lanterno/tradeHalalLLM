"""With the day-trader retired, nothing it left behind lingers."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.db.repos.trades import TradeRepoImpl
from halal_trader.trading.scheduler import TradingBot


@pytest.mark.parametrize("enabled", [True, False])
def test_reactor_entries_are_held_overnight_only_while_the_day_trader_runs(
    enabled: bool,
) -> None:
    bot = TradingBot()
    bot.settings = SimpleNamespace(  # type: ignore[assignment]
        stocks=SimpleNamespace(day_trader_enabled=enabled)
    )
    assert bot.reactor_holds_overnight() is enabled


async def test_a_rejected_buy_is_not_an_open_position(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO trades (timestamp, symbol, side, quantity, price, status, "
                "filled_quantity) VALUES (:t, :s, 'buy', :q, 100, :st, :q)"
            ),
            [
                {"t": datetime.now(UTC), "s": "HELD", "q": 5, "st": "filled"},
                {"t": datetime.now(UTC), "s": "REFUSED", "q": 0, "st": "rejected"},
                {"t": datetime.now(UTC), "s": "CANCELLED", "q": 0, "st": "canceled"},
            ],
        )

    opens = await TradeRepoImpl(engine).get_open_trades()

    assert [t.symbol for t in opens] == ["HELD"]
