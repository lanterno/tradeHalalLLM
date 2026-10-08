"""A refused or cancelled buy never reads as an open position."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.db.repos.trades import TradeRepoImpl


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
