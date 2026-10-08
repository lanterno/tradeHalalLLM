"""The shared reads of equity, book NAVs, snapshots and single heartbeats."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.heartbeat import STOCK_LEDGER, beat, read_beat
from halal_trader.execution.ledger import equity_history
from halal_trader.portfolio.snapshots import read_snapshot, read_snapshots
from halal_trader.research.forward_book import latest_weights, nav_series

D1, D2, D3 = date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7)


async def test_equity_history_skips_empty_days_and_windows(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        for day, equity in ((D1, 0.0), (D2, 1000.0), (D3, 1010.0)):
            await conn.execute(
                text(
                    "INSERT INTO broker_equity (account, day, equity, profit_loss, "
                    "profit_loss_pct, synced_at) VALUES ('core', :d, :e, 0, 0, now())"
                ),
                {"d": day, "e": equity},
            )

    assert await equity_history(engine, "core") == [(D2, 1000.0), (D3, 1010.0)]
    assert await equity_history(engine, "core", since=D3) == [(D3, 1010.0)]
    assert await equity_history(engine, "core", through=D2) == [(D2, 1000.0)]
    assert await equity_history(engine, "paper") == []


async def test_book_navs_and_latest_weights(engine: AsyncEngine) -> None:
    assert await latest_weights(engine, "core") == {}
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO forward_books (name, strategy, params, created_at) "
                "VALUES ('core', 'core-strict-cap', '{}', now())"
            )
        )
        for day, nav, weights in ((D1, 1.0, '{"MSFT": 1.0}'), (D2, 1.02, '{"NVDA": 1.0}')):
            await conn.execute(
                text(
                    "INSERT INTO forward_book_days (book, day, nav, day_return, turnover, "
                    "weights, rebalance_next, recorded_at) VALUES ('core', :d, :n, 0, 0, "
                    "CAST(:w AS JSONB), false, now())"
                ),
                {"d": day, "n": nav, "w": weights},
            )

    assert await nav_series(engine, "core") == [(D1, 1.0), (D2, 1.02)]
    assert await nav_series(engine, "core", through=D1) == [(D1, 1.0)]
    assert await latest_weights(engine, "core") == {"NVDA": 1.0}


async def test_snapshots_and_a_single_beat(engine: AsyncEngine) -> None:
    taken = datetime(2026, 10, 7, 15, 0, tzinfo=UTC)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO account_snapshots (account, taken_at, equity, cash, last_equity, "
                "positions) VALUES ('core', :t, 1000, 20, NULL, CAST(:p AS JSONB))"
            ),
            {"t": taken, "p": '[{"symbol": "MSFT"}]'},
        )

    snap = await read_snapshot(engine, "core")
    assert snap is not None
    assert (snap.taken_at, snap.equity, snap.cash, snap.last_equity) == (taken, 1000.0, 20.0, None)
    assert snap.positions == [{"symbol": "MSFT"}]
    assert list(await read_snapshots(engine)) == ["core"]
    assert await read_snapshot(engine, "paper") is None

    assert await read_beat(engine, STOCK_LEDGER) is None
    await beat(engine, STOCK_LEDGER, {"broker_fills": 3}, now=taken)
    b = await read_beat(engine, STOCK_LEDGER)
    assert b is not None and (b.beat_at, b.detail) == (taken, {"broker_fills": 3})
