"""The core's fills against the arrival price and the close."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.portfolio.execution_quality import report
from halal_trader.research.daily import _execution_quality

DAY = date(2026, 10, 5)
AT = datetime.fromisoformat("2026-10-05 15:40:10-04:00")


async def _order(
    engine: AsyncEngine, oid: str, symbol: str, side: str, qty: float, est: float
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO core_orders (submitted_at, symbol, side, qty, est_price, notional, "
                "reason, status, broker_order_id) VALUES (:t, :s, :side, :q, :p, :n, 'rebalance', "
                "'submitted', :o)"
            ),
            {"t": AT, "s": symbol, "side": side, "q": qty, "p": est, "n": qty * est, "o": oid},
        )


async def _fill(
    engine: AsyncEngine, fid: str, oid: str, symbol: str, side: str, qty: float, price: float
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO broker_activities (id, account, activity_type, transaction_time, "
                "symbol, side, qty, price, order_id, raw) VALUES (:i, 'core', 'FILL', :t, :s, "
                ":side, :q, :p, :o, '{}')"
            ),
            {"i": fid, "t": AT, "s": symbol, "side": side, "q": qty, "p": price, "o": oid},
        )


async def _close(engine: AsyncEngine, symbol: str, close: float) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES (:s, :d, 'raw', :c, :c, :c, :c, 1, now())"
            ),
            {"s": symbol, "d": DAY, "c": close},
        )


async def test_slippage_is_signed_as_a_cost_and_weighted_by_value(engine: AsyncEngine) -> None:
    # A buy split over two fills averaging 101 against an arrival of 100: +100 bps.
    await _order(engine, "o1", "AAA", "buy", 2, 100.0)
    await _fill(engine, "f1", "o1", "AAA", "buy", 1, 100.0)
    await _fill(engine, "f2", "o1", "AAA", "buy", 1, 102.0)
    await _close(engine, "AAA", 101.0)
    # A sell filled below arrival is a cost too: 49.5 vs 50 is +100 bps.
    await _order(engine, "o2", "BBB", "sell", 4, 50.0)
    await _fill(engine, "f3", "o2", "BBB", "sell", 4, 49.5)
    # An order the broker never filled.
    await _order(engine, "o3", "CCC", "buy", 1, 10.0)

    r = await report(engine, DAY)

    by = {o.symbol: o for o in r.orders}
    assert by["AAA"].fill_price == pytest.approx(101.0)
    assert by["AAA"].vs_arrival_bps == pytest.approx(100.0)
    assert by["AAA"].vs_close_bps == pytest.approx(0.0)
    assert by["BBB"].vs_arrival_bps == pytest.approx(100.0)
    assert by["BBB"].vs_close_bps is None  # no close stored for it
    assert by["CCC"].status == "unfilled"
    assert r.vs_arrival_bps == pytest.approx(100.0)
    assert r.filled_notional == pytest.approx(202 + 198)
    assert r.summary()["unfilled"] == 1


async def test_the_evening_run_reports_unfilled_orders_and_keeps_the_summary(
    engine: AsyncEngine,
) -> None:
    await _order(engine, "o1", "AAA", "buy", 1, 100.0)
    await _fill(engine, "f1", "o1", "AAA", "buy", 1, 100.1)
    await _order(engine, "o2", "CCC", "buy", 1, 10.0)

    errors = await _execution_quality(engine, DAY)

    assert errors and "CCC" in errors[0]
    async with engine.connect() as conn:
        detail = (
            await conn.execute(
                text("SELECT detail FROM heartbeats WHERE component = 'core.execution'")
            )
        ).scalar_one()
    assert detail["orders"] == 2 and detail["vs_arrival_bps"] == pytest.approx(10.0)


async def test_no_orders_writes_nothing(engine: AsyncEngine) -> None:
    assert await _execution_quality(engine, DAY) == []
