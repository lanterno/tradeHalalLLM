"""The core's live-money gate: duration, a monthly rebalance, tracking, clean orders."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.portfolio.readiness import check
from halal_trader.research.daily import _core_readiness

START = date(2026, 10, 5)


def _sessions(n: int) -> list[date]:
    out, d = [], START
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


async def _seed(engine: AsyncEngine, n: int, *, drift: float = 0.0, monthly: bool = True) -> date:
    days = _sessions(n)
    async with engine.begin() as conn:
        for i, d in enumerate(days):
            book = 1.0 * (1.001**i)
            await conn.execute(
                text(
                    "INSERT INTO broker_equity (account, day, equity, profit_loss, "
                    "profit_loss_pct, synced_at) VALUES ('core', :d, :e, 0, 0, now())"
                ),
                {"d": d, "e": 100_000 * book * (1 + drift * i)},
            )
            await conn.execute(
                text(
                    "INSERT INTO forward_book_days (book, day, nav, day_return, turnover, weights, "
                    "rebalance_next, recorded_at) VALUES ('core', :d, :n, 0, 0, '{}', false, now())"
                ),
                {"d": d, "n": book},
            )
        if monthly:
            await conn.execute(
                text(
                    "INSERT INTO core_runs (run_on, monthly, executed, equity, cash, orders, "
                    "recorded_at) VALUES (:d, true, true, 100000, 1000, 100, now())"
                ),
                {"d": days[0]},
            )
    return days[-1]


async def _book(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO forward_books (name, strategy, params, created_at) "
                "VALUES ('core', 'core-strict-cap', '{}', now())"
            )
        )


async def test_a_month_tracking_the_book_cleanly_is_ready_and_alerts_once(
    engine: AsyncEngine,
) -> None:
    await _book(engine)
    last = await _seed(engine, 22)
    r = await check(engine, today=last)
    assert r.ready, r.failures
    assert await _core_readiness(engine, last) is True
    assert await _core_readiness(engine, last) is False  # already announced


async def test_too_few_days_or_no_monthly_run_is_not_ready(engine: AsyncEngine) -> None:
    await _book(engine)
    last = await _seed(engine, 10, monthly=False)
    r = await check(engine, today=last)
    assert not r.ready
    assert any("trading days" in f for f in r.failures)
    assert any("monthly" in f for f in r.failures)


async def test_drifting_from_the_book_or_a_refused_order_is_not_ready(engine: AsyncEngine) -> None:
    await _book(engine)
    last = await _seed(engine, 22, drift=0.001)  # ~2% behind the book over the month
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO core_orders (submitted_at, symbol, side, qty, est_price, notional, "
                "reason, status) VALUES (:t, 'X', 'buy', 1, 1, 1, 'rebalance', 'refused')"
            ),
            {"t": datetime.combine(last, datetime.min.time(), UTC)},
        )
    r = await check(engine, today=last)
    assert not r.ready
    assert any("gap" in f for f in r.failures) and any("refused" in f for f in r.failures)


async def test_an_unfilled_order_is_not_clean(engine: AsyncEngine) -> None:
    from datetime import UTC, datetime

    from halal_trader.market_hours import today_eastern
    from halal_trader.portfolio.readiness import check

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO core_orders (submitted_at, symbol, side, qty, est_price, notional, "
                "reason, status, broker_order_id) VALUES (:t, 'X', 'buy', 1, 10, 10, "
                "'rebalance', 'submitted', 'never-filled')"
            ),
            {"t": datetime.now(UTC)},
        )
    r = await check(engine, today=today_eastern())
    assert r.unfilled == 1
    assert any("unfilled" in f for f in r.failures)
