"""The core's live-money gate: duration, a monthly rebalance that sold, tracking from
pre-trade equity, a run every trading day, and clean (fully filled) orders."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.market_hours import trading_days_back
from halal_trader.portfolio.readiness import check, live_gate
from halal_trader.research.daily import _core_readiness

START = date(2026, 10, 5)


def _sessions(n: int) -> list[date]:
    out, d = [], START
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _at(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time(), UTC) + timedelta(hours=19, minutes=40)


async def _seed(
    engine: AsyncEngine,
    n: int,
    *,
    drift: float = 0.0,
    monthly_sell: bool = True,
    skip_runs: tuple[int, ...] = (),
    first_day_cost: float = 0.0,
) -> list[date]:
    """``n`` sessions of account and book from START (the book's genesis the day
    before), a run on each (but ``skip_runs``), the first a buying rebalance, and
    (``monthly_sell``) a second monthly rebalance that sold."""
    days = _sessions(n)
    genesis = days[0] - timedelta(days=3)  # the Friday before
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO forward_books (name, strategy, params, created_at) "
                "VALUES ('core', 'core-strict-cap', '{}', now())"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO forward_book_days (book, day, nav, day_return, turnover, weights, "
                "rebalance_next, recorded_at) VALUES ('core', :d, 1.0, 0, 0, '{}', true, now())"
            ),
            {"d": genesis},
        )
        for i, d in enumerate(days):
            book = 1.001 ** (i + 1)
            await conn.execute(
                text(
                    "INSERT INTO broker_equity (account, day, equity, profit_loss, "
                    "profit_loss_pct, synced_at) VALUES ('core', :d, :e, 0, 0, now())"
                ),
                {"d": d, "e": 100_000 * book * (1 + drift * i) * (1 - first_day_cost)},
            )
            await conn.execute(
                text(
                    "INSERT INTO forward_book_days (book, day, nav, day_return, turnover, weights, "
                    "rebalance_next, recorded_at) VALUES ('core', :d, :n, 0, 0, '{}', false, now())"
                ),
                {"d": d, "n": book},
            )
            if i in skip_runs:
                continue
            monthly = i == 0 or (monthly_sell and i == n - 3)
            await conn.execute(
                text(
                    "INSERT INTO core_runs (account, run_on, monthly, executed, equity, cash, "
                    "orders, recorded_at) VALUES ('core', :d, :m, true, 100000, 1000, 1, now())"
                ),
                {"d": d, "m": monthly},
            )
            if monthly_sell and i == n - 3:
                await conn.execute(
                    text(
                        "INSERT INTO core_orders (account, submitted_at, symbol, side, qty, "
                        "est_price, notional, reason, status, broker_order_id) VALUES ('core', "
                        ":t, 'X', 'sell', 1, 10, 10, 'rebalance', 'submitted', 's1')"
                    ),
                    {"t": _at(d)},
                )
                await conn.execute(
                    text(
                        "INSERT INTO broker_activities (id, account, activity_type, "
                        "transaction_time, symbol, side, qty, price, order_id, raw) VALUES "
                        "('f-s1', 'core', 'FILL', :t, 'X', 'sell', 1, 10, 's1', '{}')"
                    ),
                    {"t": _at(d)},
                )
    return days


async def test_a_month_tracking_the_book_cleanly_is_ready_and_alerts_once(
    engine: AsyncEngine,
) -> None:
    days = await _seed(engine, 22)
    r = await check(engine, today=days[-1])
    assert r.ready, r.failures
    assert await _core_readiness(engine, days[-1]) is True
    assert await _core_readiness(engine, days[-1]) is False  # already announced


async def test_too_few_days_is_not_ready(engine: AsyncEngine) -> None:
    days = await _seed(engine, 10)
    r = await check(engine, today=days[-1])
    assert not r.ready and any("trading days" in f for f in r.failures)


async def test_a_rebalance_that_only_bought_does_not_count(engine: AsyncEngine) -> None:
    days = await _seed(engine, 22, monthly_sell=False)
    r = await check(engine, today=days[-1])
    assert r.monthly_runs == 0
    assert any("monthly rebalance with sells" in f for f in r.failures)


async def test_a_trading_day_without_a_run_is_not_ready(engine: AsyncEngine) -> None:
    days = await _seed(engine, 22, skip_runs=(15,))
    r = await check(engine, today=days[-1])
    assert r.missing_runs == [days[15]]
    assert any("no run on 1 trading day" in f for f in r.failures)


async def test_todays_run_is_not_missing_before_it_happens(engine: AsyncEngine) -> None:
    days = await _seed(engine, 22)
    tomorrow = trading_days_back(days[-1] + timedelta(days=5), 1)[0]  # a later session
    r = await check(engine, today=tomorrow)  # morning: no run yet today
    assert tomorrow not in r.missing_runs


async def test_the_first_days_trading_cost_is_measured_from_pre_trade_equity(
    engine: AsyncEngine,
) -> None:
    """The first close already carries the first rebalance's cost: measured from
    the account's equity before its first orders, a 2% cost shows as a gap."""
    days = await _seed(engine, 22, first_day_cost=0.02)
    r = await check(engine, today=days[-1])
    assert r.gap is not None and r.gap < -0.015
    assert any("gap" in f for f in r.failures)


async def test_drifting_from_the_book_or_a_refused_order_is_not_ready(engine: AsyncEngine) -> None:
    days = await _seed(engine, 22, drift=0.001)  # ~2% behind the book over the month
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO core_orders (submitted_at, symbol, side, qty, est_price, notional, "
                "reason, status) VALUES (:t, 'X', 'buy', 1, 1, 1, 'rebalance', 'refused')"
            ),
            {"t": _at(days[-1])},
        )
    r = await check(engine, today=days[-1])
    assert not r.ready
    assert any("gap" in f for f in r.failures) and any("refused" in f for f in r.failures)


async def test_unfilled_and_partly_filled_orders_are_not_clean(engine: AsyncEngine) -> None:
    days = await _seed(engine, 22)
    async with engine.begin() as conn:
        for oid, qty in (("never-filled", None), ("half", 0.5)):
            await conn.execute(
                text(
                    "INSERT INTO core_orders (submitted_at, symbol, side, qty, est_price, "
                    "notional, reason, status, broker_order_id) VALUES (:t, 'X', 'buy', 1, 10, "
                    "10, 'rebalance', 'submitted', :o)"
                ),
                {"t": _at(days[-1]), "o": oid},
            )
            if qty:
                await conn.execute(
                    text(
                        "INSERT INTO broker_activities (id, account, activity_type, "
                        "transaction_time, symbol, side, qty, price, order_id, raw) VALUES "
                        "(:i, 'core', 'FILL', :t, 'X', 'buy', :q, 10, :o, '{}')"
                    ),
                    {"i": f"f-{oid}", "t": _at(days[-1]), "q": qty, "o": oid},
                )
    r = await check(engine, today=days[-1])
    assert (r.unfilled, r.partial) == (1, 1)
    assert any("unfilled" in f for f in r.failures)
    assert any("partially filled" in f for f in r.failures)


async def test_a_halted_run_is_not_clean(engine: AsyncEngine) -> None:
    days = await _seed(engine, 22)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO core_runs (account, run_on, monthly, executed, equity, cash, orders, "
                "halted, recorded_at) VALUES ('core', :d, false, false, 1, 1, 0, 'stale', now())"
            ),
            {"d": days[-2]},
        )
    r = await check(engine, today=days[-1])
    assert r.halted == 1 and not r.ready


async def test_the_live_gate_reads_the_paper_record_as_of_its_last_day(
    engine: AsyncEngine,
) -> None:
    days = await _seed(engine, 22)
    assert await live_gate(engine, today=days[-1] + timedelta(days=3)) == []
    # Long after the paper record ended, a first live run must rehearse again.
    late = await live_gate(engine, today=days[-1] + timedelta(days=30))
    assert any("rehearse again" in p for p in late)
