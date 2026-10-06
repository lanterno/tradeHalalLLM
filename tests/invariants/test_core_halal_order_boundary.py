"""INVARIANT: the core portfolio buys only what a fresh strict screen holds halal.

portfolio/core_executor.py is a second order path, on its own account. Like
TradeExecutor._execute_buy, it fails closed: a stale or missing screen means
no orders at all, and every buy is re-checked at the moment of the order.
The kill-switch stops every buy; only selling what the screen no longer
holds halal gets through it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core import halt
from halal_trader.domain.models import Position
from halal_trader.portfolio import core_executor as ce
from tests._core_fakes import FakeCoreBroker, core_settings, day_trader_is, no_sleep, screen

TODAY = date(2026, 10, 5)


async def test_a_stale_screen_means_no_orders_at_all(engine: AsyncEngine) -> None:
    await screen(engine, TODAY - timedelta(days=30), {"AAA": ("halal", 10, 1e9)})
    broker = FakeCoreBroker(cash=10_000, prices={"AAA": 10})
    p = await ce.plan(engine, broker, today=TODAY, monthly=True)
    assert p.halted and "stale" in p.halted and p.orders == []
    assert await ce.execute(engine, broker, p, today=TODAY) == []
    assert broker.orders == []


async def test_the_first_monthly_run_buys_cap_weights_within_cash(engine: AsyncEngine) -> None:
    await screen(
        engine,
        TODAY - timedelta(days=3),
        {"BIG": ("halal", 100, 3e9), "SMALL": ("halal", 100, 1e9), "BAD": ("not_halal", 100, 9e9)},
    )
    broker = FakeCoreBroker(cash=10_000, prices={"BIG": 100, "SMALL": 100, "BAD": 100})
    p = await ce.plan(engine, broker, today=TODAY, monthly=True)
    assert {o.symbol for o in p.orders} == {"BIG", "SMALL"}
    spend = sum(o.notional for o in p.orders)
    assert spend == pytest.approx(10_000 * (1 - ce.CASH_BUFFER))  # cash, less the buffer
    big = next(o for o in p.orders if o.symbol == "BIG")
    assert big.notional / spend == pytest.approx(0.75)
    results = await ce.execute(engine, broker, p, today=TODAY)
    assert [r["st"] for r in results] == ["submitted", "submitted"]


async def test_a_holding_the_screen_drops_is_sold_in_full_on_any_day(engine: AsyncEngine) -> None:
    await screen(
        engine,
        TODAY - timedelta(days=1),
        {"KEEP": ("halal", 50, 1e9), "GONE": ("not_halal", 20, 1e9)},
    )
    positions = [
        Position(symbol="KEEP", qty=10, current_price=50),
        Position(symbol="GONE", qty=7.5, current_price=20),
    ]
    broker = FakeCoreBroker(cash=0, positions=positions)
    p = await ce.plan(engine, broker, today=TODAY, monthly=False)
    # Sold in full, and nothing bought with the proceeds outside the monthly run.
    assert [(o.symbol, o.side, o.qty, o.reason) for o in p.orders] == [
        ("GONE", "sell", 7.5, ce.SCREEN_SALE)
    ]


async def test_a_buy_is_refused_at_the_order_if_the_screen_changed_after_planning(
    engine: AsyncEngine,
) -> None:
    await screen(
        engine, TODAY - timedelta(days=3), {"AAA": ("halal", 10, 1e9), "BBB": ("halal", 10, 1e9)}
    )
    broker = FakeCoreBroker(cash=1_000, prices={"AAA": 10, "BBB": 10})
    p = await ce.plan(engine, broker, today=TODAY, monthly=True)
    await screen(engine, TODAY, {"AAA": ("not_halal", 10, 1e9), "BBB": ("halal", 10, 1e9)})
    results = await ce.execute(engine, broker, p, today=TODAY)
    assert {r["s"]: r["st"] for r in results} == {
        "AAA": "refused: not halal now",
        "BBB": "submitted",
    }
    assert [o[0] for o in broker.orders] == ["BBB"]


async def test_monthly_is_due_until_a_monthly_run_actually_trades(engine: AsyncEngine) -> None:
    p = ce.Plan(monthly=True, screen_as_of=TODAY, equity=1.0, cash=1.0)
    assert await ce.monthly_due(engine, TODAY)
    await ce.record_run(engine, p, today=TODAY, executed=False)  # a plan only
    assert await ce.monthly_due(engine, TODAY)
    halted = ce.Plan(monthly=True, screen_as_of=None, equity=1.0, cash=1.0, halted="stale")
    await ce.record_run(engine, halted, today=TODAY, executed=False)
    assert await ce.monthly_due(engine, TODAY)  # a halted run does not use up the month
    await ce.record_run(engine, p, today=TODAY, executed=True)
    assert not await ce.monthly_due(engine, TODAY)
    assert await ce.monthly_due(engine, date(2026, 11, 2))
    assert await ce.monthly_due(engine, TODAY, "core-live")  # per account


async def test_the_kill_switch_lets_only_forced_sales_through(engine: AsyncEngine) -> None:
    """A halt is not a reason to keep holding what the screen no longer passes,
    and it is every reason not to buy: forced sales only, recorded, monthly or not."""
    await screen(
        engine,
        TODAY - timedelta(days=1),
        {"KEEP": ("halal", 50, 1e9), "GONE": ("not_halal", 20, 1e9), "NEW": ("halal", 10, 9e9)},
    )
    await halt.set_halt(engine, reason="drill", set_by="test")
    broker = FakeCoreBroker(
        cash=5_000,
        positions=[
            Position(symbol="KEEP", qty=10, current_price=50),
            Position(symbol="GONE", qty=5, current_price=20),
        ],
        prices={"KEEP": 50, "NEW": 10},
    )
    out = await ce.run(
        engine,
        broker,
        core_settings(),
        today=TODAY,
        execute_orders=True,
        monthly=True,  # even asked for a rebalance
        day_trader=day_trader_is("day-acct"),
        sleep=no_sleep,
    )
    assert out.kill_switch
    assert broker.orders == [("GONE", "sell", 5.0)]
    async with engine.connect() as conn:
        run = (await conn.execute(text("SELECT monthly, executed, halted FROM core_runs"))).one()
    assert run.monthly is False and run.executed is True and "kill-switch" in run.halted
    assert await ce.monthly_due(engine, TODAY)  # the month's rebalance is still owed


async def test_the_scheduled_job_obeys_the_kill_switch_too(
    engine: AsyncEngine, monkeypatch
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from halal_trader.trading.scheduler import TradingBot

    await halt.set_halt(engine, reason="drill", set_by="test")
    await screen(engine, TODAY - timedelta(days=1), {"NEW": ("halal", 10, 9e9)})
    broker = FakeCoreBroker(cash=5_000, prices={"NEW": 10})
    monkeypatch.setattr(
        "halal_trader.execution.alpaca_broker.AlpacaRestBroker", lambda *a, **k: broker
    )
    monkeypatch.setattr(
        "halal_trader.core.safeguards.day_trader_account", day_trader_is("day-acct")
    )
    monkeypatch.setattr("halal_trader.trading.scheduler.today_eastern", lambda: TODAY)
    bot = TradingBot.__new__(TradingBot)
    bot._engine = engine
    bot.settings = core_settings()
    bot._alerts = SimpleNamespace(notify=AsyncMock())
    await bot.core_trade()
    assert broker.orders == []  # nothing to sell, and nothing bought
    kinds = [c.args[0] for c in bot._alerts.notify.await_args_list]
    assert "core.halted_sells" in kinds


async def test_unpaid_purification_is_held_back_from_buys(engine: AsyncEngine) -> None:
    await screen(engine, TODAY - timedelta(days=3), {"AAA": ("halal", 10, 1e9)})
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO purification_accruals (account, dividend_id, symbol, ex_date, shares, "
                "dividend, impure_ratio, amount, method, accrued_at) VALUES "
                "('core', 'd1', 'AAA', '2026-09-01', 10, 100, 0.02, 2.0, 'm', now())"
            )
        )
    broker = FakeCoreBroker(cash=1_000, prices={"AAA": 10})
    p = await ce.plan(engine, broker, today=TODAY, monthly=True)
    spend = sum(o.notional for o in p.orders)
    assert spend == pytest.approx(1_000 - ce.CASH_BUFFER * 1_000 - 2.0)
    assert any("purification" in n for n in p.notes)
