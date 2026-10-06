"""INVARIANT: one core rebalance per month, never placed twice, never counted when it did not run.

A repeated `core run` once placed ~$104k of buys a second time (the plan
ignored the first run's open orders, and every order carried a fresh
random id), and a halted run used up the month. Now:

* nothing is planned while the account has an open order;
* every order's ``client_order_id`` is ``core-<date>-<symbol>-<side>``, which
  Alpaca refuses to accept twice;
* only a run that traded (or had nothing to do) is recorded ``executed``:
  a halted or all-refused run leaves the month's rebalance owed;
* a plan (`core plan`) and a closed market record nothing.
"""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.portfolio import core_executor as ce
from tests._core_fakes import FakeCoreBroker, core_settings, day_trader_is, no_sleep, screen

TODAY = date(2026, 10, 5)


async def _run(engine: AsyncEngine, broker: FakeCoreBroker, **kw):
    kw.setdefault("execute_orders", True)
    return await ce.run(
        engine,
        broker,
        core_settings(),
        today=TODAY,
        day_trader=day_trader_is("day-acct"),
        sleep=no_sleep,
        **kw,
    )


async def _runs(engine: AsyncEngine) -> list:
    async with engine.connect() as conn:
        return list(
            await conn.execute(
                text("SELECT monthly, executed, halted, orders FROM core_runs ORDER BY id")
            )
        )


async def _screen(engine: AsyncEngine) -> None:
    await screen(
        engine, TODAY - timedelta(days=2), {"AAA": ("halal", 10, 2e9), "BBB": ("halal", 20, 1e9)}
    )


async def test_a_second_run_the_same_day_buys_nothing(engine: AsyncEngine) -> None:
    await _screen(engine)
    broker = FakeCoreBroker(cash=10_000, prices={"AAA": 10, "BBB": 20})
    first = await _run(engine, broker)
    assert len(first.submitted) == 2
    assert broker.client_ids == ["core-20261005-AAA-buy", "core-20261005-BBB-buy"]
    bought = list(broker.orders)
    second = await _run(engine, broker)  # the month is done: sells of screen failures only
    assert second.plan is not None and not second.plan.monthly
    assert broker.orders == bought


async def test_a_forced_repeat_is_refused_by_its_order_ids(engine: AsyncEngine) -> None:
    await _screen(engine)
    broker = FakeCoreBroker(cash=10_000, prices={"AAA": 10, "BBB": 20})
    await _run(engine, broker)
    broker.cash += 10_000  # even with fresh cash on the account
    again = await _run(engine, broker, monthly=True)
    assert again.submitted == [] and len(again.rejected) >= 1
    assert len(broker.orders) == 2  # Alpaca accepted each id once
    assert (await _runs(engine))[-1].executed is False  # every order refused: not a run


async def test_nothing_is_planned_while_an_order_is_open(engine: AsyncEngine) -> None:
    await _screen(engine)
    broker = FakeCoreBroker(
        cash=10_000,
        prices={"AAA": 10, "BBB": 20},
        open_orders=[{"id": "x", "side": "buy", "qty": "5", "status": "new"}],
    )
    out = await _run(engine, broker)
    assert out.plan is not None and out.plan.halted and "open" in out.plan.halted
    assert broker.orders == []
    (run,) = await _runs(engine)
    assert run.executed is False and run.halted
    assert await ce.monthly_due(engine, TODAY)  # the halted run did not use up the month


async def test_a_halted_run_leaves_the_month_owed(engine: AsyncEngine) -> None:
    await screen(engine, TODAY - timedelta(days=40), {"AAA": ("halal", 10, 1e9)})  # stale
    broker = FakeCoreBroker(cash=10_000, prices={"AAA": 10})
    out = await _run(engine, broker)
    assert out.plan is not None and out.plan.halted
    (run,) = await _runs(engine)
    assert run.monthly is True and run.executed is False
    assert await ce.monthly_due(engine, TODAY)


async def test_a_plan_records_nothing_and_a_closed_market_runs_nothing(
    engine: AsyncEngine,
) -> None:
    await _screen(engine)
    broker = FakeCoreBroker(cash=10_000, prices={"AAA": 10, "BBB": 20})
    preview = await _run(engine, broker, execute_orders=False)
    assert preview.plan is not None and len(preview.plan.orders) == 2
    assert broker.orders == [] and await _runs(engine) == []

    closed = FakeCoreBroker(cash=10_000, prices={"AAA": 10}, is_open=False)
    out = await _run(engine, closed)
    assert out.market_closed and out.plan is None
    assert closed.orders == [] and await _runs(engine) == []


async def test_order_ids_are_deterministic() -> None:
    assert ce.client_order_id(TODAY, "BRK.B", "sell") == "core-20261005-BRK.B-sell"
