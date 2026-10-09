"""The core executor's money handling: sells before buys, cash-account budgets,
prices for holdings without a live trade, and what each order is called."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.domain.models import Position
from halal_trader.portfolio import core_executor as ce
from halal_trader.portfolio.core_account import core_account
from tests._core_fakes import FakeCoreBroker, no_sleep, screen

TODAY = date(2026, 10, 5)


async def test_the_core_account_name_follows_the_environment() -> None:
    assert core_account(True) == "core"  # where the paper history already is
    assert core_account(False) == "core-live"


def test_the_smallest_trade_scales_with_the_account() -> None:
    assert ce.min_trade(1_000) == ce.MIN_TRADE_FLOOR  # $1: Alpaca's fractional minimum
    assert ce.min_trade(105_000) == pytest.approx(52.5)  # a quarter of the band floor


async def _rebalance_setup(engine: AsyncEngine, **broker_kw) -> FakeCoreBroker:
    """Holding only OLD (left the target); the target is NEW: a sell funds a buy."""
    await screen(
        engine,
        TODAY - timedelta(days=2),
        {"NEW": ("halal", 10, 9e9), "OLD": ("halal", 10, 1e3)},
    )
    return FakeCoreBroker(
        cash=0,
        positions=[Position(symbol="OLD", qty=100, current_price=10)],
        prices={"NEW": 10, "OLD": 10},
        **broker_kw,
    )


async def test_buys_wait_for_the_sells_and_spend_their_proceeds(engine: AsyncEngine) -> None:
    broker = await _rebalance_setup(engine)
    p = await ce.plan(engine, broker, today=TODAY, monthly=True, top_n=1)
    assert [(o.symbol, o.side, o.reason) for o in p.orders] == [
        ("OLD", "sell", ce.TARGET_EXIT),
        ("NEW", "buy", ce.REBALANCE),
    ]
    results = await ce.execute(engine, broker, p, today=TODAY, sleep=no_sleep)
    assert [r["st"] for r in results] == ["submitted", "submitted"]
    assert broker.polls >= 1  # the sell was confirmed filled before the buy
    assert broker.cash >= 0 and broker.cash == pytest.approx(ce.CASH_BUFFER * 1_000)


async def test_an_unfilled_sell_funds_nothing(engine: AsyncEngine) -> None:
    broker = await _rebalance_setup(engine, sells_fill=False)
    p = await ce.plan(engine, broker, today=TODAY, monthly=True, top_n=1)
    results = await ce.execute(
        engine, broker, p, today=TODAY, sell_timeout=6, poll=2, sleep=no_sleep
    )
    assert [o[1] for o in broker.orders] == ["sell"]  # no buy against unreceived proceeds
    assert [r["side"] for r in results] == ["sell"]
    assert any("not filled" in n for n in p.notes) and any("no cash" in n for n in p.notes)
    assert broker.polls == 4  # 0, 2, 4, 6 seconds, then gave up


async def test_a_margin_account_is_budgeted_as_a_cash_account(engine: AsyncEngine) -> None:
    """The paper account has 4x buying power: none of it is borrowed."""
    await screen(engine, TODAY - timedelta(days=2), {"AAA": ("halal", 10, 1e9)})
    broker = FakeCoreBroker(cash=1_000, prices={"AAA": 10}, margin=4.0)
    p = await ce.plan(engine, broker, today=TODAY, monthly=True)
    await ce.execute(engine, broker, p, today=TODAY, sleep=no_sleep)
    assert broker.cash >= 0
    assert sum(q * 10 for _, _, q in broker.orders) == pytest.approx(990)


async def test_buys_never_exceed_the_cash_left_after_open_buy_orders(engine: AsyncEngine) -> None:
    await screen(engine, TODAY - timedelta(days=2), {"AAA": ("halal", 10, 1e9)})
    broker = FakeCoreBroker(cash=1_000, prices={"AAA": 10})
    p = await ce.plan(engine, broker, today=TODAY, monthly=True)
    # An order that appears between the plan and the buys still holds its cash.
    broker.open_orders = [{"side": "buy", "notional": "400", "status": "new"}]
    await ce.execute(engine, broker, p, today=TODAY, sleep=no_sleep)
    assert sum(q * 10 for _, _, q in broker.orders) == pytest.approx(1_000 - 400 - 10)


async def test_a_holding_without_a_live_price_is_kept_not_sold(engine: AsyncEngine) -> None:
    await screen(
        engine, TODAY - timedelta(days=2), {"AAA": ("halal", 10, 1e9), "QUIET": ("halal", 10, 1e9)}
    )
    broker = FakeCoreBroker(
        cash=0,
        positions=[
            Position(symbol="AAA", qty=50, current_price=10),
            Position(symbol="QUIET", qty=50, current_price=10),
        ],
        prices={"AAA": 10},  # no IEX trade for QUIET today
    )
    p = await ce.plan(engine, broker, today=TODAY, monthly=True)
    assert not [o for o in p.orders if o.symbol == "QUIET"]
    assert any("without a live price" in n and "QUIET" in n for n in p.notes)


async def test_a_new_name_without_a_price_is_left_out_and_said_so(engine: AsyncEngine) -> None:
    await screen(
        engine, TODAY - timedelta(days=2), {"AAA": ("halal", 10, 1e9), "DARK": ("halal", 10, 1e9)}
    )
    broker = FakeCoreBroker(cash=1_000, prices={"AAA": 10})
    p = await ce.plan(engine, broker, today=TODAY, monthly=True)
    assert {o.symbol for o in p.orders} == {"AAA"}
    assert any("DARK" in n for n in p.notes)


async def test_a_daily_run_never_buys_even_with_idle_cash(engine: AsyncEngine) -> None:
    await screen(engine, TODAY - timedelta(days=2), {"AAA": ("halal", 10, 1e9)})
    broker = FakeCoreBroker(
        cash=5_000, positions=[Position(symbol="AAA", qty=10, current_price=10)], prices={}
    )
    p = await ce.plan(engine, broker, today=TODAY, monthly=False)
    assert p.orders == []


async def test_a_universe_exit_is_named_apart_from_a_screen_sale(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:  # a one-name universe: only AAA trades enough
        for m in range(12):  # 2025-10 .. 2026-09, the year before TODAY's month
            await conn.execute(
                text(
                    "INSERT INTO monthly_bars (symbol, month, close, volume) "
                    "VALUES ('AAA', :m, 10, 1e9)"
                ),
                {"m": date(2025 + (9 + m) // 12, (9 + m) % 12 + 1, 1)},
            )
    await screen(
        engine,
        TODAY - timedelta(days=2),
        {"AAA": ("halal", 10, 1e9), "THIN": ("halal", 10, 1e9), "BAD": ("not_halal", 10, 1e9)},
    )
    broker = FakeCoreBroker(
        cash=0,
        positions=[Position(symbol=s, qty=10, current_price=10) for s in ("AAA", "THIN", "BAD")],
        prices={"AAA": 10, "THIN": 10, "BAD": 10},
    )
    p = await ce.plan(engine, broker, today=TODAY, monthly=True)
    reasons = {o.symbol: o.reason for o in p.orders if o.side == "sell"}
    assert reasons == {"THIN": ce.UNIVERSE_EXIT, "BAD": ce.SCREEN_SALE}
    daily = await ce.plan(engine, broker, today=TODAY, monthly=False)
    assert [(o.symbol, o.reason) for o in daily.orders] == [("BAD", ce.SCREEN_SALE)]


async def test_a_runs_notes_are_kept_with_it(engine: AsyncEngine) -> None:
    p = ce.Plan(monthly=True, screen_as_of=TODAY, equity=1.0, cash=1.0, notes=["a", "b"])
    p.account = "core-live"
    await ce.record_run(engine, p, today=TODAY, executed=True)
    async with engine.connect() as conn:
        row = (await conn.execute(text("SELECT account, notes FROM core_runs"))).one()
    assert row.account == "core-live" and row.notes == ["a", "b"]


def test_order_quantities_never_round_up_past_the_holding() -> None:
    assert ce._floor6(7.1234567) == 7.123456  # round() would sell 7.123457
    assert ce._floor6(0.3) == 0.3  # and binary noise does not shave a share's millionth


def test_an_open_buy_of_unknown_size_blocks_every_buy() -> None:
    assert ce._open_buy_notional([{"side": "buy", "notional": "50"}]) == 50.0
    assert ce._open_buy_notional([{"side": "buy", "qty": "2", "limit_price": "10"}]) == 20.0
    assert ce._open_buy_notional([{"side": "sell", "qty": "2"}]) == 0.0
    assert ce._open_buy_notional([{"side": "buy", "qty": "2"}]) == float("inf")


async def test_the_core_holds_technology_only_and_sells_the_rest_at_the_month(
    engine: AsyncEngine,
) -> None:
    """XOM passes the screen but is not technology: the monthly plan sells it and
    buys only tech; a daily run leaves it alone (the screen still passes it)."""
    await screen(
        engine,
        TODAY - timedelta(days=2),
        {"MSFT": ("halal", 10, 9e9), "XOM": ("halal", 10, 9e9)},
        sic={"XOM": "PETROLEUM REFINING"},
    )
    broker = FakeCoreBroker(
        cash=0,
        positions=[Position(symbol="XOM", qty=100, current_price=10)],
        prices={"MSFT": 10, "XOM": 10},
    )
    monthly = await ce.plan(engine, broker, today=TODAY, monthly=True)
    assert [(o.symbol, o.side) for o in monthly.orders] == [("XOM", "sell"), ("MSFT", "buy")]
    daily = await ce.plan(engine, broker, today=TODAY, monthly=False)
    assert daily.orders == []


async def test_a_rule_change_makes_the_month_due_again_from_its_date(engine: AsyncEngine) -> None:
    changed = ce.RULE_SINCE
    before = changed - timedelta(days=4)
    await screen(engine, before - timedelta(days=2), {"MSFT": ("halal", 10, 9e9)})
    broker = FakeCoreBroker(cash=1_000, prices={"MSFT": 10})
    p = await ce.plan(engine, broker, today=before, monthly=True)
    await ce.record_run(engine, p, today=before, executed=True)
    assert not await ce.monthly_due(engine, before)  # that month's rebalance ran
    assert await ce.monthly_due(engine, changed)  # but the targets changed since
    await ce.record_run(engine, p, today=changed, executed=True)
    assert not await ce.monthly_due(engine, changed + timedelta(days=1))
