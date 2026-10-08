"""INVARIANT: the core never trades real money without every gate, and never shares an account.

CORE_PAPER=false used to be one setting away from live money. Now a live
core needs, at bot start and on every `core run`: today's dated token
(CORE_LIVE_CONFIRMATION), the paper rehearsal having passed its gate, and a
cash account (multiplier 1, no shorting, no debit: a margin debit is riba).
Its buys stop at CORE_LIVE_MAX_NOTIONAL invested. On paper or live, the
core's keys must reach a different account from the day-trader's. Every
refusal places nothing and fails closed when it cannot tell.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.safeguards import expected_core_token
from halal_trader.market_hours import trading_days_back
from halal_trader.portfolio import core_executor as ce
from tests._core_fakes import FakeCoreBroker, core_settings, day_trader_is, no_sleep, screen

TODAY = date(2026, 11, 20)
NOW = datetime(2026, 11, 20, 20, 0, tzinfo=UTC)
TOKEN = expected_core_token(NOW)


async def _passing_paper_record(engine: AsyncEngine, last: date) -> None:
    """A paper core that passed its gate on ``last``: 25 sessions on its book,
    a run every day, and a monthly rebalance that sold."""
    days = trading_days_back(last, 25)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO forward_books (name, strategy, params, created_at) "
                "VALUES ('core', 'core-strict-cap', '{}', now())"
            )
        )
        for i, d in enumerate(days):
            nav = 1.001**i
            await conn.execute(
                text(
                    "INSERT INTO forward_book_days (book, day, nav, day_return, turnover, "
                    "weights, rebalance_next, recorded_at) VALUES ('core', :d, :n, 0, 0, '{}', "
                    "false, now())"
                ),
                {"d": d, "n": nav},
            )
            await conn.execute(
                text(
                    "INSERT INTO broker_equity (account, day, equity, profit_loss, "
                    "profit_loss_pct, synced_at) VALUES ('core', :d, :e, 0, 0, now())"
                ),
                {"d": d, "e": 100_000 * nav},
            )
            await conn.execute(
                text(
                    "INSERT INTO core_runs (account, run_on, monthly, executed, equity, cash, "
                    "orders, recorded_at) VALUES ('core', :d, :m, true, 100000, 1000, 1, now())"
                ),
                {"d": d, "m": i == 10},
            )
        sold_on = datetime.combine(days[10], datetime.min.time(), UTC) + timedelta(hours=19)
        await conn.execute(
            text(
                "INSERT INTO core_orders (account, submitted_at, symbol, side, qty, est_price, "
                "notional, reason, status, broker_order_id) VALUES ('core', :t, 'X', 'sell', 1, "
                "10, 10, 'rebalance', 'submitted', 'sold-1')"
            ),
            {"t": sold_on},
        )
        await conn.execute(
            text(
                "INSERT INTO broker_activities (id, account, activity_type, transaction_time, "
                "symbol, side, qty, price, order_id, raw) VALUES ('f1', 'core', 'FILL', :t, 'X', "
                "'sell', 1, 10, 'sold-1', '{}')"
            ),
            {"t": sold_on},
        )


async def _live_run(engine: AsyncEngine, broker: FakeCoreBroker, **settings_kw):
    settings_kw.setdefault("token", TOKEN)
    return await ce.run(
        engine,
        broker,
        core_settings(paper=False, **settings_kw),
        today=TODAY,
        execute_orders=True,
        day_trader=day_trader_is("day-acct"),
        now=NOW,
        sleep=no_sleep,
    )


@pytest.fixture
async def halal_screen(engine: AsyncEngine) -> None:
    await screen(engine, TODAY - timedelta(days=2), {"AAA": ("halal", 10, 1e9)})


async def test_live_without_todays_token_places_nothing(engine, halal_screen) -> None:
    await _passing_paper_record(engine, TODAY - timedelta(days=1))
    broker = FakeCoreBroker(cash=500, prices={"AAA": 10})
    yesterday = expected_core_token(NOW - timedelta(days=1))
    for token in ("", yesterday, "I-UNDERSTAND-REAL-MONEY-" + NOW.strftime("%Y-%m-%d")):
        out = await _live_run(engine, broker, token=token)
        assert out.plan is None and any("CORE_LIVE_CONFIRMATION" in r for r in out.refused)
    assert broker.orders == []


@pytest.mark.parametrize(
    ("margin", "shorting", "cash", "why"),
    [
        (4.0, False, 500, "margin account"),
        (1.0, True, 500, "shorting"),
        (None, False, 500, "multiplier is unknown"),
        (1.0, False, -50, "margin debit"),
    ],
)
async def test_live_needs_a_cash_account(engine, halal_screen, margin, shorting, cash, why) -> None:
    await _passing_paper_record(engine, TODAY - timedelta(days=1))
    broker = FakeCoreBroker(cash=cash, prices={"AAA": 10}, margin=margin or 1.0, shorting=shorting)
    if margin is None:
        info = await broker.get_account_info()
        broker.get_account_info = AsyncMock(  # type: ignore[method-assign]
            return_value=info.model_copy(update={"multiplier": None})
        )
    out = await _live_run(engine, broker)
    assert any(why in r for r in out.refused), out.refused
    assert broker.orders == []


async def test_live_needs_the_paper_gate_passed(engine, halal_screen) -> None:
    broker = FakeCoreBroker(cash=500, prices={"AAA": 10})
    out = await _live_run(engine, broker)  # no paper record at all
    assert any("paper" in r for r in out.refused)
    assert broker.orders == []


async def test_a_stale_paper_record_does_not_arm_a_first_live_run(engine, halal_screen) -> None:
    await _passing_paper_record(engine, TODAY - timedelta(days=40))
    out = await _live_run(engine, FakeCoreBroker(cash=500, prices={"AAA": 10}))
    assert any("rehearse again" in r for r in out.refused)


async def test_with_every_gate_live_buys_stop_at_the_ceiling(engine, halal_screen) -> None:
    await _passing_paper_record(engine, TODAY - timedelta(days=1))
    broker = FakeCoreBroker(cash=5_000, prices={"AAA": 10})
    out = await _live_run(engine, broker, ceiling=1_000)
    assert out.refused == [] and out.account == "core-live"
    spent = sum(q * 10 for _, side, q in broker.orders if side == "buy")
    assert spent == pytest.approx(1_000, rel=1e-6)
    async with engine.connect() as conn:
        accounts = {
            r.account for r in await conn.execute(text("SELECT account FROM core_orders"))
        } | {r.account for r in await conn.execute(text("SELECT account FROM core_runs"))}
    assert "core-live" in accounts  # recorded apart from the paper history
    assert await ce.monthly_due(engine, TODAY, "core")  # paper's month untouched


async def test_the_core_refuses_the_day_traders_account(engine, halal_screen) -> None:
    broker = FakeCoreBroker(cash=500, prices={"AAA": 10}, account_id="shared")
    out = await ce.run(
        engine,
        broker,
        core_settings(),
        today=TODAY,
        execute_orders=True,
        day_trader=day_trader_is("shared"),
        sleep=no_sleep,
    )
    assert any("day-trader's account" in r for r in out.refused)
    same_keys = await ce.run(
        engine,
        broker,
        core_settings(day_key="k", core_key="k"),
        today=TODAY,
        execute_orders=True,
        day_trader=day_trader_is("other"),
        sleep=no_sleep,
    )
    assert any("share API keys" in r for r in same_keys.refused)
    assert broker.orders == []


async def test_an_unreadable_day_trader_account_refuses_a_live_core(engine, halal_screen) -> None:
    async def down(_settings):
        raise RuntimeError("401")

    broker = FakeCoreBroker(cash=500, prices={"AAA": 10})
    out = await ce.run(
        engine,
        broker,
        core_settings(paper=False),
        today=TODAY,
        execute_orders=True,
        day_trader=down,
        sleep=no_sleep,
    )
    assert any("cannot read the day-trader" in r for r in out.refused)
    assert broker.orders == []


async def test_an_unreadable_day_trader_account_does_not_stop_the_paper_core(
    engine, halal_screen
) -> None:
    """The retired day-trader's keys going stale must not stop the paper core's
    daily run (identical keys are still refused, above)."""

    async def down(_settings):
        raise RuntimeError("401")

    broker = FakeCoreBroker(cash=500, prices={"AAA": 10})
    out = await ce.run(
        engine,
        broker,
        core_settings(),
        today=TODAY,
        execute_orders=True,
        day_trader=down,
        sleep=no_sleep,
    )
    assert not any("cannot read the day-trader" in r for r in out.refused)


async def test_the_bot_checks_the_token_at_start_and_refuses_a_live_core_without_it(
    engine, monkeypatch
) -> None:
    from halal_trader.trading.scheduler import TradingBot

    built: list[int] = []
    monkeypatch.setattr(
        "halal_trader.execution.alpaca_broker.AlpacaRestBroker",
        lambda *a, **k: built.append(1) or FakeCoreBroker(cash=1),
    )
    monkeypatch.setattr("halal_trader.core.safeguards.day_trader_account", day_trader_is(None))
    bot = TradingBot.__new__(TradingBot)
    bot._engine = engine
    bot.settings = core_settings(paper=False, token="")
    bot._alerts = SimpleNamespace(notify=AsyncMock())
    problems = await bot.core_start_check()
    assert any("CORE_LIVE_CONFIRMATION" in p for p in problems)
    built.clear()
    await bot.core_trade()
    assert built == []  # refused before any broker was built
    kinds = [c.args[0] for c in bot._alerts.notify.await_args_list]
    assert kinds.count("core.refused") == 2  # at start, and at the run


async def test_a_live_core_never_checked_at_start_is_refused(engine, monkeypatch) -> None:
    from halal_trader.trading.scheduler import TradingBot

    built: list[int] = []
    monkeypatch.setattr(
        "halal_trader.execution.alpaca_broker.AlpacaRestBroker",
        lambda *a, **k: built.append(1),
    )
    bot = TradingBot.__new__(TradingBot)
    bot._engine = engine
    bot.settings = core_settings(paper=False, token=TOKEN)
    bot._alerts = SimpleNamespace(notify=AsyncMock())
    await bot.core_trade()
    assert built == []
