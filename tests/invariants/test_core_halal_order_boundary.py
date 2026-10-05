"""INVARIANT: the core portfolio buys only what a fresh strict screen holds halal.

portfolio/core_executor.py is a second order path, on its own account. Like
TradeExecutor._execute_buy, it fails closed: a stale or missing screen means
no orders at all, and every buy is re-checked at the moment of the order.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.domain.models import Account, Position
from halal_trader.portfolio import core_executor as ce

TODAY = date(2026, 10, 5)


class FakeBroker:
    def __init__(self, *, cash: float, positions: list[Position], prices: dict[str, float]):
        self.cash, self.positions, self.prices = cash, positions, prices
        self.orders: list[tuple[str, str, float]] = []

    async def get_account_info(self) -> Account:
        equity = self.cash + sum(p.qty * p.current_price for p in self.positions)
        return Account(equity=equity, cash=self.cash, portfolio_value=equity, status="ACTIVE")

    async def get_all_positions(self) -> list[Position]:
        return self.positions

    async def get_stock_snapshot(self, symbols: str):
        return {
            s: {"latestTrade": {"p": self.prices[s]}}
            for s in symbols.split(",")
            if s in self.prices
        }

    async def place_order(self, symbol: str, side: str, quantity: float):
        self.orders.append((symbol, side, quantity))
        return {"id": f"o{len(self.orders)}", "status": "accepted"}


async def _screen(
    engine: AsyncEngine, as_of: date, rows: dict[str, tuple[str, float, float]]
) -> None:
    """symbol -> (verdict, price, shares)."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, :s, :c, '', :v, '[]', "
                "CAST(:m AS JSONB), 't', now())"
            ),
            [
                {
                    "a": as_of,
                    "s": s,
                    "c": i + 1,
                    "v": v,
                    "m": '{"price": %f, "shares_outstanding": %f}' % (p, sh),
                }
                for i, (s, (v, p, sh)) in enumerate(rows.items())
            ],
        )


async def test_a_stale_screen_means_no_orders_at_all(engine: AsyncEngine) -> None:
    await _screen(engine, TODAY - timedelta(days=30), {"AAA": ("halal", 10, 1e9)})
    broker = FakeBroker(cash=10_000, positions=[], prices={"AAA": 10})
    p = await ce.plan(engine, broker, today=TODAY, monthly=True)
    assert p.halted and "stale" in p.halted and p.orders == []
    assert await ce.execute(engine, broker, p, today=TODAY) == []
    assert broker.orders == []


async def test_the_first_monthly_run_buys_cap_weights_within_cash(engine: AsyncEngine) -> None:
    await _screen(
        engine,
        TODAY - timedelta(days=3),
        {"BIG": ("halal", 100, 3e9), "SMALL": ("halal", 100, 1e9), "BAD": ("not_halal", 100, 9e9)},
    )
    broker = FakeBroker(cash=10_000, positions=[], prices={"BIG": 100, "SMALL": 100, "BAD": 100})
    p = await ce.plan(engine, broker, today=TODAY, monthly=True)
    assert {o.symbol for o in p.orders} == {"BIG", "SMALL"}
    spend = sum(o.notional for o in p.orders)
    assert spend == pytest.approx(10_000 * (1 - ce.CASH_BUFFER))  # cash, less the buffer
    big = next(o for o in p.orders if o.symbol == "BIG")
    assert big.notional / spend == pytest.approx(0.75)
    results = await ce.execute(engine, broker, p, today=TODAY)
    assert [r["st"] for r in results] == ["submitted", "submitted"]


async def test_a_holding_the_screen_drops_is_sold_in_full_on_any_day(engine: AsyncEngine) -> None:
    await _screen(
        engine,
        TODAY - timedelta(days=1),
        {"KEEP": ("halal", 50, 1e9), "GONE": ("not_halal", 20, 1e9)},
    )
    positions = [
        Position(symbol="KEEP", qty=10, current_price=50),
        Position(symbol="GONE", qty=7.5, current_price=20),
    ]
    broker = FakeBroker(cash=0, positions=positions, prices={})
    p = await ce.plan(engine, broker, today=TODAY, monthly=False)
    sell, *rest = p.orders
    assert (sell.symbol, sell.side, sell.qty, sell.reason) == (
        "GONE",
        "sell",
        7.5,
        "forced sale (screen)",
    )
    # The proceeds go to what remains, pro rata (as the forward book does), within cash.
    assert [(o.symbol, o.side) for o in rest] == [("KEEP", "buy")]
    assert rest[0].notional <= 7.5 * 20 - ce.CASH_BUFFER * (10 * 50 + 7.5 * 20) + 1e-9


async def test_a_buy_is_refused_at_the_order_if_the_screen_changed_after_planning(
    engine: AsyncEngine,
) -> None:
    await _screen(
        engine, TODAY - timedelta(days=3), {"AAA": ("halal", 10, 1e9), "BBB": ("halal", 10, 1e9)}
    )
    broker = FakeBroker(cash=1_000, positions=[], prices={"AAA": 10, "BBB": 10})
    p = await ce.plan(engine, broker, today=TODAY, monthly=True)
    await _screen(engine, TODAY, {"AAA": ("not_halal", 10, 1e9), "BBB": ("halal", 10, 1e9)})
    results = await ce.execute(engine, broker, p, today=TODAY)
    assert {r["s"]: r["st"] for r in results} == {
        "AAA": "refused: not halal now",
        "BBB": "submitted",
    }
    assert [o[0] for o in broker.orders] == ["BBB"]


async def test_monthly_is_due_until_an_executed_monthly_run_is_recorded(
    engine: AsyncEngine,
) -> None:
    p = ce.Plan(monthly=True, screen_as_of=TODAY, equity=1.0, cash=1.0)
    assert await ce.monthly_due(engine, TODAY)
    await ce.record_run(engine, p, today=TODAY, executed=False)  # a plan only
    assert await ce.monthly_due(engine, TODAY)
    await ce.record_run(engine, p, today=TODAY, executed=True)
    assert not await ce.monthly_due(engine, TODAY)
    assert await ce.monthly_due(engine, date(2026, 11, 2))


async def test_the_kill_switch_stops_the_core_too(engine: AsyncEngine, monkeypatch) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from halal_trader.core import halt as halt_mod
    from halal_trader.trading.scheduler import TradingBot

    monkeypatch.setattr(halt_mod, "is_halted", AsyncMock(return_value=True))
    bot = TradingBot.__new__(TradingBot)
    bot._engine = engine
    bot.settings = SimpleNamespace(
        core=SimpleNamespace(
            enabled=True, alpaca_api_key="k", alpaca_secret_key="s", paper=True, top_n=100
        )
    )
    built = []
    monkeypatch.setattr(
        "halal_trader.execution.alpaca_broker.AlpacaRestBroker",
        lambda *a, **k: built.append(1),
    )
    await bot.core_trade()
    assert built == []  # never even reached the broker


async def test_unpaid_purification_is_held_back_from_buys(engine: AsyncEngine) -> None:
    await _screen(engine, TODAY - timedelta(days=3), {"AAA": ("halal", 10, 1e9)})
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO purification_accruals (account, dividend_id, symbol, ex_date, shares, "
                "dividend, impure_ratio, amount, method, accrued_at) VALUES "
                "('core', 'd1', 'AAA', '2026-09-01', 10, 100, 0.02, 2.0, 'm', now())"
            )
        )
    broker = FakeBroker(cash=1_000, positions=[], prices={"AAA": 10})
    p = await ce.plan(engine, broker, today=TODAY, monthly=True)
    spend = sum(o.notional for o in p.orders)
    assert spend == pytest.approx(1_000 - ce.CASH_BUFFER * 1_000 - 2.0)
    assert any("purification" in n for n in p.notes)
