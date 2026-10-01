"""Broker ledger: Alpaca's record synced, our trades reconciled to it, performance from it."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.execution.alpaca_rest import (
    BrokerActivity,
    EquityPoint,
    parse_activity,
    parse_portfolio_history,
)
from halal_trader.execution.ledger import performance, reconcile_fills, sync_broker_ledger

# ── parsing (real payload shapes, 2026-10-01) ─────────────────────────


def test_parse_a_fill() -> None:
    a = parse_activity(
        {
            "id": "20261001124621373::3c61",
            "activity_type": "FILL",
            "transaction_time": "2026-10-01T16:46:21.373332Z",
            "type": "fill",
            "price": "136.31",
            "qty": "1",
            "side": "buy",
            "symbol": "NOW",
            "order_id": "3a6f",
        }
    )
    assert (a.symbol, a.side, a.qty, a.price) == ("NOW", "buy", 1.0, 136.31)
    assert a.transaction_time == datetime(2026, 10, 1, 16, 46, 21, 373332, tzinfo=UTC)


def test_parse_a_dated_non_trade_activity() -> None:
    a = parse_activity(
        {"id": "x", "activity_type": "FEE", "date": "2026-05-21", "net_amount": "-0.35"}
    )
    assert a.net_amount == -0.35
    assert a.transaction_time == datetime(2026, 5, 21, tzinfo=UTC)
    assert a.symbol is None


def test_a_payload_missing_required_fields_raises() -> None:
    with pytest.raises(KeyError):
        parse_activity({"id": "x"})  # a schema change must fail the sync, loudly


def test_portfolio_history_skips_pre_account_nulls() -> None:
    points = parse_portfolio_history(
        {
            "timestamp": [1790294400, 1790380800],
            "equity": [None, 105764.43],
            "profit_loss": [None, 238.11],
            "profit_loss_pct": [None, 0.0023],
        }
    )
    # 1790380800 = 2026-09-26 00:00 UTC (a Saturday) = Friday 2026-09-25's close
    assert points == [EquityPoint(date(2026, 9, 25), 105764.43, 238.11, 0.0023)]


def test_portfolio_history_with_ragged_arrays_raises() -> None:
    with pytest.raises(ValueError):
        parse_portfolio_history(
            {"timestamp": [1], "equity": [1.0, 2.0], "profit_loss": [0], "profit_loss_pct": [0]}
        )


# ── sync ──────────────────────────────────────────────────────────────


def _fill(id_: str, when: datetime, symbol: str, side: str, qty: float) -> BrokerActivity:
    return BrokerActivity(
        id=id_,
        activity_type="FILL",
        transaction_time=when,
        symbol=symbol,
        side=side,
        qty=qty,
        price=100.0,
        net_amount=None,
        order_id="o-" + id_,
        raw={"id": id_},
    )


class FakeAlpaca:
    def __init__(self, activities: list[BrokerActivity], equity: list[EquityPoint]) -> None:
        self._activities = activities
        self._equity = equity
        self.after_requested: list[datetime | None] = []

    async def activities(self, *, after: datetime | None = None) -> list[BrokerActivity]:
        self.after_requested.append(after)
        return [a for a in self._activities if after is None or a.transaction_time > after]

    async def equity_history(self, *, start: date, end: date | None = None) -> list[EquityPoint]:
        return [p for p in self._equity if p.day >= start]


T = datetime(2026, 10, 1, 14, 0, tzinfo=UTC)  # 10:00 ET


async def test_sync_is_idempotent_and_incremental(engine: AsyncEngine) -> None:
    # First sync backfills equity from the first activity's date onwards.
    t0 = datetime(2026, 9, 30, 14, 0, tzinfo=UTC)
    fake = FakeAlpaca(
        [_fill("a1", t0, "NVDA", "buy", 5), _fill("a2", T, "AAPL", "sell", 3)],
        [
            EquityPoint(date(2026, 9, 30), 100_000, 0, 0),
            EquityPoint(date(2026, 10, 1), 101_000, 1_000, 0.01),
        ],
    )

    first = await sync_broker_ledger(engine, fake)  # type: ignore[arg-type]
    second = await sync_broker_ledger(engine, fake)  # type: ignore[arg-type]

    assert (first.activities_new, first.equity_days) == (2, 2)
    assert second.activities_new == 0
    assert fake.after_requested[0] is None  # empty ledger: everything
    assert fake.after_requested[1] is not None  # then only recent
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT count(*) FROM broker_activities"))).scalar() == 2
        assert (await conn.execute(text("SELECT count(*) FROM broker_equity"))).scalar() == 2


# ── reconciliation ────────────────────────────────────────────────────


async def _record_trade(
    engine: AsyncEngine, symbol: str, side: str, qty: float, status: str = "filled"
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO trades (timestamp, symbol, side, quantity, status, filled_at, "
                "filled_quantity) VALUES (:t, :s, :side, :q, :status, :t, :q)"
            ),
            {"t": T, "s": symbol, "side": side, "q": qty, "status": status},
        )


async def test_matching_books_reconcile_clean(engine: AsyncEngine) -> None:
    await sync_broker_ledger(  # type: ignore[arg-type]
        engine, FakeAlpaca([_fill("p1", T, "NVDA", "buy", 3), _fill("p2", T, "NVDA", "buy", 2)], [])
    )
    await _record_trade(engine, "NVDA", "buy", 5)  # partial fills sum to the recorded fill

    rec = await reconcile_fills(engine, date(2026, 10, 1))

    assert rec.clean and rec.broker_fills == 2


async def test_a_buy_whose_position_closed_still_counts_as_filled(engine: AsyncEngine) -> None:
    """Closing a position re-labels its BUY 'closed'; the fill still happened."""
    await sync_broker_ledger(  # type: ignore[arg-type]
        engine,
        FakeAlpaca([_fill("c1", T, "SHOP", "buy", 10), _fill("c2", T, "SHOP", "sell", 10)], []),
    )
    await _record_trade(engine, "SHOP", "buy", 10, status="closed")
    await _record_trade(engine, "SHOP", "sell", 10)

    assert (await reconcile_fills(engine, date(2026, 10, 1))).clean


async def test_drift_is_reported_both_ways(engine: AsyncEngine) -> None:
    await sync_broker_ledger(  # type: ignore[arg-type]
        engine,
        FakeAlpaca([_fill("b1", T, "NVDA", "buy", 5), _fill("b2", T, "SHOP", "buy", 10)], []),
    )
    await _record_trade(engine, "NVDA", "buy", 4)  # recorded less than filled
    await _record_trade(engine, "AMD", "buy", 2)  # recorded a fill the broker never made

    rec = await reconcile_fills(engine, date(2026, 10, 1))

    drift = {(d.symbol, d.side): d.difference for d in rec.drifts}
    assert drift == {("AMD", "buy"): -2.0, ("NVDA", "buy"): 1.0, ("SHOP", "buy"): 10.0}


# ── performance ───────────────────────────────────────────────────────


async def test_performance_from_broker_equity(engine: AsyncEngine) -> None:
    days = [date(2026, 9, d) for d in (24, 25, 28, 29)]
    pcts = [0.0, 0.10, -0.20, 0.05]  # first day's pct belongs to the day before
    equity = [100.0, 110.0, 88.0, 92.4]
    await sync_broker_ledger(  # type: ignore[arg-type]
        engine, FakeAlpaca([], [EquityPoint(d, e, 0, p) for d, e, p in zip(days, equity, pcts)])
    )

    p = await performance(engine)

    assert p is not None
    assert p.days == 3
    assert p.total_return == pytest.approx(1.10 * 0.80 * 1.05 - 1)
    assert p.max_drawdown == pytest.approx(-0.20)
    assert (p.best_day, p.worst_day) == (0.10, -0.20)
    assert (p.start_equity, p.end_equity) == (100.0, 92.4)


async def test_performance_needs_two_days(engine: AsyncEngine) -> None:
    assert await performance(engine) is None


# ── the bot's after-close job ─────────────────────────────────────────


def _bot(engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, fake: FakeAlpaca | Exception):  # type: ignore[no-untyped-def]
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import halal_trader.execution.alpaca_rest as rest
    import halal_trader.trading.scheduler as scheduler
    from halal_trader.trading.scheduler import TradingBot

    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            if isinstance(fake, Exception):
                self._fake = None
            else:
                self._fake = fake

        async def activities(self, **kw: object):  # type: ignore[no-untyped-def]
            if self._fake is None:
                raise fake  # type: ignore[misc]
            return await self._fake.activities(**kw)  # type: ignore[arg-type]

        async def equity_history(self, **kw: object):  # type: ignore[no-untyped-def]
            return await self._fake.equity_history(**kw)  # type: ignore[union-attr,arg-type]

        async def aclose(self) -> None: ...

    monkeypatch.setattr(rest, "AlpacaRestClient", _Client)
    monkeypatch.setattr(scheduler, "today_eastern", lambda: date(2026, 10, 1))
    bot = TradingBot.__new__(TradingBot)
    bot._engine = engine
    bot.settings = SimpleNamespace(
        alpaca=SimpleNamespace(api_key="k", secret_key="s", paper_trade=True)
    )
    bot._alerts = SimpleNamespace(notify=AsyncMock())
    return bot


async def test_job_with_agreeing_books_beats_and_stays_quiet(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from halal_trader.core.heartbeat import STOCK_LEDGER, read_beats

    await _record_trade(engine, "NVDA", "buy", 5)
    bot = _bot(engine, monkeypatch, FakeAlpaca([_fill("j1", T, "NVDA", "buy", 5)], []))

    await bot.sync_broker_ledger()

    bot._alerts.notify.assert_not_awaited()
    assert STOCK_LEDGER in await read_beats(engine)


async def test_job_alerts_on_fill_drift(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = _bot(engine, monkeypatch, FakeAlpaca([_fill("j2", T, "SHOP", "buy", 10)], []))

    await bot.sync_broker_ledger()  # a fill the bot never recorded

    kind, detail = bot._alerts.notify.await_args.args
    assert kind == "ledger.fill_drift"
    assert "SHOP buy: broker 10 vs recorded 0" in detail


async def test_job_survives_a_broker_outage(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = _bot(engine, monkeypatch, ConnectionError("alpaca down"))

    await bot.sync_broker_ledger()  # must not raise into the scheduler

    assert bot._alerts.notify.await_args.args[0] == "ledger.sync_failed"
