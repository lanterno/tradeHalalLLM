"""Stock position monitor tests — SL/TP enforcement + price extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa

from halal_trader.db.models import Trade
from halal_trader.db.repository import Repository
from halal_trader.trading.monitor import StockPositionMonitor, _extract_last_price


def _trade(*, id_=1, symbol="AAPL", entry=200.0, sl=190.0, tp=220.0, qty=10):
    return Trade(
        id=id_,
        symbol=symbol,
        side="buy",
        quantity=qty,
        price=entry,
        filled_price=entry,
        status="open",
        stop_loss=sl,
        target_price=tp,
    )


def _monitor(repo, mcp=None):
    if mcp is None:
        mcp = MagicMock()
        mcp.place_order = AsyncMock(return_value={"id": "ord-1", "status": "filled"})
        mcp.get_stock_snapshot = AsyncMock(return_value={})
    return StockPositionMonitor(mcp=mcp, repo=repo, check_interval=1)


# ── Price extraction ────────────────────────────────────────────


def test_extract_last_price_flat_dict():
    snap = {"latestTrade": {"p": 195.5}}
    assert _extract_last_price(snap, "AAPL") == 195.5


def test_extract_last_price_nested_by_symbol():
    snap = {"AAPL": {"latestTrade": {"price": 195.5}}}
    assert _extract_last_price(snap, "AAPL") == 195.5


def test_extract_last_price_alt_keys():
    snap = {"latest_trade": {"p": "195.5"}}
    assert _extract_last_price(snap, "AAPL") == 195.5


def test_extract_last_price_missing_returns_none():
    assert _extract_last_price({}, "AAPL") is None
    assert _extract_last_price(None, "AAPL") is None
    assert _extract_last_price({"AAPL": "garbage"}, "AAPL") is None


# ── _check_trade behaviour ──────────────────────────────────────


async def test_check_trade_triggers_stop_loss(engine):
    repo = Repository(engine)
    # Seed a trade so close_trade has something to mutate.
    tid = await repo.record_trade(
        symbol="AAPL",
        side="buy",
        quantity=10,
        price=200.0,
        stop_loss=190.0,
        target_price=220.0,
    )
    mon = _monitor(repo)
    # Reload as Trade so the monitor sees the populated id.
    tr = _trade(id_=tid, sl=190.0, tp=220.0)
    await mon._check_trade(tr, price=185.0)
    # close_trade was called → status closed, exit_reason recorded.
    assert mon._mcp.place_order.await_count == 1
    async with engine.begin() as conn:
        row = await conn.execute(
            sa.text("SELECT status, exit_reason FROM trades WHERE id = :i"),
            {"i": tid},
        )
        status, reason = row.first()
        assert status == "closed"
        assert reason == "stop_loss"


async def test_check_trade_triggers_take_profit(engine):
    repo = Repository(engine)
    tid = await repo.record_trade(
        symbol="AAPL", side="buy", quantity=10, price=200.0, stop_loss=190.0, target_price=220.0
    )
    mon = _monitor(repo)
    tr = _trade(id_=tid)
    await mon._check_trade(tr, price=225.0)
    assert mon._mcp.place_order.await_count == 1
    async with engine.begin() as conn:
        row = await conn.execute(
            sa.text("SELECT exit_reason FROM trades WHERE id = :i"), {"i": tid}
        )
        assert row.first()[0] == "take_profit"


async def test_check_trade_holds_inside_band(engine):
    repo = Repository(engine)
    tid = await repo.record_trade(
        symbol="AAPL", side="buy", quantity=10, price=200.0, stop_loss=190.0, target_price=220.0
    )
    mon = _monitor(repo)
    tr = _trade(id_=tid)
    await mon._check_trade(tr, price=205.0)
    assert mon._mcp.place_order.await_count == 0
    async with engine.begin() as conn:
        row = await conn.execute(sa.text("SELECT status FROM trades WHERE id = :i"), {"i": tid})
        assert row.first()[0] != "closed"


async def test_trailing_stop_ratchets_up(engine):
    repo = Repository(engine)
    tid = await repo.record_trade(
        symbol="AAPL", side="buy", quantity=10, price=200.0, stop_loss=190.0, target_price=220.0
    )
    mon = _monitor(repo)
    # Activate trailing once price is 1% above entry; distance 0.5%.
    mon._trailing_activation_pct = 0.01
    mon._trailing_distance_pct = 0.005
    tr = _trade(id_=tid)
    await mon._check_trade(tr, price=210.0)  # +5% gain → trail activates
    async with engine.begin() as conn:
        row = await conn.execute(sa.text("SELECT stop_loss FROM trades WHERE id = :i"), {"i": tid})
        new_sl = row.first()[0]
    # new_sl should be 210 * (1 - 0.005) = 208.95
    assert new_sl == pytest.approx(208.95, rel=1e-4)


async def test_exit_skipped_when_alpaca_returns_error(engine):
    repo = Repository(engine)
    tid = await repo.record_trade(
        symbol="AAPL", side="buy", quantity=10, price=200.0, stop_loss=190.0, target_price=220.0
    )
    mcp = MagicMock()
    mcp.place_order = AsyncMock(return_value={"error": "rejected"})
    mon = _monitor(repo, mcp=mcp)
    await mon._exit(_trade(id_=tid), price=185.0, reason="stop_loss")
    # Trade should remain open since the exit failed.
    async with engine.begin() as conn:
        row = await conn.execute(sa.text("SELECT status FROM trades WHERE id = :i"), {"i": tid})
        assert row.first()[0] != "closed"


async def test_exit_swallows_mcp_exception(engine):
    repo = Repository(engine)
    tid = await repo.record_trade(
        symbol="AAPL", side="buy", quantity=10, price=200.0, stop_loss=190.0, target_price=220.0
    )
    mcp = MagicMock()
    mcp.place_order = AsyncMock(side_effect=RuntimeError("network gone"))
    mon = _monitor(repo, mcp=mcp)
    # Must not propagate.
    await mon._exit(_trade(id_=tid), price=185.0, reason="stop_loss")


# ── Repo round-trip surface ─────────────────────────────────────


async def test_exit_calls_retrainer_with_return_pct(engine):
    """When wired with a retrainer, a successful exit feeds it the realized return."""
    repo = Repository(engine)
    try:
        tid = await repo.record_trade(
            symbol="AAPL", side="buy", quantity=10, price=200.0, stop_loss=190.0, target_price=220.0
        )
        retrainer = MagicMock()
        retrainer.on_trade_closed = AsyncMock()
        mon = _monitor(repo)
        mon._retrainer = retrainer
        # Exit at $185 → -7.5% from entry of 200.
        await mon._exit(_trade(id_=tid, entry=200.0), price=185.0, reason="stop_loss")

        retrainer.on_trade_closed.assert_awaited_once()
        args, _ = retrainer.on_trade_closed.await_args
        assert args[0] == tid
        assert abs(args[1] - (-0.075)) < 1e-6
    finally:
        await engine.dispose()


async def test_exit_swallows_retrainer_exception(engine):
    """A blowing-up retrainer must not abort the close path."""
    repo = Repository(engine)
    tid = await repo.record_trade(
        symbol="AAPL", side="buy", quantity=10, price=200.0, stop_loss=190.0, target_price=220.0
    )
    retrainer = MagicMock()
    retrainer.on_trade_closed = AsyncMock(side_effect=RuntimeError("retrain dead"))
    mon = _monitor(repo)
    mon._retrainer = retrainer
    await mon._exit(_trade(id_=tid), price=185.0, reason="stop_loss")  # must not raise


async def test_get_open_trades_returns_only_unclosed_buys(engine):
    repo = Repository(engine)
    try:
        await repo.record_trade(symbol="AAPL", side="buy", quantity=10, price=200.0)
        sell_id = await repo.record_trade(symbol="AAPL", side="sell", quantity=10, price=210.0)
        closed_id = await repo.record_trade(symbol="MSFT", side="buy", quantity=5, price=420.0)
        await repo.close_trade(closed_id, exit_price=425.0, exit_reason="take_profit")

        open_trades = await repo.get_open_trades()
        ids = sorted(t.id for t in open_trades)
        assert sell_id not in ids
        assert closed_id not in ids
        # The AAPL buy should remain.
        symbols = {t.symbol for t in open_trades}
        assert symbols == {"AAPL"}
    finally:
        await engine.dispose()
