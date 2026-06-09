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


def _wash_trade_rejection(existing_order_id: str | None = "blocking-ord") -> dict:
    """Shape of an Alpaca wash-trade (40310000) rejection via the MCP server."""
    detail: dict = {
        "code": 40310000,
        "message": "potential wash trade detected. use complex orders",
    }
    if existing_order_id is not None:
        detail["existing_order_id"] = existing_order_id
    return {"error": {"message": "API rejected the order", "http_status": 403, "detail": detail}}


async def test_exit_recovers_from_wash_trade_by_cancelling_blocker(engine):
    """A wash-trade rejection names the resting order blocking the sell — the
    monitor must cancel it and retry ONCE rather than looping a doomed sell."""
    repo = Repository(engine)
    tid = await repo.record_trade(
        symbol="AAPL", side="buy", quantity=10, price=200.0, stop_loss=190.0, target_price=220.0
    )
    mcp = MagicMock()
    mcp.place_order = AsyncMock(
        side_effect=[_wash_trade_rejection("blocking-ord"), {"id": "ord-2", "status": "filled"}]
    )
    mcp.cancel_order = AsyncMock(return_value={})
    mon = _monitor(repo, mcp=mcp)

    await mon._exit(_trade(id_=tid), price=185.0, reason="stop_loss")

    # Cancelled the exact blocker, then retried the sell → trade closed.
    mcp.cancel_order.assert_awaited_once_with("AAPL", "blocking-ord")
    assert mcp.place_order.await_count == 2
    async with engine.begin() as conn:
        row = await conn.execute(sa.text("SELECT status FROM trades WHERE id = :i"), {"i": tid})
        assert row.first()[0] == "closed"


async def test_exit_wash_trade_without_blocker_id_does_not_cancel(engine):
    """A 40310000 with no existing_order_id can't be auto-recovered — abort
    cleanly (don't cancel a guessed order); the position stays open."""
    repo = Repository(engine)
    tid = await repo.record_trade(
        symbol="AAPL", side="buy", quantity=10, price=200.0, stop_loss=190.0, target_price=220.0
    )
    mcp = MagicMock()
    mcp.place_order = AsyncMock(return_value=_wash_trade_rejection(existing_order_id=None))
    mcp.cancel_order = AsyncMock(return_value={})
    mon = _monitor(repo, mcp=mcp)

    await mon._exit(_trade(id_=tid), price=185.0, reason="stop_loss")

    mcp.cancel_order.assert_not_awaited()
    assert mcp.place_order.await_count == 1
    async with engine.begin() as conn:
        row = await conn.execute(sa.text("SELECT status FROM trades WHERE id = :i"), {"i": tid})
        assert row.first()[0] != "closed"


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


async def test_exit_calls_notifier_when_wired(engine):
    """An SL/TP exit fires `notify_sl_tp` so the operator gets the same
    Telegram alert the crypto monitor already sends."""
    repo = Repository(engine)
    try:
        tid = await repo.record_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=200.0,
            stop_loss=190.0,
            target_price=220.0,
        )
        notifier = MagicMock()
        notifier.enabled = True
        notifier.notify_sl_tp = AsyncMock()
        mon = _monitor(repo)
        mon._notifier = notifier
        await mon._exit(_trade(id_=tid, entry=200.0), price=185.0, reason="stop_loss")

        notifier.notify_sl_tp.assert_awaited_once()
        kwargs = notifier.notify_sl_tp.await_args.kwargs
        assert kwargs["pair"] == "AAPL"
        assert kwargs["exit_reason"] == "stop_loss"
        assert kwargs["entry_price"] == 200.0
        assert kwargs["exit_price"] == 185.0
        # 10 shares × ($185 − $200) = −$150
        assert abs(kwargs["pnl"] - (-150.0)) < 1e-6
    finally:
        await engine.dispose()


async def test_exit_swallows_notifier_exception(engine):
    """A blowing-up notifier must not abort the close path."""
    repo = Repository(engine)
    tid = await repo.record_trade(
        symbol="AAPL", side="buy", quantity=10, price=200.0, stop_loss=190.0, target_price=220.0
    )
    notifier = MagicMock()
    notifier.enabled = True
    notifier.notify_sl_tp = AsyncMock(side_effect=RuntimeError("telegram down"))
    mon = _monitor(repo)
    mon._notifier = notifier
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


# ── reactor (slow-out) wide trailing stop ───────────────────────


def _reactor_trade(*, id_=7, symbol="NVDA", entry=200.0, sl=184.0, qty=10):
    return Trade(
        id=id_,
        symbol=symbol,
        side="buy",
        quantity=qty,
        price=entry,
        filled_price=entry,
        status="open",
        stop_loss=sl,
        entry_type="reactor_momentum",
    )


async def test_reactor_position_trails_wide_and_activates_immediately():
    """Reactor positions trail at the wide distance (8%) and ratchet from
    the first tick in profit — no activation gate, since the trailing
    stop is their only rule-based exit."""
    repo = MagicMock()
    repo.update_stock_trade_stop_loss = AsyncMock()
    mon = StockPositionMonitor(
        MagicMock(), repo, check_interval=1, reactor_trailing_stop_distance_pct=0.08
    )
    # Entry 200, price 210 (only +5% — below any normal activation gate),
    # reactor still ratchets: new stop = 210 * (1 - 0.08) = 193.2 > 184.
    await mon._update_trailing_stop(_reactor_trade(entry=200.0, sl=184.0), price=210.0)
    repo.update_stock_trade_stop_loss.assert_awaited_once()
    _, new_stop = repo.update_stock_trade_stop_loss.await_args.args
    assert new_stop == pytest.approx(210.0 * 0.92)


async def test_reactor_trailing_never_lowers_stop():
    """A pullback after a high-water mark must not lower the stop."""
    repo = MagicMock()
    repo.update_stock_trade_stop_loss = AsyncMock()
    mon = StockPositionMonitor(
        MagicMock(), repo, check_interval=1, reactor_trailing_stop_distance_pct=0.08
    )
    trade = _reactor_trade(entry=200.0, sl=200.0)  # stop already high
    # Price below the level that would produce a higher stop → no update.
    await mon._update_trailing_stop(trade, price=205.0)  # 205*0.92=188.6 < 200
    repo.update_stock_trade_stop_loss.assert_not_awaited()


async def test_non_reactor_still_respects_activation_gate():
    """Cycle positions keep the opt-in activation gate (unchanged)."""
    repo = MagicMock()
    repo.update_stock_trade_stop_loss = AsyncMock()
    mon = StockPositionMonitor(
        MagicMock(),
        repo,
        check_interval=1,
        trailing_stop_activation_pct=0.10,  # needs +10% before trailing
        trailing_stop_distance_pct=0.005,
    )
    # Plain cycle trade up only +2% → below activation → no ratchet.
    await mon._update_trailing_stop(_trade(id_=3, entry=200.0, sl=190.0), price=204.0)
    repo.update_stock_trade_stop_loss.assert_not_awaited()


async def test_non_reactor_trailing_disabled_when_activation_none():
    """Default (activation None) leaves cycle positions untouched."""
    repo = MagicMock()
    repo.update_stock_trade_stop_loss = AsyncMock()
    mon = StockPositionMonitor(MagicMock(), repo, check_interval=1)
    await mon._update_trailing_stop(_trade(id_=4, entry=200.0, sl=190.0), price=260.0)
    repo.update_stock_trade_stop_loss.assert_not_awaited()


# ── reactor trend-break exit ────────────────────────────────────


def _bars(closes):
    return {"bars": [{"c": c} for c in closes]}


async def test_trend_break_exits_winning_reactor_below_ma():
    """A winning reactor position whose price falls below the SMA exits
    with reason 'trend_break' instead of waiting for the wide stop."""
    repo = MagicMock()
    repo.close_trade = AsyncMock()
    mcp = MagicMock()
    # 20 closes averaging ~210; latest price 205 < SMA → break.
    mcp.get_stock_bars = AsyncMock(return_value=_bars([210.0] * 20))
    mcp.place_order = AsyncMock(return_value={"id": "o", "status": "filled"})
    mon = StockPositionMonitor(
        mcp, repo, check_interval=1, trend_break_ma_period=20, trend_break_enabled=True
    )
    trade = _reactor_trade(id_=11, entry=200.0, sl=184.0)
    # price 205 > entry 200 (winner) but < SMA 210 → trend break exit.
    exited = await mon._maybe_trend_break_exit(trade, price=205.0)
    assert exited is True
    repo.close_trade.assert_awaited_once()
    assert repo.close_trade.await_args.kwargs["exit_reason"] == "trend_break"


async def test_trend_break_skips_losing_position():
    """A reactor position underwater rides the hard stop, not trend-break."""
    repo = MagicMock()
    repo.close_trade = AsyncMock()
    mcp = MagicMock()
    mcp.get_stock_bars = AsyncMock(return_value=_bars([210.0] * 20))
    mon = StockPositionMonitor(mcp, repo, check_interval=1, trend_break_enabled=True)
    trade = _reactor_trade(id_=12, entry=200.0, sl=184.0)
    # price below entry → not a winner → no trend-break exit.
    exited = await mon._maybe_trend_break_exit(trade, price=195.0)
    assert exited is False
    repo.close_trade.assert_not_awaited()
    mcp.get_stock_bars.assert_not_awaited()  # bailed before fetching bars


async def test_trend_break_skips_non_reactor():
    """Cycle positions are not subject to the trend-break exit."""
    repo = MagicMock()
    repo.close_trade = AsyncMock()
    mcp = MagicMock()
    mcp.get_stock_bars = AsyncMock(return_value=_bars([210.0] * 20))
    mon = StockPositionMonitor(mcp, repo, check_interval=1, trend_break_enabled=True)
    plain = _trade(id_=13, entry=200.0, sl=190.0, tp=None)
    exited = await mon._maybe_trend_break_exit(plain, price=205.0)
    assert exited is False
    mcp.get_stock_bars.assert_not_awaited()


async def test_trend_break_holds_when_price_above_ma():
    """In profit but still above the SMA → trend intact, no exit."""
    repo = MagicMock()
    repo.close_trade = AsyncMock()
    mcp = MagicMock()
    mcp.get_stock_bars = AsyncMock(return_value=_bars([200.0] * 20))  # SMA 200
    mon = StockPositionMonitor(mcp, repo, check_interval=1, trend_break_enabled=True)
    trade = _reactor_trade(id_=14, entry=195.0, sl=180.0)
    exited = await mon._maybe_trend_break_exit(trade, price=210.0)  # > SMA 200
    assert exited is False
    repo.close_trade.assert_not_awaited()


async def test_trend_break_disabled_is_noop():
    repo = MagicMock()
    repo.close_trade = AsyncMock()
    mcp = MagicMock()
    mcp.get_stock_bars = AsyncMock(return_value=_bars([210.0] * 20))
    mon = StockPositionMonitor(mcp, repo, check_interval=1, trend_break_enabled=False)
    trade = _reactor_trade(id_=15, entry=200.0, sl=184.0)
    exited = await mon._maybe_trend_break_exit(trade, price=205.0)
    assert exited is False
    mcp.get_stock_bars.assert_not_awaited()
