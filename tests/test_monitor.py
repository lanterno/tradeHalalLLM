"""Tests for PositionMonitor — SL/TP triggers and trailing stop logic."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.crypto.monitor import PositionMonitor
from halal_trader.db.models import CryptoTrade


def _make_trade(
    trade_id=1,
    pair="BTCUSDT",
    entry_price=50000.0,
    stop_loss=49500.0,
    target_price=51000.0,
    quantity=0.001,
):
    trade = CryptoTrade(
        id=trade_id,
        pair=pair,
        side="buy",
        quantity=quantity,
        price=entry_price,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        status="filled",
        timestamp=datetime.now(UTC),
    )
    return trade


def _make_balance(asset="BTC", free=0.001, locked=0.0):
    b = MagicMock()
    b.asset = asset
    b.free = free
    b.locked = locked
    return b


def _make_monitor(
    *,
    symbol_filter_min_notional=5.0,
    trailing_activation_pct=None,
    trailing_distance_pct=0.003,
    order_result=None,
    balances=None,
):
    broker = AsyncMock()
    broker.place_order.return_value = order_result or {
        "orderId": "99",
        "status": "FILLED",
        "fills": [{"price": "50000", "qty": "0.001"}],
    }

    sf = MagicMock()
    sf.min_notional = symbol_filter_min_notional
    broker.get_symbol_filter = MagicMock(return_value=sf)
    broker.get_balances.return_value = balances if balances is not None else [
        _make_balance("BTC", 0.001),
    ]
    broker.round_quantity = MagicMock(side_effect=lambda symbol, qty: qty)

    repo = AsyncMock()
    repo.record_crypto_trade.return_value = 99
    repo.close_crypto_trade.return_value = None
    repo.update_crypto_trade_stop_loss.return_value = None

    ws = MagicMock()
    ws.get_latest_price.return_value = 50000.0

    monitor = PositionMonitor(
        broker=broker,
        repo=repo,
        ws_manager=ws,
        check_interval=0.1,
        trailing_stop_activation_pct=trailing_activation_pct,
        trailing_stop_distance_pct=trailing_distance_pct,
    )
    return monitor, broker, repo, ws


class TestCheckTrade:
    async def test_stop_loss_triggered(self):
        monitor, broker, repo, _ = _make_monitor()
        trade = _make_trade(stop_loss=50000.0)
        await monitor._check_trade(trade, price=49900.0)

        broker.place_order.assert_awaited_once()
        call_kwargs = broker.place_order.call_args.kwargs
        assert call_kwargs["side"] == "SELL"
        repo.close_crypto_trade.assert_awaited_once()
        close_args = repo.close_crypto_trade.call_args
        assert close_args.args[2] == "stop_loss"

    async def test_take_profit_triggered(self):
        monitor, broker, repo, _ = _make_monitor()
        trade = _make_trade(target_price=51000.0)
        await monitor._check_trade(trade, price=51500.0)

        broker.place_order.assert_awaited_once()
        repo.close_crypto_trade.assert_awaited_once()
        close_args = repo.close_crypto_trade.call_args
        assert close_args.args[2] == "take_profit"

    async def test_no_trigger_in_range(self):
        monitor, broker, repo, _ = _make_monitor()
        trade = _make_trade(stop_loss=49500.0, target_price=51000.0)
        await monitor._check_trade(trade, price=50500.0)

        broker.place_order.assert_not_awaited()
        repo.close_crypto_trade.assert_not_awaited()

    async def test_skip_trade_without_id(self):
        monitor, broker, _, _ = _make_monitor()
        trade = _make_trade()
        trade.id = None
        await monitor._check_trade(trade, price=0)
        broker.place_order.assert_not_awaited()

    async def test_skip_trade_without_sl_tp(self):
        monitor, broker, _, _ = _make_monitor()
        trade = _make_trade(stop_loss=None, target_price=None)
        await monitor._check_trade(trade, price=50000.0)
        broker.place_order.assert_not_awaited()


class TestExitPosition:
    async def test_records_sell_trade(self):
        monitor, broker, repo, _ = _make_monitor()
        trade = _make_trade()
        await monitor._exit_position(trade, 49000.0, "stop_loss")

        repo.record_crypto_trade.assert_awaited_once()
        call_kwargs = repo.record_crypto_trade.call_args.kwargs
        assert call_kwargs["side"] == "sell"
        assert call_kwargs["pair"] == "BTCUSDT"

    async def test_skips_below_min_notional(self):
        monitor, broker, repo, _ = _make_monitor(
            symbol_filter_min_notional=100.0,
            balances=[_make_balance("BTC", 0.0001)],
        )
        trade = _make_trade(quantity=0.0001)
        await monitor._exit_position(trade, 50000.0, "stop_loss")

        broker.place_order.assert_not_awaited()
        repo.close_crypto_trade.assert_awaited_once()
        close_args = repo.close_crypto_trade.call_args
        assert "too_small" in close_args.args[2]

    async def test_handles_order_failure_with_retry(self):
        monitor, broker, repo, _ = _make_monitor()
        broker.place_order.side_effect = RuntimeError("API error")
        trade = _make_trade()

        await monitor._exit_position(trade, 49000.0, "stop_loss")
        repo.close_crypto_trade.assert_not_awaited()
        assert monitor._exit_failures[1] == 1

        await monitor._exit_position(trade, 49000.0, "stop_loss")
        assert monitor._exit_failures[1] == 2

        await monitor._exit_position(trade, 49000.0, "stop_loss")
        repo.close_crypto_trade.assert_awaited_once()
        assert 1 not in monitor._exit_failures

    async def test_insufficient_balance_force_closes(self):
        from binance import BinanceAPIException

        monitor, broker, repo, _ = _make_monitor()
        exc = BinanceAPIException(
            response=MagicMock(status_code=400),
            status_code=400,
            text="Account has insufficient balance",
        )
        exc.code = -2010
        broker.place_order.side_effect = exc
        trade = _make_trade()
        await monitor._exit_position(trade, 49000.0, "stop_loss")

        repo.close_crypto_trade.assert_awaited_once()
        close_args = repo.close_crypto_trade.call_args
        assert "insufficient_balance" in close_args.args[2]

    async def test_zero_balance_closes_without_selling(self):
        monitor, broker, repo, _ = _make_monitor(balances=[])
        trade = _make_trade()
        await monitor._exit_position(trade, 49000.0, "stop_loss")

        broker.place_order.assert_not_awaited()
        repo.close_crypto_trade.assert_awaited_once()
        close_args = repo.close_crypto_trade.call_args
        assert "balance_exhausted" in close_args.args[2]

    async def test_cleans_high_water_on_exit(self):
        monitor, broker, repo, _ = _make_monitor()
        trade = _make_trade(trade_id=42)
        monitor._high_water[42] = 55000.0
        await monitor._exit_position(trade, 49000.0, "stop_loss")
        assert 42 not in monitor._high_water


class TestTrailingStop:
    async def test_trailing_stop_activates_above_threshold(self):
        monitor, broker, repo, _ = _make_monitor(
            trailing_activation_pct=0.005,
            trailing_distance_pct=0.003,
        )
        trade = _make_trade(entry_price=50000.0, stop_loss=49500.0)

        price_above_activation = 50000.0 * 1.006
        await monitor._update_trailing_stop(trade, price_above_activation)

        repo.update_crypto_trade_stop_loss.assert_awaited_once()
        new_sl = repo.update_crypto_trade_stop_loss.call_args.args[1]
        expected_sl = price_above_activation * (1 - 0.003)
        assert new_sl == pytest.approx(expected_sl, rel=1e-6)

    async def test_trailing_stop_does_not_activate_below_threshold(self):
        monitor, broker, repo, _ = _make_monitor(
            trailing_activation_pct=0.01,
        )
        trade = _make_trade(entry_price=50000.0, stop_loss=49500.0)

        await monitor._update_trailing_stop(trade, 50100.0)
        repo.update_crypto_trade_stop_loss.assert_not_awaited()

    async def test_trailing_stop_ratchets_up(self):
        monitor, broker, repo, _ = _make_monitor(
            trailing_activation_pct=0.005,
            trailing_distance_pct=0.003,
        )
        trade = _make_trade(entry_price=50000.0, stop_loss=49500.0)

        await monitor._update_trailing_stop(trade, 50300.0)
        first_sl = repo.update_crypto_trade_stop_loss.call_args.args[1]
        trade.stop_loss = first_sl

        await monitor._update_trailing_stop(trade, 50600.0)
        second_sl = repo.update_crypto_trade_stop_loss.call_args.args[1]
        assert second_sl > first_sl

    async def test_trailing_stop_does_not_lower(self):
        monitor, broker, repo, _ = _make_monitor(
            trailing_activation_pct=0.005,
            trailing_distance_pct=0.003,
        )
        trade = _make_trade(trade_id=1, entry_price=50000.0, stop_loss=49500.0)

        await monitor._update_trailing_stop(trade, 50500.0)
        first_sl = repo.update_crypto_trade_stop_loss.call_args.args[1]
        trade.stop_loss = first_sl

        repo.update_crypto_trade_stop_loss.reset_mock()
        await monitor._update_trailing_stop(trade, 50300.0)
        repo.update_crypto_trade_stop_loss.assert_not_awaited()


class TestStartStop:
    async def test_start_creates_task(self):
        monitor, _, _, _ = _make_monitor()
        await monitor.start()
        assert monitor._task is not None
        assert monitor._running is True
        await monitor.stop()
        assert monitor._running is False
        assert monitor._task is None
