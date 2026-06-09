"""Tests for the DB-vs-broker reconciler (core/reconcile.py)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core import reconcile
from halal_trader.db.repository import Repository
from halal_trader.notifications.telegram import AlertSink, TelegramNotifier


def _alert_sink() -> tuple[AlertSink, MagicMock]:
    notifier = MagicMock(spec=TelegramNotifier)
    notifier.enabled = True
    notifier.notify_error = AsyncMock()
    return AlertSink(notifier=notifier), notifier


def _balance(asset: str, free: float, locked: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(asset=asset, free=free, locked=locked)


def _stock_position(symbol: str, qty: float) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, qty=qty)


# ── Crypto ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_crypto_clean_when_quantities_match(engine):
    repo = Repository(engine)
    await repo.record_crypto_trade(pair="BTCUSDT", side="buy", quantity=0.5, status="filled")

    broker = MagicMock()
    broker.get_balances = AsyncMock(return_value=[_balance("BTC", 0.5)])
    broker.get_cached_price = MagicMock(return_value=68000.0)

    report = await reconcile.reconcile_crypto(engine=engine, broker=broker)
    assert not report.has_drift
    assert report.checked_symbols >= 1


@pytest.mark.asyncio
async def test_crypto_drift_above_threshold_logged(engine):
    repo = Repository(engine)
    await repo.record_crypto_trade(pair="BTCUSDT", side="buy", quantity=1.0, status="filled")

    broker = MagicMock()
    broker.get_balances = AsyncMock(return_value=[_balance("BTC", 0.7)])
    broker.get_cached_price = MagicMock(return_value=70000.0)

    sink, notifier = _alert_sink()
    report = await reconcile.reconcile_crypto(engine=engine, broker=broker, alerts=sink)
    assert report.has_drift
    drift = report.drifts[0]
    assert drift.symbol == "BTC"
    assert drift.db_quantity == 1.0
    assert drift.broker_quantity == 0.7
    assert pytest.approx(drift.drift_pct, rel=1e-3) == 0.3
    assert drift.drift_usd == pytest.approx(0.3 * 70000)

    notifier.notify_error.assert_awaited_once()
    rows = await reconcile.get_recent_logs(engine)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTC"


@pytest.mark.asyncio
async def test_crypto_drift_below_threshold_skipped(engine):
    repo = Repository(engine)
    await repo.record_crypto_trade(pair="BTCUSDT", side="buy", quantity=1.0, status="filled")

    broker = MagicMock()
    broker.get_balances = AsyncMock(return_value=[_balance("BTC", 0.998)])
    broker.get_cached_price = MagicMock(return_value=70000.0)

    report = await reconcile.reconcile_crypto(engine=engine, broker=broker, threshold_pct=0.01)
    assert not report.has_drift


@pytest.mark.asyncio
async def test_crypto_surplus_on_broker_flagged(engine):
    broker = MagicMock()
    broker.get_balances = AsyncMock(return_value=[_balance("ETH", 2.0)])
    broker.get_cached_price = MagicMock(return_value=3500.0)

    report = await reconcile.reconcile_crypto(engine=engine, broker=broker)
    assert report.has_drift
    assert report.drifts[0].symbol == "ETH"
    assert report.drifts[0].db_quantity == 0.0
    assert report.drifts[0].notes is not None


@pytest.mark.asyncio
async def test_crypto_surplus_dust_below_5usd_ignored(engine):
    broker = MagicMock()
    broker.get_balances = AsyncMock(return_value=[_balance("ETH", 0.001)])
    broker.get_cached_price = MagicMock(return_value=3500.0)

    report = await reconcile.reconcile_crypto(engine=engine, broker=broker)
    assert not report.has_drift  # 0.001 * 3500 = $3.50, below $5 dust


@pytest.mark.asyncio
async def test_crypto_ignores_usdt_balance(engine):
    broker = MagicMock()
    broker.get_balances = AsyncMock(return_value=[_balance("USDT", 1000.0), _balance("BUSD", 50.0)])
    broker.get_cached_price = MagicMock(return_value=None)

    report = await reconcile.reconcile_crypto(engine=engine, broker=broker)
    assert not report.has_drift


# ── Stocks ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stocks_clean_when_match(engine):
    repo = Repository(engine)
    await repo.record_trade(symbol="AAPL", side="buy", quantity=10, status="filled")

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(return_value=[_stock_position("AAPL", 10)])

    report = await reconcile.reconcile_stocks(engine=engine, broker=broker)
    assert not report.has_drift


@pytest.mark.asyncio
async def test_stocks_excludes_closed_buys(engine):
    """A closed BUY is an exited position, not a holding. The SL/TP/trailing
    monitor flips the BUY row to status='closed' (no SELL row), so counting it
    would phantom a long-gone position as still held — the chronic ~100%
    stock-drift bug. Broker is flat; DB must agree (no drift)."""
    repo = Repository(engine)
    bid = await repo.record_trade(symbol="AAPL", side="buy", quantity=10, status="filled")
    await repo.close_trade(bid, exit_price=205.0, exit_reason="stop_loss")

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(return_value=[])  # flat — position exited

    report = await reconcile.reconcile_stocks(engine=engine, broker=broker)
    assert not report.has_drift


@pytest.mark.asyncio
async def test_stocks_ignores_sell_legs_counts_open_buys(engine):
    """The LLM-sell path writes a 'filled' SELL row AND closes the BUY(s), so a
    SELL row never represents a holding. Only the still-open BUY is counted."""
    repo = Repository(engine)
    # An exited LLM round-trip: a closed BUY plus its filled SELL leg.
    sold = await repo.record_trade(symbol="MSFT", side="buy", quantity=8, status="filled")
    await repo.close_trade(sold, exit_price=410.0, exit_reason="llm_sell")
    await repo.record_trade(symbol="MSFT", side="sell", quantity=8, status="filled")
    # A genuinely open position on the same symbol.
    await repo.record_trade(symbol="MSFT", side="buy", quantity=5, status="filled")

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(return_value=[_stock_position("MSFT", 5)])

    report = await reconcile.reconcile_stocks(engine=engine, broker=broker)
    assert not report.has_drift


@pytest.mark.asyncio
async def test_stocks_position_with_no_trade_row(engine):
    broker = MagicMock()
    broker.get_all_positions = AsyncMock(return_value=[_stock_position("AAPL", 5)])

    sink, notifier = _alert_sink()
    report = await reconcile.reconcile_stocks(engine=engine, broker=broker, alerts=sink)
    assert report.has_drift
    assert report.drifts[0].notes is not None
    notifier.notify_error.assert_awaited_once()


# ── Persistence helper ───────────────────────────────────────


@pytest.mark.asyncio
async def test_get_recent_logs_orders_desc(engine):
    repo = Repository(engine)
    await repo.record_crypto_trade(pair="BTCUSDT", side="buy", quantity=1.0, status="filled")
    await repo.record_crypto_trade(pair="ETHUSDT", side="buy", quantity=2.0, status="filled")

    broker = MagicMock()
    broker.get_balances = AsyncMock(return_value=[_balance("BTC", 0.5), _balance("ETH", 1.0)])
    broker.get_cached_price = MagicMock(return_value=10000.0)

    await reconcile.reconcile_crypto(engine=engine, broker=broker)

    logs = await reconcile.get_recent_logs(engine, limit=10)
    assert len(logs) == 2
    assert {row["symbol"] for row in logs} == {"BTC", "ETH"}
