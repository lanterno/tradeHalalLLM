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


def _stock_position(symbol: str, qty: float) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, qty=qty)


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
async def test_stocks_signed_aggregation(engine):
    repo = Repository(engine)
    await repo.record_trade(symbol="AAPL", side="buy", quantity=10, status="filled")
    await repo.record_trade(symbol="AAPL", side="sell", quantity=4, status="filled")

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(return_value=[_stock_position("AAPL", 6)])

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


@pytest.mark.asyncio
async def test_get_recent_logs_orders_desc(engine):
    from datetime import timedelta

    repo = Repository(engine)
    await repo.record_trade(symbol="AAPL", side="buy", quantity=10, status="filled")
    await repo.record_trade(symbol="MSFT", side="buy", quantity=20, status="filled")

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(
        return_value=[_stock_position("AAPL", 5), _stock_position("MSFT", 10)]
    )

    # No settlement grace: both fills are seconds old, and the drift must
    # be persisted rather than treated as a fresh-fill race.
    await reconcile.reconcile_stocks(engine=engine, broker=broker, settlement_grace=timedelta(0))

    logs = await reconcile.get_recent_logs(engine, limit=10)
    assert len(logs) == 2
    assert {row["symbol"] for row in logs} == {"AAPL", "MSFT"}
