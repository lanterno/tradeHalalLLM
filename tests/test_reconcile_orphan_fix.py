"""Tests for the reconciler bug-fix wave:

* ``status='pending'`` Trade rows must not inflate the DB-side sum
  (this was the orphan-counting bug — yesterday's failed-validation
  AMZN x380 row was being treated as a 380-share phantom forever).
* A fresh fill (``filled_at`` within the settlement grace) must be
  flagged ``is_settling`` and never escalate to an alert / persist
  to the ``reconciliation_log`` table.
* ``fix_stocks_orphans`` resolves stale ``pending`` rows: marks
  ``rejected`` when no ``order_id`` was ever assigned, or adopts the
  broker's terminal status when one was.
* Dry-run mode never persists.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.core import reconcile
from halal_trader.db.models import Trade
from halal_trader.db.repository import Repository
from halal_trader.notifications.telegram import AlertSink, TelegramNotifier


def _stock_position(symbol: str, qty: float) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, qty=qty)


def _alert_sink() -> tuple[AlertSink, MagicMock]:
    notifier = MagicMock(spec=TelegramNotifier)
    notifier.enabled = True
    notifier.notify_error = AsyncMock()
    return AlertSink(notifier=notifier), notifier


@pytest.mark.asyncio
async def test_pending_rows_excluded_from_db_sum(engine):
    """A pending Trade row must NOT inflate the DB-side position sum.

    Regression test for the orphan-AMZN scenario: yesterday's
    place_stock_order had Pydantic validation errors → the executor
    persisted a Trade with status='pending', quantity=380,
    filled_quantity=0. The reconciler then logged 100% drift forever.
    """
    repo = Repository(engine)
    # Phantom pending row from a never-placed order
    await repo.record_trade(
        symbol="AMZN",
        side="buy",
        quantity=380,
        status="pending",
        filled_quantity=0,
    )
    # Real filled row on a different symbol
    await repo.record_trade(
        symbol="NVDA",
        side="buy",
        quantity=10,
        status="filled",
        filled_quantity=10,
    )

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(return_value=[_stock_position("NVDA", 10)])

    report = await reconcile.reconcile_stocks(engine=engine, broker=broker)
    # AMZN must not appear — the pending row contributes 0.
    drift_symbols = {d.symbol for d in report.drifts}
    assert "AMZN" not in drift_symbols
    assert not report.has_drift


@pytest.mark.asyncio
async def test_rejected_rows_excluded(engine):
    """Same filter applies to rejected / canceled / error rows."""
    repo = Repository(engine)
    await repo.record_trade(
        symbol="TSLA", side="buy", quantity=100, status="rejected", filled_quantity=0
    )
    await repo.record_trade(
        symbol="META", side="buy", quantity=50, status="canceled", filled_quantity=0
    )
    await repo.record_trade(
        symbol="GOOG", side="buy", quantity=25, status="error", filled_quantity=0
    )

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(return_value=[])

    report = await reconcile.reconcile_stocks(engine=engine, broker=broker)
    assert not report.has_drift
    assert {d.symbol for d in report.drifts} == set()


@pytest.mark.asyncio
async def test_fresh_fill_marked_settling_not_alerted(engine):
    """A filled row within the settlement grace must not trigger an alert."""
    repo = Repository(engine)
    now = datetime.now(UTC)
    await repo.record_trade(
        symbol="SHOP",
        side="buy",
        quantity=145,
        status="filled",
        filled_quantity=145,
        filled_at=now,  # just landed
    )

    broker = MagicMock()
    # Alpaca's REST cache hasn't propagated yet
    broker.get_all_positions = AsyncMock(return_value=[])

    sink, notifier = _alert_sink()
    report = await reconcile.reconcile_stocks(
        engine=engine,
        broker=broker,
        alerts=sink,
        settlement_grace=timedelta(seconds=60),
    )
    # Drift is recorded for diagnostics but flagged as settling.
    assert len(report.drifts) == 1
    drift = report.drifts[0]
    assert drift.symbol == "SHOP"
    assert drift.is_settling is True
    # has_drift filters out settling drifts so callers don't treat them as actionable.
    assert report.has_drift is False
    # No alert fired, no reconciliation_log row persisted.
    notifier.notify_error.assert_not_called()
    rows = await reconcile.get_recent_logs(engine)
    assert rows == []


@pytest.mark.asyncio
async def test_stale_fill_outside_grace_alerts_normally(engine):
    """A filled row older than the grace window must alert."""
    repo = Repository(engine)
    old = datetime.now(UTC) - timedelta(minutes=5)
    await repo.record_trade(
        symbol="SHOP",
        side="buy",
        quantity=145,
        status="filled",
        filled_quantity=145,
        filled_at=old,
    )

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(return_value=[])

    sink, notifier = _alert_sink()
    report = await reconcile.reconcile_stocks(
        engine=engine,
        broker=broker,
        alerts=sink,
        settlement_grace=timedelta(seconds=10),
    )
    assert report.has_drift is True
    assert report.drifts[0].is_settling is False
    notifier.notify_error.assert_awaited_once()


# ── fix_stocks_orphans ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_fix_orphans_no_order_id_marks_rejected(engine):
    """A stale pending row with empty order_id is marked rejected."""
    repo = Repository(engine)
    trade_id = await repo.record_trade(
        symbol="AMZN",
        side="buy",
        quantity=380,
        status="pending",
        filled_quantity=0,
        order_id=None,
    )
    # Make the row old enough to qualify
    async with AsyncSession(engine) as session:
        trade = await session.get(Trade, trade_id)
        assert trade is not None
        trade.timestamp = datetime.now(UTC) - timedelta(minutes=30)
        session.add(trade)
        await session.commit()

    report = await reconcile.fix_stocks_orphans(
        engine=engine, broker=None, min_age_minutes=5, dry_run=False
    )
    assert report.candidates == 1
    assert report.updated == 1
    assert report.fixes[0].new_status == "rejected"
    assert report.fixes[0].source == "no-order-id"

    # Verify it actually persisted
    async with AsyncSession(engine) as session:
        refreshed = await session.get(Trade, trade_id)
        assert refreshed is not None
        assert refreshed.status == "rejected"


@pytest.mark.asyncio
async def test_fix_orphans_dry_run_does_not_persist(engine):
    """Dry-run reports candidates but doesn't touch the DB."""
    repo = Repository(engine)
    trade_id = await repo.record_trade(
        symbol="AMZN",
        side="buy",
        quantity=380,
        status="pending",
        filled_quantity=0,
        order_id=None,
    )
    async with AsyncSession(engine) as session:
        trade = await session.get(Trade, trade_id)
        assert trade is not None
        trade.timestamp = datetime.now(UTC) - timedelta(minutes=30)
        session.add(trade)
        await session.commit()

    report = await reconcile.fix_stocks_orphans(
        engine=engine, broker=None, min_age_minutes=5, dry_run=True
    )
    assert report.candidates == 1
    assert report.updated == 0  # nothing persisted
    assert report.fixes[0].new_status == "rejected"

    async with AsyncSession(engine) as session:
        refreshed = await session.get(Trade, trade_id)
        assert refreshed is not None
        assert refreshed.status == "pending"  # still pending


@pytest.mark.asyncio
async def test_fix_orphans_too_recent_skipped(engine):
    """Rows younger than min_age_minutes are left alone (in-flight grace)."""
    repo = Repository(engine)
    await repo.record_trade(
        symbol="AMZN", side="buy", quantity=380, status="pending", filled_quantity=0
    )
    report = await reconcile.fix_stocks_orphans(
        engine=engine, broker=None, min_age_minutes=5, dry_run=False
    )
    assert report.candidates == 0


@pytest.mark.asyncio
async def test_fix_orphans_adopts_broker_filled_status(engine):
    """When the broker reports the order filled, adopt its truth."""
    repo = Repository(engine)
    trade_id = await repo.record_trade(
        symbol="NVDA",
        side="buy",
        quantity=67,
        status="pending",
        filled_quantity=0,
        order_id="abc-123",
    )
    async with AsyncSession(engine) as session:
        trade = await session.get(Trade, trade_id)
        assert trade is not None
        trade.timestamp = datetime.now(UTC) - timedelta(minutes=30)
        session.add(trade)
        await session.commit()

    broker = MagicMock()
    broker.get_order_by_id = AsyncMock(
        return_value={
            "id": "abc-123",
            "status": "filled",
            "filled_qty": "67",
            "filled_avg_price": "120.50",
        }
    )

    report = await reconcile.fix_stocks_orphans(
        engine=engine, broker=broker, min_age_minutes=5, dry_run=False
    )
    assert report.updated == 1
    assert report.fixes[0].new_status == "filled"
    assert report.fixes[0].source == "broker"

    async with AsyncSession(engine) as session:
        refreshed = await session.get(Trade, trade_id)
        assert refreshed is not None
        assert refreshed.status == "filled"
        assert refreshed.filled_quantity == 67.0
        assert refreshed.filled_price == 120.50


# ── Crypto reconciler status filter ─────────────────────────────


def _balance(asset: str, free: float, locked: float = 0.0):
    return SimpleNamespace(asset=asset, free=free, locked=locked)


@pytest.mark.asyncio
async def test_crypto_pending_buy_excluded_from_db_sum(engine):
    """Same orphan path as stocks: a pending crypto BUY must not count
    toward the DB-side balance for reconciliation."""
    repo = Repository(engine)
    # Phantom pending order (never settled)
    await repo.record_crypto_trade(
        pair="BTCUSDT", side="buy", quantity=1.0, status="pending", filled_quantity=0
    )
    # Real filled order
    await repo.record_crypto_trade(
        pair="BTCUSDT", side="buy", quantity=0.3, status="filled", filled_quantity=0.3
    )

    broker = MagicMock()
    broker.get_balances = AsyncMock(return_value=[_balance("BTC", 0.3)])
    broker.get_cached_price = MagicMock(return_value=70000.0)

    report = await reconcile.reconcile_crypto(engine=engine, broker=broker)
    # The pending row's 1.0 BTC must not appear — broker has 0.3, DB
    # should also see 0.3 from the filled row.
    assert not report.has_drift


@pytest.mark.asyncio
async def test_crypto_prefers_filled_over_requested_quantity(engine):
    """For partial fills, the reconciler sums the actual filled amount,
    not what the LLM originally requested."""
    repo = Repository(engine)
    await repo.record_crypto_trade(
        pair="ETHUSDT",
        side="buy",
        quantity=2.0,
        status="partially_filled",
        filled_quantity=0.5,
    )

    broker = MagicMock()
    broker.get_balances = AsyncMock(return_value=[_balance("ETH", 0.5)])
    broker.get_cached_price = MagicMock(return_value=3500.0)

    report = await reconcile.reconcile_crypto(engine=engine, broker=broker)
    # 0.5 filled vs 0.5 on exchange → clean. If it had used quantity (2.0)
    # vs broker 0.5 → drift would appear.
    assert not report.has_drift


@pytest.mark.asyncio
async def test_crypto_legacy_row_without_filled_quantity_still_counted(engine):
    """Backward-compat: a `status='filled'` row from before the
    fill-confirmer landed may not have `filled_quantity` set. The
    reconciler still trusts these by falling back to `quantity`."""
    repo = Repository(engine)
    # status='filled' but no filled_quantity — legacy shape
    await repo.record_crypto_trade(
        pair="BTCUSDT", side="buy", quantity=0.4, status="filled"
    )

    broker = MagicMock()
    broker.get_balances = AsyncMock(return_value=[_balance("BTC", 0.4)])
    broker.get_cached_price = MagicMock(return_value=70000.0)

    report = await reconcile.reconcile_crypto(engine=engine, broker=broker)
    assert not report.has_drift


@pytest.mark.asyncio
async def test_fix_orphans_broker_still_open_leaves_pending(engine):
    """When the broker reports 'new' / 'pending_new', don't touch the row."""
    repo = Repository(engine)
    trade_id = await repo.record_trade(
        symbol="QCOM", side="buy", quantity=49, status="pending", order_id="open-id"
    )
    async with AsyncSession(engine) as session:
        trade = await session.get(Trade, trade_id)
        assert trade is not None
        trade.timestamp = datetime.now(UTC) - timedelta(minutes=30)
        session.add(trade)
        await session.commit()

    broker = MagicMock()
    broker.get_order_by_id = AsyncMock(
        return_value={"id": "open-id", "status": "new", "filled_qty": "0"}
    )

    report = await reconcile.fix_stocks_orphans(
        engine=engine, broker=broker, min_age_minutes=5, dry_run=False
    )
    assert report.updated == 0
    # We still record the candidate so the operator sees we considered it.
    assert report.candidates == 1
    assert report.fixes[0].new_status == "pending"

    async with AsyncSession(engine) as session:
        refreshed = await session.get(Trade, trade_id)
        assert refreshed is not None
        assert refreshed.status == "pending"  # unchanged


# ── Reverse orphan: broker holds a position the DB never recorded ──


@pytest.mark.asyncio
async def test_fix_orphans_imports_broker_only_position(engine):
    """A position present on the broker with no DB row gets imported as
    a filled BUY at the broker's avg_entry_price, so it's tracked,
    risk-managed, and nets out in reconcile instead of showing as
    permanent broker-surplus drift."""
    broker = MagicMock()
    broker.get_order_by_id = AsyncMock(return_value={})
    broker.get_all_positions = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="AAPL", qty=12, avg_entry_price=195.0, current_price=198.0),
        ]
    )

    report = await reconcile.fix_stocks_orphans(
        engine=engine, broker=broker, min_age_minutes=5, dry_run=False
    )
    assert report.candidates == 1
    assert report.updated == 1
    fix = report.fixes[0]
    assert fix.source == "broker-import"
    assert fix.new_status == "filled"

    repo = Repository(engine)
    rows = await repo.get_recent_trades(limit=10)
    aapl = [r for r in rows if r["symbol"] == "AAPL"]
    assert len(aapl) == 1
    assert aapl[0]["side"] == "buy"
    assert aapl[0]["filled_price"] == 195.0
    assert aapl[0]["filled_quantity"] == 12.0


@pytest.mark.asyncio
async def test_fix_orphans_skips_broker_position_already_tracked(engine):
    """If the DB already nets long on a symbol the broker also holds,
    don't double-import it."""
    repo = Repository(engine)
    await repo.record_trade(
        symbol="AAPL",
        side="buy",
        quantity=12,
        status="filled",
        filled_quantity=12,
        filled_price=195.0,
        order_id="real-fill",
    )

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="AAPL", qty=12, avg_entry_price=195.0, current_price=198.0),
        ]
    )

    report = await reconcile.fix_stocks_orphans(
        engine=engine, broker=broker, min_age_minutes=5, dry_run=False
    )
    # Forward pass found nothing; reverse pass skipped AAPL (already tracked).
    assert all(f.source != "broker-import" for f in report.fixes)
    rows = await repo.get_recent_trades(limit=10)
    assert len([r for r in rows if r["symbol"] == "AAPL"]) == 1


@pytest.mark.asyncio
async def test_fix_orphans_broker_import_dry_run_does_not_persist(engine):
    """Dry-run reports the import candidate but writes no Trade row."""
    broker = MagicMock()
    broker.get_all_positions = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="TSLA", qty=5, avg_entry_price=240.0, current_price=242.0),
        ]
    )

    report = await reconcile.fix_stocks_orphans(
        engine=engine, broker=broker, min_age_minutes=5, dry_run=True
    )
    assert report.candidates == 1
    assert report.updated == 0
    repo = Repository(engine)
    rows = await repo.get_recent_trades(limit=10)
    assert [r for r in rows if r["symbol"] == "TSLA"] == []


@pytest.mark.asyncio
async def test_reconcile_db_to_broker_dry_run_proposes_no_write(engine):
    """Dry-run reports the per-symbol balancing adjustment without writing."""
    repo = Repository(engine)
    # DB net for TXN = -5 (oversold residue); broker holds 25.
    await repo.record_trade(
        symbol="TXN", side="buy", quantity=20, status="filled", filled_quantity=20
    )
    await repo.record_trade(
        symbol="TXN", side="sell", quantity=25, status="filled", filled_quantity=25
    )

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(
        return_value=[SimpleNamespace(symbol="TXN", qty=25, avg_entry_price=200.0)]
    )

    report = await reconcile.reconcile_db_to_broker(
        engine=engine, broker=broker, dry_run=True
    )
    assert report.applied_count == 0
    fix = next(f for f in report.fixes if f.symbol == "TXN")
    assert fix.db_net == -5.0
    assert fix.broker_qty == 25.0
    assert fix.delta == 30.0  # need +30 to reach broker truth
    assert fix.side == "buy"
    # nothing written
    rows = await repo.get_recent_trades(limit=20)
    assert all(r["order_id"] != "RECONCILE-ADJ" for r in rows)


@pytest.mark.asyncio
async def test_reconcile_db_to_broker_apply_balances_to_broker(engine):
    """--apply writes a closed, P&L-neutral balancing entry that brings
    the DB net to broker truth and leaves the reconciler clean."""
    repo = Repository(engine)
    await repo.record_trade(
        symbol="TXN", side="buy", quantity=20, status="filled", filled_quantity=20
    )
    await repo.record_trade(
        symbol="TXN", side="sell", quantity=25, status="filled", filled_quantity=25
    )

    broker = MagicMock()
    broker.get_all_positions = AsyncMock(
        return_value=[SimpleNamespace(symbol="TXN", qty=25, avg_entry_price=200.0)]
    )

    report = await reconcile.reconcile_db_to_broker(
        engine=engine, broker=broker, dry_run=False
    )
    assert report.applied_count == 1

    # The adjustment row is tagged, closed, and P&L-neutral.
    rows = await repo.get_recent_trades(limit=20)
    adj = [r for r in rows if r["order_id"] == "RECONCILE-ADJ"]
    assert len(adj) == 1
    assert adj[0]["entry_type"] == "reconcile_adjustment"
    assert adj[0]["side"] == "buy"
    assert adj[0]["filled_quantity"] == 30.0

    # Reconciler now reads clean (db net == broker).
    rpt2 = await reconcile.reconcile_stocks(engine=engine, broker=broker, threshold_pct=0.01)
    assert not rpt2.has_drift


@pytest.mark.asyncio
async def test_reconcile_db_to_broker_ignores_sub_threshold(engine):
    """Tiny fractional drift below threshold is left alone."""
    repo = Repository(engine)
    await repo.record_trade(
        symbol="AAPL", side="buy", quantity=10, status="filled", filled_quantity=10
    )
    broker = MagicMock()
    broker.get_all_positions = AsyncMock(
        return_value=[SimpleNamespace(symbol="AAPL", qty=10.3, avg_entry_price=190.0)]
    )
    report = await reconcile.reconcile_db_to_broker(
        engine=engine, broker=broker, threshold_shares=0.5, dry_run=False
    )
    assert report.fixes == []
    assert report.applied_count == 0


@pytest.mark.asyncio
async def test_fix_orphans_broker_import_skips_unpriceable(engine):
    """A broker position with no cost basis (avg_entry=0, current=0) is
    NOT imported with a $0 basis — that would poison P&L the same way the
    EOD $0 close did. It's reported skipped instead."""
    broker = MagicMock()
    broker.get_all_positions = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="GME", qty=3, avg_entry_price=0.0, current_price=0.0),
        ]
    )

    report = await reconcile.fix_stocks_orphans(
        engine=engine, broker=broker, min_age_minutes=5, dry_run=False
    )
    assert report.candidates == 1
    assert report.updated == 0
    assert report.fixes[0].source == "broker-import"
    assert report.fixes[0].new_status == "(skipped)"
