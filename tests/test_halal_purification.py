"""Dividend purification ledger tests."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from halal_trader.halal.purification import (
    PurificationLedger,
    compute_purification,
)


def test_compute_basic():
    entry = compute_purification(symbol="aapl", dividend_usd=100.0, haram_revenue_pct=0.05)
    assert entry.symbol == "AAPL"  # uppercased
    assert entry.dividend_usd == Decimal("100.00")
    assert entry.haram_pct == Decimal("0.05")
    assert entry.purification_usd == Decimal("5.00")
    assert entry.is_outstanding is True


def test_compute_negative_dividend_clamped_to_zero():
    """Reversals should not create a *credit* — purification is one-way."""
    entry = compute_purification(symbol="X", dividend_usd=-50.0, haram_revenue_pct=0.05)
    assert entry.dividend_usd == Decimal("0.00")
    assert entry.purification_usd == Decimal("0.00")


def test_compute_haram_pct_clamped():
    entry = compute_purification(symbol="X", dividend_usd=100, haram_revenue_pct=2.0)
    assert entry.haram_pct == Decimal("1")  # clamped to 100%
    assert entry.purification_usd == Decimal("100.00")


def test_ledger_outstanding_total_sums_unpaid():
    ledger = PurificationLedger()
    ledger.record(compute_purification(symbol="A", dividend_usd=100, haram_revenue_pct=0.05))
    ledger.record(compute_purification(symbol="B", dividend_usd=200, haram_revenue_pct=0.10))
    assert ledger.outstanding_total() == Decimal("25.00")
    assert ledger.paid_total() == Decimal("0.00")


def test_mark_paid_moves_entry_to_paid_total():
    ledger = PurificationLedger()
    ledger.record(compute_purification(symbol="A", dividend_usd=100, haram_revenue_pct=0.05))
    ledger.record(compute_purification(symbol="B", dividend_usd=200, haram_revenue_pct=0.10))
    ledger.mark_paid(0, paid_at=datetime.now(UTC))
    assert ledger.outstanding_total() == Decimal("20.00")
    assert ledger.paid_total() == Decimal("5.00")


def test_mark_paid_invalid_index_raises():
    ledger = PurificationLedger()
    with pytest.raises(IndexError):
        ledger.mark_paid(0)


def test_paid_at_timestamp_is_recorded():
    ledger = PurificationLedger()
    ledger.record(compute_purification(symbol="A", dividend_usd=10, haram_revenue_pct=0.05))
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.mark_paid(0, paid_at=ts)
    assert ledger.entries[0].paid_at == ts
    assert ledger.entries[0].is_outstanding is False
