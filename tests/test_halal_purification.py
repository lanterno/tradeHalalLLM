"""Dividend purification calculator tests."""

from __future__ import annotations

from decimal import Decimal

from halal_trader.halal.purification import compute_purification


def test_compute_basic():
    entry = compute_purification(symbol="aapl", dividend_usd=100.0, haram_revenue_pct=0.05)
    assert entry.symbol == "AAPL"  # uppercased
    assert entry.dividend_usd == Decimal("100.00")
    assert entry.haram_pct == Decimal("0.05")
    assert entry.purification_usd == Decimal("5.00")


def test_compute_negative_dividend_clamped_to_zero():
    """Reversals should not create a *credit* — purification is one-way."""
    entry = compute_purification(symbol="X", dividend_usd=-50.0, haram_revenue_pct=0.05)
    assert entry.dividend_usd == Decimal("0.00")
    assert entry.purification_usd == Decimal("0.00")


def test_compute_haram_pct_clamped():
    entry = compute_purification(symbol="X", dividend_usd=100, haram_revenue_pct=2.0)
    assert entry.haram_pct == Decimal("1")  # clamped to 100%
    assert entry.purification_usd == Decimal("100.00")
