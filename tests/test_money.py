"""Decimal money helper tests — round-trip, precision, and edge cases."""

from decimal import Decimal

from halal_trader.domain.money import (
    notional,
    pnl,
    quantize_qty,
    quantize_usd,
    return_pct,
    to_decimal,
)


def test_to_decimal_preserves_float_string_form():
    # The float 0.1 cannot be represented exactly; routing through str
    # gives us the user-facing value, not the binary expansion.
    assert to_decimal(0.1) == Decimal("0.1")
    assert to_decimal("0.1") == Decimal("0.1")
    assert to_decimal(Decimal("0.1")) == Decimal("0.1")
    assert to_decimal(5) == Decimal("5")


def test_quantize_usd_uses_bankers_rounding():
    assert quantize_usd("1.005") == Decimal("1.00")  # half to even → 0
    assert quantize_usd("1.015") == Decimal("1.02")  # half to even → 2
    assert quantize_usd("1.234") == Decimal("1.23")


def test_quantize_qty_with_explicit_step():
    # Binance BTCUSDT step is 0.00001 today; verify we round to that grid.
    step = Decimal("0.00001")
    assert quantize_qty("0.123456789", step=step) == Decimal("0.12346")
    # Default 8 dp when step omitted.
    assert quantize_qty("0.123456789") == Decimal("0.12345679")


def test_notional_round_trip():
    assert notional("0.5", "10000") == Decimal("5000.00")
    assert notional(0.01, 67234.51) == Decimal("672.35")  # rounded to cents


def test_pnl_long_winning_trade():
    assert pnl(entry=100, exit_price=110, quantity="0.5") == Decimal("5.00")


def test_pnl_long_losing_trade():
    assert pnl(entry=100, exit_price=95, quantity=2) == Decimal("-10.00")


def test_pnl_negative_quantity_models_short_side():
    # (95 - 100) * -1 = 5 — short profits when price drops.
    assert pnl(entry=100, exit_price=95, quantity=-1) == Decimal("5.00")


def test_return_pct_zero_entry_returns_zero():
    assert return_pct(0, 100) == Decimal("0")


def test_return_pct_basic():
    assert return_pct(100, 110) == Decimal("0.1")
    assert return_pct(100, 90) == Decimal("-0.1")


def test_no_float_drift_across_aggregation():
    # The classic 0.1 + 0.2 trap — accumulating floats yields 0.30000…04.
    # Decimal preserves exactly. Run 1000 sums to magnify any drift.
    total = Decimal("0")
    for _ in range(1000):
        total += to_decimal(0.1)
    assert total == Decimal("100.0")
