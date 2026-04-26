"""Decimal money helpers — single source of truth for currency/quantity math.

Floats accumulate rounding error across hundreds of micro-trades and make
fill reconciliation drift over time. Anything that crosses a money boundary
(P&L aggregation, fill notional, fee math) should pass through these helpers
so we round once, at a known precision, in a consistent direction.

We deliberately keep the API tiny — most code can stay in float and only
adopt Decimal at the boundary where it stores or compares money.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any

# Quote currencies (USD, USDT) round to the cent. Crypto base quantities
# round to 8 dp which covers BTC's smallest unit and is finer than every
# Binance lotSize we currently trade. Override at call sites for venues
# whose tick is coarser.
USD_QUANT = Decimal("0.01")
QTY_QUANT = Decimal("0.00000001")


def to_decimal(value: Any) -> Decimal:
    """Coerce float/int/str/Decimal to Decimal without float precision loss.

    Floats are routed through ``str()`` so 0.1 stays 0.1, not 0.1000…0055.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def quantize_usd(value: Any) -> Decimal:
    """Round to 0.01 using banker's rounding (matches most exchanges' fee math)."""
    return to_decimal(value).quantize(USD_QUANT, rounding=ROUND_HALF_EVEN)


def quantize_qty(value: Any, *, step: Decimal | None = None) -> Decimal:
    """Round a quantity to ``step`` (or 8 dp if step is omitted).

    Pass ``step`` from the exchange filter (Binance ``LOT_SIZE.stepSize``) to
    match venue-specific precision exactly — important so we never submit a
    sub-step quantity that the exchange will silently truncate.
    """
    quant = step if step is not None else QTY_QUANT
    return to_decimal(value).quantize(quant, rounding=ROUND_HALF_EVEN)


def notional(quantity: Any, price: Any) -> Decimal:
    """Quantity × price rounded to USD cents."""
    return quantize_usd(to_decimal(quantity) * to_decimal(price))


def pnl(entry: Any, exit_price: Any, quantity: Any) -> Decimal:
    """Realized P&L in quote currency for a long round-trip.

    For shorts, pass ``quantity`` as a negative Decimal — the formula
    ``(exit - entry) * qty`` handles direction naturally.
    """
    return quantize_usd((to_decimal(exit_price) - to_decimal(entry)) * to_decimal(quantity))


def return_pct(entry: Any, exit_price: Any) -> Decimal:
    """Return percentage as a Decimal in the range e.g. 0.0123 for 1.23%.

    Returns Decimal('0') if entry is zero (defensive — caller should not
    be computing returns on a zero-cost basis but we don't want to crash
    aggregation pipelines).
    """
    e = to_decimal(entry)
    if e == 0:
        return Decimal("0")
    return (to_decimal(exit_price) - e) / e
