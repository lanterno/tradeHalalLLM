"""The halal no-short invariant — one enforced gate.

Long-only is non-negotiable for this bot: a sell may at most take a position
flat, never below zero (which would open or deepen a short — forbidden, and on
a margin-enabled paper account an over-sized sell silently does exactly that).

Both executors enforced this independently (stock: clamp to held long qty,
refuse if none held; crypto: ``min(requested, free)``). Two copies of the most
safety-critical invariant invite drift. This is the single source of truth they
now share; callers supply their own notion of "available" (broker position
quantity for stocks, free base balance for crypto).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LongOnlySell:
    """Result of applying the no-short clamp to a requested sell."""

    quantity: float  # the safe sell quantity (0.0 when nothing is held)
    clamped: bool  # requested qty was reduced to the available holding
    blocked: bool  # nothing held/free → selling at all would open a short


def clamp_sell_to_long(requested: float, available: float) -> LongOnlySell:
    """Clamp a sell so it never exceeds the held/free quantity.

    ``requested`` — the quantity the strategy/monitor wants to sell.
    ``available`` — the held long quantity (stocks) or free base balance
    (crypto). Negative/NaN-ish availability is floored to 0.

    A long-only book can at most go flat: the returned quantity is
    ``min(max(requested, 0), max(available, 0))``. ``blocked`` is true when
    nothing is available (the caller should refuse the order rather than
    short); ``clamped`` is true when the request was reduced.
    """
    avail = available if available > 0 else 0.0
    req = requested if requested > 0 else 0.0
    qty = min(req, avail)
    return LongOnlySell(
        quantity=qty,
        clamped=qty < req,
        blocked=avail <= 0.0,
    )
