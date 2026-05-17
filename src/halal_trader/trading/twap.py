"""Time-Weighted Average Price (TWAP) execution algorithm — Round-5 Wave 12.A.

When the bot wants to buy / sell a quantity that's large relative to
recent average volume, slamming the whole order into a single market
trade leaks information + pays slippage. TWAP slices the order into
N equal-size child orders distributed evenly across a time window —
the simplest execution algorithm + the baseline every other algo
beats.

This module ships the **slicer**: given parent quantity, time window,
and slice count, it returns a deterministic schedule of child orders
the cycle / monitor consumes. Broker dispatch + fill reconciliation
live one layer up.

Pinned semantics:

- **Equal-size slices.** ``parent_quantity / n_slices`` per child,
  with the remainder added to the *first* slice (so cumulative
  quantity is exact at every checkpoint).
- **Equal-time intervals.** First slice fires at ``t0``, last at
  ``t0 + (n-1) * interval``. The window's end-time is the time of
  the last child's submission, not the time of last fill.
- **Closed-set Side ladder** (BUY / SELL).
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class TwapPolicy:
    """Operator-tunable TWAP policy."""

    min_slice_quantity: float = 0.0
    max_slices: int = 1000

    def __post_init__(self) -> None:
        if self.min_slice_quantity < 0:
            raise ValueError("min_slice_quantity must be non-negative")
        if self.max_slices <= 0:
            raise ValueError("max_slices must be positive")


@dataclass(frozen=True)
class TwapInputs:
    """Inputs for a TWAP slice."""

    parent_id: str
    symbol: str
    side: Side
    parent_quantity: float
    start_time: datetime
    end_time: datetime
    n_slices: int

    def __post_init__(self) -> None:
        if not self.parent_id or not self.parent_id.strip():
            raise ValueError("parent_id must be non-empty")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.parent_quantity <= 0:
            raise ValueError("parent_quantity must be positive")
        if self.start_time.tzinfo is None or self.end_time.tzinfo is None:
            raise ValueError("start_time + end_time must be timezone-aware")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        if self.n_slices <= 0:
            raise ValueError("n_slices must be positive")


@dataclass(frozen=True)
class ChildOrder:
    """A single child order in the TWAP schedule."""

    parent_id: str
    slice_index: int
    symbol: str
    side: Side
    quantity: float
    submit_time: datetime

    def __post_init__(self) -> None:
        if self.slice_index < 0:
            raise ValueError("slice_index must be non-negative")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.submit_time.tzinfo is None:
            raise ValueError("submit_time must be timezone-aware")


def slice_twap(
    inputs: TwapInputs, *, policy: TwapPolicy | None = None
) -> tuple[ChildOrder, ...]:
    """Slice a parent order into a TWAP schedule of child orders."""
    pol = policy if policy is not None else TwapPolicy()
    if inputs.n_slices > pol.max_slices:
        raise ValueError(f"n_slices {inputs.n_slices} exceeds max_slices {pol.max_slices}")

    base = inputs.parent_quantity / inputs.n_slices
    if base < pol.min_slice_quantity:
        raise ValueError(
            f"slice size {base:.6f} below min_slice_quantity {pol.min_slice_quantity:.6f}"
        )

    # Distribute remainder into the first slice for cumulative-exact arithmetic.
    quantities = [base] * inputs.n_slices
    total = base * inputs.n_slices
    remainder = inputs.parent_quantity - total
    quantities[0] += remainder

    if inputs.n_slices == 1:
        interval = timedelta(0)
    else:
        interval = (inputs.end_time - inputs.start_time) / (inputs.n_slices - 1)

    children = tuple(
        ChildOrder(
            parent_id=inputs.parent_id,
            slice_index=i,
            symbol=inputs.symbol,
            side=inputs.side,
            quantity=quantities[i],
            submit_time=inputs.start_time + interval * i,
        )
        for i in range(inputs.n_slices)
    )
    return children


def cumulative_quantity(children: Iterable[ChildOrder]) -> float:
    return sum(c.quantity for c in children)


def filter_due(children: Iterable[ChildOrder], *, now: datetime) -> tuple[ChildOrder, ...]:
    """Return the children whose ``submit_time`` is <= ``now``."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return tuple(c for c in children if c.submit_time <= now)


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_schedule(children: tuple[ChildOrder, ...]) -> str:
    if not children:
        return "TWAP schedule: empty"
    head = (
        f"TWAP {children[0].parent_id} {children[0].symbol} {children[0].side.value}: "
        f"{len(children)} slices, total={cumulative_quantity(children):.4f}"
    )
    lines = [head]
    for c in children:
        lines.append(
            f"  • slice {c.slice_index}: {c.quantity:.4f} @ "
            f"{c.submit_time.isoformat()}"
        )
    return _scrub("\n".join(lines))
