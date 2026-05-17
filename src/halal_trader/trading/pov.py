"""Percentage-of-Volume (POV) execution algorithm — Round-5 Wave 12.C.

Where TWAP slices on time + VWAP slices on a forecasted volume profile,
**POV** ties the parent order's pace to *realised* market volume during
execution: the operator targets, say, 10% of every minute's volume.
This minimises information leakage when intraday volume deviates from
forecast.

This module ships the **stateful POV engine**: a `POVState` carries
the parent quantity + target participation rate; calling
`tick(state, market_volume_in_period)` returns the next child order
size + the updated state.

Pinned semantics:

- **Closed-set Side ladder** — re-uses ``trading.twap.Side``.
- **Participation rate clipped to [0.001, 0.50]** — operators almost
  never want < 0.1% (too slow) or > 50% (regulator concern).
- **State updates are immutable** — `tick` returns ``(child, new_state)``.
- **Cumulative pace bounded** by `max_cumulative_qty = parent_quantity`.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass

from halal_trader.trading.twap import Side


@dataclass(frozen=True)
class POVPolicy:
    """Operator-tunable POV policy."""

    participation_rate: float = 0.10  # 10% of market volume
    min_child_quantity: float = 1.0
    max_child_quantity: float = 1_000_000.0

    def __post_init__(self) -> None:
        if not 0.001 <= self.participation_rate <= 0.50:
            raise ValueError("participation_rate must be in [0.001, 0.50]")
        if self.min_child_quantity <= 0:
            raise ValueError("min_child_quantity must be positive")
        if self.max_child_quantity <= self.min_child_quantity:
            raise ValueError("max_child_quantity must exceed min")


@dataclass(frozen=True)
class POVState:
    """Immutable snapshot of the POV engine's state."""

    parent_id: str
    symbol: str
    side: Side
    parent_quantity: float
    cumulative_filled: float
    cumulative_market_volume: float

    def __post_init__(self) -> None:
        if not self.parent_id or not self.parent_id.strip():
            raise ValueError("parent_id must be non-empty")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.parent_quantity <= 0:
            raise ValueError("parent_quantity must be positive")
        if self.cumulative_filled < 0:
            raise ValueError("cumulative_filled must be non-negative")
        if self.cumulative_market_volume < 0:
            raise ValueError("cumulative_market_volume must be non-negative")
        if self.cumulative_filled > self.parent_quantity + 1e-6:
            raise ValueError("cumulative_filled exceeds parent_quantity")

    def remaining(self) -> float:
        return max(0.0, self.parent_quantity - self.cumulative_filled)

    def is_complete(self) -> bool:
        return self.remaining() <= 1e-9


def initialise_pov(
    *,
    parent_id: str,
    symbol: str,
    side: Side,
    parent_quantity: float,
) -> POVState:
    return POVState(
        parent_id=parent_id,
        symbol=symbol,
        side=side,
        parent_quantity=parent_quantity,
        cumulative_filled=0.0,
        cumulative_market_volume=0.0,
    )


@dataclass(frozen=True)
class ChildOrder:
    """A POV child order to submit this period."""

    parent_id: str
    quantity: float
    side: Side

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")


def tick(
    state: POVState,
    market_volume: float,
    *,
    policy: POVPolicy | None = None,
) -> tuple[ChildOrder, POVState]:
    """Advance the POV engine by one period of market volume.

    Returns ``(child_order, new_state)``. If the participation rate
    yields a child quantity below ``min_child_quantity``, returns a
    zero-quantity child + state with cumulative volume updated.
    """
    if market_volume < 0:
        raise ValueError("market_volume must be non-negative")
    pol = policy if policy is not None else POVPolicy()

    new_cum_volume = state.cumulative_market_volume + market_volume
    target_cumulative = pol.participation_rate * new_cum_volume
    deficit = max(0.0, target_cumulative - state.cumulative_filled)
    deficit = min(deficit, state.remaining())

    if deficit < pol.min_child_quantity and deficit < state.remaining():
        child_qty = 0.0
    else:
        child_qty = min(deficit, pol.max_child_quantity)
        child_qty = min(child_qty, state.remaining())

    new_state = POVState(
        parent_id=state.parent_id,
        symbol=state.symbol,
        side=state.side,
        parent_quantity=state.parent_quantity,
        cumulative_filled=state.cumulative_filled + child_qty,
        cumulative_market_volume=new_cum_volume,
    )
    child = ChildOrder(parent_id=state.parent_id, quantity=child_qty, side=state.side)
    return child, new_state


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


def render_state(state: POVState) -> str:
    progress = state.cumulative_filled / state.parent_quantity if state.parent_quantity > 0 else 0
    return _scrub(
        f"📊 POV {state.parent_id} {state.symbol} {state.side.value}: "
        f"filled {state.cumulative_filled:.2f}/{state.parent_quantity:.2f} "
        f"({progress * 100:.1f}%), mkt vol seen={state.cumulative_market_volume:.0f}"
    )
