"""Iceberg-order slicer — Round-5 Wave 12.D.

An iceberg order shows only a "tip" (a small visible quantity) at any
moment; the remaining hidden quantity replenishes as fills come in.
Compared to TWAP / VWAP this minimises information leakage by
keeping the parent's full size off the visible book.

This module ships the **iceberg state machine + slicer**: given a
parent quantity + tip size + replenishment policy, it returns the
sequence of visible orders and tracks state (visible / hidden /
filled).

Pinned semantics:

- **Closed-set ReplenishStrategy ladder.** ON_FILL (replenish only when
  the visible tip fills) / TIME_BASED (replenish every N seconds
  regardless).
- **Visible tip never exceeds parent quantity.**
- **`fill_visible` returns a new state** (immutable updates).
- **Closed-set Side ladder** (re-uses ``trading.twap.Side``).
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

from halal_trader.trading.twap import Side


class ReplenishStrategy(str, Enum):
    """Closed-set replenish strategies."""

    ON_FILL = "on_fill"
    TIME_BASED = "time_based"


@dataclass(frozen=True)
class IcebergPolicy:
    """Operator-tunable iceberg policy."""

    max_visible_pct: float = 0.10  # tip ≤ 10% of parent by default
    min_tip_quantity: float = 1.0
    replenish_strategy: ReplenishStrategy = ReplenishStrategy.ON_FILL
    time_based_interval: timedelta = timedelta(seconds=30)

    def __post_init__(self) -> None:
        if not 0.0 < self.max_visible_pct <= 1.0:
            raise ValueError("max_visible_pct must be in (0, 1]")
        if self.min_tip_quantity <= 0:
            raise ValueError("min_tip_quantity must be positive")
        if self.time_based_interval <= timedelta(0):
            raise ValueError("time_based_interval must be positive")


@dataclass(frozen=True)
class IcebergState:
    """Current state of an iceberg parent order."""

    parent_id: str
    symbol: str
    side: Side
    parent_quantity: float
    visible_quantity: float
    hidden_quantity: float
    filled_quantity: float

    def __post_init__(self) -> None:
        if not self.parent_id or not self.parent_id.strip():
            raise ValueError("parent_id must be non-empty")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.parent_quantity <= 0:
            raise ValueError("parent_quantity must be positive")
        for name, val in (
            ("visible_quantity", self.visible_quantity),
            ("hidden_quantity", self.hidden_quantity),
            ("filled_quantity", self.filled_quantity),
        ):
            if val < 0:
                raise ValueError(f"{name} must be non-negative")
        total = self.visible_quantity + self.hidden_quantity + self.filled_quantity
        if abs(total - self.parent_quantity) > 1e-6:
            raise ValueError(
                "visible + hidden + filled must equal parent_quantity (got "
                f"{total} vs {self.parent_quantity})"
            )

    def is_complete(self) -> bool:
        return self.filled_quantity >= self.parent_quantity - 1e-6


def initialise_iceberg(
    *,
    parent_id: str,
    symbol: str,
    side: Side,
    parent_quantity: float,
    policy: IcebergPolicy | None = None,
) -> IcebergState:
    """Initialise an iceberg state with the first tip visible."""
    pol = policy if policy is not None else IcebergPolicy()
    if parent_quantity <= 0:
        raise ValueError("parent_quantity must be positive")
    tip = max(pol.min_tip_quantity, parent_quantity * pol.max_visible_pct)
    tip = min(tip, parent_quantity)
    return IcebergState(
        parent_id=parent_id,
        symbol=symbol,
        side=side,
        parent_quantity=parent_quantity,
        visible_quantity=tip,
        hidden_quantity=parent_quantity - tip,
        filled_quantity=0.0,
    )


def fill_visible(
    state: IcebergState,
    fill_quantity: float,
    *,
    policy: IcebergPolicy | None = None,
) -> IcebergState:
    """Apply a fill to the visible tip; replenish from hidden if needed.

    Returns a new state.
    """
    pol = policy if policy is not None else IcebergPolicy()
    if fill_quantity <= 0:
        raise ValueError("fill_quantity must be positive")
    if fill_quantity > state.visible_quantity + 1e-6:
        raise ValueError(
            f"fill_quantity {fill_quantity} exceeds visible {state.visible_quantity}"
        )

    new_filled = state.filled_quantity + fill_quantity
    remaining_visible = state.visible_quantity - fill_quantity

    if pol.replenish_strategy is ReplenishStrategy.ON_FILL:
        # Replenish if the visible tip is exhausted.
        if remaining_visible < 1e-9 and state.hidden_quantity > 0:
            tip = min(
                state.hidden_quantity,
                max(pol.min_tip_quantity, state.parent_quantity * pol.max_visible_pct),
            )
            new_visible = tip
            new_hidden = state.hidden_quantity - tip
        else:
            new_visible = remaining_visible
            new_hidden = state.hidden_quantity
    else:
        # TIME_BASED replenishment is driven externally; the fill itself doesn't
        # trigger a top-up. Caller invokes ``replenish_time`` separately.
        new_visible = remaining_visible
        new_hidden = state.hidden_quantity

    return IcebergState(
        parent_id=state.parent_id,
        symbol=state.symbol,
        side=state.side,
        parent_quantity=state.parent_quantity,
        visible_quantity=new_visible,
        hidden_quantity=new_hidden,
        filled_quantity=new_filled,
    )


def replenish_time(
    state: IcebergState, *, policy: IcebergPolicy | None = None
) -> IcebergState:
    """Top up the visible tip from hidden — for TIME_BASED replenish strategy."""
    pol = policy if policy is not None else IcebergPolicy()
    if state.hidden_quantity <= 0:
        return state
    target_tip = max(pol.min_tip_quantity, state.parent_quantity * pol.max_visible_pct)
    deficit = target_tip - state.visible_quantity
    if deficit <= 0:
        return state
    add = min(deficit, state.hidden_quantity)
    return IcebergState(
        parent_id=state.parent_id,
        symbol=state.symbol,
        side=state.side,
        parent_quantity=state.parent_quantity,
        visible_quantity=state.visible_quantity + add,
        hidden_quantity=state.hidden_quantity - add,
        filled_quantity=state.filled_quantity,
    )


def render_state(state: IcebergState) -> str:
    return (
        f"🧊 Iceberg {state.parent_id} {state.symbol} {state.side.value}: "
        f"visible={state.visible_quantity:.4f} "
        f"hidden={state.hidden_quantity:.4f} "
        f"filled={state.filled_quantity:.4f} "
        f"/ parent={state.parent_quantity:.4f}"
    )
