"""Smart order router (multi-venue) — Round-5 Wave 12.E.

When the same symbol trades on multiple venues with slightly
different bid/ask + commission profiles, the smart router picks the
venue (or splits the order across venues) to minimise expected fill
cost. This module ships the **routing decision engine** — given a
parent order and a snapshot of venue quotes, it returns a list of
venue-allocations.

Pinned semantics:

- **Closed-set RoutingMode ladder** (BEST_PRICE / SPLIT / VENUE_PINNED).
- **Best-price routing** picks the venue with the lowest effective
  price (price + per-share commission) and dispatches the full
  quantity there.
- **Split routing** sweeps the ladder of venues from cheapest to most
  expensive, taking up to ``available_quantity`` from each.
- **Venue-pinned routing** routes everything to the operator-pinned
  venue regardless of price (used when a venue has unique
  capabilities or the operator's compliance team requires it).
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from halal_trader.trading.twap import Side


class RoutingMode(str, Enum):
    BEST_PRICE = "best_price"
    SPLIT = "split"
    VENUE_PINNED = "venue_pinned"


@dataclass(frozen=True)
class VenueQuote:
    """A single venue's quote for the symbol."""

    venue: str
    symbol: str
    bid_price: float
    ask_price: float
    available_quantity: float
    per_share_commission: float = 0.0

    def __post_init__(self) -> None:
        if not self.venue or not self.venue.strip():
            raise ValueError("venue must be non-empty")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.bid_price < 0 or self.ask_price < 0:
            raise ValueError("prices must be non-negative")
        if self.bid_price > self.ask_price:
            raise ValueError("bid_price > ask_price (crossed quote)")
        if self.available_quantity < 0:
            raise ValueError("available_quantity must be non-negative")
        if self.per_share_commission < 0:
            raise ValueError("per_share_commission must be non-negative")

    def effective_price(self, side: Side) -> float:
        """Effective price including commission for the given side."""
        if side is Side.BUY:
            return self.ask_price + self.per_share_commission
        return self.bid_price - self.per_share_commission


@dataclass(frozen=True)
class RoutingPolicy:
    """Operator-tunable routing policy."""

    mode: RoutingMode = RoutingMode.BEST_PRICE
    pinned_venue: str | None = None

    def __post_init__(self) -> None:
        if self.mode is RoutingMode.VENUE_PINNED and not self.pinned_venue:
            raise ValueError("VENUE_PINNED mode requires pinned_venue")
        if self.pinned_venue is not None and not self.pinned_venue.strip():
            raise ValueError("pinned_venue, if set, must be non-empty")


@dataclass(frozen=True)
class RouterInputs:
    """Inputs for the router."""

    parent_id: str
    symbol: str
    side: Side
    parent_quantity: float
    venue_quotes: tuple[VenueQuote, ...]

    def __post_init__(self) -> None:
        if not self.parent_id or not self.parent_id.strip():
            raise ValueError("parent_id must be non-empty")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.parent_quantity <= 0:
            raise ValueError("parent_quantity must be positive")
        for q in self.venue_quotes:
            if q.symbol != self.symbol:
                raise ValueError(f"quote symbol {q.symbol} != input symbol {self.symbol}")


@dataclass(frozen=True)
class VenueAllocation:
    """A single venue's allocation in the routing decision."""

    venue: str
    quantity: float
    expected_price: float

    def __post_init__(self) -> None:
        if not self.venue:
            raise ValueError("venue must be non-empty")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.expected_price < 0:
            raise ValueError("expected_price must be non-negative")


@dataclass(frozen=True)
class RoutingDecision:
    """Result of running the router."""

    parent_id: str
    side: Side
    allocations: tuple[VenueAllocation, ...]
    unallocated_quantity: float

    def __post_init__(self) -> None:
        if self.unallocated_quantity < 0:
            raise ValueError("unallocated_quantity must be non-negative")

    def total_allocated(self) -> float:
        return sum(a.quantity for a in self.allocations)


def route(
    inputs: RouterInputs,
    *,
    policy: RoutingPolicy | None = None,
) -> RoutingDecision:
    """Run the smart router and return a routing decision."""
    pol = policy if policy is not None else RoutingPolicy()

    if pol.mode is RoutingMode.VENUE_PINNED:
        pinned = next((q for q in inputs.venue_quotes if q.venue == pol.pinned_venue), None)
        if pinned is None or pinned.available_quantity <= 0:
            return RoutingDecision(
                parent_id=inputs.parent_id,
                side=inputs.side,
                allocations=(),
                unallocated_quantity=inputs.parent_quantity,
            )
        take = min(pinned.available_quantity, inputs.parent_quantity)
        return RoutingDecision(
            parent_id=inputs.parent_id,
            side=inputs.side,
            allocations=(
                VenueAllocation(
                    venue=pinned.venue,
                    quantity=take,
                    expected_price=pinned.effective_price(inputs.side),
                ),
            ),
            unallocated_quantity=inputs.parent_quantity - take,
        )

    # Sort venues by effective price.
    if inputs.side is Side.BUY:
        sorted_quotes = sorted(inputs.venue_quotes, key=lambda q: q.effective_price(Side.BUY))
    else:
        sorted_quotes = sorted(inputs.venue_quotes, key=lambda q: -q.effective_price(Side.SELL))

    if pol.mode is RoutingMode.BEST_PRICE:
        # Find the first venue with available capacity.
        for q in sorted_quotes:
            if q.available_quantity > 0:
                take = min(q.available_quantity, inputs.parent_quantity)
                return RoutingDecision(
                    parent_id=inputs.parent_id,
                    side=inputs.side,
                    allocations=(
                        VenueAllocation(
                            venue=q.venue,
                            quantity=take,
                            expected_price=q.effective_price(inputs.side),
                        ),
                    ),
                    unallocated_quantity=inputs.parent_quantity - take,
                )
        return RoutingDecision(
            parent_id=inputs.parent_id,
            side=inputs.side,
            allocations=(),
            unallocated_quantity=inputs.parent_quantity,
        )

    # SPLIT
    remaining = inputs.parent_quantity
    allocations: list[VenueAllocation] = []
    for q in sorted_quotes:
        if remaining <= 0:
            break
        take = min(q.available_quantity, remaining)
        if take <= 0:
            continue
        allocations.append(
            VenueAllocation(
                venue=q.venue,
                quantity=take,
                expected_price=q.effective_price(inputs.side),
            )
        )
        remaining -= take

    return RoutingDecision(
        parent_id=inputs.parent_id,
        side=inputs.side,
        allocations=tuple(allocations),
        unallocated_quantity=remaining,
    )


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


def render_decision(decision: RoutingDecision) -> str:
    head = (
        f"Smart-router {decision.parent_id} {decision.side.value}: "
        f"{decision.total_allocated():.4f} allocated"
    )
    if decision.unallocated_quantity > 0:
        head = f"{head}, {decision.unallocated_quantity:.4f} UNALLOCATED"
    lines = [head]
    for a in decision.allocations:
        lines.append(f"  • {a.venue} {a.quantity:.4f} @ ${a.expected_price:.4f}")
    return _scrub("\n".join(lines))
