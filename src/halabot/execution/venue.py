"""Venue port + value types + an in-memory fake (REARCHITECTURE L6).

The :class:`Venue` Protocol is the seam every broker adapter implements (Alpaca
for stocks, Binance dormant per fork 2). Cycle/execution code depends only on
this surface, so swapping venues never touches the executor. :class:`FakeVenue`
is a deterministic in-memory venue for tests and the dormant default — it never
talks to a network.

INV-2: a quote/order failure raises a typed :class:`VenueError`; it NEVER returns
a fabricated $0 fill or quote (the ``_eod_exit_price`` bug, generalized).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

Side = Literal["buy", "sell"]


class VenueError(RuntimeError):
    """A venue/transport failure. Carries an optional broker error code so the
    executor can classify rejections (-1013/-2010) vs breaker-tripping errors."""

    def __init__(self, message: str, *, code: int | None = None, rejection: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.rejection = rejection  # a clean rejection (bad qty/funds), not a glitch


@dataclass(frozen=True)
class Order:
    asset: str
    side: Side
    quantity: float  # shares/units (already feasibility-rounded)
    client_id: str  # idempotency key
    belief_version: int = 0


@dataclass(frozen=True)
class OrderResult:
    asset: str
    side: Side
    requested_qty: float
    filled_qty: float
    filled_price: float | None  # None until/unless filled
    status: Literal["filled", "partial", "submitted", "rejected"]
    order_id: str
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    detail: str = ""

    @property
    def is_filled(self) -> bool:
        return self.status in ("filled", "partial") and self.filled_qty > 0


@dataclass(frozen=True)
class Position:
    asset: str
    quantity: float  # signed (long > 0); broker truth
    avg_price: float


@dataclass(frozen=True)
class Quote:
    asset: str
    price: float
    ts: datetime


class Venue(Protocol):
    async def place(self, order: Order) -> OrderResult: ...
    async def positions(self) -> list[Position]: ...
    async def close(self, asset: str) -> OrderResult: ...
    async def snapshot(self, asset: str) -> Quote: ...


@dataclass
class FakeVenue:
    """Deterministic in-memory venue (tests + dormant default).

    Fills at the configured ``prices`` (mark price); maintains positions. A
    ``fail_assets`` set raises :class:`VenueError` for that asset (to exercise
    the breaker + skip-not-invent paths). No network, no real money."""

    prices: dict[str, float] = field(default_factory=dict)
    _positions: dict[str, Position] = field(default_factory=dict)
    fail_assets: set[str] = field(default_factory=set)
    clock_ts: datetime | None = None
    placed: list[Order] = field(default_factory=list)
    _by_client_id: dict[str, OrderResult] = field(default_factory=dict)

    def _now(self) -> datetime:
        if self.clock_ts is None:
            raise VenueError("FakeVenue requires clock_ts for deterministic timestamps")
        return self.clock_ts

    def _price(self, asset: str) -> float:
        if asset in self.fail_assets:
            raise VenueError(f"venue error for {asset}", code=-1)  # glitch → trips breaker
        px = self.prices.get(asset)
        if px is None:
            # Cold-start "not ready", not a glitch — never invent a price (INV-2),
            # and don't quarantine the symbol for it (rejection=True).
            raise VenueError(f"no quote for {asset}", rejection=True)
        return px

    async def snapshot(self, asset: str) -> Quote:
        return Quote(asset=asset, price=self._price(asset), ts=self._now())

    async def positions(self) -> list[Position]:
        return [p for p in self._positions.values() if abs(p.quantity) > 1e-12]

    async def place(self, order: Order) -> OrderResult:
        # Idempotency on client_id (the real adapters MUST honor this too): a
        # duplicate submission returns the prior result without re-placing, so a
        # retry/overlapping tick can't double-fill (audit finding #4).
        if order.client_id in self._by_client_id:
            return self._by_client_id[order.client_id]
        self.placed.append(order)
        px = self._price(order.asset)  # raises on fail_assets — no fabricated fill
        signed = order.quantity if order.side == "buy" else -order.quantity
        prev = self._positions.get(order.asset)
        if prev is None:
            new_qty = signed
            avg = px
        else:
            new_qty = prev.quantity + signed
            avg = px if (prev.quantity == 0) else prev.avg_price
        self._positions[order.asset] = Position(order.asset, new_qty, avg)
        result = OrderResult(
            asset=order.asset,
            side=order.side,
            requested_qty=order.quantity,
            filled_qty=order.quantity,
            filled_price=px,
            status="filled",
            order_id=f"fake-{order.client_id}",
            submitted_at=self._now(),
            filled_at=self._now(),
        )
        self._by_client_id[order.client_id] = result
        return result

    async def close(self, asset: str) -> OrderResult:
        pos = self._positions.get(asset)
        qty = abs(pos.quantity) if pos else 0.0
        px = self._price(asset)  # raises on fail — never a $0 synthetic close (INV-2)
        if pos is None or qty <= 0:
            return OrderResult(
                asset=asset,
                side="sell",
                requested_qty=0.0,
                filled_qty=0.0,
                filled_price=px,
                status="filled",
                order_id=f"fake-close-{asset}",
            )
        side: Side = "sell" if pos.quantity > 0 else "buy"
        self._positions[asset] = Position(asset, 0.0, pos.avg_price)
        return OrderResult(
            asset=asset,
            side=side,
            requested_qty=qty,
            filled_qty=qty,
            filled_price=px,
            status="filled",
            order_id=f"fake-close-{asset}",
            submitted_at=self._now(),
            filled_at=self._now(),
        )
