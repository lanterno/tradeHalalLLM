"""Halal secondary-market routing — Round-5 Wave 6.F.

When a holder of a halal-screened private position wants liquidity,
this module routes the resale through a price-time order book + a
halal-screen gate on both legs:

1. The **listing** is checked against the screen at posting time
   (issuer still halal, no haram revenue recent reclassification).
2. The **counterparty** is checked at match time — buyer must be on
   the platform's KYC-verified halal-investor list (the deployment
   layer supplies the predicate).
3. The **price** must fall within a `[lower_band_pct, upper_band_pct]`
   of the latest fair-value mark; well outside the band → reject as
   gharar (excessive uncertainty about reasonable price).

This module is the **order book + matcher + price-band gate**.

Pinned semantics:

- **Closed-set OrderSide** — SELL / BUY. (No SHORT — secondary trades
  are by definition long position transfers.)
- **Closed-set OrderStatus FSM** — OPEN → PARTIALLY_FILLED → FILLED,
  with CANCELLED as alternate terminal.
- **Price band defaults to ±15%** of the last fair-value mark; outside
  → reject.
- **Match priority** is price-time: best-priced orders first, ties
  broken by earliest `posted_at`.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    """Closed-set order side ladder."""

    SELL = "sell"
    BUY = "buy"


class OrderStatus(str, Enum):
    """Closed-set order status FSM."""

    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Order:
    """One open order on the secondary book."""

    order_id: str
    asset_id: str
    """E.g. private-company identifier."""
    side: OrderSide
    user_id: str
    quantity: float
    """Original quantity at posting."""
    filled_quantity: float
    limit_price: float
    posted_at: datetime
    status: OrderStatus = OrderStatus.OPEN

    def __post_init__(self) -> None:
        if not self.order_id or not self.order_id.strip():
            raise ValueError("order_id must be non-empty")
        if not self.asset_id or not self.asset_id.strip():
            raise ValueError("asset_id must be non-empty")
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.filled_quantity < 0:
            raise ValueError("filled_quantity must be ≥ 0")
        if self.filled_quantity > self.quantity + 1e-9:
            raise ValueError("filled_quantity cannot exceed quantity")
        if self.limit_price <= 0:
            raise ValueError("limit_price must be positive")
        # Status consistency.
        if self.status is OrderStatus.FILLED and abs(self.filled_quantity - self.quantity) > 1e-9:
            raise ValueError("FILLED status requires filled_quantity ≈ quantity")
        if self.status is OrderStatus.PARTIALLY_FILLED and (
            self.filled_quantity == 0 or self.filled_quantity >= self.quantity
        ):
            raise ValueError("PARTIALLY_FILLED requires 0 < filled_quantity < quantity")

    def remaining(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)


@dataclass(frozen=True)
class FairValueMark:
    """The latest fair-value mark for an asset."""

    asset_id: str
    price: float
    marked_at: datetime

    def __post_init__(self) -> None:
        if not self.asset_id or not self.asset_id.strip():
            raise ValueError("asset_id must be non-empty")
        if self.price <= 0:
            raise ValueError("price must be positive")


@dataclass(frozen=True)
class BookPolicy:
    """Operator-tunable secondary-market policy."""

    lower_band_pct: float = 0.85
    """Order limit price must be ≥ lower_band_pct × fair_value."""
    upper_band_pct: float = 1.15
    """Order limit price must be ≤ upper_band_pct × fair_value."""
    min_quantity: float = 0.01
    """Below this fractional quantity, the dust-gate rejects."""

    def __post_init__(self) -> None:
        if not 0.0 < self.lower_band_pct < 1.0:
            raise ValueError("lower_band_pct must be in (0, 1)")
        if not 1.0 < self.upper_band_pct <= 3.0:
            raise ValueError("upper_band_pct must be in (1, 3]")
        if self.min_quantity <= 0:
            raise ValueError("min_quantity must be positive")


class PriceBandViolation(ValueError):
    """Limit price falls outside the operator's price band."""


class CounterpartyHalalError(ValueError):
    """The proposed counterparty is not on the halal-verified list."""


def assert_price_in_band(limit_price: float, fair_value: float, *, policy: BookPolicy) -> None:
    """Raise if `limit_price` falls outside the band around `fair_value`."""
    if fair_value <= 0:
        raise ValueError("fair_value must be positive")
    lo = fair_value * policy.lower_band_pct
    hi = fair_value * policy.upper_band_pct
    if limit_price < lo - 1e-9 or limit_price > hi + 1e-9:
        raise PriceBandViolation(f"limit_price {limit_price:.2f} outside band [{lo:.2f}, {hi:.2f}]")


def post_order(
    *,
    order_id: str,
    asset_id: str,
    side: OrderSide,
    user_id: str,
    quantity: float,
    limit_price: float,
    posted_at: datetime,
    fair_value: FairValueMark,
    is_asset_halal: Callable[[str], bool],
    policy: BookPolicy | None = None,
) -> Order:
    """Validate + return a freshly-posted Order.

    Pinned: rejects if asset fails halal screen, or price outside
    band, or quantity below min.
    """
    pol = policy if policy is not None else BookPolicy()
    if fair_value.asset_id != asset_id:
        raise ValueError("fair_value asset mismatch")
    if not is_asset_halal(asset_id):
        raise ValueError(f"asset {asset_id} is not halal-compliant")
    if quantity < pol.min_quantity:
        raise ValueError(f"quantity {quantity} below min {pol.min_quantity}")
    assert_price_in_band(limit_price, fair_value.price, policy=pol)
    return Order(
        order_id=order_id,
        asset_id=asset_id,
        side=side,
        user_id=user_id,
        quantity=quantity,
        filled_quantity=0.0,
        limit_price=limit_price,
        posted_at=posted_at,
        status=OrderStatus.OPEN,
    )


@dataclass(frozen=True)
class Trade:
    """A matched trade between a buy and sell order."""

    trade_id: str
    asset_id: str
    buy_order_id: str
    sell_order_id: str
    buyer_id: str
    seller_id: str
    quantity: float
    price: float
    matched_at: datetime


def _is_crossable(buy: Order, sell: Order) -> bool:
    return buy.limit_price >= sell.limit_price - 1e-9


def match_book(
    orders: Sequence[Order],
    *,
    matched_at: datetime,
    is_counterparty_halal: Callable[[str], bool],
    trade_id_prefix: str = "T-",
) -> tuple[tuple[Trade, ...], tuple[Order, ...]]:
    """Run a single matching pass.

    Returns (trades, updated_orders) where `updated_orders` reflects
    new filled_quantity + status for each touched order.

    Pinned:
    - Match priority: best price first; ties → earliest posted_at.
    - Sells sorted ascending by price; buys descending by price.
    - Cross at the *resting* (older) order's price.
    - Same `asset_id` only — different assets do not cross.
    - Each match validates `is_counterparty_halal(buyer_id)` AND
      `is_counterparty_halal(seller_id)`.
    """
    # Group by asset.
    by_asset: dict[str, list[Order]] = {}
    for o in orders:
        if o.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            continue
        by_asset.setdefault(o.asset_id, []).append(o)
    trades: list[Trade] = []
    updated_map: dict[str, Order] = {o.order_id: o for o in orders}
    trade_seq = 0
    for asset_id, asset_orders in by_asset.items():
        sells = sorted(
            (o for o in asset_orders if o.side is OrderSide.SELL),
            key=lambda o: (o.limit_price, o.posted_at),
        )
        buys = sorted(
            (o for o in asset_orders if o.side is OrderSide.BUY),
            key=lambda o: (-o.limit_price, o.posted_at),
        )
        # Mutable copies.
        sell_remain = {o.order_id: o.remaining() for o in sells}
        buy_remain = {o.order_id: o.remaining() for o in buys}
        for buy in buys:
            if buy_remain[buy.order_id] <= 1e-12:
                continue
            for sell in sells:
                if sell_remain[sell.order_id] <= 1e-12:
                    continue
                if not _is_crossable(buy, sell):
                    continue
                if buy.user_id == sell.user_id:
                    continue  # no self-cross
                if not is_counterparty_halal(buy.user_id):
                    raise CounterpartyHalalError(f"buyer {buy.user_id} not halal-verified")
                if not is_counterparty_halal(sell.user_id):
                    raise CounterpartyHalalError(f"seller {sell.user_id} not halal-verified")
                # Match price: resting (older) order wins.
                if sell.posted_at <= buy.posted_at:
                    match_price = sell.limit_price
                else:
                    match_price = buy.limit_price
                fill_qty = min(buy_remain[buy.order_id], sell_remain[sell.order_id])
                if fill_qty <= 1e-12:
                    continue
                trade_seq += 1
                trades.append(
                    Trade(
                        trade_id=f"{trade_id_prefix}{trade_seq:06d}",
                        asset_id=asset_id,
                        buy_order_id=buy.order_id,
                        sell_order_id=sell.order_id,
                        buyer_id=buy.user_id,
                        seller_id=sell.user_id,
                        quantity=fill_qty,
                        price=match_price,
                        matched_at=matched_at,
                    )
                )
                buy_remain[buy.order_id] -= fill_qty
                sell_remain[sell.order_id] -= fill_qty
                # Update maps.
                new_buy_filled = buy.quantity - buy_remain[buy.order_id]
                new_sell_filled = sell.quantity - sell_remain[sell.order_id]
                updated_map[buy.order_id] = replace(
                    updated_map[buy.order_id],
                    filled_quantity=new_buy_filled,
                    status=_new_status(new_buy_filled, buy.quantity),
                )
                updated_map[sell.order_id] = replace(
                    updated_map[sell.order_id],
                    filled_quantity=new_sell_filled,
                    status=_new_status(new_sell_filled, sell.quantity),
                )
                if buy_remain[buy.order_id] <= 1e-12:
                    break
    return tuple(trades), tuple(updated_map.values())


def _new_status(filled: float, total: float) -> OrderStatus:
    if filled <= 1e-12:
        return OrderStatus.OPEN
    if filled >= total - 1e-9:
        return OrderStatus.FILLED
    return OrderStatus.PARTIALLY_FILLED


def cancel_order(order: Order) -> Order:
    """Cancel an order. FILLED and already-CANCELLED are terminal."""
    if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
        raise ValueError(f"cancel illegal from {order.status.value}")
    return replace(order, status=OrderStatus.CANCELLED)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


def render_order(order: Order) -> str:
    return (
        f"{'🟢' if order.side is OrderSide.BUY else '🔴'} "
        f"[{order.order_id}] {order.asset_id} "
        f"{order.side.value} {order.quantity:.4f} @ "
        f"${order.limit_price:.2f} "
        f"({order.status.value}, filled={order.filled_quantity:.4f}) "
        f"by {_mask(order.user_id)}"
    )


def render_trade(trade: Trade) -> str:
    return (
        f"🤝 [{trade.trade_id}] {trade.asset_id} "
        f"{trade.quantity:.4f} @ ${trade.price:.2f} "
        f"{_mask(trade.seller_id)} → {_mask(trade.buyer_id)} "
        f"at {trade.matched_at.isoformat()}"
    )
