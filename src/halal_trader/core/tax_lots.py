"""Tax-lot tracker (HIFO / FIFO / LIFO) — Round-5 Wave 18.A.

When a position is partially closed, the tax-lot accounting determines
which slices of the open lot pool are deemed sold. The choice of
method has real consequences:

- **FIFO** (First-In-First-Out) — earliest lots sold first. Default in
  most jurisdictions; long-term-gain treatment kicks in earliest.
- **LIFO** (Last-In-First-Out) — most-recent lots sold first. Useful
  for tax-loss harvesting in declining markets.
- **HIFO** (Highest-In-First-Out) — highest-cost lots sold first.
  Optimal for minimising current-year capital gains.

This module ships the **pure-Python lot-pool accounting engine**.
Persistence + per-jurisdiction reporting (Wave 18.B–F) live above.

Pinned semantics:

- **Closed-set Method ladder** (FIFO / LIFO / HIFO).
- **Lots are immutable**; selling a partial quantity returns a new
  pool with the lot quantity decremented.
- **Round-trip realisation** preserves total quantity (within
  floating-point tolerance).
- **`apply_sale` returns ``(realised_lots, remaining_pool)``** —
  caller owns the rebuilt pool. No global state.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum


class LotMethod(str, Enum):
    """Closed-set tax-lot selection methods."""

    FIFO = "fifo"
    LIFO = "lifo"
    HIFO = "hifo"


@dataclass(frozen=True)
class TaxLot:
    """A single open tax lot."""

    lot_id: str
    symbol: str
    quantity: float
    cost_basis_per_share: float
    acquisition_date: date

    def __post_init__(self) -> None:
        if not self.lot_id or not self.lot_id.strip():
            raise ValueError("lot_id must be non-empty")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.cost_basis_per_share < 0:
            raise ValueError("cost_basis_per_share must be non-negative")


@dataclass(frozen=True)
class RealisedSlice:
    """A slice of a lot deemed sold under the tax-lot method."""

    lot_id: str
    quantity: float
    cost_basis_per_share: float
    proceeds_per_share: float
    acquisition_date: date
    sale_date: date

    @property
    def realised_pnl(self) -> float:
        return (self.proceeds_per_share - self.cost_basis_per_share) * self.quantity

    @property
    def is_long_term(self) -> bool:
        """US convention: held > 1 year → long-term capital gain."""
        return (self.sale_date - self.acquisition_date).days > 365


def _ordered_for_method(lots: Iterable[TaxLot], method: LotMethod) -> list[TaxLot]:
    """Order lots by the selection priority of the chosen method."""
    if method is LotMethod.FIFO:
        return sorted(lots, key=lambda l: (l.acquisition_date, l.lot_id))
    if method is LotMethod.LIFO:
        return sorted(lots, key=lambda l: (l.acquisition_date, l.lot_id), reverse=True)
    if method is LotMethod.HIFO:
        return sorted(lots, key=lambda l: (-l.cost_basis_per_share, l.acquisition_date, l.lot_id))
    raise AssertionError("unreachable")  # pragma: no cover


def apply_sale(
    pool: tuple[TaxLot, ...],
    *,
    quantity: float,
    proceeds_per_share: float,
    sale_date: date,
    method: LotMethod = LotMethod.FIFO,
) -> tuple[tuple[RealisedSlice, ...], tuple[TaxLot, ...]]:
    """Apply a sale to the pool. Returns (realised slices, remaining pool)."""
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if proceeds_per_share < 0:
        raise ValueError("proceeds_per_share must be non-negative")
    total_open = sum(l.quantity for l in pool)
    if quantity > total_open + 1e-9:
        raise ValueError(f"sale quantity {quantity} exceeds open pool quantity {total_open}")

    ordered = _ordered_for_method(pool, method)
    remaining = quantity
    realised: list[RealisedSlice] = []
    new_lots: list[TaxLot] = []

    for lot in ordered:
        if remaining <= 1e-9:
            new_lots.append(lot)
            continue
        take = min(lot.quantity, remaining)
        if take > 0:
            realised.append(
                RealisedSlice(
                    lot_id=lot.lot_id,
                    quantity=take,
                    cost_basis_per_share=lot.cost_basis_per_share,
                    proceeds_per_share=proceeds_per_share,
                    acquisition_date=lot.acquisition_date,
                    sale_date=sale_date,
                )
            )
            remaining -= take
        leftover = lot.quantity - take
        if leftover > 1e-9:
            new_lots.append(
                TaxLot(
                    lot_id=lot.lot_id,
                    symbol=lot.symbol,
                    quantity=leftover,
                    cost_basis_per_share=lot.cost_basis_per_share,
                    acquisition_date=lot.acquisition_date,
                )
            )

    # Restore original order by acquisition_date for the returned pool.
    new_lots.sort(key=lambda l: (l.acquisition_date, l.lot_id))
    return tuple(realised), tuple(new_lots)


def total_quantity(pool: Iterable[TaxLot]) -> float:
    return sum(l.quantity for l in pool)


def total_cost_basis(pool: Iterable[TaxLot]) -> float:
    return sum(l.quantity * l.cost_basis_per_share for l in pool)


def total_realised_pnl(slices: Iterable[RealisedSlice]) -> float:
    return sum(s.realised_pnl for s in slices)


def split_long_short(
    slices: Iterable[RealisedSlice],
) -> tuple[tuple[RealisedSlice, ...], tuple[RealisedSlice, ...]]:
    """Return (long_term, short_term) tuples — US convention."""
    longs: list[RealisedSlice] = []
    shorts: list[RealisedSlice] = []
    for s in slices:
        (longs if s.is_long_term else shorts).append(s)
    return tuple(longs), tuple(shorts)


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
    "SSN",
    "TaxID",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_pool(pool: tuple[TaxLot, ...]) -> str:
    if not pool:
        return "Tax-lot pool: empty"
    lines = [
        f"Tax-lot pool: {len(pool)} lots, "
        f"qty={total_quantity(pool):.4f}, "
        f"basis=${total_cost_basis(pool):.2f}"
    ]
    for lot in pool:
        lines.append(
            f"  • {lot.lot_id} {lot.symbol} {lot.quantity:.4f}@"
            f"${lot.cost_basis_per_share:.2f} acq {lot.acquisition_date.isoformat()}"
        )
    return _scrub("\n".join(lines))


def render_realisation(slices: tuple[RealisedSlice, ...]) -> str:
    if not slices:
        return "Realisation: empty"
    longs, shorts = split_long_short(slices)
    lines = [
        f"Realisation: {len(slices)} slices, "
        f"net=${total_realised_pnl(slices):.2f} "
        f"(LT={len(longs)} ST={len(shorts)})"
    ]
    for s in slices:
        marker = "LT" if s.is_long_term else "ST"
        lines.append(
            f"  • {marker} {s.lot_id} {s.quantity:.4f}@"
            f"${s.cost_basis_per_share:.2f}→${s.proceeds_per_share:.2f} "
            f"= ${s.realised_pnl:+.2f}"
        )
    return _scrub("\n".join(lines))
