"""Tax-loss harvesting selector — Round-5 Wave 18.G.

Tax-loss harvesting (TLH) systematically realises capital losses to
offset capital gains, reducing current-year tax liability while
maintaining market exposure (subject to wash-sale rules in the US).

This module ships the **selector**: given the current tax-lot pool +
current market prices + a wash-sale window, it ranks lots by
harvest-priority and returns the recommended sales.

Pinned semantics:

- **Wash-sale window pin** (US default 30 days). Any lot acquired
  within ``wash_sale_window_days`` of today's date in the same symbol
  is in the "wash zone" and is excluded.
- **Minimum loss threshold** filters small losses that don't justify
  trading costs.
- **Closed-set HarvestRank ladder** — losses sorted by absolute size
  (largest losses harvested first).
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from halal_trader.core.tax_lots import TaxLot


@dataclass(frozen=True)
class HarvestPolicy:
    """Operator-tunable harvest policy."""

    wash_sale_window_days: int = 30
    min_loss_amount: float = 50.0  # in account currency
    min_loss_pct: float = 0.02  # 2% min loss to be worth harvesting

    def __post_init__(self) -> None:
        if self.wash_sale_window_days < 0:
            raise ValueError("wash_sale_window_days must be non-negative")
        if self.min_loss_amount < 0:
            raise ValueError("min_loss_amount must be non-negative")
        if not 0.0 <= self.min_loss_pct < 1.0:
            raise ValueError("min_loss_pct must be in [0, 1)")


@dataclass(frozen=True)
class HarvestCandidate:
    """A lot identified as a harvest candidate."""

    lot: TaxLot
    market_price: float
    unrealised_loss: float  # positive number = loss
    loss_pct: float

    def __post_init__(self) -> None:
        if self.market_price < 0:
            raise ValueError("market_price must be non-negative")
        if self.unrealised_loss < 0:
            raise ValueError("unrealised_loss must be non-negative (positive = loss)")


def _is_in_wash_zone(
    lot: TaxLot,
    *,
    today: date,
    window_days: int,
) -> bool:
    """A lot acquired within ``window_days`` of today is in the wash zone.

    Note: the *full* wash-sale rule applies to "substantially identical"
    securities purchased within ±30 days of the sale, including
    pending future buys. This module conservatively treats only the
    *acquisition* side; future buys are operator-tracked.
    """
    threshold = today - timedelta(days=window_days)
    return lot.acquisition_date >= threshold


def select_candidates(
    pool: Iterable[TaxLot],
    market_prices: Mapping[str, float],
    *,
    today: date,
    policy: HarvestPolicy | None = None,
) -> tuple[HarvestCandidate, ...]:
    """Identify TLH candidates from the pool, ranked by absolute loss."""
    pol = policy if policy is not None else HarvestPolicy()
    candidates: list[HarvestCandidate] = []

    for lot in pool:
        if lot.symbol not in market_prices:
            continue
        market_price = market_prices[lot.symbol]
        if market_price < 0:
            continue
        loss = (lot.cost_basis_per_share - market_price) * lot.quantity
        if loss <= 0:
            continue  # gain or break-even
        if loss < pol.min_loss_amount:
            continue
        loss_pct = (lot.cost_basis_per_share - market_price) / lot.cost_basis_per_share
        if loss_pct < pol.min_loss_pct:
            continue
        if _is_in_wash_zone(lot, today=today, window_days=pol.wash_sale_window_days):
            continue
        candidates.append(
            HarvestCandidate(
                lot=lot,
                market_price=market_price,
                unrealised_loss=loss,
                loss_pct=loss_pct,
            )
        )

    # Sort by absolute loss descending (largest first)
    candidates.sort(key=lambda c: -c.unrealised_loss)
    return tuple(candidates)


def total_harvestable_loss(candidates: Iterable[HarvestCandidate]) -> float:
    return sum(c.unrealised_loss for c in candidates)


def top_n_candidates(
    candidates: Sequence[HarvestCandidate], n: int
) -> tuple[HarvestCandidate, ...]:
    if n <= 0:
        raise ValueError("n must be positive")
    return tuple(candidates[:n])


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


def render_candidates(candidates: tuple[HarvestCandidate, ...]) -> str:
    if not candidates:
        return "TLH candidates: none"
    head = (
        f"TLH candidates: {len(candidates)} lots, "
        f"total harvestable loss=${total_harvestable_loss(candidates):.2f}"
    )
    lines = [head]
    for c in candidates:
        lines.append(
            f"  • {c.lot.lot_id} {c.lot.symbol} {c.lot.quantity:.2f}@"
            f"${c.lot.cost_basis_per_share:.2f}→${c.market_price:.2f} "
            f"= -${c.unrealised_loss:.2f} ({c.loss_pct * 100:.1f}%)"
        )
    return _scrub("\n".join(lines))
