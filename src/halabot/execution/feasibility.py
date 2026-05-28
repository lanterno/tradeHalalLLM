"""Pre-order feasibility (REARCHITECTURE L6 execution gates).

Turns a desired dollar notional into a venue-legal quantity and rejects orders
that can't or shouldn't be placed: below the min-notional floor, beyond available
buying power, or rounding to zero on the lot step. Pure + deterministic — the
executor calls this before ever touching the venue.

The $50 min-notional floor is intentional and un-loosenable below the configured
value (matches the legacy executor; many tests assume it)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FeasibilityConfig:
    min_notional_usd: float = 50.0
    lot_step: float = 1.0  # 1 = whole shares (stocks); fractional for crypto
    min_qty: float = 0.0


@dataclass(frozen=True)
class Feasibility:
    ok: bool
    quantity: float
    notional_usd: float
    reason: str | None = None


def _round_to_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step


def feasible_buy(
    notional_usd: float,
    price: float,
    *,
    buying_power: float,
    cfg: FeasibilityConfig,
) -> Feasibility:
    """Largest lot-step-legal buy at ``price`` for ``notional_usd``, capped by
    buying power and floored at the min-notional. Rejects (ok=False) otherwise."""
    if not math.isfinite(price) or price <= 0:
        return Feasibility(False, 0.0, 0.0, "non-positive or non-finite price")
    if not math.isfinite(notional_usd) or not math.isfinite(buying_power):
        return Feasibility(False, 0.0, 0.0, "non-finite notional/buying-power")
    target = min(notional_usd, buying_power)
    if target < cfg.min_notional_usd:
        return Feasibility(False, 0.0, 0.0, f"below min notional ${cfg.min_notional_usd:.0f}")
    qty = _round_to_step(target / price, cfg.lot_step)
    if qty < cfg.min_qty or qty <= 0:
        return Feasibility(False, 0.0, 0.0, "rounds to zero quantity")
    notional = qty * price
    if notional < cfg.min_notional_usd:
        return Feasibility(False, qty, notional, f"below min notional ${cfg.min_notional_usd:.0f}")
    if notional > buying_power + 1e-9:
        return Feasibility(False, qty, notional, "insufficient buying power")
    return Feasibility(True, qty, notional, None)


def feasible_sell(position_qty: float, *, cfg: FeasibilityConfig) -> Feasibility:
    """Sells reduce risk → always feasible for the held quantity (no min-notional
    floor on exits). Rejects only when there's nothing to sell."""
    qty = _round_to_step(abs(position_qty), cfg.lot_step)
    if qty <= 0:
        return Feasibility(False, 0.0, 0.0, "no position to sell")
    return Feasibility(True, qty, 0.0, None)
