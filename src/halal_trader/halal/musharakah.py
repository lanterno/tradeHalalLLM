"""Musharakah co-investment pool engine — Round-5 Wave 7.B.

Musharakah is the multi-party investment partnership: all parties
contribute capital + may contribute labour; profits are shared per
agreed ratio; **losses are borne in proportion to capital contribution**
(unlike Mudarabah where loss flows to capital provider only).

This module ships the **Musharakah pool engine**:

- Pool with N partners, each carrying a capital contribution.
- Profit-share ratio per partner (ratios sum to 1).
- Loss distribution proportional to capital share.

Pinned semantics:

- **Closed-set PoolStatus ladder** (FORMING / ACTIVE / DISSOLVING /
  CLOSED).
- **Partner profit ratios sum to 1.0**; loss ratios are derived from
  capital contributions (cannot be overridden — fiqh rule).
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class PoolStatus(str, Enum):
    """Closed-set pool lifecycle status."""

    FORMING = "forming"
    ACTIVE = "active"
    DISSOLVING = "dissolving"
    CLOSED = "closed"


@dataclass(frozen=True)
class Partner:
    """A single partner's stake in the pool."""

    handle: str
    capital_contribution: float
    profit_share: float  # 0..1, sums to 1 across partners

    def __post_init__(self) -> None:
        if not self.handle or not self.handle.strip():
            raise ValueError("handle must be non-empty")
        if "@" in self.handle:
            raise ValueError("handle must be a handle, not an email")
        if self.capital_contribution <= 0:
            raise ValueError("capital_contribution must be positive")
        if not 0.0 <= self.profit_share <= 1.0:
            raise ValueError("profit_share must be in [0, 1]")


@dataclass(frozen=True)
class MusharakahPool:
    """A Musharakah pool."""

    pool_id: str
    partners: tuple[Partner, ...]
    currency: str
    formation_date: date
    status: PoolStatus

    def __post_init__(self) -> None:
        if not self.pool_id or not self.pool_id.strip():
            raise ValueError("pool_id must be non-empty")
        if len(self.partners) < 2:
            raise ValueError("Musharakah requires at least 2 partners")
        if not self.currency or len(self.currency) > 8:
            raise ValueError("currency must be a non-empty short code")
        handles = {p.handle for p in self.partners}
        if len(handles) != len(self.partners):
            raise ValueError("partners must have distinct handles")
        total_share = sum(p.profit_share for p in self.partners)
        if abs(total_share - 1.0) > 1e-9:
            raise ValueError("partner profit_shares must sum to 1.0")

    def total_capital(self) -> float:
        return sum(p.capital_contribution for p in self.partners)

    def capital_share(self, handle: str) -> float:
        for p in self.partners:
            if p.handle == handle:
                return p.capital_contribution / self.total_capital()
        raise KeyError(f"unknown partner: {handle}")


@dataclass(frozen=True)
class PoolDistribution:
    """Distribution at settlement."""

    pool_id: str
    final_pool_value: float
    profit_or_loss: float
    is_loss: bool
    per_partner: tuple[tuple[str, float], ...]  # (handle, share_amount)


def settle_pool(pool: MusharakahPool, *, final_pool_value: float) -> PoolDistribution:
    """Distribute the final pool value among partners per the rules."""
    if pool.status not in (PoolStatus.ACTIVE, PoolStatus.DISSOLVING):
        raise ValueError("can only settle ACTIVE or DISSOLVING pools")
    if final_pool_value < 0:
        raise ValueError("final_pool_value must be non-negative")

    total_capital = pool.total_capital()
    p_or_l = final_pool_value - total_capital
    is_loss = p_or_l < 0

    per_partner: list[tuple[str, float]] = []
    for p in pool.partners:
        if is_loss:
            # Loss in proportion to capital contribution.
            share = p_or_l * (p.capital_contribution / total_capital)
        else:
            share = p_or_l * p.profit_share
        per_partner.append((p.handle, share))

    return PoolDistribution(
        pool_id=pool.pool_id,
        final_pool_value=final_pool_value,
        profit_or_loss=p_or_l,
        is_loss=is_loss,
        per_partner=tuple(per_partner),
    )


def advance_status(pool: MusharakahPool, target: PoolStatus) -> MusharakahPool:
    valid = {
        PoolStatus.FORMING: {PoolStatus.ACTIVE},
        PoolStatus.ACTIVE: {PoolStatus.DISSOLVING},
        PoolStatus.DISSOLVING: {PoolStatus.CLOSED},
    }
    if target not in valid.get(pool.status, set()):
        raise ValueError(f"cannot transition {pool.status.value} → {target.value}")
    return MusharakahPool(
        pool_id=pool.pool_id,
        partners=pool.partners,
        currency=pool.currency,
        formation_date=pool.formation_date,
        status=target,
    )


def render_pool(pool: MusharakahPool) -> str:
    head = (
        f"🤝 Musharakah {pool.pool_id} [{pool.status.value}] "
        f"capital=${pool.total_capital():.2f} {pool.currency}"
    )
    lines = [head]
    for p in pool.partners:
        cap_pct = p.capital_contribution / pool.total_capital() * 100
        lines.append(
            f"  • {p.handle}: capital ${p.capital_contribution:.2f} "
            f"({cap_pct:.1f}%) | profit_share {p.profit_share * 100:.1f}%"
        )
    return "\n".join(lines)


def render_distribution(d: PoolDistribution) -> str:
    state = "loss" if d.is_loss else "profit"
    head = f"⚖ {d.pool_id} {state}: ${d.profit_or_loss:+.2f}"
    lines = [head]
    for handle, share in d.per_partner:
        lines.append(f"  • {handle}: ${share:+.2f}")
    return "\n".join(lines)
