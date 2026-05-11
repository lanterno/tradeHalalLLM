"""Mudarabah DeFi pool — Round-5 Wave 22.C.

On-chain Mudarabah pool: capital providers (rabb-al-mal) deposit
stablecoin; a manager (mudarib) deploys it into halal-screened
strategies; profits split per pre-agreed ratio; losses borne by
capital providers (AAOIFI Standard 13).

Structurally distinct from Wave 22.B Wakalah vault:
- **22.B Wakalah** = manager charges a *flat fee*; profit/loss accrues
  fully to depositors after fee.
- **22.C Mudarabah** = manager earns a *profit share* (not a flat fee);
  if there's no profit, the manager earns nothing. Loss is borne by
  depositors only.

This module is the **pool accounting primitive**: deposits, withdrawals,
profit/loss distributions per Standard 13.

Pinned semantics:

- **Closed-set PoolStatus FSM** — OPEN → PAUSED → CLOSED.
- **Profit share ratio in (0, 1)** — manager < 100% (can't take everything).
- **No flat management fee.** This is structurally what distinguishes
  Mudarabah from Wakalah.
- **Loss borne 100% by depositors.** Manager loses time only (unless
  negligence — not modelled here, scholar review path).
- **Share-based accounting.** Depositor units mint/burn at NAV.
- **Profit-distribution event** moves manager's slice into a separate
  `manager_owed_usd` bucket; depositors keep their unit count but NAV
  drops by the manager-take fraction.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from enum import Enum


class PoolStatus(str, Enum):
    """Closed-set pool FSM ladder."""

    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"


@dataclass(frozen=True)
class DepositorEntry:
    """One depositor's share balance."""

    depositor_id: str
    units: float

    def __post_init__(self) -> None:
        if not self.depositor_id or not self.depositor_id.strip():
            raise ValueError("depositor_id must be non-empty")
        if self.units < 0:
            raise ValueError("units must be non-negative")


@dataclass(frozen=True)
class MudarabahPool:
    """On-chain Mudarabah pool state."""

    pool_id: str
    manager_id: str
    manager_profit_share_pct: float
    """Manager's slice of profit (rabb's share = 1 - this)."""
    inception_on: date
    aum_usd: float
    total_units: float
    manager_owed_usd: float
    """Manager's accrued (but unpaid) profit share."""
    depositors: tuple[DepositorEntry, ...] = ()
    status: PoolStatus = PoolStatus.OPEN
    high_water_aum: float = 0.0
    """Pinned: high-water mark on AUM net of accrued manager_owed.
    Manager only earns profit share above this mark."""

    def __post_init__(self) -> None:
        if not self.pool_id or not self.pool_id.strip():
            raise ValueError("pool_id must be non-empty")
        if not self.manager_id or not self.manager_id.strip():
            raise ValueError("manager_id must be non-empty")
        if not 0.0 < self.manager_profit_share_pct < 1.0:
            raise ValueError("manager_profit_share_pct must be in (0, 1)")
        if self.aum_usd < 0:
            raise ValueError("aum_usd must be non-negative")
        if self.total_units < 0:
            raise ValueError("total_units must be non-negative")
        if self.manager_owed_usd < 0:
            raise ValueError("manager_owed_usd must be non-negative")
        if self.high_water_aum < 0:
            raise ValueError("high_water_aum must be non-negative")
        # Manager cannot be a depositor.
        for d in self.depositors:
            if d.depositor_id == self.manager_id:
                raise ValueError("manager cannot also be a depositor")
        ids = [d.depositor_id for d in self.depositors]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate depositor_id")
        sum_units = sum(d.units for d in self.depositors)
        if abs(sum_units - self.total_units) > 1e-6:
            raise ValueError(f"depositor unit sum {sum_units:.6f} ≠ total_units")

    def net_aum_for_depositors(self) -> float:
        """AUM net of manager's accrued owe — this backs depositor units."""
        return self.aum_usd - self.manager_owed_usd

    def nav_per_unit(self) -> float:
        """Per-unit NAV. Returns 1.0 when no units issued."""
        if self.total_units <= 0:
            return 1.0
        return self.net_aum_for_depositors() / self.total_units


def new_pool(
    *,
    pool_id: str,
    manager_id: str,
    manager_profit_share_pct: float,
    inception_on: date,
) -> MudarabahPool:
    return MudarabahPool(
        pool_id=pool_id,
        manager_id=manager_id,
        manager_profit_share_pct=manager_profit_share_pct,
        inception_on=inception_on,
        aum_usd=0.0,
        total_units=0.0,
        manager_owed_usd=0.0,
        depositors=(),
        status=PoolStatus.OPEN,
        high_water_aum=0.0,
    )


def _bump_units(
    depositors: tuple[DepositorEntry, ...],
    depositor_id: str,
    delta: float,
) -> tuple[DepositorEntry, ...]:
    out: list[DepositorEntry] = []
    found = False
    for d in depositors:
        if d.depositor_id == depositor_id:
            new_units = d.units + delta
            if new_units < -1e-12:
                raise ValueError(f"{depositor_id} cannot go negative units")
            if new_units > 1e-12:
                out.append(replace(d, units=new_units))
            found = True
        else:
            out.append(d)
    if not found:
        if delta < 0:
            raise ValueError(f"{depositor_id} not a depositor")
        out.append(DepositorEntry(depositor_id=depositor_id, units=delta))
    out.sort(key=lambda d: d.depositor_id)
    return tuple(out)


def deposit(
    pool: MudarabahPool,
    *,
    depositor_id: str,
    amount_usd: float,
) -> MudarabahPool:
    """Mint units at current NAV."""
    if pool.status is not PoolStatus.OPEN:
        raise ValueError(f"deposits forbidden in {pool.status.value}")
    if depositor_id == pool.manager_id:
        raise ValueError("manager cannot deposit into own pool")
    if amount_usd <= 0:
        raise ValueError("amount_usd must be positive")
    nav = pool.nav_per_unit()
    minted = amount_usd / nav
    new_d = _bump_units(pool.depositors, depositor_id, minted)
    new_aum = pool.aum_usd + amount_usd
    return replace(
        pool,
        aum_usd=new_aum,
        total_units=pool.total_units + minted,
        depositors=new_d,
        high_water_aum=max(pool.high_water_aum, pool.net_aum_for_depositors() + amount_usd),
    )


def withdraw(
    pool: MudarabahPool,
    *,
    depositor_id: str,
    amount_usd: float,
) -> MudarabahPool:
    """Burn units at current NAV."""
    if pool.status is PoolStatus.CLOSED:
        raise ValueError("CLOSED pool rejects withdrawals")
    if amount_usd <= 0:
        raise ValueError("amount_usd must be positive")
    nav = pool.nav_per_unit()
    holder = next(
        (d for d in pool.depositors if d.depositor_id == depositor_id),
        None,
    )
    if holder is None:
        raise ValueError(f"{depositor_id} not a depositor")
    max_value = holder.units * nav
    if amount_usd > max_value + 1e-9:
        raise ValueError(f"withdraw {amount_usd} exceeds balance {max_value:.2f}")
    burned = amount_usd / nav
    new_d = _bump_units(pool.depositors, depositor_id, -burned)
    return replace(
        pool,
        aum_usd=pool.aum_usd - amount_usd,
        total_units=pool.total_units - burned,
        depositors=new_d,
    )


def mark_to_market(pool: MudarabahPool, *, new_aum_usd: float) -> MudarabahPool:
    """Update AUM (strategy returns).

    Pinned: new_aum_usd ≥ manager_owed_usd (pool can't be insolvent).
    """
    if new_aum_usd < 0:
        raise ValueError("new_aum_usd must be non-negative")
    if new_aum_usd < pool.manager_owed_usd - 1e-9:
        raise ValueError("mark_to_market below manager_owed_usd → insolvent")
    return replace(pool, aum_usd=new_aum_usd)


def distribute_profit(
    pool: MudarabahPool,
) -> MudarabahPool:
    """Accrue manager's profit share against current excess over HWM.

    Pinned: manager earns profit_share × (net_aum − high_water).
    Loss path: no accrual; depositors absorb (NAV drops).
    """
    excess = pool.net_aum_for_depositors() - pool.high_water_aum
    if excess <= 0:
        # Below HWM → loss or flat; no manager accrual; just update
        # high_water? No — HWM stays at prior peak.
        return pool
    manager_take = excess * pool.manager_profit_share_pct
    new_manager_owed = pool.manager_owed_usd + manager_take
    # New HWM = current net_aum_for_depositors after manager take is
    # accrued (since the manager_take moves into a separate bucket,
    # net_aum_for_depositors drops by manager_take).
    new_hw = pool.high_water_aum + (excess - manager_take)
    return replace(
        pool,
        manager_owed_usd=new_manager_owed,
        high_water_aum=new_hw,
    )


def pay_manager(pool: MudarabahPool) -> MudarabahPool:
    """Pay accrued manager_owed_usd out to the manager."""
    if pool.manager_owed_usd <= 0:
        return pool
    return replace(
        pool,
        aum_usd=pool.aum_usd - pool.manager_owed_usd,
        manager_owed_usd=0.0,
    )


_LEGAL_TRANSITIONS: dict[PoolStatus, set[PoolStatus]] = {
    PoolStatus.OPEN: {PoolStatus.PAUSED, PoolStatus.CLOSED},
    PoolStatus.PAUSED: {PoolStatus.OPEN, PoolStatus.CLOSED},
    PoolStatus.CLOSED: set(),
}


def transition_status(pool: MudarabahPool, *, new_status: PoolStatus) -> MudarabahPool:
    if new_status not in _LEGAL_TRANSITIONS[pool.status]:
        raise ValueError(f"illegal transition {pool.status.value} → {new_status.value}")
    return replace(pool, status=new_status)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_STATUS_EMOJI: dict[PoolStatus, str] = {
    PoolStatus.OPEN: "🟢",
    PoolStatus.PAUSED: "🟡",
    PoolStatus.CLOSED: "🔴",
}


def render_pool(pool: MudarabahPool) -> str:
    return (
        f"{_STATUS_EMOJI[pool.status]} {pool.pool_id} "
        f"[{pool.status.value}] manager={_mask(pool.manager_id)} "
        f"(profit-share {pool.manager_profit_share_pct * 100:.0f}%)\n"
        f"  AUM ${pool.aum_usd:,.2f} | "
        f"units {pool.total_units:,.4f} | "
        f"NAV ${pool.nav_per_unit():.4f} | "
        f"HWM ${pool.high_water_aum:,.2f} | "
        f"manager owed ${pool.manager_owed_usd:,.2f}"
    )
