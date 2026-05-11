"""Wakalah DeFi vault — Round-5 Wave 22.B.

User-deposits-stablecoin → halal-vault → manager-allocates flow.
Manager (the Wakil) charges a fixed-percentage Wakalah fee on AUM —
NOT a performance carry. Profits accrue to depositors pro-rata; the
vault is structurally non-riba because:

1. The vault holds principal-yielding strategies that are themselves
   halal-screened (caller passes the screen predicate).
2. The Wakalah fee is a service fee, capped at 3%/yr in policy.
3. There is no guaranteed return to depositors.

This module is the **vault accounting primitive**: shares minted/burned
on deposit/withdrawal, AUM marked to market, Wakalah fee accrued, and
the share-price-NAV reconciler.

Pinned semantics:

- **Share-based accounting.** Each depositor holds N shares; per-share
  NAV is `(aum - accrued_fees) / total_shares`.
- **First deposit pins NAV at 1.0 USD per share.**
- **Closed-set VaultStatus FSM** — OPEN → PAUSED → CLOSED.
  PAUSED rejects deposits but allows withdrawals (emergency drain).
  CLOSED is terminal (no deposits, no withdrawals).
- **Fee accrual is daily, simple interest.** Pinned to avoid the
  compound-Wakalah ambiguity.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from enum import Enum


class VaultStatus(str, Enum):
    """Closed-set vault FSM ladder."""

    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"


@dataclass(frozen=True)
class WakalahPolicy:
    """Operator-tunable vault policy."""

    annual_fee_pct: float = 0.015
    """Wakalah fee as a fraction of AUM per year. ≤ 3% pinned."""
    min_deposit_usd: float = 100.0
    max_aum_usd: float = 100_000_000.0
    """Hard cap; deposits beyond reject."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.annual_fee_pct < 0.03:
            raise ValueError("annual_fee_pct must be in [0, 0.03); higher reads as carry")
        if self.min_deposit_usd <= 0:
            raise ValueError("min_deposit_usd must be positive")
        if self.max_aum_usd <= 0:
            raise ValueError("max_aum_usd must be positive")


@dataclass(frozen=True)
class ShareEntry:
    """One depositor's share balance."""

    depositor_id: str
    shares: float

    def __post_init__(self) -> None:
        if not self.depositor_id or not self.depositor_id.strip():
            raise ValueError("depositor_id must be non-empty")
        if self.shares < 0:
            raise ValueError("shares must be non-negative")


@dataclass(frozen=True)
class WakalahVault:
    """A halal Wakalah DeFi vault."""

    vault_id: str
    manager_id: str
    policy: WakalahPolicy
    inception_on: date
    aum_usd: float
    total_shares: float
    accrued_fee_usd: float
    """Wakalah fees accrued but not yet paid out to the manager."""
    last_accrual_on: date
    holders: tuple[ShareEntry, ...] = ()
    status: VaultStatus = VaultStatus.OPEN

    def __post_init__(self) -> None:
        if not self.vault_id or not self.vault_id.strip():
            raise ValueError("vault_id must be non-empty")
        if not self.manager_id or not self.manager_id.strip():
            raise ValueError("manager_id must be non-empty")
        if self.aum_usd < 0:
            raise ValueError("aum_usd must be non-negative")
        if self.total_shares < 0:
            raise ValueError("total_shares must be non-negative")
        if self.accrued_fee_usd < 0:
            raise ValueError("accrued_fee_usd must be non-negative")
        if self.last_accrual_on < self.inception_on:
            raise ValueError("last_accrual_on must be ≥ inception_on")
        # Holders must reconcile to total_shares.
        sum_shares = sum(h.shares for h in self.holders)
        if abs(sum_shares - self.total_shares) > 1e-6:
            raise ValueError(f"holders sum {sum_shares:.6f} ≠ total_shares {self.total_shares:.6f}")
        # Unique holders.
        ids = [h.depositor_id for h in self.holders]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate depositor_id")
        # Manager cannot also be a depositor (anti-self-dealing).
        for h in self.holders:
            if h.depositor_id == self.manager_id:
                raise ValueError("manager cannot also be a depositor")

    def nav_per_share(self) -> float:
        """Net asset value per share. Returns 1.0 when total_shares=0."""
        if self.total_shares <= 0:
            return 1.0
        net = self.aum_usd - self.accrued_fee_usd
        return net / self.total_shares


def new_vault(
    *,
    vault_id: str,
    manager_id: str,
    inception_on: date,
    policy: WakalahPolicy | None = None,
) -> WakalahVault:
    return WakalahVault(
        vault_id=vault_id,
        manager_id=manager_id,
        policy=policy if policy is not None else WakalahPolicy(),
        inception_on=inception_on,
        aum_usd=0.0,
        total_shares=0.0,
        accrued_fee_usd=0.0,
        last_accrual_on=inception_on,
        holders=(),
        status=VaultStatus.OPEN,
    )


def _bump_holder_shares(
    holders: tuple[ShareEntry, ...],
    depositor_id: str,
    delta: float,
) -> tuple[ShareEntry, ...]:
    """Return new tuple with `delta` shares added to `depositor_id`."""
    out: list[ShareEntry] = []
    found = False
    for h in holders:
        if h.depositor_id == depositor_id:
            new_shares = h.shares + delta
            if new_shares < -1e-12:
                raise ValueError(f"{depositor_id} cannot go negative shares")
            if new_shares > 1e-12:
                out.append(replace(h, shares=new_shares))
            found = True
        else:
            out.append(h)
    if not found:
        if delta < 0:
            raise ValueError(f"{depositor_id} not a holder")
        out.append(ShareEntry(depositor_id=depositor_id, shares=delta))
    out.sort(key=lambda h: h.depositor_id)
    return tuple(out)


def deposit(
    vault: WakalahVault,
    *,
    depositor_id: str,
    amount_usd: float,
) -> WakalahVault:
    """Mint shares at current NAV.

    Pinned:
    - Vault must be OPEN.
    - Manager cannot deposit (anti-self-dealing).
    - amount_usd ≥ policy.min_deposit_usd.
    - aum + amount ≤ policy.max_aum_usd.
    """
    if vault.status is not VaultStatus.OPEN:
        raise ValueError(f"deposits forbidden in {vault.status.value} state")
    if depositor_id == vault.manager_id:
        raise ValueError("manager cannot deposit into own vault")
    if amount_usd < vault.policy.min_deposit_usd:
        raise ValueError(f"deposit {amount_usd} < min {vault.policy.min_deposit_usd}")
    if vault.aum_usd + amount_usd > vault.policy.max_aum_usd + 1e-9:
        raise ValueError("deposit would exceed max_aum_usd")
    nav = vault.nav_per_share()
    minted = amount_usd / nav
    new_holders = _bump_holder_shares(vault.holders, depositor_id, minted)
    return replace(
        vault,
        aum_usd=vault.aum_usd + amount_usd,
        total_shares=vault.total_shares + minted,
        holders=new_holders,
    )


def withdraw(
    vault: WakalahVault,
    *,
    depositor_id: str,
    amount_usd: float,
) -> WakalahVault:
    """Burn shares at current NAV.

    Pinned: PAUSED allows withdrawals (emergency drain). CLOSED rejects.
    """
    if vault.status is VaultStatus.CLOSED:
        raise ValueError("CLOSED vault rejects withdrawals")
    if amount_usd <= 0:
        raise ValueError("amount_usd must be positive")
    if vault.aum_usd <= 0:
        raise ValueError("vault has no AUM")
    holder = next(
        (h for h in vault.holders if h.depositor_id == depositor_id),
        None,
    )
    if holder is None:
        raise ValueError(f"{depositor_id} not a holder")
    nav = vault.nav_per_share()
    max_withdrawable = holder.shares * nav
    if amount_usd > max_withdrawable + 1e-9:
        raise ValueError(f"withdrawal {amount_usd} exceeds holder balance {max_withdrawable:.2f}")
    burned = amount_usd / nav
    new_holders = _bump_holder_shares(vault.holders, depositor_id, -burned)
    return replace(
        vault,
        aum_usd=vault.aum_usd - amount_usd,
        total_shares=vault.total_shares - burned,
        holders=new_holders,
    )


def mark_to_market(vault: WakalahVault, *, new_aum_usd: float) -> WakalahVault:
    """Update AUM (strategy returns) without minting/burning shares.

    Pinned: must be ≥ accrued_fee (vault cannot be insolvent).
    """
    if new_aum_usd < 0:
        raise ValueError("new_aum_usd must be non-negative")
    if new_aum_usd < vault.accrued_fee_usd - 1e-9:
        raise ValueError("mark_to_market below accrued_fee — would make vault insolvent")
    return replace(vault, aum_usd=new_aum_usd)


def accrue_fee(vault: WakalahVault, *, on_date: date) -> WakalahVault:
    """Accrue Wakalah fee for days since `last_accrual_on`.

    Pinned simple-interest math: `accrued += pct × aum × days/365`.
    """
    if on_date < vault.last_accrual_on:
        raise ValueError("on_date cannot precede last_accrual_on")
    days = (on_date - vault.last_accrual_on).days
    if days == 0:
        return vault
    extra = vault.aum_usd * vault.policy.annual_fee_pct * (days / 365.0)
    new_accrued = vault.accrued_fee_usd + extra
    if new_accrued > vault.aum_usd + 1e-9:
        # Cap at AUM — vault cannot be insolvent.
        new_accrued = vault.aum_usd
    return replace(
        vault,
        accrued_fee_usd=new_accrued,
        last_accrual_on=on_date,
    )


def pay_fee_to_manager(vault: WakalahVault) -> WakalahVault:
    """Pay accrued fees out to the manager. Resets `accrued_fee_usd` to 0.

    Pinned: AUM decreases by the paid fee.
    """
    if vault.accrued_fee_usd <= 0:
        return vault
    return replace(
        vault,
        aum_usd=vault.aum_usd - vault.accrued_fee_usd,
        accrued_fee_usd=0.0,
    )


_LEGAL_TRANSITIONS: dict[VaultStatus, set[VaultStatus]] = {
    VaultStatus.OPEN: {VaultStatus.PAUSED, VaultStatus.CLOSED},
    VaultStatus.PAUSED: {VaultStatus.OPEN, VaultStatus.CLOSED},
    VaultStatus.CLOSED: set(),
}


def transition_status(vault: WakalahVault, *, new_status: VaultStatus) -> WakalahVault:
    if new_status not in _LEGAL_TRANSITIONS[vault.status]:
        raise ValueError(f"illegal transition {vault.status.value} → {new_status.value}")
    return replace(vault, status=new_status)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_STATUS_EMOJI: dict[VaultStatus, str] = {
    VaultStatus.OPEN: "🟢",
    VaultStatus.PAUSED: "🟡",
    VaultStatus.CLOSED: "🔴",
}


def render_vault(vault: WakalahVault) -> str:
    return (
        f"{_STATUS_EMOJI[vault.status]} {vault.vault_id} "
        f"[{vault.status.value}] manager={_mask(vault.manager_id)}\n"
        f"  AUM ${vault.aum_usd:,.2f} | "
        f"shares {vault.total_shares:,.4f} | "
        f"NAV ${vault.nav_per_share():.4f}/share | "
        f"accrued fee ${vault.accrued_fee_usd:,.2f} | "
        f"{len(vault.holders)} holders"
    )
