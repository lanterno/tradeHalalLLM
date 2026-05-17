"""Mudarabah profit-sharing account engine — Round-5 Wave 7.A.

Mudarabah is the classical fiqh profit-sharing partnership: the
**capital provider** (rabb-ul-mal) supplies funds; the **manager**
(mudarib) supplies expertise + labour. Profits are shared per a
pre-agreed split; **losses are borne entirely by the capital
provider** unless the loss results from the manager's negligence.

This module ships the **Mudarabah account state engine + profit/loss
distribution math**.

Pinned semantics:

- **Closed-set MudarabahStatus ladder** (DRAFT / ACTIVE / SETTLING /
  CLOSED).
- **Profit split must sum to 1.0** (rabb_share + mudarib_share = 1).
- **On loss, loss flows to capital provider** unless `manager_negligent=True`.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class MudarabahStatus(str, Enum):
    """Closed-set status ladder."""

    DRAFT = "draft"
    ACTIVE = "active"
    SETTLING = "settling"
    CLOSED = "closed"


@dataclass(frozen=True)
class MudarabahContract:
    """A Mudarabah contract."""

    contract_id: str
    rabb_handle: str  # capital provider
    mudarib_handle: str  # manager
    capital_amount: float
    currency: str
    rabb_profit_share: float  # 0..1
    mudarib_profit_share: float
    start_date: date
    expected_end_date: date
    status: MudarabahStatus

    def __post_init__(self) -> None:
        if not self.contract_id or not self.contract_id.strip():
            raise ValueError("contract_id must be non-empty")
        if not self.rabb_handle.strip() or not self.mudarib_handle.strip():
            raise ValueError("party handles must be non-empty")
        if "@" in self.rabb_handle or "@" in self.mudarib_handle:
            raise ValueError("handles must be handles, not emails")
        if self.rabb_handle == self.mudarib_handle:
            raise ValueError("rabb and mudarib must be different parties")
        if self.capital_amount <= 0:
            raise ValueError("capital_amount must be positive")
        if not self.currency or len(self.currency) > 8:
            raise ValueError("currency must be a non-empty short code")
        if not 0.0 < self.rabb_profit_share < 1.0:
            raise ValueError("rabb_profit_share must be in (0, 1)")
        if not 0.0 < self.mudarib_profit_share < 1.0:
            raise ValueError("mudarib_profit_share must be in (0, 1)")
        if abs(self.rabb_profit_share + self.mudarib_profit_share - 1.0) > 1e-9:
            raise ValueError("profit shares must sum to 1.0")
        if self.expected_end_date <= self.start_date:
            raise ValueError("expected_end_date must be after start_date")


@dataclass(frozen=True)
class Settlement:
    """Settlement of profit/loss at end of contract."""

    contract_id: str
    final_capital_value: float
    profit_or_loss: float
    rabb_share: float
    mudarib_share: float
    is_loss: bool
    manager_negligent: bool

    def __post_init__(self) -> None:
        if self.final_capital_value < 0:
            raise ValueError("final_capital_value cannot be negative")


def settle(
    contract: MudarabahContract,
    *,
    final_capital_value: float,
    manager_negligent: bool = False,
) -> Settlement:
    """Compute settlement: distribute profit per ratio; loss to rabb (or both if negligent)."""
    if contract.status not in (MudarabahStatus.ACTIVE, MudarabahStatus.SETTLING):
        raise ValueError("can only settle ACTIVE or SETTLING contracts")
    if final_capital_value < 0:
        raise ValueError("final_capital_value must be non-negative")

    p_or_l = final_capital_value - contract.capital_amount
    is_loss = p_or_l < 0

    if is_loss:
        if manager_negligent:
            # Negligent manager bears the entire loss
            rabb_share = 0.0
            mudarib_share = p_or_l
        else:
            # Standard rule: loss to rabb-ul-mal alone
            rabb_share = p_or_l
            mudarib_share = 0.0
    else:
        rabb_share = p_or_l * contract.rabb_profit_share
        mudarib_share = p_or_l * contract.mudarib_profit_share

    return Settlement(
        contract_id=contract.contract_id,
        final_capital_value=final_capital_value,
        profit_or_loss=p_or_l,
        rabb_share=rabb_share,
        mudarib_share=mudarib_share,
        is_loss=is_loss,
        manager_negligent=manager_negligent,
    )


def advance_status(
    contract: MudarabahContract, target: MudarabahStatus
) -> MudarabahContract:
    """Advance the contract through its lifecycle."""
    valid_transitions = {
        MudarabahStatus.DRAFT: {MudarabahStatus.ACTIVE},
        MudarabahStatus.ACTIVE: {MudarabahStatus.SETTLING},
        MudarabahStatus.SETTLING: {MudarabahStatus.CLOSED},
    }
    allowed = valid_transitions.get(contract.status, set())
    if target not in allowed:
        raise ValueError(
            f"cannot transition {contract.status.value} → {target.value}"
        )
    return MudarabahContract(
        contract_id=contract.contract_id,
        rabb_handle=contract.rabb_handle,
        mudarib_handle=contract.mudarib_handle,
        capital_amount=contract.capital_amount,
        currency=contract.currency,
        rabb_profit_share=contract.rabb_profit_share,
        mudarib_profit_share=contract.mudarib_profit_share,
        start_date=contract.start_date,
        expected_end_date=contract.expected_end_date,
        status=target,
    )


def render_contract(contract: MudarabahContract) -> str:
    return (
        f"📜 Mudarabah {contract.contract_id} [{contract.status.value}]: "
        f"{contract.rabb_handle} ↔ {contract.mudarib_handle} "
        f"capital={contract.capital_amount:.2f} {contract.currency}, "
        f"split {contract.rabb_profit_share * 100:.0f}/{contract.mudarib_profit_share * 100:.0f}"
    )


def render_settlement(s: Settlement) -> str:
    state = "loss" if s.is_loss else "profit"
    flag = " [NEGLIGENT]" if s.manager_negligent else ""
    return (
        f"⚖ Settlement {s.contract_id}: {state} ${s.profit_or_loss:+.2f}{flag} "
        f"→ rabb=${s.rabb_share:+.2f}, mudarib=${s.mudarib_share:+.2f}"
    )
