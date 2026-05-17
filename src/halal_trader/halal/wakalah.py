"""Wakalah (agency) account engine — Round-5 Wave 7.C.

Wakalah is the classical fiqh agency relationship: the **principal**
(muwakkil) appoints an **agent** (wakil) to manage funds for a fixed
fee (wakil fee). Profits accrue entirely to the principal; the agent
receives only their pre-agreed fee. Losses (absent agent negligence)
flow to the principal.

This module ships the **Wakalah account state engine + fee
distribution math**.

Pinned semantics:

- **Closed-set WakalahStatus ladder** (DRAFT / ACTIVE / SETTLING /
  CLOSED).
- **Wakil fee** is operator-set; either fixed amount or fixed
  percentage of capital — not a percentage of profit (would convert
  it to Mudarabah).
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class WakalahStatus(str, Enum):
    """Closed-set Wakalah status ladder."""

    DRAFT = "draft"
    ACTIVE = "active"
    SETTLING = "settling"
    CLOSED = "closed"


class FeeStructure(str, Enum):
    """Closed-set Wakil fee structures."""

    FIXED_AMOUNT = "fixed_amount"
    FIXED_PCT_OF_CAPITAL = "fixed_pct_of_capital"


@dataclass(frozen=True)
class WakalahContract:
    """A Wakalah contract."""

    contract_id: str
    principal_handle: str  # muwakkil
    agent_handle: str  # wakil
    capital_amount: float
    currency: str
    fee_structure: FeeStructure
    fee_value: float  # amount or fraction depending on structure
    start_date: date
    expected_end_date: date
    status: WakalahStatus

    def __post_init__(self) -> None:
        if not self.contract_id or not self.contract_id.strip():
            raise ValueError("contract_id must be non-empty")
        if not self.principal_handle.strip() or not self.agent_handle.strip():
            raise ValueError("party handles must be non-empty")
        if "@" in self.principal_handle or "@" in self.agent_handle:
            raise ValueError("handles must be handles, not emails")
        if self.principal_handle == self.agent_handle:
            raise ValueError("principal and agent must be different parties")
        if self.capital_amount <= 0:
            raise ValueError("capital_amount must be positive")
        if not self.currency or len(self.currency) > 8:
            raise ValueError("currency must be a non-empty short code")
        if self.fee_value < 0:
            raise ValueError("fee_value must be non-negative")
        if (
            self.fee_structure is FeeStructure.FIXED_PCT_OF_CAPITAL
            and not 0.0 <= self.fee_value <= 1.0
        ):
            raise ValueError("fee_value as pct must be in [0, 1]")
        if self.expected_end_date <= self.start_date:
            raise ValueError("expected_end_date must be after start_date")


def calculate_fee(contract: WakalahContract) -> float:
    """Compute the wakil fee from the contract."""
    if contract.fee_structure is FeeStructure.FIXED_AMOUNT:
        return contract.fee_value
    return contract.capital_amount * contract.fee_value


@dataclass(frozen=True)
class Settlement:
    """Settlement: principal gets net P&L, agent gets fixed fee."""

    contract_id: str
    final_capital_value: float
    profit_or_loss: float
    is_loss: bool
    agent_fee: float
    principal_net: float
    agent_negligent: bool

    def __post_init__(self) -> None:
        if self.final_capital_value < 0:
            raise ValueError("final_capital_value cannot be negative")
        if self.agent_fee < 0:
            raise ValueError("agent_fee must be non-negative")


def settle(
    contract: WakalahContract,
    *,
    final_capital_value: float,
    agent_negligent: bool = False,
) -> Settlement:
    """Compute settlement: principal gets net P&L; agent gets fee or 0 if negligent."""
    if contract.status not in (WakalahStatus.ACTIVE, WakalahStatus.SETTLING):
        raise ValueError("can only settle ACTIVE or SETTLING contracts")
    if final_capital_value < 0:
        raise ValueError("final_capital_value must be non-negative")

    p_or_l = final_capital_value - contract.capital_amount
    is_loss = p_or_l < 0

    if agent_negligent:
        # Negligent agent forfeits fee
        agent_fee = 0.0
    else:
        agent_fee = calculate_fee(contract)

    # Agent fee is paid from final value; principal gets the rest.
    # If the fee exceeds final value (shouldn't happen with reasonable fees),
    # cap fee at final value and zero principal.
    if agent_fee > final_capital_value:
        agent_fee = final_capital_value
        principal_net = 0.0
    else:
        principal_net = final_capital_value - agent_fee

    return Settlement(
        contract_id=contract.contract_id,
        final_capital_value=final_capital_value,
        profit_or_loss=p_or_l,
        is_loss=is_loss,
        agent_fee=agent_fee,
        principal_net=principal_net,
        agent_negligent=agent_negligent,
    )


def advance_status(
    contract: WakalahContract, target: WakalahStatus
) -> WakalahContract:
    valid = {
        WakalahStatus.DRAFT: {WakalahStatus.ACTIVE},
        WakalahStatus.ACTIVE: {WakalahStatus.SETTLING},
        WakalahStatus.SETTLING: {WakalahStatus.CLOSED},
    }
    if target not in valid.get(contract.status, set()):
        raise ValueError(
            f"cannot transition {contract.status.value} → {target.value}"
        )
    return WakalahContract(
        contract_id=contract.contract_id,
        principal_handle=contract.principal_handle,
        agent_handle=contract.agent_handle,
        capital_amount=contract.capital_amount,
        currency=contract.currency,
        fee_structure=contract.fee_structure,
        fee_value=contract.fee_value,
        start_date=contract.start_date,
        expected_end_date=contract.expected_end_date,
        status=target,
    )


def render_contract(c: WakalahContract) -> str:
    fee_desc = (
        f"{c.fee_value:.2f} {c.currency} (fixed)"
        if c.fee_structure is FeeStructure.FIXED_AMOUNT
        else f"{c.fee_value * 100:.2f}% of capital"
    )
    return (
        f"📜 Wakalah {c.contract_id} [{c.status.value}]: "
        f"{c.principal_handle} → {c.agent_handle} "
        f"capital={c.capital_amount:.2f} {c.currency}, fee={fee_desc}"
    )


def render_settlement(s: Settlement) -> str:
    state = "loss" if s.is_loss else "profit"
    flag = " [NEGLIGENT]" if s.agent_negligent else ""
    return (
        f"⚖ Wakalah settlement {s.contract_id}: {state} ${s.profit_or_loss:+.2f}{flag} "
        f"→ agent_fee=${s.agent_fee:.2f}, principal_net=${s.principal_net:.2f}"
    )
