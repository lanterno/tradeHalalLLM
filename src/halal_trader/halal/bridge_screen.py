"""Cross-chain bridge halal screen — Round-5 Wave 22.G.

Cross-chain bridges move assets between blockchains. Halal compliance
hinges on:

1. **Underlying asset** — bridging a halal asset is fine; bridging a
   haram asset never is, regardless of bridge mechanism.
2. **Bridge mechanism** — lock-and-mint vs. burn-and-mint vs.
   liquidity-pool. Lock-and-mint with proper attestation is generally
   permissible; some liquidity-pool models involve fractional reserve
   or interest-bearing collateral, which raises riba concerns.
3. **Counterparty risk + custodian** — third-party custodian holding
   the locked tokens must be itself halal-compliant.

This module ships the **bridge halal screen**.

Pinned semantics:

- **Closed-set BridgeMechanism ladder** — 4 documented types.
- **Closed-set BridgeIssue ladder** — 6 specific halal concerns.
- **`screen_bridge`** is pure.
- **No-secret-leak pin** — never includes wallet addresses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BridgeMechanism(str, Enum):
    """Closed-set bridge mechanisms."""

    LOCK_AND_MINT = "lock_and_mint"
    BURN_AND_MINT = "burn_and_mint"
    LIQUIDITY_POOL = "liquidity_pool"
    ATOMIC_SWAP = "atomic_swap"


class BridgeIssue(str, Enum):
    """Closed-set bridge halal issues."""

    UNDERLYING_ASSET_NOT_HALAL = "underlying_asset_not_halal"
    INTEREST_BEARING_COLLATERAL = "interest_bearing_collateral"
    FRACTIONAL_RESERVE_RISK = "fractional_reserve_risk"
    NO_PROOF_OF_RESERVES = "no_proof_of_reserves"
    CUSTODIAN_NOT_HALAL = "custodian_not_halal"
    UNAUDITED_BRIDGE_CONTRACT = "unaudited_bridge_contract"


@dataclass(frozen=True)
class BridgePolicy:
    """Operator-tunable bridge policy."""

    require_proof_of_reserves: bool = True
    block_liquidity_pool_models: bool = False
    require_audit: bool = True

    def __post_init__(self) -> None:
        pass


@dataclass(frozen=True)
class BridgeInputs:
    """Inputs for a bridge screen."""

    bridge_name: str
    source_chain: str
    target_chain: str
    asset_symbol: str
    asset_is_halal: bool
    mechanism: BridgeMechanism
    custodian_is_halal: bool
    has_proof_of_reserves: bool
    is_audited: bool
    interest_bearing_collateral: bool
    fractional_reserve_in_use: bool

    def __post_init__(self) -> None:
        if not self.bridge_name or not self.bridge_name.strip():
            raise ValueError("bridge_name must be non-empty")
        if not self.source_chain or not self.source_chain.strip():
            raise ValueError("source_chain must be non-empty")
        if not self.target_chain or not self.target_chain.strip():
            raise ValueError("target_chain must be non-empty")
        if self.source_chain == self.target_chain:
            raise ValueError("source_chain and target_chain must differ")
        if not self.asset_symbol or not self.asset_symbol.strip():
            raise ValueError("asset_symbol must be non-empty")


@dataclass(frozen=True)
class BridgeAssessment:
    """Result of running a bridge through the halal screen."""

    bridge_name: str
    asset_symbol: str
    issues: frozenset[BridgeIssue]
    is_compliant: bool

    def __post_init__(self) -> None:
        if self.is_compliant and self.issues:
            raise ValueError("is_compliant=True but issues non-empty")
        if (not self.is_compliant) and not self.issues:
            raise ValueError("is_compliant=False but issues empty")


def screen_bridge(
    inputs: BridgeInputs, *, policy: BridgePolicy | None = None
) -> BridgeAssessment:
    """Run the bridge through the halal screen."""
    pol = policy if policy is not None else BridgePolicy()
    issues: set[BridgeIssue] = set()

    if not inputs.asset_is_halal:
        issues.add(BridgeIssue.UNDERLYING_ASSET_NOT_HALAL)
    if inputs.interest_bearing_collateral:
        issues.add(BridgeIssue.INTEREST_BEARING_COLLATERAL)
    if inputs.fractional_reserve_in_use:
        issues.add(BridgeIssue.FRACTIONAL_RESERVE_RISK)
    if pol.block_liquidity_pool_models and inputs.mechanism is BridgeMechanism.LIQUIDITY_POOL:
        issues.add(BridgeIssue.FRACTIONAL_RESERVE_RISK)
    if pol.require_proof_of_reserves and not inputs.has_proof_of_reserves:
        issues.add(BridgeIssue.NO_PROOF_OF_RESERVES)
    if not inputs.custodian_is_halal:
        issues.add(BridgeIssue.CUSTODIAN_NOT_HALAL)
    if pol.require_audit and not inputs.is_audited:
        issues.add(BridgeIssue.UNAUDITED_BRIDGE_CONTRACT)

    return BridgeAssessment(
        bridge_name=inputs.bridge_name,
        asset_symbol=inputs.asset_symbol,
        issues=frozenset(issues),
        is_compliant=len(issues) == 0,
    )


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
    "wallet_address",
    "private_key",
    "0x",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_assessment(a: BridgeAssessment) -> str:
    emoji = "✅" if a.is_compliant else "❌"
    head = f"{emoji} bridge {a.bridge_name} ({a.asset_symbol})"
    lines = [head]
    for issue in sorted(a.issues, key=lambda x: x.value):
        lines.append(f"  • {issue.value}")
    return _scrub("\n".join(lines))
