"""On-chain sukuk integration screen — Round-5 Wave 22.D.

On-chain sukuk tokens (e.g. Wahed-issued, Blossom Finance, Sukuk
Capital) represent shariah-compliant fixed-income on a blockchain.
They differ from off-chain sukuk in two load-bearing ways:

1. **Smart-contract terms** — the structure can encode rules
   (revenue distribution, default handling) directly in code; we
   need to verify those terms match AAOIFI Standard 17.
2. **On-chain provenance** — the issuer's wallet must be verifiable;
   the bot must check token contract is the canonical one.

This module ships the **on-chain-specific screen** that complements
``halal/aaoifi_standard_17.py`` (off-chain rule encoding).

Pinned semantics:

- **Closed-set OnChainIssue ladder** — 7 specific on-chain risks.
- **Composes with Standard 17 screen** — calls into existing
  `screen_sukuk` for the structural compliance check.
- **No-secret-leak pin** on render output — never includes wallet
  addresses or private keys.
"""

from __future__ import annotations

from dataclasses import dataclass

from halal_trader.halal.aaoifi_standard_17 import (
    SukukIssuanceInputs,
    SukukType,
    screen_sukuk,
)


class OnChainIssue(str, Enum := __import__("enum").Enum):
    """Closed-set on-chain-specific issues."""

    UNVERIFIED_CONTRACT = "unverified_contract"
    UPGRADEABLE_PROXY_RISK = "upgradeable_proxy_risk"
    NO_AUDIT = "no_audit"
    SHARIAH_BOARD_OFFCHAIN_ONLY = "shariah_board_offchain_only"
    ORACLE_DEPENDENCY = "oracle_dependency"
    SECONDARY_TRADING_VIA_AMM = "secondary_trading_via_amm"
    NON_HALAL_FALLBACK_ASSET = "non_halal_fallback_asset"


@dataclass(frozen=True)
class OnChainScreenPolicy:
    """Operator-tunable policy for on-chain sukuk screens."""

    require_audit: bool = True
    require_upgradeable_pause: bool = True
    block_on_oracle_dependency: bool = False  # operator's risk choice

    def __post_init__(self) -> None:
        # No numeric thresholds; just bool toggles
        pass


@dataclass(frozen=True)
class OnChainSukukInputs:
    """Inputs for an on-chain sukuk screen."""

    token_symbol: str
    chain: str
    structural: SukukIssuanceInputs
    contract_verified_on_explorer: bool
    is_upgradeable_proxy: bool
    has_third_party_audit: bool
    shariah_board_signature_on_chain: bool
    relies_on_external_oracle: bool
    secondary_via_amm: bool
    fallback_asset_is_halal: bool

    def __post_init__(self) -> None:
        if not self.token_symbol or not self.token_symbol.strip():
            raise ValueError("token_symbol must be non-empty")
        if not self.chain or not self.chain.strip():
            raise ValueError("chain must be non-empty")


@dataclass(frozen=True)
class OnChainAssessment:
    """Result of running an on-chain sukuk screen."""

    token_symbol: str
    chain: str
    structural_compliant: bool
    on_chain_issues: frozenset[OnChainIssue]
    is_compliant: bool


def screen_onchain(
    inputs: OnChainSukukInputs,
    *,
    policy: OnChainScreenPolicy | None = None,
) -> OnChainAssessment:
    """Run the on-chain screen + compose with Standard 17 structural check."""
    pol = policy if policy is not None else OnChainScreenPolicy()

    structural = screen_sukuk(inputs.structural)

    issues: set[OnChainIssue] = set()
    if not inputs.contract_verified_on_explorer:
        issues.add(OnChainIssue.UNVERIFIED_CONTRACT)
    if inputs.is_upgradeable_proxy and pol.require_upgradeable_pause:
        issues.add(OnChainIssue.UPGRADEABLE_PROXY_RISK)
    if not inputs.has_third_party_audit and pol.require_audit:
        issues.add(OnChainIssue.NO_AUDIT)
    if not inputs.shariah_board_signature_on_chain:
        issues.add(OnChainIssue.SHARIAH_BOARD_OFFCHAIN_ONLY)
    if inputs.relies_on_external_oracle and pol.block_on_oracle_dependency:
        issues.add(OnChainIssue.ORACLE_DEPENDENCY)
    if inputs.secondary_via_amm:
        # AMM trading of debt-only sukuk would re-introduce price
        # variation that violates Standard 17 cl. 5.1.8.
        if inputs.structural.sukuk_type in (SukukType.MURABAHA, SukukType.SALAM):
            issues.add(OnChainIssue.SECONDARY_TRADING_VIA_AMM)
    if not inputs.fallback_asset_is_halal:
        issues.add(OnChainIssue.NON_HALAL_FALLBACK_ASSET)

    is_compliant = structural.is_compliant and len(issues) == 0

    return OnChainAssessment(
        token_symbol=inputs.token_symbol,
        chain=inputs.chain,
        structural_compliant=structural.is_compliant,
        on_chain_issues=frozenset(issues),
        is_compliant=is_compliant,
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
    "0x",  # block hex addresses
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_assessment(a: OnChainAssessment) -> str:
    emoji = "✅" if a.is_compliant else "❌"
    head = f"{emoji} {a.token_symbol} on {a.chain}: structural={a.structural_compliant}"
    lines = [head]
    for issue in sorted(a.on_chain_issues, key=lambda x: x.value):
        lines.append(f"  • {issue.value}")
    return _scrub("\n".join(lines))
