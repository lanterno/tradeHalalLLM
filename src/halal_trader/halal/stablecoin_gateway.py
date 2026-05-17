"""Halal stablecoin gateway — Round-5 Wave 22.A.

Stablecoins back their peg with reserves; the **nature of the reserve**
determines halal compliance:

- **Gold-backed** (PAXG, XAUT) — backed by physical gold. Permissible
  if reserves are genuinely held + audited + the issuer doesn't
  earn riba on cash buffers.
- **USD-backed with T-bills** (USDC, USDT) — backed primarily by
  short-term US Treasuries that pay interest. Operator earns implicit
  riba via the issuer's yield. Standard scholar opinion: not
  permissible unless held briefly for transactional necessity, with
  the riba portion purified.
- **Crypto-collateralised** (DAI) — backed by other crypto + has
  stability-fee mechanism resembling interest. Mostly impermissible.
- **Algorithmic** (collapsed UST etc.) — high gharar (excessive
  uncertainty). Not permissible.
- **Salam-based** (synthetic-halal proposals) — emerging class;
  permissible if structured per Salam rules.

This module ships the **stablecoin halal screen**.

Pinned semantics:

- **Closed-set BackingType ladder** — 6 types.
- **Closed-set GatewayDecision ladder** — APPROVED / TRANSACTIONAL_ONLY
  / BLOCKED.
- **TRANSACTIONAL_ONLY** is a constrained allow: hold for ≤
  `max_hold_hours`, with riba-purification for any yield while held.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BackingType(str, Enum):
    """Closed-set stablecoin backing types."""

    GOLD = "gold"
    SILVER = "silver"
    USD_TBILL = "usd_tbill"
    CRYPTO_COLLATERAL = "crypto_collateral"
    ALGORITHMIC = "algorithmic"
    SALAM_BASED = "salam_based"


class GatewayDecision(str, Enum):
    """Closed-set gateway decisions."""

    APPROVED = "approved"
    TRANSACTIONAL_ONLY = "transactional_only"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class StablecoinPolicy:
    """Operator-tunable policy."""

    allow_transactional_use: bool = True
    max_transactional_hold_hours: float = 24.0
    require_attestation: bool = True
    require_segregated_reserves: bool = True

    def __post_init__(self) -> None:
        if self.max_transactional_hold_hours <= 0:
            raise ValueError("max_transactional_hold_hours must be positive")


@dataclass(frozen=True)
class StablecoinInputs:
    """Inputs for screening a stablecoin."""

    coin_symbol: str
    backing_type: BackingType
    issuer: str
    has_third_party_attestation: bool
    reserves_segregated: bool
    issuer_earns_riba_on_cash: bool

    def __post_init__(self) -> None:
        if not self.coin_symbol or not self.coin_symbol.strip():
            raise ValueError("coin_symbol must be non-empty")
        if not self.issuer or not self.issuer.strip():
            raise ValueError("issuer must be non-empty")


@dataclass(frozen=True)
class GatewayAssessment:
    """Result of running a stablecoin through the gateway."""

    coin_symbol: str
    backing_type: BackingType
    decision: GatewayDecision
    reasons: tuple[str, ...]


def screen(
    inputs: StablecoinInputs,
    *,
    policy: StablecoinPolicy | None = None,
) -> GatewayAssessment:
    """Run the stablecoin through the halal gateway."""
    pol = policy if policy is not None else StablecoinPolicy()
    reasons: list[str] = []

    # Backing-type ladder
    if inputs.backing_type is BackingType.ALGORITHMIC:
        reasons.append("algorithmic stablecoin: excessive gharar")
        return GatewayAssessment(
            coin_symbol=inputs.coin_symbol,
            backing_type=inputs.backing_type,
            decision=GatewayDecision.BLOCKED,
            reasons=tuple(reasons),
        )

    if inputs.backing_type is BackingType.CRYPTO_COLLATERAL:
        reasons.append("crypto-collateralised: stability fee resembles riba")
        return GatewayAssessment(
            coin_symbol=inputs.coin_symbol,
            backing_type=inputs.backing_type,
            decision=GatewayDecision.BLOCKED,
            reasons=tuple(reasons),
        )

    # USD-T-bill backed → transactional-only at best
    if inputs.backing_type is BackingType.USD_TBILL:
        if pol.allow_transactional_use:
            reasons.append("USD-T-bill backed: transactional use only with riba purification")
            return GatewayAssessment(
                coin_symbol=inputs.coin_symbol,
                backing_type=inputs.backing_type,
                decision=GatewayDecision.TRANSACTIONAL_ONLY,
                reasons=tuple(reasons),
            )
        reasons.append("USD-T-bill backed: operator policy disallows transactional use")
        return GatewayAssessment(
            coin_symbol=inputs.coin_symbol,
            backing_type=inputs.backing_type,
            decision=GatewayDecision.BLOCKED,
            reasons=tuple(reasons),
        )

    # Gold / Silver / Salam-based — operationally clean if attestation + segregation pass
    if pol.require_attestation and not inputs.has_third_party_attestation:
        reasons.append("third-party attestation required but not present")
    if pol.require_segregated_reserves and not inputs.reserves_segregated:
        reasons.append("reserves are commingled, not segregated")
    if inputs.issuer_earns_riba_on_cash:
        reasons.append("issuer earns riba on cash buffer")

    if reasons:
        # Fail soft: block if backing requires attestation + missing
        return GatewayAssessment(
            coin_symbol=inputs.coin_symbol,
            backing_type=inputs.backing_type,
            decision=GatewayDecision.BLOCKED,
            reasons=tuple(reasons),
        )
    reasons.append(f"{inputs.backing_type.value}-backed with attestation")
    return GatewayAssessment(
        coin_symbol=inputs.coin_symbol,
        backing_type=inputs.backing_type,
        decision=GatewayDecision.APPROVED,
        reasons=tuple(reasons),
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
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_assessment(a: GatewayAssessment) -> str:
    emoji = {
        GatewayDecision.APPROVED: "✅",
        GatewayDecision.TRANSACTIONAL_ONLY: "🟡",
        GatewayDecision.BLOCKED: "❌",
    }[a.decision]
    head = f"{emoji} {a.coin_symbol} ({a.backing_type.value}) → {a.decision.value}"
    lines = [head]
    for r in a.reasons:
        lines.append(f"  • {r}")
    return _scrub("\n".join(lines))
