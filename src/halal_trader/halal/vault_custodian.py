"""Halal vault custodian screen — Round-5 Wave 5.A.

Spot trading of physical gold + silver requires a vault custodian
who: (a) holds the physical metal, not paper; (b) does not lend the
metal out for interest; (c) provides audited proof of holdings; (d)
permits constructive possession transfer at the time of trade per
AAOIFI Standard 38 + Standard 30.

This module ships the **custodian screen**.

Pinned semantics:

- **Closed-set CustodianTier ladder** — TIER_1 (best — full audit +
  segregation + AAOIFI compliance) → TIER_4 (rejected).
- **Closed-set VaultIssue ladder** — 6 specific issues.
- **`screen_custodian`** is pure.
- **No-secret-leak pin** — never includes vault address.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CustodianTier(str, Enum):
    """Closed-set custodian tiers."""

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    REJECTED = "rejected"


class VaultIssue(str, Enum):
    """Closed-set vault halal issues."""

    NO_AUDIT = "no_audit"
    METAL_LENT_OUT = "metal_lent_out"
    PAPER_BACKED_NOT_PHYSICAL = "paper_backed_not_physical"
    NOT_SEGREGATED = "not_segregated"
    NO_CONSTRUCTIVE_POSSESSION = "no_constructive_possession"
    INTEREST_BEARING_CASH_BUFFER = "interest_bearing_cash_buffer"


@dataclass(frozen=True)
class CustodianPolicy:
    """Operator-tunable thresholds."""

    require_aaoifi_certification: bool = False  # nice-to-have, not hard requirement

    def __post_init__(self) -> None:
        pass


@dataclass(frozen=True)
class VaultInputs:
    """Inputs for a custodian screen."""

    custodian_name: str
    jurisdiction: str
    metal: str  # "gold" / "silver" / etc.
    has_independent_audit: bool
    metal_lent_for_interest: bool
    fully_physical_backed: bool
    holdings_segregated_per_client: bool
    permits_constructive_possession: bool
    interest_bearing_cash_buffer: bool
    aaoifi_certified: bool

    def __post_init__(self) -> None:
        if not self.custodian_name or not self.custodian_name.strip():
            raise ValueError("custodian_name must be non-empty")
        if not self.jurisdiction or not self.jurisdiction.strip():
            raise ValueError("jurisdiction must be non-empty")
        if not self.metal or not self.metal.strip():
            raise ValueError("metal must be non-empty")


@dataclass(frozen=True)
class CustodianAssessment:
    """Result of running a custodian screen."""

    custodian_name: str
    metal: str
    issues: frozenset[VaultIssue]
    tier: CustodianTier


def screen_custodian(
    inputs: VaultInputs, *, policy: CustodianPolicy | None = None
) -> CustodianAssessment:
    """Run the custodian screen + classify into a tier."""
    pol = policy if policy is not None else CustodianPolicy()
    issues: set[VaultIssue] = set()

    if not inputs.has_independent_audit:
        issues.add(VaultIssue.NO_AUDIT)
    if inputs.metal_lent_for_interest:
        issues.add(VaultIssue.METAL_LENT_OUT)
    if not inputs.fully_physical_backed:
        issues.add(VaultIssue.PAPER_BACKED_NOT_PHYSICAL)
    if not inputs.holdings_segregated_per_client:
        issues.add(VaultIssue.NOT_SEGREGATED)
    if not inputs.permits_constructive_possession:
        issues.add(VaultIssue.NO_CONSTRUCTIVE_POSSESSION)
    if inputs.interest_bearing_cash_buffer:
        issues.add(VaultIssue.INTEREST_BEARING_CASH_BUFFER)

    # Tier laddering
    n_issues = len(issues)
    if n_issues == 0 and (not pol.require_aaoifi_certification or inputs.aaoifi_certified):
        tier = CustodianTier.TIER_1
    elif n_issues <= 1 and not (
        VaultIssue.METAL_LENT_OUT in issues or VaultIssue.PAPER_BACKED_NOT_PHYSICAL in issues
    ):
        tier = CustodianTier.TIER_2
    elif n_issues <= 3 and VaultIssue.METAL_LENT_OUT not in issues:
        tier = CustodianTier.TIER_3
    else:
        tier = CustodianTier.REJECTED

    return CustodianAssessment(
        custodian_name=inputs.custodian_name,
        metal=inputs.metal,
        issues=frozenset(issues),
        tier=tier,
    )


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
    "vault_address",
    "serial_number",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_assessment(a: CustodianAssessment) -> str:
    emoji = {
        CustodianTier.TIER_1: "🥇",
        CustodianTier.TIER_2: "🥈",
        CustodianTier.TIER_3: "🥉",
        CustodianTier.REJECTED: "❌",
    }[a.tier]
    head = f"{emoji} {a.custodian_name} ({a.metal}) → {a.tier.value}"
    lines = [head]
    for issue in sorted(a.issues, key=lambda x: x.value):
        lines.append(f"  • {issue.value}")
    return _scrub("\n".join(lines))
