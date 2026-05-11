"""Halal SPAC screen — Round-5 Wave 6.E.

Special-Purpose Acquisition Companies are usually structured with two
features that conflict with halal compliance:

1. **Trust holds interest-bearing T-bills** while waiting for a target.
   The interest is riba; SPACs that DON'T sweep the interest into the
   trust (or convert to halal money-market substitutes) fail the
   screen.
2. **Sponsor "promote" shares** (typically 20% of the post-merger
   equity for nominal capital) — this is permissible only if
   structured as a Mudarabah-style profit-share rather than a free
   warrant grant.

Additional flags:
- **Target sector** — alcohol / gambling / conventional banking / etc.
  rejected up-front.
- **Redemption rights** — investors must have redemption at issue
  price + ANY share of *halal* return; if the redemption pool earns
  interest, the redemption is contaminated.

This module is the **pre-screen + verdict**.

Pinned semantics:

- **Closed-set TrustHolding** — RIBA_BEARING_TBILLS / HALAL_MMF /
  GOLD_VAULT / SUKUK_BASKET / CASH_NO_INTEREST.
- **Closed-set SponsorPromoteKind** — FREE_WARRANT / MUDARABAH_SHARE /
  CAPITAL_PROPORTIONAL.
- **Closed-set SectorRestriction** — HARAM (auto-reject) / HALAL /
  AMBIGUOUS.
- **Closed-set Verdict** — APPROVED / FLAGGED / REJECTED.
- **REJECTED is sticky** — a single hard-fail flips the verdict.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class TrustHolding(str, Enum):
    """Closed-set ladder for how the SPAC trust holds capital."""

    RIBA_BEARING_TBILLS = "riba_bearing_tbills"
    HALAL_MMF = "halal_mmf"
    """Sharia-compliant money-market fund (sukuk-backed)."""
    GOLD_VAULT = "gold_vault"
    SUKUK_BASKET = "sukuk_basket"
    CASH_NO_INTEREST = "cash_no_interest"


_HARAM_TRUST: frozenset[TrustHolding] = frozenset({TrustHolding.RIBA_BEARING_TBILLS})


class SponsorPromoteKind(str, Enum):
    """Closed-set sponsor-promote structure ladder."""

    FREE_WARRANT = "free_warrant"
    """Sponsor receives shares for nominal capital → gharar / asymmetric upside."""
    MUDARABAH_SHARE = "mudarabah_share"
    """Profit-share aligned with capital — halal-compatible."""
    CAPITAL_PROPORTIONAL = "capital_proportional"
    """Sponsor invests proportionally → halal-clean."""


class SectorRestriction(str, Enum):
    """Closed-set sector ladder for the target."""

    HARAM = "haram"
    HALAL = "halal"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


class Verdict(str, Enum):
    """Closed-set verdict ladder."""

    APPROVED = "approved"
    FLAGGED = "flagged"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SPACProfile:
    """A SPAC's structural profile against the halal screen."""

    spac_id: str
    ticker: str
    trust_holding: TrustHolding
    sponsor_promote_kind: SponsorPromoteKind
    sponsor_promote_pct: float
    """Sponsor's share of post-merger equity. Typically 0.20."""
    target_sector: SectorRestriction
    redemption_returns_interest: bool
    """True if redemption pool earns riba interest before close."""
    redemption_at_issue_or_better: bool
    """True if redemption price ≥ issue price."""
    target_announced: bool = False

    def __post_init__(self) -> None:
        if not self.spac_id or not self.spac_id.strip():
            raise ValueError("spac_id must be non-empty")
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        if not 0.0 <= self.sponsor_promote_pct <= 0.50:
            raise ValueError("sponsor_promote_pct must be in [0, 0.50]")


@dataclass(frozen=True)
class ScreenResult:
    """Output of `screen_spac`."""

    spac_id: str
    verdict: Verdict
    failures: tuple[str, ...]
    flags: tuple[str, ...]


# Operator-tunable thresholds.
_DEFAULT_SPONSOR_PROMOTE_HARD = 0.25
_DEFAULT_SPONSOR_PROMOTE_FLAG = 0.20


def screen_spac(
    profile: SPACProfile,
    *,
    max_sponsor_promote_pct: float = _DEFAULT_SPONSOR_PROMOTE_HARD,
    flag_sponsor_promote_pct: float = _DEFAULT_SPONSOR_PROMOTE_FLAG,
) -> ScreenResult:
    """Apply the four-layer screen.

    Pinned: failures + flags are accumulated; the final verdict is
    REJECTED iff any failure fires; FLAGGED if no failures but ≥1 flag;
    else APPROVED.
    """
    if not 0.0 < max_sponsor_promote_pct <= 1.0:
        raise ValueError("max_sponsor_promote_pct must be in (0, 1]")
    if not 0.0 <= flag_sponsor_promote_pct <= max_sponsor_promote_pct:
        raise ValueError("flag_sponsor_promote_pct must be in [0, max_sponsor_promote_pct]")

    failures: list[str] = []
    flags: list[str] = []

    # 1. Trust holding.
    if profile.trust_holding in _HARAM_TRUST:
        failures.append(f"trust holds {profile.trust_holding.value} (riba)")

    # 2. Target sector.
    if profile.target_sector is SectorRestriction.HARAM:
        failures.append("target sector is HARAM (auto-reject)")
    elif profile.target_sector is SectorRestriction.AMBIGUOUS:
        flags.append("target sector is AMBIGUOUS — scholar review required")
    elif profile.target_sector is SectorRestriction.UNKNOWN:
        flags.append("target sector UNKNOWN — flag for operator review")

    # 3. Sponsor promote.
    if profile.sponsor_promote_kind is SponsorPromoteKind.FREE_WARRANT:
        failures.append("sponsor_promote_kind=FREE_WARRANT is gharar / asymmetric upside")
    if profile.sponsor_promote_pct > max_sponsor_promote_pct + 1e-9:
        failures.append(
            f"sponsor_promote_pct {profile.sponsor_promote_pct * 100:.2f}% "
            f"> hard cap {max_sponsor_promote_pct * 100:.0f}%"
        )
    elif profile.sponsor_promote_pct > flag_sponsor_promote_pct + 1e-9:
        flags.append(
            f"sponsor_promote_pct {profile.sponsor_promote_pct * 100:.2f}% "
            f"in flag band ({flag_sponsor_promote_pct * 100:.0f}–"
            f"{max_sponsor_promote_pct * 100:.0f}%)"
        )

    # 4. Redemption.
    if profile.redemption_returns_interest:
        failures.append("redemption pool earns riba")
    if not profile.redemption_at_issue_or_better:
        failures.append("redemption price < issue price")

    if failures:
        verdict = Verdict.REJECTED
    elif flags:
        verdict = Verdict.FLAGGED
    else:
        verdict = Verdict.APPROVED
    return ScreenResult(
        spac_id=profile.spac_id,
        verdict=verdict,
        failures=tuple(failures),
        flags=tuple(flags),
    )


def screen_batch(
    profiles: Iterable[SPACProfile],
    **kwargs: float,
) -> tuple[ScreenResult, ...]:
    return tuple(screen_spac(p, **kwargs) for p in profiles)


def filter_approved(
    profiles: Iterable[SPACProfile],
) -> tuple[SPACProfile, ...]:
    return tuple(p for p in profiles if screen_spac(p).verdict is Verdict.APPROVED)


_VERDICT_EMOJI: dict[Verdict, str] = {
    Verdict.APPROVED: "✅",
    Verdict.FLAGGED: "🟡",
    Verdict.REJECTED: "❌",
}


def render_result(result: ScreenResult) -> str:
    head = f"{_VERDICT_EMOJI[result.verdict]} {result.spac_id}: {result.verdict.value}"
    if result.failures:
        head += f" ({len(result.failures)} fail)"
    if result.flags:
        head += f" ({len(result.flags)} flag)"
    lines = [head]
    for f in result.failures:
        lines.append(f"  ❌ {f}")
    for f in result.flags:
        lines.append(f"  🟡 {f}")
    return "\n".join(lines)
