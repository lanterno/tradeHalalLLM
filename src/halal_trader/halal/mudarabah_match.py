"""Trader-to-trader Mudarabah matching — Round-5 Wave 17.H.

Some users have capital but no time/skill to trade. Others have skill
but no capital. Mudarabah is the classical fiqh structure for this
arrangement: the capital provider (rabb-al-mal) places funds with a
manager (mudarib); profit is split per a pre-agreed ratio; loss is
borne by the capital provider only (the manager loses time/effort).

This module is the **matching engine**: capital + skill listings come
in, FIFO-matched against compatible counterparts. Capacity-aware so a
manager doesn't get over-allocated; minimum-amount-aware so dust
matches don't waste fees.

Pinned semantics:

- **Closed-set ListingType ladder** — CAPITAL_OFFER / SKILL_OFFER.
- **FIFO match** — oldest listing wins on tie.
- **Risk-tolerance compatibility** — match only if listings overlap
  (e.g. conservative capital won't match aggressive skill).
- **Capacity-aware** — a skill listing with capacity 5 can match up
  to 5 capital listings simultaneously.
- **Minimum match amount = $500** by default.
- **Profit-share ratio negotiated at listing time** — not at match;
  ratios must overlap by at least 5 pp.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — user IDs masked.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum


class ListingType(str, Enum):
    """Closed-set listing-type ladder."""

    CAPITAL_OFFER = "capital_offer"  # rabb-al-mal posting
    SKILL_OFFER = "skill_offer"  # mudarib posting


class RiskTolerance(str, Enum):
    """Closed-set risk-tolerance ladder. Listings match iff overlapping."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


# Compatibility table: capital provider (row) is OK with manager (col)?
# Conservative-capital can't tolerate aggressive-manager etc.
_RISK_COMPATIBLE: dict[tuple[RiskTolerance, RiskTolerance], bool] = {
    (RiskTolerance.CONSERVATIVE, RiskTolerance.CONSERVATIVE): True,
    (RiskTolerance.CONSERVATIVE, RiskTolerance.BALANCED): False,
    (RiskTolerance.CONSERVATIVE, RiskTolerance.AGGRESSIVE): False,
    (RiskTolerance.BALANCED, RiskTolerance.CONSERVATIVE): True,
    (RiskTolerance.BALANCED, RiskTolerance.BALANCED): True,
    (RiskTolerance.BALANCED, RiskTolerance.AGGRESSIVE): False,
    (RiskTolerance.AGGRESSIVE, RiskTolerance.CONSERVATIVE): True,
    (RiskTolerance.AGGRESSIVE, RiskTolerance.BALANCED): True,
    (RiskTolerance.AGGRESSIVE, RiskTolerance.AGGRESSIVE): True,
}


@dataclass(frozen=True)
class CapitalListing:
    """A user's capital offer (rabb-al-mal posting)."""

    listing_id: str
    user_id: str
    amount_usd: float
    risk_tolerance: RiskTolerance
    min_profit_share_for_capital: float
    """Floor on the rabb-al-mal's share of profit. e.g. 0.50."""
    listed_at: date
    horizon_months: int = 12

    def __post_init__(self) -> None:
        if not self.listing_id or not self.listing_id.strip():
            raise ValueError("listing_id must be non-empty")
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if self.amount_usd <= 0:
            raise ValueError("amount_usd must be positive")
        if not 0.0 < self.min_profit_share_for_capital < 1.0:
            raise ValueError("min_profit_share_for_capital must be in (0, 1)")
        if self.horizon_months <= 0:
            raise ValueError("horizon_months must be positive")


@dataclass(frozen=True)
class SkillListing:
    """A user's skill offer (mudarib posting)."""

    listing_id: str
    user_id: str
    capital_capacity_usd: float
    """Max capital this manager can deploy effectively."""
    risk_tolerance: RiskTolerance
    max_profit_share_for_capital: float
    """Ceiling on the rabb-al-mal's share — i.e. 1 - manager's-min."""
    listed_at: date
    track_record_score: float = 0.0
    """Operator-supplied 0-1 score. Used for tie-breaking but not
    matching."""
    horizon_months: int = 12

    def __post_init__(self) -> None:
        if not self.listing_id or not self.listing_id.strip():
            raise ValueError("listing_id must be non-empty")
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if self.capital_capacity_usd <= 0:
            raise ValueError("capital_capacity_usd must be positive")
        if not 0.0 < self.max_profit_share_for_capital < 1.0:
            raise ValueError("max_profit_share_for_capital must be in (0, 1)")
        if not 0.0 <= self.track_record_score <= 1.0:
            raise ValueError("track_record_score must be in [0, 1]")
        if self.horizon_months <= 0:
            raise ValueError("horizon_months must be positive")


@dataclass(frozen=True)
class MudarabahMatch:
    """Output of `match` — one matched pair."""

    capital_listing_id: str
    skill_listing_id: str
    capital_user_id: str
    skill_user_id: str
    matched_amount_usd: float
    agreed_profit_share_capital: float
    """Final negotiated rabb-al-mal share. The midpoint of overlap."""
    horizon_months: int


@dataclass(frozen=True)
class MatchRound:
    """Output of `run_match_round`."""

    matches: tuple[MudarabahMatch, ...]
    unmatched_capital: tuple[CapitalListing, ...]
    unmatched_skill: tuple[SkillListing, ...]


def is_risk_compatible(capital: RiskTolerance, skill: RiskTolerance) -> bool:
    """True iff a capital provider with `capital` tolerance can pair
    with a manager whose strategy risk is `skill`."""
    return _RISK_COMPATIBLE[(capital, skill)]


def profit_share_overlap(cap: CapitalListing, skill: SkillListing) -> float | None:
    """Return the negotiated share for capital (the midpoint of the
    overlap interval), or None if there's no overlap.

    Capital wants ≥ `min_profit_share_for_capital`.
    Skill caps capital at `max_profit_share_for_capital`.
    Overlap = [min_share, max_share].
    """
    lo = cap.min_profit_share_for_capital
    hi = skill.max_profit_share_for_capital
    if lo > hi:
        return None
    if hi - lo < 0.05:
        # Less than 5pp overlap — too narrow, surface as no-match.
        return None
    return (lo + hi) / 2


def can_match(
    cap: CapitalListing,
    skill: SkillListing,
    *,
    min_match_amount: float = 500.0,
) -> bool:
    """Quick predicate: would these two listings produce a viable match?"""
    if not is_risk_compatible(cap.risk_tolerance, skill.risk_tolerance):
        return False
    if profit_share_overlap(cap, skill) is None:
        return False
    if cap.amount_usd < min_match_amount:
        return False
    if skill.capital_capacity_usd < min_match_amount:
        return False
    if cap.horizon_months > skill.horizon_months:
        return False
    return True


def run_match_round(
    capital_listings: Iterable[CapitalListing],
    skill_listings: Iterable[SkillListing],
    *,
    min_match_amount: float = 500.0,
) -> MatchRound:
    """Run one matching round.

    FIFO over both sides:
    - Sort capital + skill listings by `listed_at` ascending.
    - For each capital listing (oldest first), find the first skill
      listing with remaining capacity that can_match.
    - Match an amount = min(capital, remaining_capacity).
    - Continue until exhausted.

    Capital listings cannot be split across multiple skill listings —
    matched amount = min(capital_amount, remaining_capacity). If
    capital > capacity, the *full* capital listing is consumed and the
    delta becomes unmatched (operator can re-list for the residual).
    """
    cap_sorted = sorted(capital_listings, key=lambda c: c.listed_at)
    skill_sorted = sorted(skill_listings, key=lambda s: s.listed_at)
    skill_remaining: dict[str, float] = {s.listing_id: s.capital_capacity_usd for s in skill_sorted}
    matches: list[MudarabahMatch] = []
    matched_capital_ids: set[str] = set()
    for cap in cap_sorted:
        for skill in skill_sorted:
            if skill_remaining[skill.listing_id] < min_match_amount:
                continue
            if not can_match(cap, skill, min_match_amount=min_match_amount):
                continue
            share = profit_share_overlap(cap, skill)
            assert share is not None
            amount = min(cap.amount_usd, skill_remaining[skill.listing_id])
            if amount < min_match_amount:
                continue
            matches.append(
                MudarabahMatch(
                    capital_listing_id=cap.listing_id,
                    skill_listing_id=skill.listing_id,
                    capital_user_id=cap.user_id,
                    skill_user_id=skill.user_id,
                    matched_amount_usd=amount,
                    agreed_profit_share_capital=share,
                    horizon_months=min(cap.horizon_months, skill.horizon_months),
                )
            )
            skill_remaining[skill.listing_id] -= amount
            matched_capital_ids.add(cap.listing_id)
            break
    unmatched_cap = tuple(c for c in cap_sorted if c.listing_id not in matched_capital_ids)
    unmatched_skill = tuple(
        s for s in skill_sorted if skill_remaining[s.listing_id] >= min_match_amount
    )
    return MatchRound(
        matches=tuple(matches),
        unmatched_capital=unmatched_cap,
        unmatched_skill=unmatched_skill,
    )


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


def render_round(match_round: MatchRound) -> str:
    """Operator-readable summary of one matching round."""
    head = (
        f"🤝 Match round: {len(match_round.matches)} matched, "
        f"{len(match_round.unmatched_capital)} unmatched capital, "
        f"{len(match_round.unmatched_skill)} unmatched skill"
    )
    lines = [head]
    for m in match_round.matches:
        lines.append(
            f"  • [{m.capital_listing_id}↔{m.skill_listing_id}] "
            f"{_mask(m.capital_user_id)} → {_mask(m.skill_user_id)}: "
            f"${m.matched_amount_usd:,.0f} @ "
            f"{m.agreed_profit_share_capital * 100:.2f}%/"
            f"{(1 - m.agreed_profit_share_capital) * 100:.2f}% "
            f"({m.horizon_months}mo)"
        )
    return "\n".join(lines)
