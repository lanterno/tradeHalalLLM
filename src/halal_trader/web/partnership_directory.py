"""Halal-fintech partnership directory + integration-readiness engine.

The roadmap pins Wave 10.G: "Strategic integrations with: Wahed
Invest (managed accounts), Aghaz (robo-advisor), Amana Mutual Funds.
They send users; we provide high-frequency / active-management
capability they don't." This module is the **pure-Python directory
+ integration-readiness aggregator** the BD operator consults to
track each prospective partner's profile, their complementary
capabilities, and progression through the integration handshake.

Picked a focused engine over a "spreadsheet of partners" approach
because (a) the partnership funnel has a strict ordering (a partner
moves from `INITIAL_OUTREACH` → `MUTUAL_INTEREST` → `SCOPE_ALIGNED`
→ `LEGAL_REVIEW` → `INTEGRATION_BUILD` → `LIVE`), and an opaque
spreadsheet drifts; encoding the stages once means dashboards +
exports + reports all consult the same source of truth, (b) the
"complementarity" check (do we provide active-management capability
they lack?) is a pure function of (their_capabilities, our_capabilities)
— pinning it in code lets the operator quickly screen "is this
partner worth pursuing or do we duplicate?" rather than re-deriving
the analysis per partner, (c) the no-secret render contract matters
because partner profiles travel over BD email + Slack to investors;
the operator's audit-trail must not leak partner-side proprietary
information (revenue figures, internal contact emails, strategic
docs).

Pinned semantics:
- **Stages complete in canonical order; can't skip ahead.** A
  partner can't move to LEGAL_REVIEW without being SCOPE_ALIGNED
  first; pinned via `advance_stage` raising on out-of-order.
- **Partner capabilities are a closed enum.** Adding a new
  capability is a code review change so the complementarity-score
  math doesn't drift silently when a contributor adds a free-form
  string.
- **Halal certification level is a closed ladder.** Five levels
  from NONE through SHARIAH_BOARD_CERTIFIED — operators sort
  partners by certification level for due diligence prioritisation.
- **Partner can be flagged inactive without losing audit history.**
  An inactive partner is excluded from active-funnel views but
  the audit trail (stage transitions, dates, notes) is preserved
  so a partnership that fizzled can be revived later.
- **Render output never includes partner internal contact info,
  revenue, or proprietary docs.** Mirrors no-secret patterns of
  upstream waves; the partner profile carries only public-facing
  fields (display name, public website, capabilities, halal cert).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Capability(str, Enum):
    """Closed-set partner / our-side capabilities for complementarity scoring.

    Pinned string values; adding a capability is a code review change.
    """

    MANAGED_PORTFOLIOS = "managed_portfolios"
    ROBO_ADVISOR = "robo_advisor"
    MUTUAL_FUNDS = "mutual_funds"
    ACTIVE_MANAGEMENT = "active_management"
    HIGH_FREQUENCY_TRADING = "high_frequency_trading"
    HALAL_SCREENING = "halal_screening"
    PURIFICATION_LEDGER = "purification_ledger"
    LLM_REASONING = "llm_reasoning"
    BACKTESTING = "backtesting"
    BROKER_API = "broker_api"
    USER_BASE = "user_base"


# Our-side capabilities — what halal-trader brings to a partnership.
OUR_CAPABILITIES: frozenset[Capability] = frozenset(
    {
        Capability.ACTIVE_MANAGEMENT,
        Capability.HIGH_FREQUENCY_TRADING,
        Capability.HALAL_SCREENING,
        Capability.PURIFICATION_LEDGER,
        Capability.LLM_REASONING,
        Capability.BACKTESTING,
    }
)


class HalalCertLevel(str, Enum):
    """Closed-ladder halal certification levels.

    Pinned string values + ordering: NONE < SELF_DECLARED <
    THIRD_PARTY_AUDITED < SCHOLAR_REVIEWED < SHARIAH_BOARD_CERTIFIED.
    """

    NONE = "none"
    SELF_DECLARED = "self_declared"
    THIRD_PARTY_AUDITED = "third_party_audited"
    SCHOLAR_REVIEWED = "scholar_reviewed"
    SHARIAH_BOARD_CERTIFIED = "shariah_board_certified"


_CERT_ORDER: dict[HalalCertLevel, int] = {
    HalalCertLevel.NONE: 0,
    HalalCertLevel.SELF_DECLARED: 1,
    HalalCertLevel.THIRD_PARTY_AUDITED: 2,
    HalalCertLevel.SCHOLAR_REVIEWED: 3,
    HalalCertLevel.SHARIAH_BOARD_CERTIFIED: 4,
}


class IntegrationStage(str, Enum):
    """Partnership funnel stages, in canonical order.

    Pinned string values. The ordering is hard-pinned via
    `_STAGE_ORDER` below — a stage skip surfaces as
    `StageOutOfOrderError`.
    """

    INITIAL_OUTREACH = "initial_outreach"
    MUTUAL_INTEREST = "mutual_interest"
    SCOPE_ALIGNED = "scope_aligned"
    LEGAL_REVIEW = "legal_review"
    INTEGRATION_BUILD = "integration_build"
    LIVE = "live"
    PAUSED = "paused"  # off-funnel terminal; can revive to a prior stage


_STAGE_ORDER: tuple[IntegrationStage, ...] = (
    IntegrationStage.INITIAL_OUTREACH,
    IntegrationStage.MUTUAL_INTEREST,
    IntegrationStage.SCOPE_ALIGNED,
    IntegrationStage.LEGAL_REVIEW,
    IntegrationStage.INTEGRATION_BUILD,
    IntegrationStage.LIVE,
)


class StageOutOfOrderError(Exception):
    """Raised when an advance_stage call skips a prerequisite stage."""

    def __init__(
        self,
        from_stage: IntegrationStage,
        to_stage: IntegrationStage,
    ) -> None:
        super().__init__(f"cannot advance from {from_stage.value} to {to_stage.value}")
        self.from_stage = from_stage
        self.to_stage = to_stage


@dataclass(frozen=True)
class StageTransition:
    """Audit row for a single stage transition."""

    from_stage: IntegrationStage | None  # None = first stage entry
    to_stage: IntegrationStage
    decided_at: datetime
    notes: str = ""

    def __post_init__(self) -> None:
        if self.decided_at.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")


@dataclass(frozen=True)
class Partner:
    """One partner profile.

    Carries only public-facing fields: display_name + public_url +
    capabilities + halal_cert_level + current_stage + transitions
    + active flag. Specifically does NOT carry: internal contact
    emails, revenue figures, internal Slack channels, NDA-protected
    docs. The no-secret-leak contract is structural.
    """

    partner_id: str
    display_name: str
    public_url: str
    capabilities: frozenset[Capability]
    halal_cert_level: HalalCertLevel
    current_stage: IntegrationStage
    transitions: tuple[StageTransition, ...]
    active: bool = True

    def __post_init__(self) -> None:
        if not self.partner_id or not self.partner_id.strip():
            raise ValueError("partner_id must be non-empty")
        if not self.display_name or not self.display_name.strip():
            raise ValueError("display_name must be non-empty")
        if not self.public_url or not self.public_url.strip():
            raise ValueError("public_url must be non-empty")
        if not (self.public_url.startswith("https://") or self.public_url.startswith("http://")):
            raise ValueError(f"public_url {self.public_url!r} must be http(s)://")


def _canonical_index(stage: IntegrationStage) -> int | None:
    """Return canonical-order index, or None for off-funnel (PAUSED)."""

    if stage is IntegrationStage.PAUSED:
        return None
    return _STAGE_ORDER.index(stage)


def create_partner(
    *,
    partner_id: str,
    display_name: str,
    public_url: str,
    capabilities: Iterable[Capability],
    halal_cert_level: HalalCertLevel,
    now: datetime,
) -> Partner:
    """Create a fresh partner profile at INITIAL_OUTREACH stage."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    initial = StageTransition(
        from_stage=None,
        to_stage=IntegrationStage.INITIAL_OUTREACH,
        decided_at=now,
    )
    return Partner(
        partner_id=partner_id,
        display_name=display_name,
        public_url=public_url,
        capabilities=frozenset(capabilities),
        halal_cert_level=halal_cert_level,
        current_stage=IntegrationStage.INITIAL_OUTREACH,
        transitions=(initial,),
    )


def advance_stage(
    partner: Partner,
    to_stage: IntegrationStage,
    *,
    now: datetime,
    notes: str = "",
) -> Partner:
    """Advance the partner to a new stage with order enforcement.

    Forward moves must be one stage at a time along the canonical
    order. PAUSED can be entered from any stage. Returning from
    PAUSED re-enters at any prior stage (operator's call where to
    pick up).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if to_stage is partner.current_stage:
        raise ValueError(f"already at {to_stage.value}")

    if to_stage is IntegrationStage.PAUSED:
        # PAUSED can be entered from anywhere
        pass
    elif partner.current_stage is IntegrationStage.PAUSED:
        # Returning from PAUSED — operator picks any non-PAUSED stage
        if to_stage is IntegrationStage.PAUSED:
            raise ValueError("already paused")
    else:
        # Forward funnel move — must be exactly the next canonical stage
        cur_idx = _canonical_index(partner.current_stage)
        new_idx = _canonical_index(to_stage)
        if cur_idx is None or new_idx is None:
            raise ValueError("invalid funnel transition")
        if new_idx != cur_idx + 1:
            raise StageOutOfOrderError(partner.current_stage, to_stage)

    transition = StageTransition(
        from_stage=partner.current_stage,
        to_stage=to_stage,
        decided_at=now,
        notes=notes,
    )
    return Partner(
        partner_id=partner.partner_id,
        display_name=partner.display_name,
        public_url=partner.public_url,
        capabilities=partner.capabilities,
        halal_cert_level=partner.halal_cert_level,
        current_stage=to_stage,
        transitions=partner.transitions + (transition,),
        active=partner.active,
    )


def deactivate(partner: Partner, *, now: datetime, notes: str = "") -> Partner:
    """Mark a partner inactive (preserves audit history)."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return Partner(
        partner_id=partner.partner_id,
        display_name=partner.display_name,
        public_url=partner.public_url,
        capabilities=partner.capabilities,
        halal_cert_level=partner.halal_cert_level,
        current_stage=partner.current_stage,
        transitions=partner.transitions,
        active=False,
    )


def complementarity_score(
    partner: Partner,
    *,
    our_capabilities: frozenset[Capability] = OUR_CAPABILITIES,
) -> float:
    """Return [0.0, 1.0] complementarity: fraction of partner's
    capabilities we DON'T have plus fraction of our capabilities the
    partner DOESN'T have, normalised by total unique capabilities.

    A score of 1.0 means perfectly disjoint (everything they have
    is what we lack and vice versa); 0.0 means perfectly redundant
    (we have all the same capabilities). Operators target high-
    complementarity partners — they have user reach we lack and
    we have execution capabilities they lack.
    """

    if not partner.capabilities and not our_capabilities:
        return 0.0
    union = partner.capabilities | our_capabilities
    intersection = partner.capabilities & our_capabilities
    return 1.0 - (len(intersection) / len(union))


def filter_active(partners: Iterable[Partner]) -> tuple[Partner, ...]:
    """Return only the active partners (preserves order)."""

    return tuple(p for p in partners if p.active)


def filter_at_stage(partners: Iterable[Partner], stage: IntegrationStage) -> tuple[Partner, ...]:
    return tuple(p for p in partners if p.current_stage is stage)


def cert_meets_minimum(partner: Partner, *, minimum: HalalCertLevel) -> bool:
    """True if partner's halal cert is >= minimum on the closed ladder."""

    return _CERT_ORDER[partner.halal_cert_level] >= _CERT_ORDER[minimum]


@dataclass(frozen=True)
class PartnershipFunnel:
    """Aggregate funnel view-model.

    Counts per stage + complementarity-weighted live count.
    """

    counts_by_stage: tuple[tuple[IntegrationStage, int], ...]
    total_active: int
    total_live: int

    def count_at(self, stage: IntegrationStage) -> int:
        for s, n in self.counts_by_stage:
            if s is stage:
                return n
        return 0


def build_funnel(partners: Iterable[Partner]) -> PartnershipFunnel:
    """Aggregate per-stage counts (active partners only)."""

    active = list(filter_active(partners))
    counts: dict[IntegrationStage, int] = {s: 0 for s in IntegrationStage}
    for p in active:
        counts[p.current_stage] += 1
    breakdown = tuple((stage, counts[stage]) for stage in IntegrationStage)
    return PartnershipFunnel(
        counts_by_stage=breakdown,
        total_active=len(active),
        total_live=counts[IntegrationStage.LIVE],
    )


_STAGE_EMOJI: dict[IntegrationStage, str] = {
    IntegrationStage.INITIAL_OUTREACH: "📨",
    IntegrationStage.MUTUAL_INTEREST: "🤝",
    IntegrationStage.SCOPE_ALIGNED: "📋",
    IntegrationStage.LEGAL_REVIEW: "⚖️",
    IntegrationStage.INTEGRATION_BUILD: "🔨",
    IntegrationStage.LIVE: "✅",
    IntegrationStage.PAUSED: "⏸️",
}


_CERT_EMOJI: dict[HalalCertLevel, str] = {
    HalalCertLevel.NONE: "❓",
    HalalCertLevel.SELF_DECLARED: "📝",
    HalalCertLevel.THIRD_PARTY_AUDITED: "🔍",
    HalalCertLevel.SCHOLAR_REVIEWED: "📚",
    HalalCertLevel.SHARIAH_BOARD_CERTIFIED: "🕌",
}


def render_partner(partner: Partner) -> str:
    """Format a partner profile for ops display.

    Pinned no-secret-leak: never includes internal contact emails /
    revenue figures / NDA docs. Shows only public fields.
    """

    stage_emoji = _STAGE_EMOJI[partner.current_stage]
    cert_emoji = _CERT_EMOJI[partner.halal_cert_level]
    inactive_marker = " 🚫inactive" if not partner.active else ""
    score = complementarity_score(partner)
    cap_list = ", ".join(sorted(c.value for c in partner.capabilities))
    lines = [
        f"{stage_emoji}{cert_emoji} {partner.display_name} ({partner.partner_id}){inactive_marker}",
        f"  url: {partner.public_url}",
        f"  stage: {partner.current_stage.value}",
        f"  cert: {partner.halal_cert_level.value}",
        f"  complementarity: {score:.0%}",
        f"  capabilities: {cap_list}",
    ]
    return "\n".join(lines)


def render_funnel(funnel: PartnershipFunnel) -> str:
    """Format the funnel view-model."""

    lines = [f"🤝 Partnership funnel — {funnel.total_active} active, {funnel.total_live} live"]
    for stage, count in funnel.counts_by_stage:
        if count == 0:
            continue
        emoji = _STAGE_EMOJI[stage]
        lines.append(f"  {emoji} {stage.value}: {count}")
    return "\n".join(lines)


__all__ = [
    "OUR_CAPABILITIES",
    "Capability",
    "HalalCertLevel",
    "IntegrationStage",
    "Partner",
    "PartnershipFunnel",
    "StageOutOfOrderError",
    "StageTransition",
    "advance_stage",
    "build_funnel",
    "cert_meets_minimum",
    "complementarity_score",
    "create_partner",
    "deactivate",
    "filter_active",
    "filter_at_stage",
    "render_funnel",
    "render_partner",
]
