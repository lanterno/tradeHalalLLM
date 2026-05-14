"""Halal venture-capital allocation gate.

For users participating in halal private-market opportunities
(Wahed Ventures, Curate Capital's halal funds, GCC sovereign-
adjacent halal VCs), the bot needs a deal-screening + accredited-
investor + concentration gate before any allocation. The roadmap
defers full integration to a follow-up — actually pulling deal
flow from a partner like Wahed Ventures requires an API contract
that doesn't exist yet — but the **screening logic** is operator-
supplied or CSV-fed pure-Python, exactly the isolated-module
pattern of Wave 1.G commodities, 1.H sukuk, 1.I REIT, 2.G
regulator-index, and 12.A robo-advisor.

VC has rules that public-equity screeners don't catch:

- Accredited-investor jurisdiction varies (US: SEC Reg D
  $1M / $200k; UK: HNW / sophisticated; UAE: VARA-defined
  qualified investor).
- Lockup periods up to 10+ years are common; user MUST be
  warned and consent before allocating to long-lockup
  positions — illiquidity is a feature not a bug.
- Founder / management's shariah credentials matter even when
  the operating business is halal — a tobacco startup with a
  halal-supply-chain founder doesn't suddenly become halal.
- Use of proceeds tracking: a halal-business that uses raised
  capital to retire interest-bearing debt is funding riba via
  the back door.
- Sector compliance is broader than public equity (early-stage
  startups frequently pivot; the screener has to capture both
  current and stated future sector).
- Concentration limits matter more than for liquid public
  positions — a 30% portfolio bet on an illiquid 7-year-lockup
  deal is a different risk profile.

This module is the deal-screen + concentration-gate; the
accredited-investor check delegates to the Wave 11.C KYC
state engine (HIGH risk + EDD layered on top per FATF-aligned
private-market rules).

Pinned semantics:
- **Closed sector enum.** `HalalSector` lists every sector the
  screener accepts — the same closed-set guarantee as the Wave
  12.A robo-advisor closed asset enum. Conventional banking,
  alcohol, gambling, weapons, adult industries, pork-related,
  tobacco are all categorically absent — the engine can't
  approve them because the type system rejects them. Adding a
  sector is a code-review change.
- **DOUBTFUL_PIVOT for early-stage startups.** Pre-product /
  seed-stage startups frequently pivot; a "halal SaaS" that's
  pre-revenue could genuinely become an "ad network for adult
  content" by Series A. The engine flags pre-product deals as
  DOUBTFUL even when the stated sector is halal — operators
  consult their scholar profile via Wave 2.F scholar-review
  for these.
- **Use-of-proceeds disclosure required.** A deal whose `use_
  of_proceeds` field is empty or undisclosed lands in
  INSUFFICIENT_DATA, not silent HALAL — pinned because operators
  need to verify the proceeds aren't going to retire conventional
  debt (riba via the back door) or finance haram operations.
- **Per-deal concentration cap default 10%.** Operators can
  bump per-deal cap up to 25% (a high-conviction allocation)
  but the engine refuses > 50% by construction — putting half
  a portfolio in one illiquid 7-year-lockup deal is a category
  error.
- **Lockup disclosure required.** Every deal carries explicit
  `lockup_years`; operators can't silently allocate to an
  undisclosed-lockup deal because the user-consent flow needs
  the number to render the warning.
- **Render output never includes user portfolio totals.** The
  receipt summarises action + verdict + concentration %, never
  the absolute user balance — mirrors the no-USD pattern of
  Wave 12.A robo-advisor + Wave 11.D privacy + Wave 3.B vault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class HalalSector(str, Enum):
    """Halal-compliant VC sector categories.

    Closed set: a future contributor adding a new sector needs to
    (a) prove halal compliance, (b) extend the enum. The
    structural friction prevents accidentally allowing a non-halal
    sector at runtime via config.
    """

    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    AGRITECH = "agritech"
    CLEAN_ENERGY = "clean_energy"
    HALAL_FINTECH = "halal_fintech"
    SAAS_B2B = "saas_b2b"
    LOGISTICS = "logistics"
    HALAL_FOOD = "halal_food"
    MODEST_FASHION = "modest_fashion"
    PROPTECH = "proptech"
    DEVELOPER_TOOLS = "developer_tools"
    BIOTECH = "biotech"


# Sectors that need extra scrutiny because they often touch
# adjacent non-halal verticals — not auto-rejected, but flagged.
_SCRUTINY_SECTORS: frozenset[HalalSector] = frozenset(
    {
        HalalSector.HALAL_FINTECH,  # adjacent to conventional banking
        HalalSector.HALAL_FOOD,  # adjacent to alcohol / pork
        HalalSector.MODEST_FASHION,  # adjacent to mainstream apparel
    }
)


class DealStage(str, Enum):
    """Funding stage.

    `PRE_SEED` and `SEED` are flagged DOUBTFUL_PIVOT because
    pre-product startups frequently pivot business model. `SERIES_A`
    onward typically have a defined product and revenue stream.
    """

    PRE_SEED = "pre_seed"
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C = "series_c"
    GROWTH = "growth"
    PRE_IPO = "pre_ipo"


_PRE_PRODUCT_STAGES: frozenset[DealStage] = frozenset({DealStage.PRE_SEED, DealStage.SEED})


class UseOfProceeds(str, Enum):
    """How the raised capital will be deployed.

    Pinned: `RETIRE_DEBT` is the classic riba-via-back-door case
    — a halal business that uses raised equity to pay down
    interest-bearing debt is funding riba. Categorical NOT_HALAL.
    `UNDISCLOSED` returns INSUFFICIENT_DATA — operator must
    verify before allocating.
    """

    PRODUCT_DEVELOPMENT = "product_development"
    HIRING = "hiring"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    ACQUISITIONS = "acquisitions"
    RETIRE_DEBT = "retire_debt"
    UNDISCLOSED = "undisclosed"


_FORBIDDEN_USES: frozenset[UseOfProceeds] = frozenset({UseOfProceeds.RETIRE_DEBT})


class FounderShariahCompliance(str, Enum):
    """Founder's shariah-credibility tier.

    The roadmap notes that a tobacco startup with a halal-supply-
    chain founder doesn't become halal — but the inverse also
    applies: a halal-fintech founder with no shariah-board
    connection is weaker on credibility than one connected to
    AAOIFI-recognised scholars.
    """

    SCHOLAR_BOARD_BACKED = "scholar_board_backed"
    SELF_DECLARED_HALAL = "self_declared_halal"
    UNKNOWN = "unknown"


class VCDealVerdict(str, Enum):
    """Screen verdict.

    Pinned string values for JSON / DB serialisation; the
    dashboard + exception-queue UI key on these literals.
    """

    HALAL = "halal"
    NOT_HALAL = "not_halal"
    DOUBTFUL = "doubtful"
    DOUBTFUL_PIVOT = "doubtful_pivot"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class VCAllocationPolicy:
    """Operator-tunable allocation policy.

    `max_per_deal_pct` defaults to 10% — operators bump to 25%
    for high-conviction concentrated allocations but the engine
    refuses > 50% at construction (category-error guard).
    `min_lockup_disclosure_years` floors at 0 (a 0-year lockup
    is rare in private markets but liquid public-style
    secondaries do exist).
    """

    max_per_deal_pct: float = 10.0
    min_lockup_disclosure_years: int = 0
    require_scholar_board_for_pre_product: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.max_per_deal_pct <= 50.0:
            raise ValueError(f"max_per_deal_pct must be in (0, 50], got {self.max_per_deal_pct}")
        if self.min_lockup_disclosure_years < 0:
            raise ValueError("min_lockup_disclosure_years must be non-negative")


DEFAULT_POLICY = VCAllocationPolicy()


@dataclass(frozen=True)
class VCDeal:
    """One private-market opportunity to screen.

    `lockup_years` is mandatory because the user-consent flow
    needs the number to render the illiquidity warning;
    `use_of_proceeds` is mandatory because the screener can't
    silently approve undisclosed proceeds.
    """

    deal_id: str
    company_name: str
    sector: HalalSector
    stage: DealStage
    use_of_proceeds: UseOfProceeds
    founder_shariah_compliance: FounderShariahCompliance
    lockup_years: int
    minimum_check_usd: float
    has_scholar_board_review: bool

    def __post_init__(self) -> None:
        if not self.deal_id or not self.deal_id.strip():
            raise ValueError("deal_id must be non-empty")
        if not self.company_name or not self.company_name.strip():
            raise ValueError("company_name must be non-empty")
        if self.lockup_years < 0:
            raise ValueError("lockup_years must be non-negative")
        if self.minimum_check_usd < 0:
            raise ValueError("minimum_check_usd must be non-negative")


@dataclass(frozen=True)
class VCAllocationRequest:
    """A user wants to allocate to a deal.

    `requested_pct` is the share of the user's portfolio they
    want to commit to this deal. The engine validates against
    the per-deal cap and surfaces concentration warnings.
    """

    user_id: str
    deal_id: str
    requested_pct: float

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not self.deal_id or not self.deal_id.strip():
            raise ValueError("deal_id must be non-empty")
        if not 0.0 < self.requested_pct <= 100.0:
            raise ValueError(f"requested_pct must be in (0, 100], got {self.requested_pct}")


@dataclass(frozen=True)
class VCDealScreenResult:
    """Screen verdict + supporting flags + audit notes."""

    deal_id: str
    company_name: str
    sector: HalalSector
    verdict: VCDealVerdict
    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VCAllocationDecision:
    """Outcome of a per-user allocation request."""

    user_id: str
    deal_id: str
    allowed: bool
    requested_pct: float
    cap_pct: float
    reason: str
    deal_verdict: VCDealVerdict


def screen_deal(
    deal: VCDeal,
    *,
    policy: VCAllocationPolicy = DEFAULT_POLICY,
) -> VCDealScreenResult:
    """Apply the halal-VC deal screen.

    Returns a `VCDealScreenResult` with verdict + per-rule
    failure / warning lists for the audit trail.
    """

    failures: list[str] = []
    warnings: list[str] = []

    # Hard rejections.
    if deal.use_of_proceeds in _FORBIDDEN_USES:
        failures.append(
            f"use_of_proceeds={deal.use_of_proceeds.value}: "
            "raising equity to retire interest-bearing debt funds riba"
        )

    # INSUFFICIENT_DATA gate.
    if deal.use_of_proceeds is UseOfProceeds.UNDISCLOSED:
        return VCDealScreenResult(
            deal_id=deal.deal_id,
            company_name=deal.company_name,
            sector=deal.sector,
            verdict=VCDealVerdict.INSUFFICIENT_DATA,
            failures=tuple(failures),
            warnings=("use_of_proceeds is UNDISCLOSED — operator must verify before allocating",),
        )

    if failures:
        return VCDealScreenResult(
            deal_id=deal.deal_id,
            company_name=deal.company_name,
            sector=deal.sector,
            verdict=VCDealVerdict.NOT_HALAL,
            failures=tuple(failures),
            warnings=tuple(warnings),
        )

    # Soft warnings — drive DOUBTFUL or DOUBTFUL_PIVOT.
    if deal.stage in _PRE_PRODUCT_STAGES:
        if policy.require_scholar_board_for_pre_product and not deal.has_scholar_board_review:
            warnings.append(
                f"{deal.stage.value} stage with no scholar-board review: "
                "pre-product startups frequently pivot; "
                "shariah board strongly recommended"
            )
        else:
            warnings.append(
                f"{deal.stage.value} stage: pre-product startups "
                "frequently pivot — verify business model has not changed "
                "before subsequent rounds"
            )

    if deal.sector in _SCRUTINY_SECTORS:
        warnings.append(
            f"{deal.sector.value} sector requires extra scrutiny (adjacent to non-halal verticals)"
        )

    if deal.founder_shariah_compliance is FounderShariahCompliance.UNKNOWN:
        warnings.append("founder shariah credibility UNKNOWN — verify before allocating")
    elif deal.founder_shariah_compliance is FounderShariahCompliance.SELF_DECLARED_HALAL:
        warnings.append("founder shariah credibility is self-declared (no scholar-board backing)")

    if not deal.has_scholar_board_review:
        warnings.append("no scholar-board review documented for this deal")

    if warnings:
        if deal.stage in _PRE_PRODUCT_STAGES:
            verdict = VCDealVerdict.DOUBTFUL_PIVOT
        else:
            verdict = VCDealVerdict.DOUBTFUL
    else:
        verdict = VCDealVerdict.HALAL

    return VCDealScreenResult(
        deal_id=deal.deal_id,
        company_name=deal.company_name,
        sector=deal.sector,
        verdict=verdict,
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


def evaluate_allocation(
    request: VCAllocationRequest,
    *,
    deal: VCDeal,
    policy: VCAllocationPolicy = DEFAULT_POLICY,
) -> VCAllocationDecision:
    """Decide whether the requested allocation is permitted.

    Combines the deal screen verdict + concentration check + the
    requested-pct vs cap-pct comparison. The user-side
    accredited-investor check is delegated to the Wave 11.C KYC
    engine and isn't re-implemented here.
    """

    if request.deal_id != deal.deal_id:
        raise ValueError(
            f"request deal_id {request.deal_id!r} does not match deal {deal.deal_id!r}"
        )

    deal_result = screen_deal(deal, policy=policy)

    if deal_result.verdict is VCDealVerdict.NOT_HALAL:
        return VCAllocationDecision(
            user_id=request.user_id,
            deal_id=deal.deal_id,
            allowed=False,
            requested_pct=request.requested_pct,
            cap_pct=policy.max_per_deal_pct,
            reason=("deal failed halal screen: " + "; ".join(deal_result.failures)),
            deal_verdict=deal_result.verdict,
        )

    if deal_result.verdict is VCDealVerdict.INSUFFICIENT_DATA:
        return VCAllocationDecision(
            user_id=request.user_id,
            deal_id=deal.deal_id,
            allowed=False,
            requested_pct=request.requested_pct,
            cap_pct=policy.max_per_deal_pct,
            reason="deal has insufficient data for halal screening",
            deal_verdict=deal_result.verdict,
        )

    if request.requested_pct > policy.max_per_deal_pct:
        return VCAllocationDecision(
            user_id=request.user_id,
            deal_id=deal.deal_id,
            allowed=False,
            requested_pct=request.requested_pct,
            cap_pct=policy.max_per_deal_pct,
            reason=(
                f"requested {request.requested_pct:.2f}% exceeds "
                f"per-deal cap {policy.max_per_deal_pct:.2f}%"
            ),
            deal_verdict=deal_result.verdict,
        )

    # DOUBTFUL / DOUBTFUL_PIVOT proceed but with the verdict
    # carried in the decision so the user-facing flow renders the
    # warnings prominently.
    return VCAllocationDecision(
        user_id=request.user_id,
        deal_id=deal.deal_id,
        allowed=True,
        requested_pct=request.requested_pct,
        cap_pct=policy.max_per_deal_pct,
        reason=(
            f"deal verdict {deal_result.verdict.value}; "
            f"requested {request.requested_pct:.2f}% within cap "
            f"{policy.max_per_deal_pct:.2f}%"
        ),
        deal_verdict=deal_result.verdict,
    )


_VERDICT_EMOJI: dict[VCDealVerdict, str] = {
    VCDealVerdict.HALAL: "✅",
    VCDealVerdict.NOT_HALAL: "❌",
    VCDealVerdict.DOUBTFUL: "⚠️",
    VCDealVerdict.DOUBTFUL_PIVOT: "🌱",
    VCDealVerdict.INSUFFICIENT_DATA: "❓",
}


def render_screen_result(result: VCDealScreenResult) -> str:
    """Format the screen result for ops display.

    Pinned no-USD contract: deal screen never includes the
    minimum check size or user portfolio totals — operator's
    audit dashboard renders the financial details separately.
    """

    emoji = _VERDICT_EMOJI[result.verdict]
    lines = [
        f"{emoji} {result.deal_id} ({result.company_name}) — {result.verdict.value.upper()}",
        f"  sector: {result.sector.value}",
    ]
    if result.failures:
        lines.append("  failures:")
        for f in result.failures:
            lines.append(f"    · {f}")
    if result.warnings:
        lines.append("  warnings:")
        for w in result.warnings:
            lines.append(f"    · {w}")
    return "\n".join(lines)


def render_allocation_decision(decision: VCAllocationDecision) -> str:
    """Format the allocation decision for the user-facing receipt."""

    emoji = "✅" if decision.allowed else "🚫"
    line = (
        f"{emoji} {decision.user_id} → {decision.deal_id} "
        f"— {'ALLOWED' if decision.allowed else 'BLOCKED'}"
    )
    return (
        f"{line}\n"
        f"  requested: {decision.requested_pct:.2f}% / "
        f"cap: {decision.cap_pct:.2f}%\n"
        f"  deal verdict: {decision.deal_verdict.value}\n"
        f"  reason: {decision.reason}"
    )


__all__ = [
    "DEFAULT_POLICY",
    "DealStage",
    "FounderShariahCompliance",
    "HalalSector",
    "UseOfProceeds",
    "VCAllocationDecision",
    "VCAllocationPolicy",
    "VCAllocationRequest",
    "VCDeal",
    "VCDealScreenResult",
    "VCDealVerdict",
    "evaluate_allocation",
    "render_allocation_decision",
    "render_screen_result",
    "screen_deal",
]
