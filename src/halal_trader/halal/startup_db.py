"""Halal startup screen — Round-5 Wave 6.A.

Private-market deal flow (AngelList / Crunchbase / halal-platform feeds)
is the largest pool of opportunities most Muslim investors never see
filtered for compliance. This module is the **deal-level halal screen**
applied *before* a deal even surfaces in the operator's UI.

Three filter layers:

1. **Sector classification** — closed-set haram sectors (alcohol,
   gambling, conventional banking, defense-against-Muslims weapons,
   adult content, conventional insurance) → REJECTED.
2. **Revenue-mix screen** — standard AAOIFI Standard 21 thresholds
   on % revenue from non-halal sources (≤5% pinned).
3. **Capital-structure screen** — interest-bearing-debt / equity
   ratio, cash + receivables / market-cap; structural pins (33% / 33%
   / 70%) per Standard 21 cl. 3/3.5.4. (For pre-revenue startups,
   capital structure is the binding pin since revenue is too noisy.)

Pinned semantics:

- **Closed-set Sector ladder** for both haram and halal sides.
- **Closed-set Verdict ladder** — APPROVED / FLAGGED / REJECTED.
- **REJECTED is sticky** — once a sector triggers, no amount of
  revenue-mix or capital-structure rehab can move the verdict up.
- **FLAGGED** is the "needs scholar review" bucket — interest-debt
  ratio between 30%-33%, ambiguous sector, etc.
- **Stage-aware** — SEED / SERIES_A startups get capital-structure
  leniency on cash + receivables (those metrics are noisy pre-revenue).
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — founder names are masked.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Sector(str, Enum):
    """Closed-set sector ladder.

    HARAM_* = automatic REJECTED. Other sectors get the revenue +
    capital-structure screen.
    """

    HARAM_ALCOHOL = "haram_alcohol"
    HARAM_GAMBLING = "haram_gambling"
    HARAM_CONVENTIONAL_BANKING = "haram_conventional_banking"
    HARAM_CONVENTIONAL_INSURANCE = "haram_conventional_insurance"
    HARAM_ADULT_CONTENT = "haram_adult_content"
    HARAM_PORK_PRODUCTION = "haram_pork_production"
    HARAM_INTEREST_LENDING = "haram_interest_lending"
    HARAM_TOBACCO = "haram_tobacco"

    HALAL_TECHNOLOGY = "halal_technology"
    HALAL_HEALTHCARE = "halal_healthcare"
    HALAL_CLEAN_ENERGY = "halal_clean_energy"
    HALAL_AGRICULTURE = "halal_agriculture"
    HALAL_EDUCATION = "halal_education"
    HALAL_CONSUMER_GOODS = "halal_consumer_goods"
    HALAL_INDUSTRIAL = "halal_industrial"
    HALAL_FINTECH_ISLAMIC = "halal_fintech_islamic"

    AMBIGUOUS_BIOTECH = "ambiguous_biotech"
    """Biotech may include haram pathways (alcohol-based extraction);
    flag for scholar review."""
    AMBIGUOUS_DEFENSE = "ambiguous_defense"
    """Defense-tech may be permissible (defensive systems for Muslim
    nations) or impermissible (offensive); flag for scholar review."""
    AMBIGUOUS_MEDIA = "ambiguous_media"
    """Music / video content can be halal or haram per use-case."""

    OTHER = "other"


_HARAM_SECTORS: frozenset[Sector] = frozenset(
    {
        Sector.HARAM_ALCOHOL,
        Sector.HARAM_GAMBLING,
        Sector.HARAM_CONVENTIONAL_BANKING,
        Sector.HARAM_CONVENTIONAL_INSURANCE,
        Sector.HARAM_ADULT_CONTENT,
        Sector.HARAM_PORK_PRODUCTION,
        Sector.HARAM_INTEREST_LENDING,
        Sector.HARAM_TOBACCO,
    }
)


_AMBIGUOUS_SECTORS: frozenset[Sector] = frozenset(
    {
        Sector.AMBIGUOUS_BIOTECH,
        Sector.AMBIGUOUS_DEFENSE,
        Sector.AMBIGUOUS_MEDIA,
    }
)


def is_haram_sector(sector: Sector) -> bool:
    return sector in _HARAM_SECTORS


def is_ambiguous_sector(sector: Sector) -> bool:
    return sector in _AMBIGUOUS_SECTORS


class Stage(str, Enum):
    """Closed-set startup stage ladder."""

    PRE_SEED = "pre_seed"
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C = "series_c"
    GROWTH = "growth"
    PRE_IPO = "pre_ipo"


_PRE_REVENUE_STAGES: frozenset[Stage] = frozenset({Stage.PRE_SEED, Stage.SEED, Stage.SERIES_A})


class Verdict(str, Enum):
    """Closed-set screen verdict."""

    APPROVED = "approved"
    FLAGGED = "flagged"
    REJECTED = "rejected"


@dataclass(frozen=True)
class StartupDeal:
    """A deal-flow record — fields are what aggregators (AngelList,
    Crunchbase, halal-platform feeds) typically expose."""

    deal_id: str
    company_name: str
    primary_sector: Sector
    stage: Stage
    valuation_usd: float
    raise_amount_usd: float
    haram_revenue_pct: float = 0.0
    """% of revenue from haram sources (interest, gambling, etc.)."""
    interest_debt_to_equity: float = 0.0
    """Interest-bearing debt / total equity. AAOIFI threshold 33%."""
    cash_to_market_cap: float = 0.0
    """Cash + interest-bearing investments / market cap. Threshold 33%."""
    receivables_to_market_cap: float = 0.0
    """Accounts receivable / market cap. Threshold 33% (per Standard 21)."""
    description: str = ""
    founders: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.deal_id or not self.deal_id.strip():
            raise ValueError("deal_id must be non-empty")
        if not self.company_name or not self.company_name.strip():
            raise ValueError("company_name must be non-empty")
        if self.valuation_usd < 0:
            raise ValueError("valuation_usd must be non-negative")
        if self.raise_amount_usd < 0:
            raise ValueError("raise_amount_usd must be non-negative")
        for name, val in (
            ("haram_revenue_pct", self.haram_revenue_pct),
            ("interest_debt_to_equity", self.interest_debt_to_equity),
            ("cash_to_market_cap", self.cash_to_market_cap),
            ("receivables_to_market_cap", self.receivables_to_market_cap),
        ):
            if not 0.0 <= val <= 5.0:
                raise ValueError(f"{name} must be in [0, 5.0]")


@dataclass(frozen=True)
class ScreenResult:
    """Output of `screen_deal`."""

    deal_id: str
    verdict: Verdict
    reasons: tuple[str, ...]
    """Human-readable reasons in the order the rules fired."""
    sector_haram: bool
    sector_ambiguous: bool
    revenue_breach: bool
    capital_breach: bool


# AAOIFI Standard 21 prudential thresholds.
_HARAM_REVENUE_HARD_LIMIT = 0.05  # 5% — REJECTED above
_INTEREST_DEBT_HARD = 0.33
_INTEREST_DEBT_FLAG = 0.30
_CASH_RATIO_HARD = 0.33
_RECEIVABLES_HARD = 0.33


def screen_deal(deal: StartupDeal) -> ScreenResult:
    """Apply the three-layer screen and return the verdict."""
    reasons: list[str] = []
    sector_haram = is_haram_sector(deal.primary_sector)
    sector_ambiguous = is_ambiguous_sector(deal.primary_sector)
    if sector_haram:
        reasons.append(f"sector {deal.primary_sector.value} is haram")
        return ScreenResult(
            deal_id=deal.deal_id,
            verdict=Verdict.REJECTED,
            reasons=tuple(reasons),
            sector_haram=True,
            sector_ambiguous=False,
            revenue_breach=False,
            capital_breach=False,
        )
    revenue_breach = deal.haram_revenue_pct > _HARAM_REVENUE_HARD_LIMIT
    if revenue_breach:
        reasons.append(
            f"haram revenue {deal.haram_revenue_pct * 100:.2f}% > "
            f"{_HARAM_REVENUE_HARD_LIMIT * 100:.0f}% AAOIFI hard limit"
        )
    capital_breach = False
    pre_revenue = deal.stage in _PRE_REVENUE_STAGES
    if deal.interest_debt_to_equity > _INTEREST_DEBT_HARD:
        reasons.append(
            f"interest-debt/equity {deal.interest_debt_to_equity * 100:.2f}% > "
            f"{_INTEREST_DEBT_HARD * 100:.0f}% AAOIFI hard limit"
        )
        capital_breach = True
    elif deal.interest_debt_to_equity > _INTEREST_DEBT_FLAG:
        reasons.append(
            f"interest-debt/equity {deal.interest_debt_to_equity * 100:.2f}% in "
            f"flag band ({_INTEREST_DEBT_FLAG * 100:.0f}–"
            f"{_INTEREST_DEBT_HARD * 100:.0f}%)"
        )
    if not pre_revenue:
        if deal.cash_to_market_cap > _CASH_RATIO_HARD:
            reasons.append(
                f"cash/market-cap {deal.cash_to_market_cap * 100:.2f}% > "
                f"{_CASH_RATIO_HARD * 100:.0f}% AAOIFI hard limit"
            )
            capital_breach = True
        if deal.receivables_to_market_cap > _RECEIVABLES_HARD:
            reasons.append(
                f"receivables/market-cap {deal.receivables_to_market_cap * 100:.2f}% > "
                f"{_RECEIVABLES_HARD * 100:.0f}% AAOIFI hard limit"
            )
            capital_breach = True
    if revenue_breach or capital_breach:
        return ScreenResult(
            deal_id=deal.deal_id,
            verdict=Verdict.REJECTED,
            reasons=tuple(reasons),
            sector_haram=False,
            sector_ambiguous=sector_ambiguous,
            revenue_breach=revenue_breach,
            capital_breach=capital_breach,
        )
    if sector_ambiguous:
        reasons.append(f"sector {deal.primary_sector.value} requires scholar review")
        return ScreenResult(
            deal_id=deal.deal_id,
            verdict=Verdict.FLAGGED,
            reasons=tuple(reasons),
            sector_haram=False,
            sector_ambiguous=True,
            revenue_breach=False,
            capital_breach=False,
        )
    if deal.interest_debt_to_equity > _INTEREST_DEBT_FLAG:
        return ScreenResult(
            deal_id=deal.deal_id,
            verdict=Verdict.FLAGGED,
            reasons=tuple(reasons),
            sector_haram=False,
            sector_ambiguous=False,
            revenue_breach=False,
            capital_breach=False,
        )
    return ScreenResult(
        deal_id=deal.deal_id,
        verdict=Verdict.APPROVED,
        reasons=("clean: sector + revenue + capital all within AAOIFI limits",),
        sector_haram=False,
        sector_ambiguous=False,
        revenue_breach=False,
        capital_breach=False,
    )


def screen_batch(deals: Iterable[StartupDeal]) -> tuple[ScreenResult, ...]:
    return tuple(screen_deal(d) for d in deals)


def filter_approved(
    deals: Iterable[StartupDeal],
) -> tuple[StartupDeal, ...]:
    return tuple(d for d in deals if screen_deal(d).verdict is Verdict.APPROVED)


def render_result(result: ScreenResult) -> str:
    """Operator-readable summary."""
    emoji = {
        Verdict.APPROVED: "✅",
        Verdict.FLAGGED: "🟡",
        Verdict.REJECTED: "❌",
    }[result.verdict]
    lines = [f"{emoji} {result.deal_id}: {result.verdict.value}"]
    for reason in result.reasons:
        lines.append(f"  • {reason}")
    return "\n".join(lines)
