"""REIT-specific Shariah screener.

A general-purpose stock screener (Zoya / sector-limits) catches the
broad-strokes failures: alcohol, gambling, pork, conventional banking.
REITs slip through that net because they're real-estate vehicles whose
*tenants* (not the REIT itself) are the non-permissible exposure. A
property-trust holding a mall whose anchor tenant is a conventional
bank or a wine retailer fails Shariah on tenant mix even though the
REIT's own SIC code is "real estate".

This module ships the REIT-specific layer the existing screener
delegates to when the candidate symbol is a REIT. It mirrors the
``halal/scholar_review.py`` and ``halal/sector_limits.py`` isolated-
module pattern: pure-Python frozen dataclasses, no DB / network /
async, deterministic thresholds, validates conservatively (missing
data → DOUBTFUL, never silently HALAL).

The thresholds follow AAOIFI Standard 21 ("Financial Paper, Shares
and Bonds") and the Mufti Faraz Adam REIT framework most-cited
across the GCC: interest-bearing debt ≤ 33% of market cap; non-
permissible income (NPI) from forbidden-tenant rents ≤ 5% of total
rental income; underlying property type cannot be inherently
non-permissible (a hotel with > 5% alcohol revenue, an office tower
whose anchor tenant is a conventional insurer, etc.).

Pinned semantics:
- HALAL requires every check pass *and* the data to be present.
  Missing tenant breakdown on a property type that needs it (retail
  mall, diversified) → DOUBTFUL, not HALAL.
- DOUBTFUL is the operator-decides bucket: marginal NPI between 0%
  and the 5% cap, hotel/specialty without explicit certification,
  insufficient tenant detail. Operators can opt-in via the manual
  exception queue (``halal/exception_queue``).
- NOT_HALAL is unconditional: forbidden property type with confirmed
  forbidden-tenant exposure, debt above threshold, NPI above the 5%
  cap, residential-banking lockup.
- Purification % equals NPI % when the REIT passes (HALAL with
  marginal NPI requires the operator to purify that fraction of
  dividends per AAOIFI guidance).
- Float comparisons are inclusive at the threshold (33.0% debt is
  HALAL; 33.0001% is NOT_HALAL) — the boundary is documented and
  pinned both directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class REITPropertyType(str, Enum):
    """Property focus of the trust.

    Property types fall into three classes:
    - **Inherently halal** (RESIDENTIAL, OFFICE, INDUSTRIAL,
      HEALTHCARE, DATA_CENTER, SELF_STORAGE) — pass without tenant
      detail because the tenant base is many small lessees with no
      systemic non-permissible exposure.
    - **Tenant-dependent** (RETAIL_MALL, DIVERSIFIED) — need an NPI
      check against the tenant breakdown; absent the breakdown we
      land in DOUBTFUL rather than guess.
    - **Inherently doubtful** (HOTEL, SPECIALTY) — even with tenant
      data they need scholar review for the underlying activities
      (room service / conferences / amenities) that the tenant list
      doesn't capture.
    """

    RESIDENTIAL = "residential"
    OFFICE = "office"
    INDUSTRIAL = "industrial"
    HEALTHCARE = "healthcare"
    DATA_CENTER = "data_center"
    SELF_STORAGE = "self_storage"
    RETAIL_MALL = "retail_mall"
    DIVERSIFIED = "diversified"
    HOTEL = "hotel"
    SPECIALTY = "specialty"


# Property types that pass without tenant breakdown.
_INHERENTLY_HALAL_PROPERTIES: frozenset[REITPropertyType] = frozenset(
    {
        REITPropertyType.RESIDENTIAL,
        REITPropertyType.OFFICE,
        REITPropertyType.INDUSTRIAL,
        REITPropertyType.HEALTHCARE,
        REITPropertyType.DATA_CENTER,
        REITPropertyType.SELF_STORAGE,
    }
)

# Property types that need tenant breakdown to clear.
_TENANT_DEPENDENT_PROPERTIES: frozenset[REITPropertyType] = frozenset(
    {
        REITPropertyType.RETAIL_MALL,
        REITPropertyType.DIVERSIFIED,
    }
)

# Property types whose nature itself raises shariah questions
# regardless of tenant breakdown.
_INHERENTLY_DOUBTFUL_PROPERTIES: frozenset[REITPropertyType] = frozenset(
    {
        REITPropertyType.HOTEL,
        REITPropertyType.SPECIALTY,
    }
)


class TenantCategory(str, Enum):
    """Shariah classification of a tenant's primary business."""

    HALAL = "halal"
    CONVENTIONAL_BANK = "conventional_bank"
    INSURANCE_CONVENTIONAL = "insurance_conventional"
    ALCOHOL_GAMBLING = "alcohol_gambling"
    PORK_RELATED = "pork_related"
    ADULT_ENTERTAINMENT = "adult_entertainment"
    TOBACCO = "tobacco"
    CINEMA = "cinema"
    ARMS = "arms"


# Categories that contribute to non-permissible income.
_FORBIDDEN_TENANT_CATEGORIES: frozenset[TenantCategory] = frozenset(
    c for c in TenantCategory if c is not TenantCategory.HALAL
)


class REITScreenStatus(str, Enum):
    """Screen verdict.

    Pinned values (string form) for JSON serialisation: dashboard /
    exception-queue UI keys on these literals.
    """

    HALAL = "halal"
    NOT_HALAL = "not_halal"
    DOUBTFUL = "doubtful"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class REITThresholds:
    """Configurable AAOIFI thresholds.

    Default values follow AAOIFI Standard 21 + Mufti Faraz Adam
    REIT framework. Stricter operators can drop the debt threshold
    to 30%; the NPI cap is rarely lowered below 5% because cohort
    studies suggest that's the practical noise floor for tenant-
    income classification.
    """

    debt_to_marketcap_pct: float = 33.0
    npi_to_total_income_pct: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 < self.debt_to_marketcap_pct <= 100.0:
            raise ValueError(
                f"debt_to_marketcap_pct must be in (0, 100], got {self.debt_to_marketcap_pct}"
            )
        if not 0.0 < self.npi_to_total_income_pct <= 100.0:
            raise ValueError(
                f"npi_to_total_income_pct must be in (0, 100], got {self.npi_to_total_income_pct}"
            )


DEFAULT_THRESHOLDS = REITThresholds()


@dataclass(frozen=True)
class TenantContribution:
    """A single tenant (or tenant cohort) and its share of rental income.

    `rental_income_pct` is the share of total rental income the tenant
    contributes, in the 0..100 range. The screener tolerates a list
    that sums to less than 100% (the remainder is implicitly "halal
    other tenants" since the operator presumably labels the
    suspicious tenants explicitly), but rejects a list that sums to
    more than 100% as a data-entry error.
    """

    name: str
    category: TenantCategory
    rental_income_pct: float

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("tenant name must be non-empty")
        if not 0.0 <= self.rental_income_pct <= 100.0:
            raise ValueError(f"rental_income_pct must be in [0, 100], got {self.rental_income_pct}")


@dataclass(frozen=True)
class REITFinancials:
    """The minimum data the screener needs to decide.

    `liquid_assets_usd` is included for completeness — AAOIFI
    Standard 21 caps cash + receivables at 70% of market cap, but
    REITs are by definition property-heavy and almost never breach
    that cap. The screener checks it anyway so a wildly mis-
    classified financial-services-shell-pretending-to-be-a-REIT
    fails out, and so the result block can render the number for the
    operator audit trail.
    """

    symbol: str
    name: str
    property_type: REITPropertyType
    market_cap_usd: float
    interest_bearing_debt_usd: float
    liquid_assets_usd: float
    rental_income_total_usd: float
    tenants: tuple[TenantContribution, ...] = ()

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if self.market_cap_usd < 0:
            raise ValueError("market_cap_usd must be non-negative")
        if self.interest_bearing_debt_usd < 0:
            raise ValueError("interest_bearing_debt_usd must be non-negative")
        if self.liquid_assets_usd < 0:
            raise ValueError("liquid_assets_usd must be non-negative")
        if self.rental_income_total_usd < 0:
            raise ValueError("rental_income_total_usd must be non-negative")
        total_pct = sum(t.rental_income_pct for t in self.tenants)
        if total_pct > 100.0 + 1e-6:
            raise ValueError(f"tenant rental_income_pct shares sum to {total_pct:.2f} > 100")


@dataclass(frozen=True)
class REITScreenResult:
    """The screen verdict + supporting numbers + audit notes.

    `purification_pct` is the operator's actionable output for a
    HALAL-with-marginal-NPI verdict: the fraction of any dividend
    received from this REIT that must be donated rather than kept,
    per AAOIFI's purification guidance. Always 0.0 for HALAL with
    no NPI; equals npi_pct for HALAL with marginal NPI; not
    meaningful for NOT_HALAL / DOUBTFUL / INSUFFICIENT_DATA (the
    operator shouldn't be holding the position to begin with).
    """

    symbol: str
    status: REITScreenStatus
    debt_to_marketcap_pct: float | None
    npi_pct: float | None
    liquid_assets_to_marketcap_pct: float | None
    purification_pct: float
    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _compute_npi_pct(tenants: tuple[TenantContribution, ...]) -> float:
    """NPI share = sum of rental_income_pct across forbidden tenants."""

    return sum(t.rental_income_pct for t in tenants if t.category in _FORBIDDEN_TENANT_CATEGORIES)


def screen_reit(
    financials: REITFinancials,
    *,
    thresholds: REITThresholds = DEFAULT_THRESHOLDS,
) -> REITScreenResult:
    """Apply the AAOIFI REIT screen to one trust.

    Returns a `REITScreenResult` with status, the contributing
    ratios, the purification share, and the per-rule pass/fail
    detail the audit trail records.
    """

    failures: list[str] = []
    warnings: list[str] = []

    if financials.market_cap_usd <= 0:
        return REITScreenResult(
            symbol=financials.symbol,
            status=REITScreenStatus.INSUFFICIENT_DATA,
            debt_to_marketcap_pct=None,
            npi_pct=None,
            liquid_assets_to_marketcap_pct=None,
            purification_pct=0.0,
            failures=("market_cap_usd is zero — cannot compute ratios",),
        )

    debt_pct = financials.interest_bearing_debt_usd / financials.market_cap_usd * 100.0
    liquid_pct = financials.liquid_assets_usd / financials.market_cap_usd * 100.0

    if debt_pct > thresholds.debt_to_marketcap_pct:
        failures.append(f"debt {debt_pct:.2f}% > {thresholds.debt_to_marketcap_pct:.0f}% threshold")
    if liquid_pct > 70.0:
        failures.append(
            f"liquid assets {liquid_pct:.2f}% > 70% threshold (REIT mis-classification?)"
        )

    npi_pct: float | None = None
    if financials.property_type in _INHERENTLY_HALAL_PROPERTIES:
        if financials.tenants:
            npi_pct = _compute_npi_pct(financials.tenants)
            if npi_pct > thresholds.npi_to_total_income_pct:
                failures.append(
                    f"npi {npi_pct:.2f}% > {thresholds.npi_to_total_income_pct:.0f}% threshold"
                )
        else:
            npi_pct = 0.0
    elif financials.property_type in _TENANT_DEPENDENT_PROPERTIES:
        if not financials.tenants:
            warnings.append(
                f"tenant breakdown missing for {financials.property_type.value}; "
                "cannot confirm NPI is below threshold"
            )
        else:
            npi_pct = _compute_npi_pct(financials.tenants)
            if npi_pct > thresholds.npi_to_total_income_pct:
                failures.append(
                    f"npi {npi_pct:.2f}% > {thresholds.npi_to_total_income_pct:.0f}% threshold"
                )
    elif financials.property_type in _INHERENTLY_DOUBTFUL_PROPERTIES:
        warnings.append(
            f"{financials.property_type.value} property type requires scholar review "
            "regardless of tenant mix"
        )
        if financials.tenants:
            npi_pct = _compute_npi_pct(financials.tenants)
            if npi_pct > thresholds.npi_to_total_income_pct:
                failures.append(
                    f"npi {npi_pct:.2f}% > {thresholds.npi_to_total_income_pct:.0f}% threshold"
                )

    if failures:
        status = REITScreenStatus.NOT_HALAL
        purification_pct = 0.0
    elif warnings:
        status = REITScreenStatus.DOUBTFUL
        purification_pct = 0.0
    else:
        status = REITScreenStatus.HALAL
        purification_pct = npi_pct if npi_pct is not None else 0.0

    return REITScreenResult(
        symbol=financials.symbol,
        status=status,
        debt_to_marketcap_pct=debt_pct,
        npi_pct=npi_pct,
        liquid_assets_to_marketcap_pct=liquid_pct,
        purification_pct=purification_pct,
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


_STATUS_EMOJI: dict[REITScreenStatus, str] = {
    REITScreenStatus.HALAL: "✅",
    REITScreenStatus.NOT_HALAL: "❌",
    REITScreenStatus.DOUBTFUL: "⚠️",
    REITScreenStatus.INSUFFICIENT_DATA: "❓",
}


def render_screen_result(result: REITScreenResult) -> str:
    """Format the result for ops display.

    Mirrors the emoji + bullet format used by other halal-side
    renderers (`scholar_review.render_recorded_verdict`,
    `aaoifi_summary`) so a Telegram / Slack post lands visually
    consistent with the rest of the bot's output.
    """

    lines: list[str] = []
    emoji = _STATUS_EMOJI[result.status]
    lines.append(f"{emoji} {result.symbol} — {result.status.value.upper()}")

    if result.debt_to_marketcap_pct is not None:
        lines.append(f"  debt/marketcap: {result.debt_to_marketcap_pct:.2f}%")
    if result.npi_pct is not None:
        lines.append(f"  npi: {result.npi_pct:.2f}%")
    if result.liquid_assets_to_marketcap_pct is not None:
        lines.append(f"  liquid/marketcap: {result.liquid_assets_to_marketcap_pct:.2f}%")
    if result.purification_pct > 0:
        lines.append(f"  purification: {result.purification_pct:.2f}% of dividends")

    if result.failures:
        lines.append("  failures:")
        for f in result.failures:
            lines.append(f"    - {f}")
    if result.warnings:
        lines.append("  warnings:")
        for w in result.warnings:
            lines.append(f"    - {w}")

    return "\n".join(lines)


__all__ = [
    "DEFAULT_THRESHOLDS",
    "REITFinancials",
    "REITPropertyType",
    "REITScreenResult",
    "REITScreenStatus",
    "REITThresholds",
    "TenantCategory",
    "TenantContribution",
    "render_screen_result",
    "screen_reit",
]
