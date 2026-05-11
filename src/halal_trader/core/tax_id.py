"""Indonesia tax classification — Round-5 Wave 18.F.

Indonesia taxes capital gains and dividends differently from Malaysia:

- **Listed-equity disposals** (IDX): a flat **0.1% final tax on gross
  proceeds** (PPh Pasal 4 ayat 2). Computed on `proceeds`, not gain.
  Founders/major shareholders pay an additional 0.5% on listing.
- **Dividends** (resident individual): **10% final tax on gross**
  (PPh Pasal 4 ayat 2). Resident corporates with ≥25% ownership are
  exempt from this withholding.
- **Bond / sukuk coupon**: **10% final tax** for residents.
- **Foreign disposals**: subject to the regular progressive PPh
  rate (5%-35%); not handled as a final tax.
- **Frequent-trader business-income**: a daily-trade-count threshold
  flips the classification to BUSINESS_INCOME, where regular
  progressive PPh applies (the platform must surface a warning so
  the operator engages a tax accountant).

This module ships the **classifier + summary** following the same
structure as `core/tax_my.py`. Pure-Python, no I/O, deterministic.

Pinned semantics:

- **Closed-set TaxStatus ladder.**
- **Listed-equity tax is on PROCEEDS, not gain.** Pin in tests.
- **Dividend tax is 10% gross.** Pin in tests.
- **Founder uplift = 0.5% on top of 0.1%.** Pin.
- **Operator-tunable rates** for future legislative changes.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum


class TaxStatus(str, Enum):
    """Closed-set Indonesia tax statuses."""

    LISTED_EQUITY_FINAL = "listed_equity_final"  # 0.1% on proceeds
    LISTED_EQUITY_FOUNDER = "listed_equity_founder"  # 0.6% on proceeds
    DIVIDEND_FINAL = "dividend_final"  # 10% on proceeds
    DIVIDEND_EXEMPT_CORPORATE = "dividend_exempt_corporate"  # ≥25% holding
    BOND_COUPON_FINAL = "bond_coupon_final"  # 10% on coupon
    FOREIGN_PROGRESSIVE = "foreign_progressive"  # progressive PPh
    BUSINESS_INCOME = "business_income"  # frequent-trader → progressive


class AssetCategory(str, Enum):
    """Closed-set asset categories for Indonesian tax."""

    LISTED_EQUITY_IDX = "listed_equity_idx"
    LISTED_EQUITY_FOREIGN = "listed_equity_foreign"
    DIVIDEND = "dividend"
    BOND_COUPON = "bond_coupon"
    OTHER = "other"


@dataclass(frozen=True)
class TaxPolicy:
    """Operator-tunable Indonesia tax rates + thresholds."""

    listed_equity_final_rate: float = 0.001  # 0.1%
    listed_equity_founder_extra: float = 0.005  # +0.5% for founders
    dividend_final_rate: float = 0.10
    bond_coupon_final_rate: float = 0.10
    business_income_trades_per_day: int = 20
    corporate_dividend_threshold: float = 0.25
    """Resident-corporate ownership ≥ this fraction → dividend exempt."""

    def __post_init__(self) -> None:
        for name, val in (
            ("listed_equity_final_rate", self.listed_equity_final_rate),
            ("listed_equity_founder_extra", self.listed_equity_founder_extra),
            ("dividend_final_rate", self.dividend_final_rate),
            ("bond_coupon_final_rate", self.bond_coupon_final_rate),
        ):
            if not 0.0 <= val < 0.50:
                raise ValueError(f"{name} must be in [0, 0.50)")
        if self.business_income_trades_per_day <= 0:
            raise ValueError("business_income_trades_per_day must be positive")
        if not 0.0 < self.corporate_dividend_threshold <= 1.0:
            raise ValueError("corporate_dividend_threshold must be in (0, 1]")


@dataclass(frozen=True)
class DisposalEvent:
    """A single disposal / dividend / coupon event."""

    event_id: str
    asset_category: AssetCategory
    proceeds: float
    cost_basis: float
    event_date: date
    daily_trade_count: int = 0
    is_founder: bool = False
    """For listed-equity disposals: founders / pre-IPO shareholders pay
    the additional 0.5% on listing. Default False."""
    corporate_holder_pct: float | None = None
    """For DIVIDEND events: the resident-corporate holder's ownership
    fraction in the issuer. None = individual or unknown. ≥ threshold →
    exempt."""

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        if self.proceeds < 0:
            raise ValueError("proceeds must be non-negative")
        if self.cost_basis < 0:
            raise ValueError("cost_basis must be non-negative")
        if self.daily_trade_count < 0:
            raise ValueError("daily_trade_count must be non-negative")
        if self.corporate_holder_pct is not None:
            if not 0.0 <= self.corporate_holder_pct <= 1.0:
                raise ValueError("corporate_holder_pct must be in [0, 1]")


@dataclass(frozen=True)
class TaxClassification:
    """Classification result."""

    event_id: str
    status: TaxStatus
    taxable_amount: float
    """The amount the tax is computed on (proceeds / coupon / gain)."""
    tax_due: float

    def __post_init__(self) -> None:
        if self.taxable_amount < 0:
            raise ValueError("taxable_amount must be non-negative")
        if self.tax_due < 0:
            raise ValueError("tax_due must be non-negative")


def classify(event: DisposalEvent, *, policy: TaxPolicy | None = None) -> TaxClassification:
    """Classify a disposal / dividend / coupon event by Indonesian tax status."""
    pol = policy if policy is not None else TaxPolicy()

    if event.asset_category is AssetCategory.DIVIDEND:
        if (
            event.corporate_holder_pct is not None
            and event.corporate_holder_pct >= pol.corporate_dividend_threshold
        ):
            return TaxClassification(
                event_id=event.event_id,
                status=TaxStatus.DIVIDEND_EXEMPT_CORPORATE,
                taxable_amount=0.0,
                tax_due=0.0,
            )
        return TaxClassification(
            event_id=event.event_id,
            status=TaxStatus.DIVIDEND_FINAL,
            taxable_amount=event.proceeds,
            tax_due=event.proceeds * pol.dividend_final_rate,
        )
    if event.asset_category is AssetCategory.BOND_COUPON:
        return TaxClassification(
            event_id=event.event_id,
            status=TaxStatus.BOND_COUPON_FINAL,
            taxable_amount=event.proceeds,
            tax_due=event.proceeds * pol.bond_coupon_final_rate,
        )
    if event.asset_category is AssetCategory.LISTED_EQUITY_IDX:
        # Frequent-trader → BUSINESS_INCOME (progressive PPh; we don't
        # compute the bracket here — surface for accountant follow-up).
        if event.daily_trade_count >= pol.business_income_trades_per_day:
            gain = max(0.0, event.proceeds - event.cost_basis)
            return TaxClassification(
                event_id=event.event_id,
                status=TaxStatus.BUSINESS_INCOME,
                taxable_amount=gain,
                tax_due=0.0,  # progressive — accountant computes
            )
        rate = pol.listed_equity_final_rate
        if event.is_founder:
            rate += pol.listed_equity_founder_extra
        status = (
            TaxStatus.LISTED_EQUITY_FOUNDER if event.is_founder else TaxStatus.LISTED_EQUITY_FINAL
        )
        return TaxClassification(
            event_id=event.event_id,
            status=status,
            taxable_amount=event.proceeds,
            tax_due=event.proceeds * rate,
        )
    if event.asset_category is AssetCategory.LISTED_EQUITY_FOREIGN:
        gain = max(0.0, event.proceeds - event.cost_basis)
        return TaxClassification(
            event_id=event.event_id,
            status=TaxStatus.FOREIGN_PROGRESSIVE,
            taxable_amount=gain,
            tax_due=0.0,  # progressive — accountant computes
        )
    # OTHER → progressive (conservative)
    gain = max(0.0, event.proceeds - event.cost_basis)
    return TaxClassification(
        event_id=event.event_id,
        status=TaxStatus.FOREIGN_PROGRESSIVE,
        taxable_amount=gain,
        tax_due=0.0,
    )


def classify_batch(
    events: Iterable[DisposalEvent], *, policy: TaxPolicy | None = None
) -> tuple[TaxClassification, ...]:
    return tuple(classify(e, policy=policy) for e in events)


def total_final_tax(classifications: Iterable[TaxClassification]) -> float:
    """Sum of tax_due across all final-tax events (excludes BUSINESS_INCOME
    and FOREIGN_PROGRESSIVE which need accountant computation)."""
    final_statuses = {
        TaxStatus.LISTED_EQUITY_FINAL,
        TaxStatus.LISTED_EQUITY_FOUNDER,
        TaxStatus.DIVIDEND_FINAL,
        TaxStatus.BOND_COUPON_FINAL,
    }
    return sum(c.tax_due for c in classifications if c.status in final_statuses)


def needs_accountant(
    classifications: Iterable[TaxClassification],
) -> tuple[TaxClassification, ...]:
    """Return events that need progressive PPh computation by an accountant."""
    flag_statuses = {TaxStatus.BUSINESS_INCOME, TaxStatus.FOREIGN_PROGRESSIVE}
    return tuple(c for c in classifications if c.status in flag_statuses)


def render_classification(c: TaxClassification) -> str:
    return (
        f"{c.event_id}: {c.status.value} base=Rp{c.taxable_amount:,.0f} tax_due=Rp{c.tax_due:,.0f}"
    )
