"""Malaysia tax-exemption tracker — Round-5 Wave 18.E.

Malaysia exempts capital gains on listed equities for individuals.
Dividends are taxed via the single-tier system (already tax-paid at
the company level → no further tax on the recipient). The bot's
Malaysia-resident operator therefore needs:

- A list of disposals classified as **EXEMPT** (listed equity,
  individual taxpayer) vs. **POTENTIALLY_TAXABLE** (real-property
  gains, business-line transactions, frequent-trader-classification).
- A summary of tax-exempt dividend receipts.

This module ships the **classifier + summary**.

Pinned semantics:

- **Closed-set TaxStatus ladder** — EXEMPT_LISTED / EXEMPT_DIVIDEND /
  RPGT_REAL_PROPERTY / BUSINESS_INCOME / FOREIGN_TAXABLE.
- **Frequent-trader threshold** — operator-tunable; the classifier
  flags candidates as BUSINESS_INCOME when daily-trade-count exceeds
  threshold.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum


class TaxStatus(str, Enum):
    """Closed-set Malaysia tax statuses."""

    EXEMPT_LISTED = "exempt_listed"
    EXEMPT_DIVIDEND = "exempt_dividend"
    RPGT_REAL_PROPERTY = "rpgt_real_property"
    BUSINESS_INCOME = "business_income"
    FOREIGN_TAXABLE = "foreign_taxable"


class AssetCategory(str, Enum):
    """Closed-set asset categories."""

    LISTED_EQUITY_BURSA = "listed_equity_bursa"
    LISTED_EQUITY_FOREIGN = "listed_equity_foreign"
    REAL_PROPERTY_RPGT = "real_property_rpgt"
    DIVIDEND = "dividend"
    OTHER = "other"


@dataclass(frozen=True)
class TaxPolicy:
    """Operator-tunable thresholds for Malaysia tax classification."""

    business_income_trades_per_day: int = 20

    def __post_init__(self) -> None:
        if self.business_income_trades_per_day <= 0:
            raise ValueError("business_income_trades_per_day must be positive")


@dataclass(frozen=True)
class DisposalEvent:
    """A single disposal / dividend event."""

    event_id: str
    asset_category: AssetCategory
    proceeds: float
    cost_basis: float
    event_date: date
    daily_trade_count: int = 0  # disposals on this date

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        if self.proceeds < 0:
            raise ValueError("proceeds must be non-negative")
        if self.cost_basis < 0:
            raise ValueError("cost_basis must be non-negative")
        if self.daily_trade_count < 0:
            raise ValueError("daily_trade_count must be non-negative")


@dataclass(frozen=True)
class TaxClassification:
    """Classification result for a single event."""

    event_id: str
    status: TaxStatus
    taxable_amount: float

    def __post_init__(self) -> None:
        if self.taxable_amount < 0:
            raise ValueError("taxable_amount must be non-negative")


def classify(event: DisposalEvent, *, policy: TaxPolicy | None = None) -> TaxClassification:
    """Classify a disposal/dividend event by Malaysia tax status."""
    pol = policy if policy is not None else TaxPolicy()

    if event.asset_category is AssetCategory.DIVIDEND:
        return TaxClassification(
            event_id=event.event_id,
            status=TaxStatus.EXEMPT_DIVIDEND,
            taxable_amount=0.0,
        )
    if event.asset_category is AssetCategory.REAL_PROPERTY_RPGT:
        gain = max(0.0, event.proceeds - event.cost_basis)
        return TaxClassification(
            event_id=event.event_id,
            status=TaxStatus.RPGT_REAL_PROPERTY,
            taxable_amount=gain,
        )
    if event.asset_category is AssetCategory.LISTED_EQUITY_FOREIGN:
        gain = max(0.0, event.proceeds - event.cost_basis)
        return TaxClassification(
            event_id=event.event_id,
            status=TaxStatus.FOREIGN_TAXABLE,
            taxable_amount=gain,
        )
    if event.asset_category is AssetCategory.LISTED_EQUITY_BURSA:
        if event.daily_trade_count >= pol.business_income_trades_per_day:
            gain = max(0.0, event.proceeds - event.cost_basis)
            return TaxClassification(
                event_id=event.event_id,
                status=TaxStatus.BUSINESS_INCOME,
                taxable_amount=gain,
            )
        return TaxClassification(
            event_id=event.event_id,
            status=TaxStatus.EXEMPT_LISTED,
            taxable_amount=0.0,
        )
    # OTHER → conservative: flag as taxable
    gain = max(0.0, event.proceeds - event.cost_basis)
    return TaxClassification(
        event_id=event.event_id,
        status=TaxStatus.FOREIGN_TAXABLE,
        taxable_amount=gain,
    )


def classify_batch(
    events: Iterable[DisposalEvent], *, policy: TaxPolicy | None = None
) -> tuple[TaxClassification, ...]:
    return tuple(classify(e, policy=policy) for e in events)


def total_taxable(classifications: Iterable[TaxClassification]) -> float:
    return sum(c.taxable_amount for c in classifications)


def total_exempt_dividends(
    events: Iterable[DisposalEvent],
) -> float:
    """Sum exempt dividend receipts."""
    return sum(e.proceeds for e in events if e.asset_category is AssetCategory.DIVIDEND)


def render_classification(c: TaxClassification) -> str:
    return f"{c.event_id}: {c.status.value} taxable=RM{c.taxable_amount:.2f}"
