"""Time-weighted purification for partial holdings.

Round-5 Wave 1.I primitive. The platform's existing
`halal/purification.py` covers the standard case: receive a
dividend, compute impure_pct × dividend → purification owed.
That assumes the holder has held the shares for the full
revenue-generation period (typically a quarter for US equities).

For actively-traded portfolios this assumption breaks: a user who
buys XYZ on day 88 of a 91-day quarter, holds through ex-date
on day 90, sells on day 92 — under the standard rule, owes the
full impure portion of the dividend. Some scholars argue the
purification should be prorated by the holding fraction (3/91
in this example) on the reasoning that the impure earnings
accrued over the full period; the user's economic share of those
earnings is proportional to their holding window.

This module ships both methodologies as operator-selectable so
the user picks based on their scholar's preference. The default
is FULL_AMOUNT (the more conservative methodology — over-paying
purification is just generosity; under-paying is a religious
obligation gap).

Picked a pure-functional calculator over an in-place mutator
because (a) the calculation is keyed on a holding + a dividend
event — both inputs are stable point-in-time data; (b)
operators run the calculation per-trade after dividend confirm,
so one-shot pure functions match the call pattern; (c) the two
methodologies share enough math that fusing them into a single
function with a method parameter keeps the surface area small.

Pinned semantics:
- **Closed-set PurificationMethod ladder.** FULL_AMOUNT (standard)
  / HOLDING_PRORATED (operator-selectable). Adding a method is a
  code review change.
- **Default is FULL_AMOUNT.** The more conservative methodology
  by default; operators wanting prorating opt in explicitly.
- **Eligibility is binary on ex-date.** A holding is eligible for
  the dividend iff `start_date <= ex_date <= end_date` (or
  end_date is None for still-held). The broker pays the dividend
  based on this; our calculator mirrors that determination.
- **Holding fraction is `min(days_held, days_in_period) /
  days_in_period`.** Capped at 1.0 because a holding longer
  than the period doesn't earn extra purification.
- **Render output never includes the per-trade buy/sell prices
  or P&L.** Only the dividend impure portion + purification
  owed; the trade economics live in the operator-side ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class PurificationMethod(str, Enum):
    """Operator-selectable purification methodology.

    Pinned string values for JSON / DB persistence stability.
    FULL_AMOUNT is the standard conservative methodology;
    HOLDING_PRORATED prorates by days held in the period.
    """

    FULL_AMOUNT = "full_amount"
    HOLDING_PRORATED = "holding_prorated"


@dataclass(frozen=True)
class HoldingPeriod:
    """One position's holding window.

    `end_date` is None when the position is still held. The
    holding is valid only when `start_date <= end_date` (when
    end_date is not None).
    """

    holding_id: str
    start_date: date
    end_date: date | None
    share_count: float

    def __post_init__(self) -> None:
        if not self.holding_id or not self.holding_id.strip():
            raise ValueError("holding_id must be non-empty")
        if self.share_count <= 0:
            raise ValueError("share_count must be > 0")
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")


@dataclass(frozen=True)
class DividendEvent:
    """One dividend declaration with its revenue period.

    `period_start` and `period_end` bracket the underlying
    revenue-generation period (e.g., the fiscal quarter the
    dividend covers); `ex_date` is the cutoff for shareholder
    eligibility; `amount_per_share` is the per-share payment;
    `impure_revenue_pct` ∈ [0.0, 1.0] is the screening provider's
    estimate of the haram revenue fraction.
    """

    period_start: date
    period_end: date
    ex_date: date
    amount_per_share: float
    impure_revenue_pct: float

    def __post_init__(self) -> None:
        if self.period_end < self.period_start:
            raise ValueError("period_end must be >= period_start")
        if not (self.period_start <= self.ex_date <= self.period_end):
            raise ValueError("ex_date must be within [period_start, period_end]")
        if self.amount_per_share < 0:
            raise ValueError("amount_per_share must be >= 0")
        if not (0.0 <= self.impure_revenue_pct <= 1.0):
            raise ValueError("impure_revenue_pct must be in [0.0, 1.0]")

    @property
    def days_in_period(self) -> int:
        """Inclusive day count of the revenue period."""

        return (self.period_end - self.period_start).days + 1


@dataclass(frozen=True)
class PurificationAssessment:
    """Output of the calculator for one (holding, dividend) pair."""

    holding_id: str
    eligible: bool
    gross_dividend: float
    impure_amount_full: float
    purification_owed: float
    method_used: PurificationMethod
    days_held_in_period: int
    days_in_period: int
    holding_fraction: float

    def __post_init__(self) -> None:
        if not self.holding_id or not self.holding_id.strip():
            raise ValueError("holding_id must be non-empty")
        if self.gross_dividend < 0:
            raise ValueError("gross_dividend must be >= 0")
        if self.impure_amount_full < 0:
            raise ValueError("impure_amount_full must be >= 0")
        if self.purification_owed < 0:
            raise ValueError("purification_owed must be >= 0")
        if self.purification_owed > self.impure_amount_full + 1e-9:
            raise ValueError("purification_owed cannot exceed impure_amount_full")
        if not (0.0 <= self.holding_fraction <= 1.0):
            raise ValueError("holding_fraction must be in [0.0, 1.0]")
        if self.days_held_in_period < 0:
            raise ValueError("days_held_in_period must be >= 0")
        if self.days_in_period < 1:
            raise ValueError("days_in_period must be >= 1")
        if self.eligible and self.gross_dividend == 0 and self.impure_amount_full == 0:
            # Edge case: zero-dividend event — that's allowed (companies
            # sometimes declare zero special dividends). We don't reject.
            pass
        if not self.eligible and (self.gross_dividend > 0 or self.purification_owed > 0):
            raise ValueError(
                "ineligible holding cannot have gross_dividend or purification_owed > 0"
            )


def _holding_overlap_days(
    holding: HoldingPeriod,
    period_start: date,
    period_end: date,
    *,
    today: date,
) -> int:
    """Inclusive day count where holding overlaps [period_start, period_end].

    A still-held position (end_date=None) is treated as held through
    `today`. Overlap is clamped to [0, days_in_period].
    """

    eff_end = holding.end_date if holding.end_date is not None else today
    overlap_start = max(holding.start_date, period_start)
    overlap_end = min(eff_end, period_end)
    if overlap_end < overlap_start:
        return 0
    return (overlap_end - overlap_start).days + 1


def calculate_purification(
    holding: HoldingPeriod,
    dividend: DividendEvent,
    *,
    today: date,
    method: PurificationMethod = PurificationMethod.FULL_AMOUNT,
) -> PurificationAssessment:
    """Compute purification owed for one (holding, dividend) pair.

    The eligibility check is binary on ex-date: the holding must
    cover the ex-date for the broker to have paid the dividend.
    The purification calculation depends on the method:
    FULL_AMOUNT → impure_pct × full_dividend; HOLDING_PRORATED →
    impure_pct × full_dividend × (days_held_in_period / days_in_period).
    """

    eff_end = holding.end_date if holding.end_date is not None else today
    eligible = holding.start_date <= dividend.ex_date <= eff_end

    days_in_period = dividend.days_in_period
    days_held = _holding_overlap_days(
        holding,
        dividend.period_start,
        dividend.period_end,
        today=today,
    )
    # Cap at days_in_period (a holding longer than the period
    # doesn't earn extra purification).
    days_held = min(days_held, days_in_period)
    holding_fraction = days_held / days_in_period if days_in_period > 0 else 0.0

    if not eligible:
        return PurificationAssessment(
            holding_id=holding.holding_id,
            eligible=False,
            gross_dividend=0.0,
            impure_amount_full=0.0,
            purification_owed=0.0,
            method_used=method,
            days_held_in_period=days_held,
            days_in_period=days_in_period,
            holding_fraction=holding_fraction,
        )

    gross = dividend.amount_per_share * holding.share_count
    impure_full = gross * dividend.impure_revenue_pct

    if method is PurificationMethod.HOLDING_PRORATED:
        owed = impure_full * holding_fraction
    else:
        owed = impure_full

    return PurificationAssessment(
        holding_id=holding.holding_id,
        eligible=True,
        gross_dividend=gross,
        impure_amount_full=impure_full,
        purification_owed=owed,
        method_used=method,
        days_held_in_period=days_held,
        days_in_period=days_in_period,
        holding_fraction=holding_fraction,
    )


def total_owed(assessments: list[PurificationAssessment]) -> float:
    """Sum purification_owed across many assessments."""

    return sum(a.purification_owed for a in assessments)


_METHOD_LABEL: dict[PurificationMethod, str] = {
    PurificationMethod.FULL_AMOUNT: "full amount",
    PurificationMethod.HOLDING_PRORATED: "holding prorated",
}


def render_assessment(assessment: PurificationAssessment) -> str:
    """Format one assessment for ops display.

    No-secret-leak: shows only holding id + eligibility + summary
    numbers. Per-trade buy/sell prices + P&L live in the
    operator-side ledger.
    """

    if not assessment.eligible:
        return f"⏸  {assessment.holding_id}: not eligible (didn't hold ex-date)"
    method_label = _METHOD_LABEL[assessment.method_used]
    return (
        f"💧 {assessment.holding_id}: "
        f"purify {assessment.purification_owed:.4f} "
        f"(of {assessment.impure_amount_full:.4f} impure, "
        f"method {method_label}, "
        f"{assessment.days_held_in_period}/{assessment.days_in_period} days)"
    )


__all__ = [
    "DividendEvent",
    "HoldingPeriod",
    "PurificationAssessment",
    "PurificationMethod",
    "calculate_purification",
    "render_assessment",
    "total_owed",
]
