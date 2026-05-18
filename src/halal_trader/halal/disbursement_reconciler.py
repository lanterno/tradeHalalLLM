"""Charity disbursement reconciliation.

Auxiliary primitive for Wave 2.D purification scheduler. Wave
2.D's `purification_schedule.py` ships the owed-side: it groups
purification entries into quarterly receipts and tells the
operator "you owe $X this quarter for charity disbursement". This
module ships the **paid-side reconciliation**: matches disbursement
receipts (the operator's actual bank wire confirmations) against
the owed-receipt totals, computes per-period shortfall or
overpayment, and flags periods that are overdue.

Picked a focused reconciler over a generic accounting tool because
(a) the halal-compliance audit trail must be inspectable: a
scholar reviewing the bot's purification record asks "did the
operator actually pay $487.32 to charity in 2026-Q1?" and the
answer must be a single deterministic match against the operator's
bank-wire confirmation, not require cross-referencing multiple
spreadsheets, (b) the shortfall threshold (default $0.01 — full
payment required modulo cents rounding) is the load-bearing audit
attribute — anything more permissive would let the operator
under-pay and still appear reconciled, which would defeat the
purpose; (c) overdue detection (default 90 days past period end)
flags periods where the operator hasn't yet disbursed, which is
the most common operational failure (operator gets the receipt,
forgets to wire, period drifts months out).

Pinned semantics:
- **PERIOD_RECONCILED status requires zero shortfall** within the
  $0.01 cents-rounding tolerance. Any shortfall above that is
  PERIOD_UNDERPAID. Pinned via test against silent shortfall
  drift.
- **Overpayment is allowed.** If operator pays $500 against a
  $487 owed receipt, the period is reconciled with $13 credit
  (carries forward to the next period at operator-side discretion).
- **Overdue threshold default 90 days.** A period whose
  `ends_at + 90d <= now` and is still UNDERPAID/UNRECONCILED
  flags overdue.
- **One disbursement record per bank wire.** Multiple wires
  against a single period sum together.
- **Render output never includes the operator's bank account
  number, charity recipient bank details, or wire-transfer
  reference IDs.** Mirrors no-secret patterns of upstream waves.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class ReconciliationStatus(str, Enum):
    """Per-period reconciliation status.

    Pinned string values for JSON / DB stability.
    """

    UNRECONCILED = "unreconciled"  # No disbursements yet
    UNDERPAID = "underpaid"  # Some disbursed but < owed
    RECONCILED = "reconciled"  # Fully paid (within tolerance)
    OVERPAID = "overpaid"  # Paid more than owed (credit forward)


_DEFAULT_CENTS_TOLERANCE = 0.01
_DEFAULT_OVERDUE_THRESHOLD = timedelta(days=90)


@dataclass(frozen=True)
class ReconcilerPolicy:
    """Operator-tunable reconciliation policy."""

    cents_tolerance_usd: float = _DEFAULT_CENTS_TOLERANCE
    overdue_threshold: timedelta = _DEFAULT_OVERDUE_THRESHOLD

    def __post_init__(self) -> None:
        if self.cents_tolerance_usd < 0:
            raise ValueError("cents_tolerance_usd must be non-negative")
        if self.cents_tolerance_usd > 1.0:
            raise ValueError(
                f"cents_tolerance_usd {self.cents_tolerance_usd} too lax "
                f"(> $1 means we allow underpayment of more than a dollar)"
            )
        if self.overdue_threshold <= timedelta(0):
            raise ValueError("overdue_threshold must be positive")


DEFAULT_POLICY = ReconcilerPolicy()


@dataclass(frozen=True)
class OwedPeriod:
    """A single period's owed amount (from Wave 2.D scheduler).

    Mirrors the shape of Wave 2.D `DisbursementReceipt` but with the
    minimum fields needed for reconciliation. Operators bridge by
    extracting these fields from the scheduler output.
    """

    period_label: str  # "2026-Q1", "2026-03", "2026", etc
    owed_usd: float
    starts_at: datetime  # period start (inclusive)
    ends_at: datetime  # period end (exclusive)

    def __post_init__(self) -> None:
        if not self.period_label or not self.period_label.strip():
            raise ValueError("period_label must be non-empty")
        if self.owed_usd < 0:
            raise ValueError("owed_usd must be non-negative")
        if self.starts_at.tzinfo is None:
            raise ValueError("starts_at must be timezone-aware")
        if self.ends_at.tzinfo is None:
            raise ValueError("ends_at must be timezone-aware")
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be after starts_at")


@dataclass(frozen=True)
class DisbursementReceipt:
    """One bank-wire confirmation that a disbursement happened.

    `period_label` matches the OwedPeriod the wire was paying down.
    `wire_reference` is operator-provided audit metadata (NOT
    rendered — operators store the actual wire reference in their
    accounting system, the dataclass just keeps the link).
    """

    receipt_id: str
    period_label: str
    amount_usd: float
    wired_at: datetime
    wire_reference: str = ""  # operator audit tag; not rendered

    def __post_init__(self) -> None:
        if not self.receipt_id or not self.receipt_id.strip():
            raise ValueError("receipt_id must be non-empty")
        if not self.period_label or not self.period_label.strip():
            raise ValueError("period_label must be non-empty")
        if self.amount_usd <= 0:
            raise ValueError("amount_usd must be positive")
        if self.wired_at.tzinfo is None:
            raise ValueError("wired_at must be timezone-aware")


@dataclass(frozen=True)
class PeriodReconciliation:
    """Computed reconciliation for one period."""

    period_label: str
    owed_usd: float
    paid_usd: float  # sum of receipts for this period
    shortfall_usd: float  # owed - paid (positive = under; negative = over)
    status: ReconciliationStatus
    is_overdue: bool
    receipt_count: int

    def __post_init__(self) -> None:
        if not self.period_label or not self.period_label.strip():
            raise ValueError("period_label must be non-empty")
        if self.owed_usd < 0:
            raise ValueError("owed_usd must be non-negative")
        if self.paid_usd < 0:
            raise ValueError("paid_usd must be non-negative")
        if self.receipt_count < 0:
            raise ValueError("receipt_count must be non-negative")


def _classify(
    *,
    owed: float,
    paid: float,
    tolerance: float,
) -> ReconciliationStatus:
    """Determine reconciliation status."""

    if paid <= 0:
        return ReconciliationStatus.UNRECONCILED
    shortfall = owed - paid
    # Reconciled: shortfall is within tolerance (positive or negative)
    if -tolerance <= shortfall <= tolerance:
        return ReconciliationStatus.RECONCILED
    if shortfall > tolerance:
        return ReconciliationStatus.UNDERPAID
    # shortfall < -tolerance → overpaid
    return ReconciliationStatus.OVERPAID


def reconcile_period(
    period: OwedPeriod,
    receipts: Iterable[DisbursementReceipt],
    *,
    now: datetime,
    policy: ReconcilerPolicy = DEFAULT_POLICY,
) -> PeriodReconciliation:
    """Compute reconciliation for one period given a set of receipts.

    `receipts` may include receipts from other periods; only those
    matching `period.period_label` count toward the paid amount.
    The function is pure: deterministic for given (period, receipts,
    now, policy).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    matching = [r for r in receipts if r.period_label == period.period_label]
    paid = sum(r.amount_usd for r in matching)
    shortfall = period.owed_usd - paid
    status = _classify(
        owed=period.owed_usd,
        paid=paid,
        tolerance=policy.cents_tolerance_usd,
    )
    is_overdue = (
        status in (ReconciliationStatus.UNRECONCILED, ReconciliationStatus.UNDERPAID)
        and now >= period.ends_at + policy.overdue_threshold
    )
    return PeriodReconciliation(
        period_label=period.period_label,
        owed_usd=period.owed_usd,
        paid_usd=paid,
        shortfall_usd=shortfall,
        status=status,
        is_overdue=is_overdue,
        receipt_count=len(matching),
    )


def reconcile_all(
    periods: Iterable[OwedPeriod],
    receipts: Iterable[DisbursementReceipt],
    *,
    now: datetime,
    policy: ReconcilerPolicy = DEFAULT_POLICY,
) -> tuple[PeriodReconciliation, ...]:
    """Reconcile every period (sorted by period start ascending)."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    receipt_list = list(receipts)
    sorted_periods = sorted(periods, key=lambda p: p.starts_at)
    return tuple(reconcile_period(p, receipt_list, now=now, policy=policy) for p in sorted_periods)


def total_outstanding(
    reconciliations: Iterable[PeriodReconciliation],
) -> float:
    """Sum of positive shortfall across reconciliations.

    Operator's "what do I still owe across all periods?" answer.
    Negative shortfalls (overpayments) are NOT netted — operators
    don't get to subtract overpayment from a different period's
    underpayment without explicit operator-side accounting decision.
    """

    return sum(max(0.0, r.shortfall_usd) for r in reconciliations)


def overdue_periods(
    reconciliations: Iterable[PeriodReconciliation],
) -> tuple[PeriodReconciliation, ...]:
    """Return reconciliations flagged overdue (load-bearing audit tile)."""

    return tuple(r for r in reconciliations if r.is_overdue)


_STATUS_EMOJI: dict[ReconciliationStatus, str] = {
    ReconciliationStatus.UNRECONCILED: "❓",
    ReconciliationStatus.UNDERPAID: "⚠️",
    ReconciliationStatus.RECONCILED: "✅",
    ReconciliationStatus.OVERPAID: "💚",
}


def render_reconciliation(reconciliation: PeriodReconciliation) -> str:
    """Format a per-period reconciliation for ops display.

    No-secret-leak: never includes wire references / bank account
    numbers / charity recipient details. Shows period label + owed
    + paid + shortfall + status emoji.
    """

    emoji = _STATUS_EMOJI[reconciliation.status]
    overdue_marker = " ⏰ OVERDUE" if reconciliation.is_overdue else ""
    return (
        f"{emoji} {reconciliation.period_label}{overdue_marker}\n"
        f"  owed: ${reconciliation.owed_usd:.2f}\n"
        f"  paid: ${reconciliation.paid_usd:.2f} "
        f"({reconciliation.receipt_count} receipts)\n"
        f"  shortfall: ${reconciliation.shortfall_usd:+.2f}\n"
        f"  status: {reconciliation.status.value}"
    )


def render_summary(
    reconciliations: Iterable[PeriodReconciliation],
) -> str:
    """Format a summary across all periods.

    Aggregates outstanding + overdue counts.
    """

    rec_list = list(reconciliations)
    outstanding = total_outstanding(rec_list)
    overdue = overdue_periods(rec_list)
    counts: dict[ReconciliationStatus, int] = {s: 0 for s in ReconciliationStatus}
    for r in rec_list:
        counts[r.status] += 1
    return (
        f"💰 Charity disbursement summary — {len(rec_list)} periods\n"
        f"  ❓ unreconciled: {counts[ReconciliationStatus.UNRECONCILED]}\n"
        f"  ⚠️ underpaid: {counts[ReconciliationStatus.UNDERPAID]}\n"
        f"  ✅ reconciled: {counts[ReconciliationStatus.RECONCILED]}\n"
        f"  💚 overpaid: {counts[ReconciliationStatus.OVERPAID]}\n"
        f"  ⏰ overdue: {len(overdue)}\n"
        f"  total outstanding: ${outstanding:.2f}"
    )


__all__ = [
    "DEFAULT_POLICY",
    "DisbursementReceipt",
    "OwedPeriod",
    "PeriodReconciliation",
    "ReconcilerPolicy",
    "ReconciliationStatus",
    "overdue_periods",
    "reconcile_all",
    "reconcile_period",
    "render_reconciliation",
    "render_summary",
    "total_outstanding",
]
