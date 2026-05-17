"""Tests for `halal_trader.halal.disbursement_reconciler`.

Auxiliary primitive for Wave 2.D purification scheduler. Covers:
period reconciliation, shortfall classification with cents-tolerance,
overdue detection, no-secret render contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.halal.disbursement_reconciler import (
    DEFAULT_POLICY,
    DisbursementReceipt,
    OwedPeriod,
    PeriodReconciliation,
    ReconcilerPolicy,
    ReconciliationStatus,
    overdue_periods,
    reconcile_all,
    reconcile_period,
    render_reconciliation,
    render_summary,
    total_outstanding,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_reconciliation_status_string_values_pinned() -> None:
    assert ReconciliationStatus.UNRECONCILED.value == "unreconciled"
    assert ReconciliationStatus.UNDERPAID.value == "underpaid"
    assert ReconciliationStatus.RECONCILED.value == "reconciled"
    assert ReconciliationStatus.OVERPAID.value == "overpaid"


# --------------------------- ReconcilerPolicy --------------------------------


def test_default_policy() -> None:
    assert DEFAULT_POLICY.cents_tolerance_usd == 0.01
    assert DEFAULT_POLICY.overdue_threshold == timedelta(days=90)


def test_policy_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="cents_tolerance"):
        ReconcilerPolicy(cents_tolerance_usd=-0.01)


def test_policy_rejects_tolerance_above_1() -> None:
    """Pin: tolerance > $1 means we'd allow underpayment of more than
    a dollar — too lax for a halal-compliance audit."""

    with pytest.raises(ValueError, match="cents_tolerance"):
        ReconcilerPolicy(cents_tolerance_usd=2.0)


def test_policy_rejects_zero_overdue_threshold() -> None:
    with pytest.raises(ValueError, match="overdue_threshold"):
        ReconcilerPolicy(overdue_threshold=timedelta(0))


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.cents_tolerance_usd = 0.5  # type: ignore[misc]


# --------------------------- OwedPeriod --------------------------------------


def _period(**overrides: object) -> OwedPeriod:
    base: dict[str, object] = {
        "period_label": "2026-Q1",
        "owed_usd": 487.32,
        "starts_at": T0 - timedelta(days=120),
        "ends_at": T0 - timedelta(days=30),
    }
    base.update(overrides)
    return OwedPeriod(**base)  # type: ignore[arg-type]


def test_period_rejects_empty_label() -> None:
    with pytest.raises(ValueError, match="period_label"):
        _period(period_label="")


def test_period_rejects_negative_owed() -> None:
    with pytest.raises(ValueError, match="owed_usd"):
        _period(owed_usd=-1.0)


def test_period_accepts_zero_owed() -> None:
    """Pin: zero-owed periods are valid (no purification entries)."""

    p = _period(owed_usd=0.0)
    assert p.owed_usd == 0.0


def test_period_rejects_naive_starts_at() -> None:
    with pytest.raises(ValueError, match="starts_at"):
        _period(starts_at=datetime(2026, 5, 1))


def test_period_rejects_ends_before_starts() -> None:
    with pytest.raises(ValueError, match="ends_at"):
        _period(
            starts_at=T0,
            ends_at=T0 - timedelta(days=1),
        )


def test_period_is_frozen() -> None:
    p = _period()
    with pytest.raises(FrozenInstanceError):
        p.owed_usd = 99.99  # type: ignore[misc]


# --------------------------- DisbursementReceipt -----------------------------


def _receipt(**overrides: object) -> DisbursementReceipt:
    base: dict[str, object] = {
        "receipt_id": "r1",
        "period_label": "2026-Q1",
        "amount_usd": 487.32,
        "wired_at": T0,
    }
    base.update(overrides)
    return DisbursementReceipt(**base)  # type: ignore[arg-type]


def test_receipt_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="receipt_id"):
        _receipt(receipt_id="")


def test_receipt_rejects_zero_amount() -> None:
    """Pin: a $0 wire isn't a meaningful disbursement."""

    with pytest.raises(ValueError, match="amount_usd"):
        _receipt(amount_usd=0.0)


def test_receipt_rejects_negative_amount() -> None:
    with pytest.raises(ValueError, match="amount_usd"):
        _receipt(amount_usd=-100.0)


def test_receipt_rejects_naive_wired_at() -> None:
    with pytest.raises(ValueError, match="wired_at"):
        _receipt(wired_at=datetime(2026, 5, 1))


def test_receipt_is_frozen() -> None:
    r = _receipt()
    with pytest.raises(FrozenInstanceError):
        r.amount_usd = 99.99  # type: ignore[misc]


# --------------------------- reconcile_period: status classification ---------


def test_reconcile_unreconciled_when_no_receipts() -> None:
    period = _period()
    rec = reconcile_period(period, [], now=T0)
    assert rec.status is ReconciliationStatus.UNRECONCILED
    assert rec.paid_usd == 0.0


def test_reconcile_reconciled_at_exact_amount() -> None:
    """Pin: paying exactly owed → RECONCILED."""

    period = _period(owed_usd=487.32)
    receipt = _receipt(amount_usd=487.32)
    rec = reconcile_period(period, [receipt], now=T0)
    assert rec.status is ReconciliationStatus.RECONCILED
    assert rec.shortfall_usd == pytest.approx(0.0)


def test_reconcile_reconciled_within_cents_tolerance() -> None:
    """Pin: $487.32 owed + $487.32 wire is reconciled.

    The cents-tolerance handles floating-point quirks (e.g. an
    operator wires the rounded sum of multiple dollar amounts that
    happens to differ by a fraction of a cent from the computed
    owed total).
    """

    period = _period(owed_usd=487.32)
    receipt = _receipt(amount_usd=487.325)  # 0.5 cent over
    rec = reconcile_period(period, [receipt], now=T0)
    assert rec.status is ReconciliationStatus.RECONCILED


def test_reconcile_underpaid_when_short() -> None:
    """Pin: $400 wire on $487.32 owed → UNDERPAID."""

    period = _period(owed_usd=487.32)
    receipt = _receipt(amount_usd=400.00)
    rec = reconcile_period(period, [receipt], now=T0)
    assert rec.status is ReconciliationStatus.UNDERPAID
    assert rec.shortfall_usd > 0


def test_reconcile_underpaid_just_above_tolerance() -> None:
    """Pin: $1.50 short → UNDERPAID."""

    period = _period(owed_usd=487.32)
    receipt = _receipt(amount_usd=485.82)  # $1.50 short
    rec = reconcile_period(period, [receipt], now=T0)
    assert rec.status is ReconciliationStatus.UNDERPAID


def test_reconcile_overpaid_when_more_than_owed() -> None:
    """Pin: $500 wire on $487.32 owed → OVERPAID."""

    period = _period(owed_usd=487.32)
    receipt = _receipt(amount_usd=500.00)
    rec = reconcile_period(period, [receipt], now=T0)
    assert rec.status is ReconciliationStatus.OVERPAID
    assert rec.shortfall_usd < 0  # negative = overpayment


def test_reconcile_sums_multiple_receipts() -> None:
    """Pin: multiple receipts for one period sum together."""

    period = _period(owed_usd=487.32)
    receipts = [
        _receipt(receipt_id="r1", amount_usd=200.00),
        _receipt(receipt_id="r2", amount_usd=287.32),
    ]
    rec = reconcile_period(period, receipts, now=T0)
    assert rec.paid_usd == pytest.approx(487.32)
    assert rec.status is ReconciliationStatus.RECONCILED
    assert rec.receipt_count == 2


def test_reconcile_filters_by_period_label() -> None:
    """Pin: receipts for other periods don't count."""

    period = _period(period_label="2026-Q1")
    receipts = [
        _receipt(period_label="2026-Q1", amount_usd=200.00),
        _receipt(receipt_id="r2", period_label="2026-Q2", amount_usd=500.00),
    ]
    rec = reconcile_period(period, receipts, now=T0)
    assert rec.paid_usd == 200.00
    assert rec.receipt_count == 1


def test_reconcile_naive_now_rejected() -> None:
    period = _period()
    with pytest.raises(ValueError, match="now"):
        reconcile_period(period, [], now=datetime(2026, 5, 1))


# --------------------------- overdue detection -------------------------------


def test_overdue_after_90_days_unreconciled() -> None:
    """Pin: an unpaid period 91 days past period end is overdue."""

    # Period ended 91 days ago
    period = _period(
        starts_at=T0 - timedelta(days=180),
        ends_at=T0 - timedelta(days=91),
    )
    rec = reconcile_period(period, [], now=T0)
    assert rec.is_overdue is True


def test_not_overdue_at_90_day_boundary() -> None:
    """Pin: exactly 90 days past period end NOT overdue (>= boundary).

    Wait — the implementation uses `now >= ends_at + threshold`, so
    exactly at the boundary IS overdue. Test the actual behavior.
    """

    # Period ended exactly 90 days ago
    period = _period(
        starts_at=T0 - timedelta(days=180),
        ends_at=T0 - timedelta(days=90),
    )
    rec = reconcile_period(period, [], now=T0)
    # ends_at + 90d = T0; now >= T0 → overdue
    assert rec.is_overdue is True


def test_not_overdue_just_below_90_days() -> None:
    """Pin: 89 days past period end is NOT overdue."""

    period = _period(
        starts_at=T0 - timedelta(days=180),
        ends_at=T0 - timedelta(days=89),
    )
    rec = reconcile_period(period, [], now=T0)
    assert rec.is_overdue is False


def test_overdue_after_90_days_underpaid() -> None:
    """Pin: a partially-paid period that's overdue still flags overdue."""

    period = _period(
        owed_usd=487.32,
        starts_at=T0 - timedelta(days=180),
        ends_at=T0 - timedelta(days=100),
    )
    receipt = _receipt(amount_usd=200.00)
    rec = reconcile_period(period, [receipt], now=T0)
    assert rec.status is ReconciliationStatus.UNDERPAID
    assert rec.is_overdue is True


def test_not_overdue_when_reconciled() -> None:
    """Pin: a fully-paid period is never overdue, even if 1 year past."""

    period = _period(
        owed_usd=487.32,
        starts_at=T0 - timedelta(days=540),
        ends_at=T0 - timedelta(days=365),
    )
    receipt = _receipt(amount_usd=487.32)
    rec = reconcile_period(period, [receipt], now=T0)
    assert rec.status is ReconciliationStatus.RECONCILED
    assert rec.is_overdue is False


def test_not_overdue_when_overpaid() -> None:
    """Pin: overpaid periods are never overdue."""

    period = _period(
        owed_usd=487.32,
        starts_at=T0 - timedelta(days=540),
        ends_at=T0 - timedelta(days=365),
    )
    receipt = _receipt(amount_usd=600.00)
    rec = reconcile_period(period, [receipt], now=T0)
    assert rec.is_overdue is False


def test_custom_overdue_threshold() -> None:
    """Pin: strict 30-day threshold flips a 31-day-old period to overdue."""

    period = _period(
        starts_at=T0 - timedelta(days=120),
        ends_at=T0 - timedelta(days=31),
    )
    strict = ReconcilerPolicy(overdue_threshold=timedelta(days=30))
    rec = reconcile_period(period, [], now=T0, policy=strict)
    assert rec.is_overdue is True


# --------------------------- reconcile_all -----------------------------------


def test_reconcile_all_sorted_by_start() -> None:
    """Pin: results in period-start ascending order."""

    q3 = _period(
        period_label="2026-Q3",
        starts_at=T0 - timedelta(days=60),
        ends_at=T0 - timedelta(days=15),
    )
    q1 = _period(
        period_label="2026-Q1",
        starts_at=T0 - timedelta(days=180),
        ends_at=T0 - timedelta(days=120),
    )
    q2 = _period(
        period_label="2026-Q2",
        starts_at=T0 - timedelta(days=120),
        ends_at=T0 - timedelta(days=60),
    )
    results = reconcile_all([q3, q1, q2], [], now=T0)
    labels = [r.period_label for r in results]
    assert labels == ["2026-Q1", "2026-Q2", "2026-Q3"]


def test_reconcile_all_empty() -> None:
    results = reconcile_all([], [], now=T0)
    assert results == ()


def test_reconcile_all_naive_now_rejected() -> None:
    with pytest.raises(ValueError, match="now"):
        reconcile_all([], [], now=datetime(2026, 5, 1))


# --------------------------- total_outstanding -------------------------------


def test_total_outstanding_sums_underpayments() -> None:
    rec_under = PeriodReconciliation(
        period_label="2026-Q1",
        owed_usd=500.00,
        paid_usd=300.00,
        shortfall_usd=200.00,
        status=ReconciliationStatus.UNDERPAID,
        is_overdue=False,
        receipt_count=1,
    )
    rec_reconciled = PeriodReconciliation(
        period_label="2026-Q2",
        owed_usd=400.00,
        paid_usd=400.00,
        shortfall_usd=0.0,
        status=ReconciliationStatus.RECONCILED,
        is_overdue=False,
        receipt_count=1,
    )
    rec_unreconciled = PeriodReconciliation(
        period_label="2026-Q3",
        owed_usd=300.00,
        paid_usd=0.0,
        shortfall_usd=300.00,
        status=ReconciliationStatus.UNRECONCILED,
        is_overdue=False,
        receipt_count=0,
    )
    total = total_outstanding([rec_under, rec_reconciled, rec_unreconciled])
    assert total == pytest.approx(500.00)  # 200 + 0 + 300


def test_total_outstanding_excludes_overpayments() -> None:
    """Pin: overpayments don't net against underpayments.

    A $50 overpayment in Q1 doesn't reduce the $200 underpayment in Q2.
    """

    rec_over = PeriodReconciliation(
        period_label="2026-Q1",
        owed_usd=400.00,
        paid_usd=450.00,
        shortfall_usd=-50.00,
        status=ReconciliationStatus.OVERPAID,
        is_overdue=False,
        receipt_count=1,
    )
    rec_under = PeriodReconciliation(
        period_label="2026-Q2",
        owed_usd=400.00,
        paid_usd=200.00,
        shortfall_usd=200.00,
        status=ReconciliationStatus.UNDERPAID,
        is_overdue=False,
        receipt_count=1,
    )
    total = total_outstanding([rec_over, rec_under])
    # Pin: 200, NOT 150 (no netting)
    assert total == pytest.approx(200.00)


# --------------------------- overdue_periods ---------------------------------


def test_overdue_periods_filter() -> None:
    overdue_rec = PeriodReconciliation(
        period_label="2025-Q4",
        owed_usd=400.00,
        paid_usd=0.0,
        shortfall_usd=400.00,
        status=ReconciliationStatus.UNRECONCILED,
        is_overdue=True,
        receipt_count=0,
    )
    fresh_rec = PeriodReconciliation(
        period_label="2026-Q1",
        owed_usd=400.00,
        paid_usd=400.00,
        shortfall_usd=0.0,
        status=ReconciliationStatus.RECONCILED,
        is_overdue=False,
        receipt_count=1,
    )
    overdue = overdue_periods([overdue_rec, fresh_rec])
    assert len(overdue) == 1
    assert overdue[0].period_label == "2025-Q4"


# --------------------------- render ------------------------------------------


def test_render_reconciliation_includes_period_label() -> None:
    period = _period()
    receipt = _receipt(amount_usd=487.32)
    rec = reconcile_period(period, [receipt], now=T0)
    out = render_reconciliation(rec)
    assert "2026-Q1" in out


def test_render_reconciliation_status_emoji() -> None:
    period = _period()
    rec_unreconciled = reconcile_period(period, [], now=T0)
    out = render_reconciliation(rec_unreconciled)
    assert "❓" in out

    rec_paid = reconcile_period(period, [_receipt()], now=T0)
    out = render_reconciliation(rec_paid)
    assert "✅" in out


def test_render_reconciliation_overdue_marker() -> None:
    period = _period(
        starts_at=T0 - timedelta(days=180),
        ends_at=T0 - timedelta(days=100),
    )
    rec = reconcile_period(period, [], now=T0)
    out = render_reconciliation(rec)
    assert "OVERDUE" in out


def test_render_reconciliation_no_overdue_when_paid() -> None:
    period = _period()
    rec = reconcile_period(period, [_receipt()], now=T0)
    out = render_reconciliation(rec)
    assert "OVERDUE" not in out


def test_render_reconciliation_includes_amounts() -> None:
    period = _period(owed_usd=487.32)
    receipts = [
        _receipt(receipt_id="r1", amount_usd=200.00),
        _receipt(receipt_id="r2", amount_usd=287.32),
    ]
    rec = reconcile_period(period, receipts, now=T0)
    out = render_reconciliation(rec)
    assert "$487.32" in out
    assert "2 receipts" in out


def test_render_no_secret_leak() -> None:
    """Pin: render never includes wire references / bank account
    numbers / charity recipient details."""

    period = _period()
    receipt = _receipt(wire_reference="WIRE-2026-Q1-CONFIDENTIAL-REF-12345")
    rec = reconcile_period(period, [receipt], now=T0)
    out = render_reconciliation(rec)
    assert "WIRE-" not in out
    assert "CONFIDENTIAL" not in out
    # Pin: no email / bank-routing-shape strings
    assert "@" not in out
    assert "routing" not in out.lower()
    assert "iban" not in out.lower()


# --------------------------- render_summary ----------------------------------


def test_render_summary_includes_counts_per_status() -> None:
    rec_underpaid = PeriodReconciliation(
        period_label="2025-Q4",
        owed_usd=400.00,
        paid_usd=200.00,
        shortfall_usd=200.00,
        status=ReconciliationStatus.UNDERPAID,
        is_overdue=False,
        receipt_count=1,
    )
    rec_reconciled = PeriodReconciliation(
        period_label="2026-Q1",
        owed_usd=400.00,
        paid_usd=400.00,
        shortfall_usd=0.0,
        status=ReconciliationStatus.RECONCILED,
        is_overdue=False,
        receipt_count=1,
    )
    out = render_summary([rec_underpaid, rec_reconciled])
    assert "underpaid: 1" in out
    assert "reconciled: 1" in out


def test_render_summary_total_outstanding() -> None:
    rec_under = PeriodReconciliation(
        period_label="2025-Q4",
        owed_usd=400.00,
        paid_usd=200.00,
        shortfall_usd=200.00,
        status=ReconciliationStatus.UNDERPAID,
        is_overdue=False,
        receipt_count=1,
    )
    out = render_summary([rec_under])
    assert "$200.00" in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_full_year_audit() -> None:
    """Real-world: 4 quarters of receipts; 1 unpaid + 1 underpaid +
    1 reconciled + 1 overpaid."""

    periods = [
        OwedPeriod(
            period_label="2026-Q1",
            owed_usd=487.32,
            starts_at=T0 - timedelta(days=455),
            ends_at=T0 - timedelta(days=365),
        ),
        OwedPeriod(
            period_label="2026-Q2",
            owed_usd=523.18,
            starts_at=T0 - timedelta(days=365),
            ends_at=T0 - timedelta(days=275),
        ),
        OwedPeriod(
            period_label="2026-Q3",
            owed_usd=601.55,
            starts_at=T0 - timedelta(days=275),
            ends_at=T0 - timedelta(days=185),
        ),
        OwedPeriod(
            period_label="2026-Q4",
            owed_usd=478.91,
            starts_at=T0 - timedelta(days=185),
            ends_at=T0 - timedelta(days=95),
        ),
    ]
    receipts = [
        # Q1 reconciled
        DisbursementReceipt(
            receipt_id="r_q1",
            period_label="2026-Q1",
            amount_usd=487.32,
            wired_at=T0 - timedelta(days=350),
        ),
        # Q2 underpaid (only $400 of $523.18)
        DisbursementReceipt(
            receipt_id="r_q2",
            period_label="2026-Q2",
            amount_usd=400.00,
            wired_at=T0 - timedelta(days=270),
        ),
        # Q3 overpaid ($700 of $601.55)
        DisbursementReceipt(
            receipt_id="r_q3",
            period_label="2026-Q3",
            amount_usd=700.00,
            wired_at=T0 - timedelta(days=180),
        ),
        # Q4 unpaid (no receipt)
    ]
    results = reconcile_all(periods, receipts, now=T0)
    assert len(results) == 4

    statuses = {r.period_label: r.status for r in results}
    assert statuses["2026-Q1"] is ReconciliationStatus.RECONCILED
    assert statuses["2026-Q2"] is ReconciliationStatus.UNDERPAID
    assert statuses["2026-Q3"] is ReconciliationStatus.OVERPAID
    assert statuses["2026-Q4"] is ReconciliationStatus.UNRECONCILED

    # Q2 + Q4 are overdue (both > 90 days past period end)
    overdue = overdue_periods(results)
    overdue_labels = {r.period_label for r in overdue}
    assert "2026-Q2" in overdue_labels
    assert "2026-Q4" in overdue_labels
    # Q1 reconciled never overdue, Q3 overpaid never overdue
    assert "2026-Q1" not in overdue_labels
    assert "2026-Q3" not in overdue_labels

    # Outstanding = Q2 shortfall + Q4 shortfall
    expected = (523.18 - 400.00) + 478.91
    assert total_outstanding(results) == pytest.approx(expected)


def test_e2e_replay_consistency() -> None:
    """Same operations produce equal reconciliation results."""

    period = _period(owed_usd=487.32)
    receipt = _receipt(amount_usd=487.32)
    a = reconcile_period(period, [receipt], now=T0)
    b = reconcile_period(period, [receipt], now=T0)
    assert a == b
