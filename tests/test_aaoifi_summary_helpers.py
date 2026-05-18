"""Tests for the pure helpers + dataclass in `halal/aaoifi_summary.py`.

The DB-aggregating `compute_aaoifi_summary` runs against the live
Postgres engine and is covered by the integration suite. This file
pins the in-memory surface — boundary helpers + the `AAOIFISummary`
status / compliance / outstanding-purification properties — so a
refactor that flips a comparison breaks here first.
"""

from __future__ import annotations

from datetime import UTC, datetime

from halal_trader.halal.aaoifi_summary import (
    AAOIFISummary,
    _month_start_utc,
    _quarter_start_utc,
    _today_start_utc,
)

# ── Boundary helpers ──────────────────────────────────────


def test_quarter_start_for_january_returns_january_first():
    """Q1: months 1, 2, 3 all map to Jan 1."""
    for month in (1, 2, 3):
        now = datetime(2026, month, 15, 12, 0, tzinfo=UTC)
        assert _quarter_start_utc(now) == datetime(2026, 1, 1, tzinfo=UTC)


def test_quarter_start_for_april_returns_april_first():
    """Q2: months 4, 5, 6 map to Apr 1."""
    for month in (4, 5, 6):
        now = datetime(2026, month, 20, 9, 30, tzinfo=UTC)
        assert _quarter_start_utc(now) == datetime(2026, 4, 1, tzinfo=UTC)


def test_quarter_start_for_july_returns_july_first():
    for month in (7, 8, 9):
        now = datetime(2026, month, 1, tzinfo=UTC)
        assert _quarter_start_utc(now) == datetime(2026, 7, 1, tzinfo=UTC)


def test_quarter_start_for_october_returns_october_first():
    for month in (10, 11, 12):
        # Use day=15 (always valid across all months) to avoid
        # 30-vs-31-day month confusion.
        now = datetime(2026, month, 15, 23, 59, tzinfo=UTC)
        assert _quarter_start_utc(now) == datetime(2026, 10, 1, tzinfo=UTC)


def test_quarter_start_preserves_year():
    """Quarter boundary doesn't drift across the calendar year."""
    now = datetime(2025, 8, 15, tzinfo=UTC)
    out = _quarter_start_utc(now)
    assert out.year == 2025
    assert out.month == 7


def test_quarter_start_returns_utc_tzaware():
    """Return value is always tz-aware UTC — pinned so the SQL
    comparison against tz-aware columns works correctly."""
    now = datetime(2026, 5, 1, tzinfo=UTC)
    assert _quarter_start_utc(now).tzinfo is UTC


def test_month_start_drops_day_and_time():
    now = datetime(2026, 4, 25, 14, 30, 45, tzinfo=UTC)
    assert _month_start_utc(now) == datetime(2026, 4, 1, tzinfo=UTC)


def test_today_start_drops_time():
    now = datetime(2026, 4, 25, 14, 30, 45, tzinfo=UTC)
    assert _today_start_utc(now) == datetime(2026, 4, 25, tzinfo=UTC)


def test_today_start_returns_utc_tzaware():
    now = datetime(2026, 4, 25, 14, 0, tzinfo=UTC)
    assert _today_start_utc(now).tzinfo is UTC


# ── AAOIFISummary properties ──────────────────────────────


def _summary(
    *,
    non_halal_fills: int = 0,
    accrued: float = 0.0,
    disbursed: float = 0.0,
) -> AAOIFISummary:
    """Build a summary with controllable invariants for the property tests."""
    return AAOIFISummary(
        quarter_start=datetime(2026, 4, 1, tzinfo=UTC),
        month_start=datetime(2026, 4, 1, tzinfo=UTC),
        today_start=datetime(2026, 4, 25, tzinfo=UTC),
        trades_today=0,
        trades_this_month=0,
        trades_this_quarter=0,
        halal_screenings_quarter=0,
        doubtful_screenings_quarter=0,
        not_halal_screenings_quarter=0,
        non_halal_fills_quarter=non_halal_fills,
        purification_accrued_usd=accrued,
        purification_disbursed_usd=disbursed,
    )


def test_outstanding_is_accrued_minus_disbursed():
    s = _summary(accrued=100.0, disbursed=30.0)
    assert s.purification_outstanding_usd == 70.0


def test_outstanding_floors_at_zero_when_disbursed_exceeds_accrued():
    """Defensive: if the operator over-disbursed (or a refund flow
    later gets wired), outstanding stays >= 0 rather than going
    negative."""
    s = _summary(accrued=10.0, disbursed=50.0)
    assert s.purification_outstanding_usd == 0.0


def test_outstanding_zero_when_both_zero():
    s = _summary()
    assert s.purification_outstanding_usd == 0.0


def test_is_compliant_true_when_zero_non_halal_fills():
    s = _summary(non_halal_fills=0)
    assert s.is_compliant is True


def test_is_compliant_false_with_any_non_halal_fill():
    """A single non-halal fill flips compliance to False — the whole
    point of the tile. Pin so a refactor doesn't accidentally
    threshold this."""
    s = _summary(non_halal_fills=1)
    assert s.is_compliant is False


def test_status_violation_takes_priority_over_attention():
    """Even with outstanding purification, a non-halal fill renders
    as 'violation' — the more severe state wins."""
    s = _summary(non_halal_fills=1, accrued=100.0, disbursed=0.0)
    assert s.status == "violation"


def test_status_attention_when_outstanding_purification():
    """No violations + outstanding purification → 'attention' (amber tile)."""
    s = _summary(non_halal_fills=0, accrued=100.0, disbursed=0.0)
    assert s.status == "attention"


def test_status_attention_when_partially_disbursed():
    s = _summary(non_halal_fills=0, accrued=100.0, disbursed=50.0)
    assert s.status == "attention"


def test_status_compliant_when_all_disbursed():
    s = _summary(non_halal_fills=0, accrued=100.0, disbursed=100.0)
    assert s.status == "compliant"


def test_status_compliant_when_no_purification_owed():
    s = _summary()
    assert s.status == "compliant"


def test_status_treats_sub_one_cent_as_compliant():
    """Defensive: floating-point residue under $0.01 is rounding
    noise; tile shouldn't go amber for a rounding-error
    outstanding amount."""
    s = _summary(accrued=100.001, disbursed=100.0)
    assert s.status == "compliant"


def test_status_treats_one_cent_or_more_as_attention():
    """A real residual disbursement obligation flips to attention."""
    s = _summary(accrued=100.02, disbursed=100.0)
    assert s.status == "attention"


def test_summary_dataclass_is_frozen():
    s = _summary()
    import pytest

    with pytest.raises(Exception):
        s.trades_today = 999  # type: ignore[misc]
