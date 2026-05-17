"""Tests for `halal/purification_schedule.py`.

Pins the period-bucketing math (UTC quarter / month / year edges),
the include_paid filter contract, the per-symbol concentration
sort, the markdown receipt body, the upcoming-due convenience, and
the explicit "scheduler MUST NOT auto-mark paid" invariant.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from halal_trader.halal.purification import PurificationEntry
from halal_trader.halal.purification_schedule import (
    DisbursementReceipt,
    Period,
    SymbolBreakdown,
    schedule_disbursements,
    upcoming_due,
)


def _entry(
    *,
    symbol: str = "AAPL",
    purification_usd: float = 1.50,
    received_at: datetime,
    paid_at: datetime | None = None,
) -> PurificationEntry:
    return PurificationEntry(
        symbol=symbol,
        dividend_usd=Decimal("100.00"),
        haram_pct=Decimal("0.015"),
        purification_usd=Decimal(str(purification_usd)).quantize(Decimal("0.01")),
        received_at=received_at,
        paid_at=paid_at,
    )


# ── period bucketing ─────────────────────────────────────


def test_quarterly_bucket_label_is_year_q():
    """Pin the slug format — `2026-Q1` is the audit-trail key
    every receipt and PDF filename will share."""
    e = _entry(received_at=datetime(2026, 2, 15, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert r.label == "2026-Q1"


def test_quarterly_january_lands_in_q1():
    e = _entry(received_at=datetime(2026, 1, 1, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert r.label == "2026-Q1"


def test_quarterly_april_lands_in_q2():
    """Pin the (month-1)//3 + 1 math — April → Q2, not Q1."""
    e = _entry(received_at=datetime(2026, 4, 1, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert r.label == "2026-Q2"


def test_quarterly_december_lands_in_q4():
    e = _entry(received_at=datetime(2026, 12, 31, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert r.label == "2026-Q4"


def test_monthly_label_is_zero_padded_year_month():
    e = _entry(received_at=datetime(2026, 3, 15, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.MONTHLY)
    assert r.label == "2026-03"


def test_yearly_label_is_just_year():
    e = _entry(received_at=datetime(2026, 7, 15, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.YEARLY)
    assert r.label == "2026"


def test_bucket_groups_multiple_entries_in_same_period():
    e1 = _entry(received_at=datetime(2026, 1, 5, tzinfo=UTC))
    e2 = _entry(received_at=datetime(2026, 2, 20, tzinfo=UTC))
    e3 = _entry(received_at=datetime(2026, 3, 10, tzinfo=UTC))
    receipts = schedule_disbursements([e1, e2, e3], period=Period.QUARTERLY)
    assert len(receipts) == 1
    assert receipts[0].entry_count == 3


def test_bucket_splits_entries_across_periods():
    e1 = _entry(received_at=datetime(2026, 2, 15, tzinfo=UTC))  # Q1
    e2 = _entry(received_at=datetime(2026, 5, 10, tzinfo=UTC))  # Q2
    e3 = _entry(received_at=datetime(2026, 11, 20, tzinfo=UTC))  # Q4
    receipts = schedule_disbursements([e1, e2, e3], period=Period.QUARTERLY)
    labels = [r.label for r in receipts]
    assert labels == ["2026-Q1", "2026-Q2", "2026-Q4"]


def test_bucket_orders_periods_chronologically():
    """Receipts must come back oldest first so the dashboard /
    CLI doesn't need to sort. Pin against accidental dict-order."""
    e1 = _entry(received_at=datetime(2026, 11, 1, tzinfo=UTC))  # Q4
    e2 = _entry(received_at=datetime(2026, 2, 1, tzinfo=UTC))  # Q1
    receipts = schedule_disbursements([e1, e2], period=Period.QUARTERLY)
    assert [r.label for r in receipts] == ["2026-Q1", "2026-Q4"]


def test_naive_received_at_is_treated_as_utc():
    """Legacy ledger rows might be tz-naive. Pin: scheduler tolerates
    them by coercing to UTC, doesn't crash."""
    e = _entry(received_at=datetime(2026, 1, 15))
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert r.label == "2026-Q1"


# ── include_paid filter ──────────────────────────────────


def test_paid_entries_excluded_by_default():
    """Default behaviour is "outstanding only" — operator wants to
    see what's still due, not what's settled."""
    paid = _entry(
        received_at=datetime(2026, 1, 15, tzinfo=UTC),
        paid_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    outstanding = _entry(received_at=datetime(2026, 1, 20, tzinfo=UTC))
    receipts = schedule_disbursements([paid, outstanding], period=Period.QUARTERLY)
    assert len(receipts) == 1
    assert receipts[0].entry_count == 1
    assert receipts[0].entries[0] is outstanding


def test_include_paid_includes_settled_entries():
    """Backfill / audit runs need the full picture; pin the opt-in."""
    paid = _entry(
        received_at=datetime(2026, 1, 15, tzinfo=UTC),
        paid_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    outstanding = _entry(received_at=datetime(2026, 1, 20, tzinfo=UTC))
    receipts = schedule_disbursements(
        [paid, outstanding], period=Period.QUARTERLY, include_paid=True
    )
    assert receipts[0].entry_count == 2


def test_no_outstanding_entries_returns_empty_list():
    """All entries paid + default filter → nothing to render."""
    paid = _entry(
        received_at=datetime(2026, 1, 15, tzinfo=UTC),
        paid_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    receipts = schedule_disbursements([paid], period=Period.QUARTERLY)
    assert receipts == []


def test_empty_input_returns_empty_list():
    receipts = schedule_disbursements([], period=Period.QUARTERLY)
    assert receipts == []


# ── totals + breakdown ───────────────────────────────────


def test_total_usd_sums_purification_per_period():
    e1 = _entry(purification_usd=2.00, received_at=datetime(2026, 1, 5, tzinfo=UTC))
    e2 = _entry(purification_usd=3.50, received_at=datetime(2026, 2, 5, tzinfo=UTC))
    [r] = schedule_disbursements([e1, e2], period=Period.QUARTERLY)
    assert r.total_usd == Decimal("5.50")


def test_breakdown_aggregates_per_symbol():
    e1 = _entry(symbol="AAPL", purification_usd=2.00, received_at=datetime(2026, 1, 5, tzinfo=UTC))
    e2 = _entry(symbol="AAPL", purification_usd=1.00, received_at=datetime(2026, 2, 5, tzinfo=UTC))
    e3 = _entry(symbol="MSFT", purification_usd=4.50, received_at=datetime(2026, 1, 10, tzinfo=UTC))
    [r] = schedule_disbursements([e1, e2, e3], period=Period.QUARTERLY)
    by_symbol = {b.symbol: b for b in r.breakdown}
    assert by_symbol["AAPL"].entry_count == 2
    assert by_symbol["AAPL"].purification_usd == Decimal("3.00")
    assert by_symbol["MSFT"].entry_count == 1
    assert by_symbol["MSFT"].purification_usd == Decimal("4.50")


def test_breakdown_sorted_by_descending_usd():
    """Concentration (largest contributor first) must pop visually
    so the operator can spot a single-symbol dominance fast."""
    big = _entry(symbol="BIG", purification_usd=10.00, received_at=datetime(2026, 1, 1, tzinfo=UTC))
    small = _entry(
        symbol="SML", purification_usd=1.00, received_at=datetime(2026, 1, 1, tzinfo=UTC)
    )
    [r] = schedule_disbursements([big, small], period=Period.QUARTERLY)
    assert r.breakdown[0].symbol == "BIG"
    assert r.breakdown[1].symbol == "SML"


# ── period bounds ────────────────────────────────────────


def test_quarterly_starts_at_first_day_of_quarter():
    e = _entry(received_at=datetime(2026, 5, 15, tzinfo=UTC))  # Q2
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert r.starts_at == datetime(2026, 4, 1, tzinfo=UTC)


def test_quarterly_ends_at_first_day_of_next_quarter():
    """Pin: `ends_at` is exclusive — the start of the *next* quarter.
    Half-open intervals are easier to reason about than inclusive
    ones."""
    e = _entry(received_at=datetime(2026, 5, 15, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert r.ends_at == datetime(2026, 7, 1, tzinfo=UTC)


def test_q4_ends_at_jan_1_of_next_year():
    e = _entry(received_at=datetime(2026, 11, 15, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert r.ends_at == datetime(2027, 1, 1, tzinfo=UTC)


def test_monthly_december_wraps_to_next_year():
    e = _entry(received_at=datetime(2026, 12, 15, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.MONTHLY)
    assert r.starts_at == datetime(2026, 12, 1, tzinfo=UTC)
    assert r.ends_at == datetime(2027, 1, 1, tzinfo=UTC)


def test_yearly_bounds_match_calendar_year():
    e = _entry(received_at=datetime(2026, 7, 15, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.YEARLY)
    assert r.starts_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert r.ends_at == datetime(2027, 1, 1, tzinfo=UTC)


# ── auto-mark-paid invariant ─────────────────────────────


def test_scheduler_does_not_mutate_entries():
    """Pin the safety invariant: scheduler MUST NOT auto-mark
    entries as paid — that's a one-way audit-trail commitment that
    needs explicit human consent. A future refactor that passes
    entries by reference and "helpfully" sets `paid_at` would be a
    bug; this test catches it."""
    e = _entry(received_at=datetime(2026, 1, 15, tzinfo=UTC))
    schedule_disbursements([e], period=Period.QUARTERLY)
    assert e.paid_at is None
    assert e.is_outstanding is True


# ── markdown rendering ───────────────────────────────────


def test_markdown_includes_label_and_total():
    e = _entry(received_at=datetime(2026, 1, 15, tzinfo=UTC), purification_usd=12.50)
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert "2026-Q1" in r.markdown
    assert "$12.50" in r.markdown


def test_markdown_includes_charity_target_when_supplied():
    e = _entry(received_at=datetime(2026, 1, 15, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.QUARTERLY, charity="Islamic Relief")
    assert "Islamic Relief" in r.markdown


def test_markdown_omits_charity_line_when_not_supplied():
    e = _entry(received_at=datetime(2026, 1, 15, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert "Disbursement target" not in r.markdown


def test_markdown_includes_per_symbol_breakdown_table():
    e1 = _entry(symbol="AAPL", purification_usd=2.00, received_at=datetime(2026, 1, 5, tzinfo=UTC))
    e2 = _entry(symbol="MSFT", purification_usd=3.00, received_at=datetime(2026, 2, 5, tzinfo=UTC))
    [r] = schedule_disbursements([e1, e2], period=Period.QUARTERLY)
    assert "| AAPL |" in r.markdown
    assert "| MSFT |" in r.markdown
    assert "## Breakdown by symbol" in r.markdown


def test_markdown_includes_audit_trail_reminder():
    """The 'mark each entry paid manually' note is the linchpin of
    the "no auto-mark" invariant — pin its presence in the user-
    facing receipt so it doesn't quietly drop out of the template."""
    e = _entry(received_at=datetime(2026, 1, 15, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert "does not auto-mark" in r.markdown


# ── is_empty + upcoming_due ──────────────────────────────


def test_is_empty_property_reflects_total():
    e = _entry(received_at=datetime(2026, 1, 15, tzinfo=UTC), purification_usd=1.00)
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert r.is_empty is False


def test_upcoming_due_returns_none_when_no_entries_in_current_period():
    e = _entry(received_at=datetime(2025, 1, 15, tzinfo=UTC))  # last year
    now = datetime(2026, 5, 1, tzinfo=UTC)
    result = upcoming_due([e], now=now, period=Period.QUARTERLY)
    assert result is None


def test_upcoming_due_returns_current_period_receipt():
    e = _entry(received_at=datetime(2026, 5, 15, tzinfo=UTC))  # Q2
    now = datetime(2026, 5, 30, tzinfo=UTC)
    result = upcoming_due([e], now=now, period=Period.QUARTERLY)
    assert result is not None
    assert result.label == "2026-Q2"


def test_upcoming_due_uses_now_default_when_unspecified():
    """Calling without `now` uses datetime.now(UTC); pin the
    fallback path so a refactor doesn't drop the default."""
    # Use a far-future entry so we don't accidentally match "now".
    e = _entry(received_at=datetime(2099, 1, 1, tzinfo=UTC))
    result = upcoming_due([e], period=Period.QUARTERLY)
    assert result is None


# ── output structure ─────────────────────────────────────


def test_receipt_is_immutable():
    e = _entry(received_at=datetime(2026, 1, 15, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert isinstance(r, DisbursementReceipt)
    with pytest.raises(Exception):
        r.label = "tampered"  # type: ignore[misc]


def test_breakdown_dataclass_is_typed():
    e = _entry(received_at=datetime(2026, 1, 15, tzinfo=UTC))
    [r] = schedule_disbursements([e], period=Period.QUARTERLY)
    assert all(isinstance(b, SymbolBreakdown) for b in r.breakdown)


def test_unknown_period_value_is_rejected():
    """Pin: passing a bogus enum value must raise rather than
    silently bucket entries together."""
    e = _entry(received_at=datetime(2026, 1, 15, tzinfo=UTC))
    with pytest.raises(ValueError, match="unknown period"):
        # Forge a Period-like with an unrecognised value to hit
        # the catch-all branch.
        from enum import Enum

        class _Bogus(str, Enum):
            BAD = "bogus"

        schedule_disbursements([e], period=_Bogus.BAD)  # type: ignore[arg-type]
