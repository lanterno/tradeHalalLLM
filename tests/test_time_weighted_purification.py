"""Tests for time-weighted purification."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.time_weighted_purification import (
    DividendEvent,
    HoldingPeriod,
    PurificationAssessment,
    PurificationMethod,
    calculate_purification,
    render_assessment,
    total_owed,
)

# Standard fixture: a Q1 2026 dividend ($1.00/share, 5% impure).
# Quarter is Jan 1 → Mar 31 (90 days inclusive); ex-date Mar 15.
Q1_DIVIDEND = DividendEvent(
    period_start=date(2026, 1, 1),
    period_end=date(2026, 3, 31),
    ex_date=date(2026, 3, 15),
    amount_per_share=1.00,
    impure_revenue_pct=0.05,
)
TODAY = date(2026, 6, 1)  # well after the Q1 period


# --- Enum string-value pins ---------------------------------------------------


def test_method_string_values():
    assert PurificationMethod.FULL_AMOUNT.value == "full_amount"
    assert PurificationMethod.HOLDING_PRORATED.value == "holding_prorated"


# --- HoldingPeriod validation -------------------------------------------------


def test_holding_basic():
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 4, 1),
        share_count=100,
    )
    assert h.share_count == 100


def test_holding_immutable():
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 1, 1),
        end_date=None,
        share_count=100,
    )
    with pytest.raises(Exception):
        h.share_count = 50  # type: ignore[misc]


def test_empty_holding_id_rejected():
    with pytest.raises(ValueError, match="holding_id"):
        HoldingPeriod(
            holding_id="",
            start_date=date(2026, 1, 1),
            end_date=None,
            share_count=100,
        )


def test_zero_shares_rejected():
    with pytest.raises(ValueError, match="share_count"):
        HoldingPeriod(
            holding_id="POS_1",
            start_date=date(2026, 1, 1),
            end_date=None,
            share_count=0,
        )


def test_end_before_start_rejected():
    with pytest.raises(ValueError, match="end_date"):
        HoldingPeriod(
            holding_id="POS_1",
            start_date=date(2026, 3, 1),
            end_date=date(2026, 2, 1),
            share_count=100,
        )


def test_still_held_position():
    """Pin: end_date=None means still held."""
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 1, 1),
        end_date=None,
        share_count=100,
    )
    assert h.end_date is None


# --- DividendEvent validation -------------------------------------------------


def test_dividend_basic():
    assert Q1_DIVIDEND.amount_per_share == 1.00
    assert Q1_DIVIDEND.days_in_period == 90  # Jan 1 → Mar 31 inclusive


def test_dividend_immutable():
    with pytest.raises(Exception):
        Q1_DIVIDEND.amount_per_share = 2.0  # type: ignore[misc]


def test_period_end_before_start_rejected():
    with pytest.raises(ValueError, match="period_end"):
        DividendEvent(
            period_start=date(2026, 3, 31),
            period_end=date(2026, 1, 1),
            ex_date=date(2026, 2, 15),
            amount_per_share=1.0,
            impure_revenue_pct=0.05,
        )


def test_ex_date_outside_period_rejected():
    """Pin: ex_date must be within [period_start, period_end]."""
    with pytest.raises(ValueError, match="ex_date"):
        DividendEvent(
            period_start=date(2026, 1, 1),
            period_end=date(2026, 3, 31),
            ex_date=date(2026, 4, 15),  # after period_end
            amount_per_share=1.0,
            impure_revenue_pct=0.05,
        )
    with pytest.raises(ValueError, match="ex_date"):
        DividendEvent(
            period_start=date(2026, 1, 1),
            period_end=date(2026, 3, 31),
            ex_date=date(2025, 12, 15),  # before period_start
            amount_per_share=1.0,
            impure_revenue_pct=0.05,
        )


def test_negative_amount_per_share_rejected():
    with pytest.raises(ValueError, match="amount_per_share"):
        DividendEvent(
            period_start=date(2026, 1, 1),
            period_end=date(2026, 3, 31),
            ex_date=date(2026, 2, 15),
            amount_per_share=-1.0,
            impure_revenue_pct=0.05,
        )


def test_impure_pct_outside_unit_range_rejected():
    """Pin: impure_revenue_pct must be in [0.0, 1.0]."""
    with pytest.raises(ValueError, match="impure_revenue_pct"):
        DividendEvent(
            period_start=date(2026, 1, 1),
            period_end=date(2026, 3, 31),
            ex_date=date(2026, 2, 15),
            amount_per_share=1.0,
            impure_revenue_pct=-0.01,
        )
    with pytest.raises(ValueError, match="impure_revenue_pct"):
        DividendEvent(
            period_start=date(2026, 1, 1),
            period_end=date(2026, 3, 31),
            ex_date=date(2026, 2, 15),
            amount_per_share=1.0,
            impure_revenue_pct=1.01,
        )


def test_impure_pct_zero_allowed():
    """Pin: 0.0 impure means a fully-clean dividend."""
    d = DividendEvent(
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        ex_date=date(2026, 2, 15),
        amount_per_share=1.0,
        impure_revenue_pct=0.0,
    )
    assert d.impure_revenue_pct == 0.0


def test_impure_pct_one_allowed():
    """Pin: 1.0 impure means fully-haram dividend (extreme)."""
    d = DividendEvent(
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        ex_date=date(2026, 2, 15),
        amount_per_share=1.0,
        impure_revenue_pct=1.0,
    )
    assert d.impure_revenue_pct == 1.0


def test_days_in_period_inclusive():
    """Pin: days_in_period is inclusive of both endpoints."""
    d = DividendEvent(
        period_start=date(2026, 1, 1),
        period_end=date(2026, 1, 1),  # single day
        ex_date=date(2026, 1, 1),
        amount_per_share=1.0,
        impure_revenue_pct=0.05,
    )
    assert d.days_in_period == 1


# --- Eligibility (binary on ex-date) -----------------------------------------


def test_eligible_when_holding_covers_ex_date():
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 3, 1),
        end_date=date(2026, 3, 31),
        share_count=100,
    )
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    assert a.eligible is True


def test_ineligible_when_bought_after_ex_date():
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 3, 16),  # day after ex-date
        end_date=date(2026, 3, 31),
        share_count=100,
    )
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    assert a.eligible is False


def test_ineligible_when_sold_before_ex_date():
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 3, 1),
        end_date=date(2026, 3, 14),  # day before ex-date
        share_count=100,
    )
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    assert a.eligible is False


def test_eligible_when_still_held_through_today():
    """Pin: end_date=None treats as still held → eligible if today >= ex-date."""
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 1, 1),
        end_date=None,
        share_count=100,
    )
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    assert a.eligible is True


def test_eligible_at_ex_date_inclusive():
    """Pin: holding START on ex_date is eligible (inclusive)."""
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 3, 15),
        end_date=date(2026, 3, 31),
        share_count=100,
    )
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    assert a.eligible is True


def test_eligible_with_END_on_ex_date_inclusive():
    """Pin: holding END on ex_date is eligible (inclusive)."""
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 3, 1),
        end_date=date(2026, 3, 15),
        share_count=100,
    )
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    assert a.eligible is True


def test_ineligible_returns_zero_dividend():
    """Pin: ineligible → gross_dividend=0, purification_owed=0."""
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 5, 1),
        share_count=100,
    )
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    assert a.gross_dividend == 0
    assert a.purification_owed == 0


# --- FULL_AMOUNT method -------------------------------------------------------


def test_full_amount_eligible_holding():
    """Hold full quarter → 100 shares × $1 × 5% = $5 purification."""
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        share_count=100,
    )
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    assert a.method_used is PurificationMethod.FULL_AMOUNT  # default
    assert a.gross_dividend == 100.0
    assert a.impure_amount_full == pytest.approx(5.0)
    assert a.purification_owed == pytest.approx(5.0)


def test_full_amount_brief_holding_pays_full():
    """Pin: even a 3-day holding spanning ex-date pays full purification."""
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 3, 14),
        end_date=date(2026, 3, 16),
        share_count=100,
    )
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    assert a.purification_owed == pytest.approx(5.0)  # full amount


def test_default_method_is_full_amount():
    """Pin: default method is FULL_AMOUNT (more conservative)."""
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        share_count=100,
    )
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    assert a.method_used is PurificationMethod.FULL_AMOUNT


# --- HOLDING_PRORATED method --------------------------------------------------


def test_holding_prorated_full_period_equals_full_amount():
    """Holding the full period prorates to 1.0 → equal to full amount."""
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 31),
        share_count=100,
    )
    a = calculate_purification(
        h, Q1_DIVIDEND, today=TODAY, method=PurificationMethod.HOLDING_PRORATED
    )
    assert a.holding_fraction == pytest.approx(1.0)
    assert a.purification_owed == pytest.approx(5.0)


def test_holding_prorated_partial_period():
    """Hold 30 days of 90-day quarter → 30/90 × $5 = $1.667."""
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 3, 1),  # 31 days ago
        end_date=date(2026, 3, 30),
        share_count=100,
    )
    a = calculate_purification(
        h, Q1_DIVIDEND, today=TODAY, method=PurificationMethod.HOLDING_PRORATED
    )
    # Mar 1 → Mar 30 = 30 days inclusive
    assert a.days_held_in_period == 30
    assert a.days_in_period == 90
    assert a.holding_fraction == pytest.approx(30 / 90)
    assert a.purification_owed == pytest.approx(5.0 * 30 / 90)


def test_holding_prorated_three_day_window():
    """3 days held over 90-day quarter → 3/90 × $5 = $0.167."""
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 3, 14),
        end_date=date(2026, 3, 16),
        share_count=100,
    )
    a = calculate_purification(
        h, Q1_DIVIDEND, today=TODAY, method=PurificationMethod.HOLDING_PRORATED
    )
    assert a.days_held_in_period == 3
    assert a.purification_owed == pytest.approx(5.0 * 3 / 90)


def test_holding_prorated_caps_at_period_length():
    """Pin: holding longer than period caps at period (fraction stays 1.0)."""
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2025, 12, 1),  # before period start
        end_date=date(2026, 4, 30),  # after period end
        share_count=100,
    )
    a = calculate_purification(
        h, Q1_DIVIDEND, today=TODAY, method=PurificationMethod.HOLDING_PRORATED
    )
    assert a.holding_fraction == pytest.approx(1.0)
    assert a.purification_owed == pytest.approx(5.0)


def test_holding_prorated_overlaps_partially():
    """Hold from Feb 15 → end of period → overlap = Feb 15-Mar 31 = 45 days."""
    h = HoldingPeriod(
        holding_id="POS_1",
        start_date=date(2026, 2, 15),
        end_date=date(2026, 4, 30),
        share_count=100,
    )
    a = calculate_purification(
        h, Q1_DIVIDEND, today=TODAY, method=PurificationMethod.HOLDING_PRORATED
    )
    # Feb 15 → Mar 31 = 45 days inclusive (Feb has 28 days in 2026 → 14 days remaining
    # in Feb + 31 days in March = 45 days)
    assert a.days_held_in_period == 45
    assert a.purification_owed == pytest.approx(5.0 * 45 / 90)


# --- PurificationAssessment validation ---------------------------------------


def test_assessment_immutable():
    h = HoldingPeriod("POS_1", date(2026, 1, 1), date(2026, 3, 31), 100)
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    with pytest.raises(Exception):
        a.purification_owed = 99  # type: ignore[misc]


def test_purification_cannot_exceed_impure_full():
    """Pin: purification_owed cannot exceed impure_amount_full."""
    with pytest.raises(ValueError, match="purification_owed"):
        PurificationAssessment(
            holding_id="X",
            eligible=True,
            gross_dividend=100,
            impure_amount_full=5,
            purification_owed=10,  # > impure
            method_used=PurificationMethod.FULL_AMOUNT,
            days_held_in_period=90,
            days_in_period=90,
            holding_fraction=1.0,
        )


def test_holding_fraction_outside_unit_range_rejected():
    with pytest.raises(ValueError, match="holding_fraction"):
        PurificationAssessment(
            holding_id="X",
            eligible=True,
            gross_dividend=0,
            impure_amount_full=0,
            purification_owed=0,
            method_used=PurificationMethod.FULL_AMOUNT,
            days_held_in_period=0,
            days_in_period=90,
            holding_fraction=1.5,
        )


def test_ineligible_with_dividend_rejected():
    """Pin: ineligible holdings can't have gross_dividend > 0."""
    with pytest.raises(ValueError, match="ineligible holding"):
        PurificationAssessment(
            holding_id="X",
            eligible=False,
            gross_dividend=10,  # impossible
            impure_amount_full=0,
            purification_owed=0,
            method_used=PurificationMethod.FULL_AMOUNT,
            days_held_in_period=0,
            days_in_period=90,
            holding_fraction=0.0,
        )


def test_negative_days_held_rejected():
    with pytest.raises(ValueError, match="days_held_in_period"):
        PurificationAssessment(
            holding_id="X",
            eligible=False,
            gross_dividend=0,
            impure_amount_full=0,
            purification_owed=0,
            method_used=PurificationMethod.FULL_AMOUNT,
            days_held_in_period=-1,
            days_in_period=90,
            holding_fraction=0.0,
        )


def test_zero_days_in_period_rejected():
    with pytest.raises(ValueError, match="days_in_period"):
        PurificationAssessment(
            holding_id="X",
            eligible=False,
            gross_dividend=0,
            impure_amount_full=0,
            purification_owed=0,
            method_used=PurificationMethod.FULL_AMOUNT,
            days_held_in_period=0,
            days_in_period=0,
            holding_fraction=0.0,
        )


# --- total_owed --------------------------------------------------------------


def test_total_owed_sums_correctly():
    h1 = HoldingPeriod("POS_1", date(2026, 1, 1), date(2026, 3, 31), 100)
    h2 = HoldingPeriod("POS_2", date(2026, 1, 1), date(2026, 3, 31), 200)
    a1 = calculate_purification(h1, Q1_DIVIDEND, today=TODAY)
    a2 = calculate_purification(h2, Q1_DIVIDEND, today=TODAY)
    assert total_owed([a1, a2]) == pytest.approx(15.0)  # $5 + $10


def test_total_owed_empty():
    assert total_owed([]) == 0.0


# --- Render -------------------------------------------------------------------


def test_render_eligible_shows_amount():
    h = HoldingPeriod("POS_1", date(2026, 1, 1), date(2026, 3, 31), 100)
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    out = render_assessment(a)
    assert "💧" in out
    assert "POS_1" in out
    assert "5.0000" in out
    assert "full amount" in out


def test_render_ineligible_shows_pause():
    h = HoldingPeriod("POS_1", date(2026, 4, 1), date(2026, 5, 1), 100)
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    out = render_assessment(a)
    assert "⏸" in out
    assert "not eligible" in out


def test_render_includes_method_label():
    h = HoldingPeriod("POS_1", date(2026, 3, 1), date(2026, 3, 30), 100)
    a = calculate_purification(
        h, Q1_DIVIDEND, today=TODAY, method=PurificationMethod.HOLDING_PRORATED
    )
    out = render_assessment(a)
    assert "holding prorated" in out


def test_render_no_secret_leak():
    """Pin: render output never includes per-trade buy/sell prices or P&L."""
    h = HoldingPeriod("POS_42", date(2026, 1, 1), date(2026, 3, 31), 100)
    a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    out = render_assessment(a)
    forbidden = ["buy_price", "sell_price", "P&L", "cost_basis", "Authorization"]
    for word in forbidden:
        assert word not in out


# --- E2E flows ----------------------------------------------------------------


def test_e2e_active_trader_dividend_capture():
    """Active trader buys 3 days before ex-date, sells 1 day after.
    With FULL_AMOUNT: owes full 5% purification.
    With HOLDING_PRORATED: owes ~5/90 of that amount."""
    h = HoldingPeriod(
        holding_id="ACTIVE_1",
        start_date=date(2026, 3, 12),
        end_date=date(2026, 3, 16),
        share_count=100,
    )
    full_a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    prorated_a = calculate_purification(
        h, Q1_DIVIDEND, today=TODAY, method=PurificationMethod.HOLDING_PRORATED
    )
    assert full_a.purification_owed == pytest.approx(5.0)
    # 5 days held / 90 day period
    assert prorated_a.purification_owed == pytest.approx(5.0 * 5 / 90)
    # Conservative (FULL_AMOUNT) owes more
    assert full_a.purification_owed > prorated_a.purification_owed


def test_e2e_long_term_holder_methods_agree():
    """Long-term holder spanning full period: both methods give same answer."""
    h = HoldingPeriod(
        holding_id="LT_1",
        start_date=date(2025, 1, 1),
        end_date=None,
        share_count=100,
    )
    full_a = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    prorated_a = calculate_purification(
        h, Q1_DIVIDEND, today=TODAY, method=PurificationMethod.HOLDING_PRORATED
    )
    assert full_a.purification_owed == pytest.approx(prorated_a.purification_owed)


def test_e2e_replay_consistency():
    """Pin: same inputs → equal assessment."""
    h = HoldingPeriod("POS_1", date(2026, 3, 1), date(2026, 3, 30), 100)
    a1 = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    a2 = calculate_purification(h, Q1_DIVIDEND, today=TODAY)
    assert a1 == a2


def test_e2e_zero_impure_dividend_no_purification_owed():
    """Pin: clean dividend (impure_pct=0) → zero purification regardless of method."""
    clean_div = DividendEvent(
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        ex_date=date(2026, 3, 15),
        amount_per_share=1.00,
        impure_revenue_pct=0.0,
    )
    h = HoldingPeriod("POS_1", date(2026, 1, 1), date(2026, 3, 31), 100)
    a = calculate_purification(h, clean_div, today=TODAY)
    assert a.gross_dividend == 100.0
    assert a.impure_amount_full == 0.0
    assert a.purification_owed == 0.0
