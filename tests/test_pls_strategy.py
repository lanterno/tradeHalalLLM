"""Tests for halal/pls_strategy.py — Round-5 Wave 7.D."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.pls_strategy import (
    CumulativeReport,
    FeeStructure,
    PLSAgreement,
    compute_period_fee,
    render_cumulative,
    render_period,
    run_full_history,
)


def _agr(
    fee_structure: FeeStructure = FeeStructure.HURDLE_HWM,
    hurdle: float = 0.04,
    profit_share: float = 0.20,
    base_fee: float = 0.0,
    manager_capital_pct: float = 0.10,
) -> PLSAgreement:
    return PLSAgreement(
        agreement_id="A1",
        investor_id="alice",
        manager_id="bob",
        starting_capital=1_000_000.0,
        manager_capital_pct=manager_capital_pct,
        hurdle_rate_annual=hurdle,
        profit_share_pct=profit_share,
        fee_structure=fee_structure,
        base_management_fee_annual=base_fee,
    )


# --- PLSAgreement validation ---------------------------------------------


def test_agreement_valid():
    a = _agr()
    assert a.starting_capital == 1_000_000.0


def test_agreement_self_dealing_rejected():
    with pytest.raises(ValueError):
        PLSAgreement(
            agreement_id="A1",
            investor_id="x",
            manager_id="x",
            starting_capital=1000.0,
            manager_capital_pct=0.10,
            hurdle_rate_annual=0.04,
            profit_share_pct=0.20,
        )


def test_agreement_negative_capital_rejected():
    with pytest.raises(ValueError):
        PLSAgreement(
            agreement_id="A1",
            investor_id="alice",
            manager_id="bob",
            starting_capital=-100.0,
            manager_capital_pct=0.10,
            hurdle_rate_annual=0.04,
            profit_share_pct=0.20,
        )


def test_agreement_manager_capital_at_one_rejected():
    """Manager cannot own 100% of capital — investor needs a stake."""
    with pytest.raises(ValueError):
        PLSAgreement(
            agreement_id="A1",
            investor_id="alice",
            manager_id="bob",
            starting_capital=1000.0,
            manager_capital_pct=1.0,
            hurdle_rate_annual=0.04,
            profit_share_pct=0.20,
        )


def test_agreement_profit_share_at_one_rejected():
    """Manager cannot take 100% of profits."""
    with pytest.raises(ValueError):
        _agr(profit_share=1.0)


def test_agreement_unreasonable_hurdle_rejected():
    with pytest.raises(ValueError):
        _agr(hurdle=0.50)
    with pytest.raises(ValueError):
        _agr(hurdle=-0.10)


def test_agreement_unreasonable_base_fee_rejected():
    with pytest.raises(ValueError):
        _agr(base_fee=0.20)


def test_agreement_immutable():
    a = _agr()
    with pytest.raises(AttributeError):
        a.profit_share_pct = 0.10  # type: ignore[misc]


# --- compute_period_fee — HURDLE_ONLY ------------------------------------


def test_hurdle_only_above_hurdle_charges_perf_fee():
    a = _agr(fee_structure=FeeStructure.HURDLE_ONLY, hurdle=0.04)
    rep = compute_period_fee(
        a,
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        starting_nav=1_000_000.0,
        ending_nav=1_100_000.0,
        hwm_at_start=1_000_000.0,
    )
    # Period return = 10%, hurdle = 4% × 364/365 ≈ 3.99%, excess ≈ 6.01%.
    # Perf fee = excess × starting × 0.20 = 0.0601 × 1M × 0.20 ≈ 12,000.
    assert rep.performance_fee > 11_000
    assert rep.performance_fee < 13_000


def test_hurdle_only_below_hurdle_no_perf_fee():
    a = _agr(fee_structure=FeeStructure.HURDLE_ONLY, hurdle=0.10)
    rep = compute_period_fee(
        a,
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        starting_nav=1_000_000.0,
        ending_nav=1_050_000.0,
        hwm_at_start=1_000_000.0,
    )
    # Return 5% < hurdle 10% — no perf fee.
    assert rep.performance_fee == 0.0


# --- compute_period_fee — HURDLE_HWM ------------------------------------


def test_hwm_above_prior_peak_charges_perf_fee():
    a = _agr(fee_structure=FeeStructure.HURDLE_HWM)
    rep = compute_period_fee(
        a,
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        starting_nav=1_000_000.0,
        ending_nav=1_100_000.0,
        hwm_at_start=1_000_000.0,
    )
    # excess_dollar = 1.1M - 1M = 100k; perf = 100k × 0.20 = 20,000.
    assert rep.performance_fee == pytest.approx(20_000.0)
    assert rep.hwm_at_end == 1_100_000.0


def test_hwm_below_prior_peak_no_perf_fee():
    """Underwater positions don't accrue perf fee even if return > hurdle."""
    a = _agr(fee_structure=FeeStructure.HURDLE_HWM)
    rep = compute_period_fee(
        a,
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        starting_nav=900_000.0,  # Below prior peak.
        ending_nav=950_000.0,  # Recovered but still below HWM.
        hwm_at_start=1_000_000.0,
    )
    assert rep.performance_fee == 0.0
    # HWM stays at the prior peak.
    assert rep.hwm_at_end == 1_000_000.0


def test_hwm_monotone_pin():
    """Pin: HWM never decreases period-over-period."""
    a = _agr()
    rep = compute_period_fee(
        a,
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        starting_nav=1_000_000.0,
        ending_nav=900_000.0,
        hwm_at_start=1_100_000.0,
    )
    assert rep.hwm_at_end >= rep.hwm_at_start


# --- compute_period_fee — HURDLE_HWM_LOSS_SHARE -------------------------


def test_loss_share_charges_manager_when_underwater():
    a = _agr(
        fee_structure=FeeStructure.HURDLE_HWM_LOSS_SHARE,
        manager_capital_pct=0.10,
    )
    rep = compute_period_fee(
        a,
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        starting_nav=1_000_000.0,
        ending_nav=900_000.0,
        hwm_at_start=1_000_000.0,
    )
    # Drawdown = 100k; manager pays 10% × 100k = 10,000 into the pool.
    assert rep.manager_loss_share == pytest.approx(10_000.0)


def test_loss_share_zero_when_above_hwm():
    a = _agr(
        fee_structure=FeeStructure.HURDLE_HWM_LOSS_SHARE,
        manager_capital_pct=0.10,
    )
    rep = compute_period_fee(
        a,
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        starting_nav=1_000_000.0,
        ending_nav=1_100_000.0,
        hwm_at_start=1_000_000.0,
    )
    assert rep.manager_loss_share == 0.0


def test_loss_share_proportional_to_manager_capital():
    """Pin: AAOIFI Standard 13 — loss ratio follows capital exposure."""
    a_low = _agr(
        fee_structure=FeeStructure.HURDLE_HWM_LOSS_SHARE,
        manager_capital_pct=0.05,
    )
    a_high = _agr(
        fee_structure=FeeStructure.HURDLE_HWM_LOSS_SHARE,
        manager_capital_pct=0.20,
    )
    base = dict(
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        starting_nav=1_000_000.0,
        ending_nav=900_000.0,
        hwm_at_start=1_000_000.0,
    )
    r_low = compute_period_fee(a_low, **base)
    r_high = compute_period_fee(a_high, **base)
    assert r_high.manager_loss_share == 4 * r_low.manager_loss_share


# --- Base Wakalah fee ---------------------------------------------------


def test_base_fee_charged_independent_of_performance():
    a = _agr(base_fee=0.02)
    rep = compute_period_fee(
        a,
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        starting_nav=1_000_000.0,
        ending_nav=900_000.0,  # Loss period.
        hwm_at_start=1_000_000.0,
    )
    # 2% × 1M × 364/365 ≈ 19,945.
    assert rep.base_fee > 19_000
    assert rep.base_fee < 20_500


# --- Validation ---------------------------------------------------------


def test_negative_starting_nav_rejected():
    a = _agr()
    with pytest.raises(ValueError):
        compute_period_fee(
            a,
            period_start=date(2026, 1, 1),
            period_end=date(2026, 12, 31),
            starting_nav=-1.0,
            ending_nav=900_000.0,
            hwm_at_start=1_000_000.0,
        )


def test_period_end_before_start_rejected():
    a = _agr()
    with pytest.raises(ValueError):
        compute_period_fee(
            a,
            period_start=date(2026, 12, 31),
            period_end=date(2026, 1, 1),
            starting_nav=1_000_000.0,
            ending_nav=900_000.0,
            hwm_at_start=1_000_000.0,
        )


# --- run_full_history ---------------------------------------------------


def test_full_history_basic_three_periods():
    a = _agr(fee_structure=FeeStructure.HURDLE_HWM, hurdle=0.0)
    bookends = [
        (date(2026, 1, 1), 1_000_000.0),
        (date(2026, 4, 1), 1_050_000.0),
        (date(2026, 7, 1), 1_100_000.0),
        (date(2026, 10, 1), 1_080_000.0),  # Slight drawdown — no perf, HWM holds.
    ]
    cum = run_full_history(a, bookends)
    assert isinstance(cum, CumulativeReport)
    assert len(cum.period_reports) == 3
    # Final HWM = highest peak = 1.10M.
    assert cum.final_hwm == 1_100_000.0
    # Performance fee accrued only when above HWM at period close.
    assert cum.total_performance_fees > 0


def test_full_history_inception_nav_must_match_starting_capital():
    a = _agr()
    bookends = [
        (date(2026, 1, 1), 999_000.0),  # Wrong NAV.
        (date(2026, 4, 1), 1_050_000.0),
    ]
    with pytest.raises(ValueError):
        run_full_history(a, bookends)


def test_full_history_dates_must_be_increasing():
    a = _agr()
    bookends = [
        (date(2026, 1, 1), 1_000_000.0),
        (date(2026, 4, 1), 1_050_000.0),
        (date(2026, 3, 1), 1_100_000.0),  # Out of order.
    ]
    with pytest.raises(ValueError):
        run_full_history(a, bookends)


def test_full_history_too_few_bookends_rejected():
    a = _agr()
    with pytest.raises(ValueError):
        run_full_history(a, [(date(2026, 1, 1), 1_000_000.0)])


def test_full_history_no_perf_below_hwm_pin():
    """Pin: a recovery period that ends below HWM accrues no perf fee."""
    a = _agr(fee_structure=FeeStructure.HURDLE_HWM, hurdle=0.0)
    bookends = [
        (date(2026, 1, 1), 1_000_000.0),
        (date(2026, 4, 1), 1_200_000.0),  # New peak.
        (date(2026, 7, 1), 1_000_000.0),  # Drawdown.
        (date(2026, 10, 1), 1_150_000.0),  # Recovery but still below HWM.
    ]
    cum = run_full_history(a, bookends)
    # Only one period above HWM (the first period). Recovery period
    # accrues no perf fee.
    assert cum.period_reports[2].performance_fee == 0.0


# --- Render -------------------------------------------------------------


def test_render_period_contains_summary():
    a = _agr()
    rep = compute_period_fee(
        a,
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        starting_nav=1_000_000.0,
        ending_nav=1_100_000.0,
        hwm_at_start=1_000_000.0,
    )
    out = render_period(rep)
    assert "📈" in out
    assert "Investor net" in out
    assert "perf fee" in out


def test_render_cumulative():
    a = _agr()
    bookends = [
        (date(2026, 1, 1), 1_000_000.0),
        (date(2026, 12, 31), 1_100_000.0),
    ]
    cum = run_full_history(a, bookends)
    out = render_cumulative(cum)
    assert "Cumulative" in out
    assert "Total perf fees" in out
