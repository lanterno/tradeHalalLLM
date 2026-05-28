"""Significance tests — Welch t / incomplete beta / Cohen's d / promotion gate."""

from __future__ import annotations

import pytest

from halabot.analysis.significance import (
    betai,
    cohens_d,
    promotion_gate,
    student_t_sf_two_sided,
    welch_t_test,
)


def test_betai_symmetry_and_bounds():
    assert betai(2.0, 3.0, 0.0) == 0.0
    assert betai(2.0, 3.0, 1.0) == 1.0
    # I_x(a,b) = 1 - I_{1-x}(b,a)
    assert betai(2.0, 3.0, 0.4) == pytest.approx(1.0 - betai(3.0, 2.0, 0.6), abs=1e-9)


def test_student_t_pvalue_known_values():
    # t=0 → p=1; large t → p≈0.
    assert student_t_sf_two_sided(0.0, 10) == pytest.approx(1.0, abs=1e-9)
    assert student_t_sf_two_sided(50.0, 10) < 1e-6
    # Classic table value: t=2.228, df=10 → two-sided p ≈ 0.05.
    assert student_t_sf_two_sided(2.228, 10) == pytest.approx(0.05, abs=2e-3)


def test_welch_identical_samples_p_is_one():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = welch_t_test(a, list(a))
    assert res is not None
    assert res.t == pytest.approx(0.0)
    assert res.p_two_sided == pytest.approx(1.0, abs=1e-9)


def test_welch_detects_clear_difference():
    a = [10.0, 11.0, 9.0, 10.5, 10.2, 9.8] * 5  # mean ~10
    b = [1.0, 1.2, 0.8, 1.1, 0.9, 1.05] * 5  # mean ~1
    res = welch_t_test(a, b)
    assert res is not None and res.t > 0
    assert res.p_two_sided < 0.001
    assert res.p_one_sided_a_greater < 0.001


def test_cohens_d_sign_and_magnitude():
    a = [2.0, 2.1, 1.9, 2.0]
    b = [1.0, 1.1, 0.9, 1.0]
    d = cohens_d(a, b)
    assert d is not None and d > 1.0  # large positive effect


def test_welch_none_when_undersized():
    assert welch_t_test([1.0], [1.0, 2.0]) is None
    assert cohens_d([1.0], [1.0, 2.0]) is None


# ── promotion gate ──
def test_gate_holds_on_insufficient_samples():
    v = promotion_gate([0.01, 0.02], [0.01, 0.0], churn_reduction=0.5)
    assert v.promote is False
    assert any("insufficient" in r for r in v.reasons)


def test_gate_holds_when_churn_not_reduced():
    shadow = [0.01] * 40
    live = [0.01] * 40
    v = promotion_gate(shadow, live, churn_reduction=0.0)  # no churn reduction
    assert v.promote is False
    assert any("churn" in r for r in v.reasons)


def test_gate_promotes_lower_churn_at_parity_pnl():
    shadow = [0.01, 0.012, 0.008, 0.011] * 12  # ~parity with live
    live = [0.01, 0.009, 0.011, 0.010] * 12
    v = promotion_gate(shadow, live, churn_reduction=0.5, min_n=30)
    assert v.promote is True
    assert any("PASS" in r for r in v.reasons)


def test_gate_holds_when_shadow_significantly_worse():
    # Realistic spread so the t-test is defined; shadow mean clearly below live.
    shadow = [0.001 + 0.0006 * (i % 5 - 2) for i in range(50)]  # ~0.001
    live = [0.020 + 0.0006 * (i % 5 - 2) for i in range(50)]  # ~0.020
    v = promotion_gate(shadow, live, churn_reduction=0.5, min_n=30)
    assert v.promote is False
    assert any("worse" in r for r in v.reasons)
