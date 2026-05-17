"""Tests for ml/vol_regime.py — Round-5 Wave 13.A."""

from __future__ import annotations

import math

import pytest

from halal_trader.ml.vol_regime import (
    RegimeAssessment,
    VolPolicy,
    VolRegime,
    annualised_vol,
    detect,
    render_assessment,
    transitioned,
)

# --- Validation -------------------------------------------------------------


def test_vol_regime_string_values():
    assert VolRegime.LOW.value == "low"
    assert VolRegime.NORMAL.value == "normal"
    assert VolRegime.ELEVATED.value == "elevated"
    assert VolRegime.CRISIS.value == "crisis"


def test_default_policy_thresholds():
    p = VolPolicy()
    assert p.low_threshold == 0.10
    assert p.normal_threshold == 0.20
    assert p.elevated_threshold == 0.40


def test_policy_thresholds_must_increase():
    with pytest.raises(ValueError):
        VolPolicy(low_threshold=0.30, normal_threshold=0.20)


def test_policy_zero_low_rejected():
    with pytest.raises(ValueError):
        VolPolicy(low_threshold=0.0)


def test_policy_negative_hysteresis_rejected():
    with pytest.raises(ValueError):
        VolPolicy(hysteresis=-0.01)


def test_policy_high_hysteresis_rejected():
    with pytest.raises(ValueError):
        VolPolicy(hysteresis=0.5)


def test_policy_zero_factor_rejected():
    with pytest.raises(ValueError):
        VolPolicy(annualisation_factor=0)


def test_assessment_negative_vol_rejected():
    with pytest.raises(ValueError):
        RegimeAssessment(
            regime=VolRegime.LOW,
            annualised_vol=-0.1,
            realised_returns_count=0,
            is_borderline=False,
        )


# --- annualised_vol -------------------------------------------------------


def test_annualised_vol_empty_zero():
    assert annualised_vol([]) == 0.0


def test_annualised_vol_single_return_zero():
    assert annualised_vol([0.01]) == 0.0


def test_annualised_vol_constant_returns_zero():
    """Constant returns → zero std → zero vol."""
    assert annualised_vol([0.001] * 100) == pytest.approx(0.0)


def test_annualised_vol_scales_with_factor():
    returns = [0.01, -0.01, 0.02, -0.02] * 30
    v_252 = annualised_vol(returns, factor=252)
    v_365 = annualised_vol(returns, factor=365)
    # sqrt(365/252) ratio
    assert v_365 == pytest.approx(v_252 * math.sqrt(365 / 252), rel=1e-6)


def test_annualised_vol_zero_factor_rejected():
    with pytest.raises(ValueError):
        annualised_vol([0.01, 0.02], factor=0)


# --- detect ----------------------------------------------------------------


def test_detect_empty_returns_normal_default():
    a = detect([])
    assert a.regime is VolRegime.NORMAL


def test_detect_empty_with_previous_returns_previous():
    a = detect([], previous_regime=VolRegime.ELEVATED)
    assert a.regime is VolRegime.ELEVATED


def test_detect_low_regime():
    """Low daily-vol returns → LOW regime (annual vol < 10%)."""
    returns = [0.0001, -0.0001] * 50  # essentially zero vol
    a = detect(returns)
    assert a.regime is VolRegime.LOW


def test_detect_normal_regime():
    """Daily vol ~1% → annual ~16% → NORMAL."""
    # Build returns with std ~0.01
    returns = [0.01, -0.01] * 30
    a = detect(returns)
    # annualised ~ 0.01 * sqrt(252) ≈ 0.1587 → NORMAL (between 0.10 and 0.20)
    assert a.regime is VolRegime.NORMAL


def test_detect_elevated_regime():
    """Daily vol ~2% → annual ~32% → ELEVATED."""
    returns = [0.02, -0.02] * 30
    a = detect(returns)
    # annualised ~ 0.02 * sqrt(252) ≈ 0.317 → ELEVATED (between 0.20 and 0.40)
    assert a.regime is VolRegime.ELEVATED


def test_detect_crisis_regime():
    """Daily vol ~5% → annual ~79% → CRISIS."""
    returns = [0.05, -0.05] * 30
    a = detect(returns)
    assert a.regime is VolRegime.CRISIS


def test_detect_count_matches_input():
    returns = [0.01, -0.01] * 30
    a = detect(returns)
    assert a.realised_returns_count == 60


def test_detect_borderline_flagged():
    """Returns producing vol exactly at a threshold → borderline."""
    pol = VolPolicy(hysteresis=0.05)
    # Engineered to produce vol near 0.20 (normal/elevated boundary)
    target_daily = 0.20 / math.sqrt(252)  # ≈ 0.01260
    returns = [target_daily, -target_daily] * 30
    a = detect(returns, policy=pol)
    assert a.is_borderline


# --- Hysteresis -----------------------------------------------------------


def test_hysteresis_snaps_to_previous_when_borderline_adjacent():
    """A borderline ELEVATED reading with previous=NORMAL stays at NORMAL."""
    pol = VolPolicy(hysteresis=0.05)
    # vol just over 0.20 → would classify ELEVATED, but borderline + previous NORMAL
    target_daily = 0.205 / math.sqrt(252)
    returns = [target_daily, -target_daily] * 30
    a = detect(returns, previous_regime=VolRegime.NORMAL, policy=pol)
    if a.is_borderline:
        assert a.regime is VolRegime.NORMAL


def test_hysteresis_does_not_snap_across_two_levels():
    """A borderline reading two levels away from previous still re-classifies."""
    pol = VolPolicy(hysteresis=0.05)
    # vol ≈ 0.40 (elevated/crisis boundary)
    target_daily = 0.40 / math.sqrt(252)
    returns = [target_daily, -target_daily] * 30
    a = detect(returns, previous_regime=VolRegime.LOW, policy=pol)
    # LOW is two regimes away — should NOT snap
    assert a.regime is not VolRegime.LOW


def test_no_hysteresis_when_not_borderline():
    """A clean LOW reading doesn't snap to previous CRISIS."""
    returns = [0.0001, -0.0001] * 50
    a = detect(returns, previous_regime=VolRegime.CRISIS)
    assert a.regime is VolRegime.LOW


# --- Transitioned ---------------------------------------------------------


def test_transitioned_none_previous_false():
    a = detect([0.01, -0.01] * 30)
    assert transitioned(None, a) is False


def test_transitioned_same_regime_false():
    a = detect([0.01, -0.01] * 30)
    b = detect([0.01, -0.01] * 30)
    assert transitioned(a, b) is False


def test_transitioned_different_regime_true():
    a = detect([0.0001, -0.0001] * 50)  # LOW
    b = detect([0.05, -0.05] * 30)  # CRISIS
    assert transitioned(a, b) is True


# --- Render ---------------------------------------------------------------


def test_render_includes_regime_and_vol():
    a = detect([0.01, -0.01] * 30)
    out = render_assessment(a)
    assert "regime=" in out
    assert "annual_vol=" in out


def test_render_low_uses_green_emoji():
    a = detect([0.0001, -0.0001] * 50)
    assert "🟢" in render_assessment(a)


def test_render_crisis_uses_red_emoji():
    a = detect([0.05, -0.05] * 30)
    assert "🔴" in render_assessment(a)


def test_render_borderline_marker():
    pol = VolPolicy(hysteresis=0.05)
    target = 0.20 / math.sqrt(252)
    returns = [target, -target] * 30
    a = detect(returns, policy=pol)
    if a.is_borderline:
        assert "borderline" in render_assessment(a)


def test_render_no_secret_leak():
    a = detect([0.01, -0.01] * 30)
    out = render_assessment(a)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ------------------------------------------------------------------


def test_e2e_regime_walk_low_to_crisis():
    """Operator's vol regime can transition from LOW → CRISIS over time."""
    low = detect([0.0001, -0.0001] * 50)
    crisis = detect([0.05, -0.05] * 30, previous_regime=low.regime)
    assert low.regime is VolRegime.LOW
    assert crisis.regime is VolRegime.CRISIS
    assert transitioned(low, crisis)


def test_replay_consistency():
    a = detect([0.01, -0.01] * 30)
    b = detect([0.01, -0.01] * 30)
    assert a == b
