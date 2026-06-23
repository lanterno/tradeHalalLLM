"""Tests for VaR / CVaR (Expected Shortfall)."""

from __future__ import annotations

import numpy as np

from halal_trader.core.risk_metrics import (
    conditional_value_at_risk,
    value_at_risk,
)


def test_var_is_the_alpha_quantile():
    returns = [-0.10, -0.05, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    # 5% VaR ~ the low extreme of this 10-point sample.
    var = value_at_risk(returns, alpha=0.05)
    assert var < 0
    assert var <= value_at_risk(returns, alpha=0.5)  # deeper tail = worse


def test_cvar_not_above_var():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.02, 1000)
    var = value_at_risk(returns, alpha=0.05)
    cvar = conditional_value_at_risk(returns, alpha=0.05)
    # Expected Shortfall is the mean of the tail beyond VaR → <= VaR.
    assert cvar <= var


def test_cvar_negative_for_loss_tail():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.001, 0.03, 2000)
    assert conditional_value_at_risk(returns, alpha=0.05) < 0


def test_empty_returns_zero():
    assert value_at_risk([]) == 0.0
    assert conditional_value_at_risk([]) == 0.0


def test_nan_filtered():
    returns = [float("nan"), -0.05, -0.02, 0.01, 0.03, float("inf")]
    cvar = conditional_value_at_risk(returns, alpha=0.5)
    assert np.isfinite(cvar)


def test_fatter_tail_has_worse_cvar():
    rng = np.random.default_rng(2)
    thin = rng.normal(0.0, 0.01, 2000)
    fat = rng.standard_t(3, 2000) * 0.01  # heavy-tailed
    assert conditional_value_at_risk(fat, 0.05) < conditional_value_at_risk(thin, 0.05)


def test_cvar_tiny_sample_falls_back_to_var():
    # 2 points, alpha small → empty strict tail → falls back to the VaR point.
    returns = [-0.03, 0.02]
    cvar = conditional_value_at_risk(returns, alpha=0.01)
    assert np.isfinite(cvar)
