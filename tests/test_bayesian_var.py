"""Tests for ml/bayesian_var.py — Round-5 Wave 14.A."""

from __future__ import annotations

import math

import pytest

from halal_trader.ml.bayesian_var import (
    BayesianVarResult,
    bayesian_var,
    cornish_fisher_quantile,
    excess_kurtosis,
    render_result,
    skewness,
    stddev,
)

# --- Helpers ----------------------------------------------------------------


def test_stddev_matches_known():
    assert stddev([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(math.sqrt(2.5))


def test_stddev_empty_zero():
    assert stddev([]) == 0.0


def test_stddev_single_zero():
    assert stddev([1.0]) == 0.0


def test_skewness_symmetric_zero():
    """Symmetric around mean → skewness zero."""
    assert abs(skewness([-1.0, -0.5, 0.0, 0.5, 1.0])) < 1e-10


def test_skewness_negative_skewed():
    """Long left tail → negative skew."""
    returns = [-5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert skewness(returns) < 0


def test_skewness_positive_skewed():
    returns = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 5.0]
    assert skewness(returns) > 0


def test_excess_kurtosis_normal_near_zero():
    """A roughly Gaussian sample has excess kurtosis ~ 0."""
    # Sample with low excess kurtosis
    returns = [-2.0, -1.0, -0.5, -0.1, 0.0, 0.0, 0.1, 0.5, 1.0, 2.0]
    assert abs(excess_kurtosis(returns)) < 2.0


def test_excess_kurtosis_fat_tailed_positive():
    """Fat-tailed: many small returns + occasional big ones → high excess kurtosis."""
    returns = [0.001] * 99 + [10.0]
    assert excess_kurtosis(returns) > 10.0


# --- Cornish-Fisher quantile -------------------------------------------------


def test_cornish_fisher_no_skew_no_kurt_returns_z():
    z = -1.6449  # 95% one-sided
    assert cornish_fisher_quantile(z, skew=0.0, excess_kurt=0.0) == pytest.approx(z)


def test_cornish_fisher_negative_skew_pushes_quantile_lower():
    """Negative skew should worsen (more negative) the quantile."""
    z = -1.6449
    cf = cornish_fisher_quantile(z, skew=-1.0, excess_kurt=0.0)
    assert cf < z  # more negative


def test_cornish_fisher_positive_excess_kurtosis_at_deep_tail_pushes_lower():
    """At 1% (z=-2.326), the kurtosis correction pushes the quantile lower."""
    z = -2.326  # 1%
    cf = cornish_fisher_quantile(z, skew=0.0, excess_kurt=3.0)
    assert cf < z


# --- bayesian_var -----------------------------------------------------------


def test_bayesian_var_empty_returns_zero():
    r = bayesian_var([])
    assert r.var_normal == 0
    assert r.var_cornish_fisher == 0
    assert r.n_samples == 0


def test_bayesian_var_single_sample_returns_zero():
    r = bayesian_var([0.01])
    assert r.var_normal == 0
    assert r.var_cornish_fisher == 0


def test_bayesian_var_alpha_zero_rejected():
    with pytest.raises(ValueError):
        bayesian_var([0.01, 0.02], alpha=0.0)


def test_bayesian_var_alpha_one_rejected():
    with pytest.raises(ValueError):
        bayesian_var([0.01, 0.02], alpha=1.0)


def test_bayesian_var_returns_positive_for_loss_distribution():
    """Negative-mean sample → positive VaR."""
    returns = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03] * 10
    r = bayesian_var(returns, alpha=0.05)
    assert r.var_normal > 0


def test_bayesian_var_cf_higher_for_negative_skew():
    """Negative-skew sample → CF VaR > normal VaR."""
    # Construct: many small +0.01, occasional big negative
    returns = [0.01] * 95 + [-0.30] * 5
    r = bayesian_var(returns, alpha=0.05)
    assert r.sample_skewness < 0
    assert r.var_cornish_fisher >= r.var_normal - 1e-6


def test_bayesian_var_records_alpha():
    r = bayesian_var([0.01, -0.02, 0.0, 0.01, -0.01], alpha=0.10)
    assert r.alpha == 0.10


def test_bayesian_var_records_n_samples():
    r = bayesian_var([0.01] * 50)
    assert r.n_samples == 50


# --- Result invariants -----------------------------------------------------


def test_result_invalid_alpha_rejected():
    with pytest.raises(ValueError):
        BayesianVarResult(
            alpha=1.5,
            var_normal=0.01,
            var_cornish_fisher=0.01,
            sample_skewness=0.0,
            sample_excess_kurtosis=0.0,
            n_samples=10,
        )


def test_result_negative_var_rejected():
    with pytest.raises(ValueError):
        BayesianVarResult(
            alpha=0.05,
            var_normal=-0.01,
            var_cornish_fisher=0.01,
            sample_skewness=0.0,
            sample_excess_kurtosis=0.0,
            n_samples=10,
        )


# --- Render ---------------------------------------------------------------


def test_render_includes_components():
    r = bayesian_var([0.01, -0.02, 0.0, 0.01, -0.01] * 10)
    out = render_result(r)
    assert "Bayesian VaR" in out
    assert "normal=" in out
    assert "CF=" in out
    assert "skew=" in out
    assert "excess_kurt=" in out


def test_render_no_secret_leak():
    r = bayesian_var([0.01, -0.02, 0.0, 0.01, -0.01] * 10)
    out = render_result(r)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ------------------------------------------------------------------


def test_e2e_bayesian_vs_normal_moderate_skew():
    """A mildly negatively-skewed sample — both VaRs positive, both finite.

    Note: with extreme skew/kurt, the Cornish-Fisher expansion can become
    non-monotone; this test uses a moderate-tail sample where the
    expansion is well-behaved, and just confirms that both VaR
    measures emerge non-negative + the sample summary statistics are
    captured.
    """
    returns = [0.01, 0.005, 0.0, -0.005, -0.01, -0.02] * 30
    r = bayesian_var(returns, alpha=0.05)
    assert r.var_normal >= 0
    assert r.var_cornish_fisher >= 0
    assert r.n_samples == 180


def test_replay_consistency():
    returns = [0.01, -0.02, 0.0] * 20
    a = bayesian_var(returns, alpha=0.05)
    b = bayesian_var(returns, alpha=0.05)
    assert a == b
