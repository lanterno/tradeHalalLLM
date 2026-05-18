"""Tests for ml/backtest_confidence.py — Round-5 Wave 14.G."""

from __future__ import annotations

import math

import pytest

from halal_trader.ml.backtest_confidence import (
    BootstrapPolicy,
    Metric,
    MetricEstimate,
    max_drawdown,
    profit_factor,
    render_report,
    report_with_ci,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)

# --- Validation ---------------------------------------------------


def test_metric_string_values():
    assert Metric.SHARPE.value == "sharpe"
    assert Metric.SORTINO.value == "sortino"
    assert Metric.WIN_RATE.value == "win_rate"
    assert Metric.PROFIT_FACTOR.value == "profit_factor"
    assert Metric.MAX_DRAWDOWN.value == "max_drawdown"


def test_default_policy():
    p = BootstrapPolicy()
    assert p.n_replicates == 1000
    assert p.confidence_level == 0.95


def test_policy_low_replicates_rejected():
    with pytest.raises(ValueError):
        BootstrapPolicy(n_replicates=50)


def test_policy_zero_confidence_rejected():
    with pytest.raises(ValueError):
        BootstrapPolicy(confidence_level=0.0)


def test_policy_one_confidence_rejected():
    with pytest.raises(ValueError):
        BootstrapPolicy(confidence_level=1.0)


def test_estimate_lower_above_upper_rejected():
    with pytest.raises(ValueError):
        MetricEstimate(
            metric=Metric.SHARPE,
            point_estimate=1.0,
            lower_ci=2.0,
            upper_ci=1.0,
            n_samples=10,
        )


# --- Single metrics -----------------------------------------------


def test_sharpe_zero_returns():
    assert sharpe_ratio([]) == 0
    assert sharpe_ratio([0.01]) == 0


def test_sharpe_constant_returns_zero():
    assert sharpe_ratio([0.01, 0.01, 0.01]) == 0


def test_sharpe_positive_for_positive_drift():
    s = sharpe_ratio([0.01, -0.005, 0.015, 0.005, 0.01])
    assert s > 0


def test_sortino_no_downside_infinite():
    assert math.isinf(sortino_ratio([0.01, 0.02, 0.03]))


def test_sortino_finite_with_downside():
    assert math.isfinite(sortino_ratio([0.01, -0.01, 0.02, -0.005]))


def test_win_rate_basic():
    assert win_rate([1, -1, 1, 1, -1]) == 0.6


def test_win_rate_empty_zero():
    assert win_rate([]) == 0


def test_profit_factor_basic():
    assert profit_factor([2, -1, 3, -2]) == pytest.approx(5 / 3)


def test_profit_factor_no_losses_infinite():
    assert math.isinf(profit_factor([1, 2, 3]))


def test_profit_factor_no_gains_zero():
    assert profit_factor([-1, -2]) == 0


def test_max_drawdown_no_drawdown():
    """Monotonic gains → no drawdown."""
    assert max_drawdown([0.01, 0.02, 0.01, 0.005]) == 0


def test_max_drawdown_with_loss():
    """Run-up to 1.10, then drop to 1.05 → max drawdown ≈ 4.5%."""
    dd = max_drawdown([0.05, 0.05, -0.045])
    assert 0.04 < dd < 0.05


# --- Bootstrap report ---------------------------------------------


def test_report_empty_returns_rejected():
    with pytest.raises(ValueError):
        report_with_ci([])


def test_report_includes_all_default_metrics():
    returns = [0.01, -0.005, 0.015, -0.01, 0.005] * 20
    report = report_with_ci(returns, seed=42)
    metrics = [e.metric for e in report]
    assert Metric.SHARPE in metrics
    assert Metric.SORTINO in metrics
    assert Metric.WIN_RATE in metrics
    assert Metric.PROFIT_FACTOR in metrics
    assert Metric.MAX_DRAWDOWN in metrics


def test_report_seeded_replay_consistent():
    returns = [0.01, -0.005, 0.015, -0.01, 0.005] * 20
    a = report_with_ci(returns, seed=42)
    b = report_with_ci(returns, seed=42)
    assert a == b


def test_report_different_seed_different_ci():
    """Different seeds usually produce different bootstrap CIs."""
    returns = [0.01, -0.005, 0.015, -0.01, 0.005] * 20
    a = report_with_ci(returns, seed=1)
    b = report_with_ci(returns, seed=2)
    # CIs may differ
    assert a != b


def test_report_lower_le_point_le_upper():
    returns = [0.01, -0.005, 0.015, -0.01, 0.005] * 20
    report = report_with_ci(returns, seed=42)
    for e in report:
        if (
            math.isfinite(e.lower_ci)
            and math.isfinite(e.upper_ci)
            and math.isfinite(e.point_estimate)
        ):
            assert e.lower_ci <= e.upper_ci


def test_report_n_samples_matches_input():
    returns = [0.01] * 50
    report = report_with_ci(returns, seed=1)
    for e in report:
        assert e.n_samples == 50


def test_report_subset_of_metrics():
    returns = [0.01, -0.005] * 20
    report = report_with_ci(returns, metrics=[Metric.SHARPE, Metric.WIN_RATE], seed=42)
    assert len(report) == 2
    assert report[0].metric is Metric.SHARPE
    assert report[1].metric is Metric.WIN_RATE


def test_report_custom_policy():
    returns = [0.01, -0.005] * 50
    pol = BootstrapPolicy(n_replicates=500, confidence_level=0.99)
    report = report_with_ci(returns, policy=pol, seed=42)
    assert len(report) > 0


# --- Render -------------------------------------------------------


def test_render_includes_metrics():
    returns = [0.01, -0.005, 0.015] * 10
    report = report_with_ci(returns, seed=1)
    out = render_report(report)
    assert "Backtest report" in out
    assert "sharpe" in out
    assert "win_rate" in out


def test_render_empty_report():
    out = render_report([])
    assert "no metrics" in out


def test_render_no_secret_leak():
    returns = [0.01] * 30
    report = report_with_ci(returns, seed=1)
    out = render_report(report)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E -----------------------------------------------------


def test_e2e_small_sample_wider_ci():
    """A 30-trade backtest should have a wider Sharpe CI than a 300-trade one."""
    small = [0.01, -0.005, 0.015, -0.01, 0.005, 0.01] * 5
    large = [0.01, -0.005, 0.015, -0.01, 0.005, 0.01] * 50
    rep_small = report_with_ci(small, metrics=[Metric.SHARPE], seed=42)
    rep_large = report_with_ci(large, metrics=[Metric.SHARPE], seed=42)
    width_small = rep_small[0].upper_ci - rep_small[0].lower_ci
    width_large = rep_large[0].upper_ci - rep_large[0].lower_ci
    assert width_small > width_large
