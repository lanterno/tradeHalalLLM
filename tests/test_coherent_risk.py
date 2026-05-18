"""Tests for ml/coherent_risk.py — Round-5 Wave 14.E."""

from __future__ import annotations

import pytest

from halal_trader.ml.coherent_risk import (
    TailRiskReport,
    expected_shortfall,
    render_report,
    spectral_risk_measure,
    tail_risk_report,
    value_at_risk,
)

# --- Validation -------------------------------------------------------------


def test_var_alpha_zero_rejected():
    with pytest.raises(ValueError):
        value_at_risk([0.01, -0.02], alpha=0.0)


def test_var_alpha_one_rejected():
    with pytest.raises(ValueError):
        value_at_risk([0.01, -0.02], alpha=1.0)


def test_es_alpha_zero_rejected():
    with pytest.raises(ValueError):
        expected_shortfall([0.01], alpha=0.0)


def test_var_empty_returns_zero():
    assert value_at_risk([]) == 0.0


def test_es_empty_returns_zero():
    assert expected_shortfall([]) == 0.0


# --- VaR -----------------------------------------------------------------


def test_var_positive_loss_for_negative_return():
    """Sample with multiple losses → VaR captures them at α=0.10."""
    returns = [0.01] * 90 + [-0.10] * 10
    var = value_at_risk(returns, alpha=0.10)
    assert var > 0


def test_var_zero_when_all_gains():
    """All returns positive → no loss → VaR=0."""
    assert value_at_risk([0.01, 0.02, 0.03], alpha=0.05) == 0.0


def test_var_monotonic_in_alpha():
    """VaR at smaller α (deeper tail) should be ≥ VaR at larger α."""
    # 100 returns including some big losses
    returns = [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03] * 12 + [-0.10] * 4
    v_01 = value_at_risk(returns, alpha=0.01)
    v_05 = value_at_risk(returns, alpha=0.05)
    assert v_01 >= v_05


# --- Expected Shortfall ---------------------------------------------------


def test_es_at_least_var():
    """ES (mean of tail) >= VaR (quantile of tail) by definition."""
    returns = [-0.10, -0.08, -0.05, -0.03, 0.0, 0.01, 0.02, 0.03] * 25
    var = value_at_risk(returns, alpha=0.05)
    es = expected_shortfall(returns, alpha=0.05)
    assert es >= var


def test_es_zero_when_all_gains():
    assert expected_shortfall([0.01, 0.02, 0.03], alpha=0.05) == 0.0


def test_es_extreme_loss_dominates():
    """A single huge loss in the tail spikes ES."""
    base = [0.01] * 99
    es_clean = expected_shortfall(base + [0.0], alpha=0.05)
    es_dirty = expected_shortfall(base + [-1.0], alpha=0.05)
    assert es_dirty > es_clean


# --- Spectral risk measure -------------------------------------------------


def test_spectral_risk_zero_n_rejected():
    with pytest.raises(ValueError):
        spectral_risk_measure([0.01], weight_fn=lambda u: 1.0, n_quantiles=0)


def test_spectral_risk_negative_weight_rejected():
    with pytest.raises(ValueError):
        spectral_risk_measure([0.01], weight_fn=lambda u: -1.0)


def test_spectral_risk_uniform_weight_returns_neg_mean():
    """Uniform weight over all quantiles → -mean of returns (clamped to 0)."""
    returns = [0.01, 0.02, -0.05, -0.03, 0.0]
    spec = spectral_risk_measure(returns, weight_fn=lambda u: 1.0, n_quantiles=100)
    assert spec >= 0


def test_spectral_risk_empty():
    assert spectral_risk_measure([], weight_fn=lambda u: 1.0) == 0.0


def test_spectral_risk_es_equivalence():
    """Spectral with weight = 1[u<alpha]/alpha should approximate ES at that alpha."""
    returns = [-0.08, -0.05, -0.03, -0.01, 0.01, 0.02, 0.03] * 20
    alpha = 0.10
    spec = spectral_risk_measure(
        returns, weight_fn=lambda u: 1.0 if u < alpha else 0.0, n_quantiles=200
    )
    es = expected_shortfall(returns, alpha=alpha)
    # Rough approximation
    assert abs(spec - es) < 0.05


# --- Report ---------------------------------------------------------------


def test_tail_risk_report_combines_alphas():
    returns = [-0.08, -0.05, -0.03, -0.01, 0.01, 0.02, 0.03] * 20
    report = tail_risk_report(returns)
    assert report.n_returns == 140
    assert len(report.var_at) == 3
    assert len(report.es_at) == 3
    # ES >= VaR at each alpha
    for v, e in zip(report.var_at, report.es_at):
        assert e >= v


def test_tail_risk_report_validation_alpha_out_of_range():
    with pytest.raises(ValueError):
        TailRiskReport(alphas=(1.5,), var_at=(0.01,), es_at=(0.01,), n_returns=10)


def test_tail_risk_report_validation_negative_var():
    with pytest.raises(ValueError):
        TailRiskReport(alphas=(0.05,), var_at=(-0.01,), es_at=(0.01,), n_returns=10)


def test_tail_risk_report_validation_length_mismatch():
    with pytest.raises(ValueError):
        TailRiskReport(alphas=(0.05,), var_at=(0.01, 0.02), es_at=(0.01,), n_returns=10)


# --- Render ---------------------------------------------------------------


def test_render_report_includes_alphas():
    returns = [-0.05, 0.01, 0.02, 0.03] * 30
    report = tail_risk_report(returns)
    out = render_report(report)
    assert "Tail-risk" in out
    assert "α=0.010" in out
    assert "α=0.050" in out
    assert "VaR=" in out
    assert "ES=" in out


# --- E2E -----------------------------------------------------------------


def test_e2e_es_vs_var_real_distribution():
    """1000 returns with fat-tailed spikes — ES >= VaR."""
    import math

    # Simulate fat-tailed: small Gaussian + occasional spikes of varying sizes
    returns: list[float] = []
    for i in range(990):
        returns.append(0.001 * math.sin(i))  # bounded
    # 10 losses of varying severity
    for severity in (-0.30, -0.25, -0.20, -0.18, -0.15, -0.13, -0.10, -0.08, -0.05, -0.03):
        returns.append(severity)
    var = value_at_risk(returns, alpha=0.01)
    es = expected_shortfall(returns, alpha=0.01)
    # ES (mean of tail) should be at least VaR (single quantile point)
    assert es >= var
    assert es > 0.10  # tail dominated by the -0.30 ... events
