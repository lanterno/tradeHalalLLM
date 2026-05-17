"""Tests for ml/tail_dependence.py — Round-5 Wave 14.F."""

from __future__ import annotations

import random

import pytest

from halal_trader.ml.tail_dependence import (
    TailDependenceReport,
    TailDirection,
    clayton_copula_cdf,
    clayton_lower_tail,
    empirical_tail_dependence,
    estimate_tail_dependence,
    gaussian_copula_cdf,
    gaussian_copula_lower_tail,
    render_report,
)

# --- Validation ---------------------------------------------------


def test_direction_string_values():
    assert TailDirection.LOWER.value == "lower"
    assert TailDirection.UPPER.value == "upper"


def test_report_lower_outside_range_rejected():
    with pytest.raises(ValueError):
        TailDependenceReport(
            lower_tail=1.5,
            upper_tail=0.5,
            n_observations=100,
            quantile_threshold=0.05,
        )


def test_report_zero_observations_rejected():
    with pytest.raises(ValueError):
        TailDependenceReport(
            lower_tail=0.5,
            upper_tail=0.5,
            n_observations=0,
            quantile_threshold=0.05,
        )


# --- Empirical tail dependence -----------------------------------


def test_empirical_perfectly_correlated_high_tail_dep():
    """Identical series → tail dependence = 1."""
    a = [random.Random(1).gauss(0, 1) for _ in range(200)]
    b = list(a)
    val = empirical_tail_dependence(a, b, q=0.05, direction=TailDirection.LOWER)
    assert val == 1.0


def test_empirical_independent_low_tail_dep():
    """Independent gaussian → tail dependence near 0."""
    rng_a = random.Random(1)
    rng_b = random.Random(2)
    a = [rng_a.gauss(0, 1) for _ in range(500)]
    b = [rng_b.gauss(0, 1) for _ in range(500)]
    val = empirical_tail_dependence(a, b, q=0.05, direction=TailDirection.LOWER)
    assert val < 0.30


def test_empirical_anticorrelated_zero_lower_tail():
    """Perfect anti-correlation → no lower-tail co-movement."""
    a = list(range(100))
    b = [-x for x in a]
    val = empirical_tail_dependence(
        a, b, q=0.10, direction=TailDirection.LOWER
    )
    # When A is in lower tail (smallest), B is in upper tail → λ_L should be 0
    assert val == 0.0


def test_empirical_length_mismatch_rejected():
    with pytest.raises(ValueError):
        empirical_tail_dependence([1.0] * 30, [1.0] * 25)


def test_empirical_too_few_observations_rejected():
    with pytest.raises(ValueError):
        empirical_tail_dependence([1.0, 2.0], [1.0, 2.0])


def test_empirical_q_zero_rejected():
    with pytest.raises(ValueError):
        empirical_tail_dependence([1.0] * 50, [1.0] * 50, q=0.0)


def test_empirical_q_one_rejected():
    with pytest.raises(ValueError):
        empirical_tail_dependence([1.0] * 50, [1.0] * 50, q=1.0)


# --- Gaussian copula --------------------------------------------


def test_gaussian_copula_zero_correlation_independent():
    """rho=0 → C(u, v) = u * v."""
    u, v = 0.5, 0.5
    cdf = gaussian_copula_cdf(u, v, rho=0.0)
    assert cdf == pytest.approx(u * v, abs=0.01)


def test_gaussian_copula_u_v_in_unit_interval_required():
    with pytest.raises(ValueError):
        gaussian_copula_cdf(0.0, 0.5, rho=0.5)


def test_gaussian_copula_rho_at_boundary_rejected():
    with pytest.raises(ValueError):
        gaussian_copula_cdf(0.5, 0.5, rho=1.0)


def test_gaussian_copula_lower_tail_is_zero():
    """Gaussian copula has zero asymptotic tail dependence."""
    assert gaussian_copula_lower_tail(0.7) == 0.0


def test_gaussian_lower_tail_invalid_rho_rejected():
    with pytest.raises(ValueError):
        gaussian_copula_lower_tail(2.0)


# --- Clayton copula ---------------------------------------------


def test_clayton_copula_basic():
    cdf = clayton_copula_cdf(0.5, 0.5, theta=2.0)
    assert 0.0 < cdf < 1.0


def test_clayton_copula_negative_theta_rejected():
    with pytest.raises(ValueError):
        clayton_copula_cdf(0.5, 0.5, theta=-1.0)


def test_clayton_copula_uv_outside_rejected():
    with pytest.raises(ValueError):
        clayton_copula_cdf(0.0, 0.5, theta=2.0)


def test_clayton_lower_tail_known_value():
    """Clayton θ=2 → λ_L = 2^(-0.5) ≈ 0.707."""
    assert clayton_lower_tail(2.0) == pytest.approx(0.7071, abs=0.001)


def test_clayton_lower_tail_higher_theta_higher_dep():
    """Higher θ → stronger lower-tail dependence."""
    assert clayton_lower_tail(5.0) > clayton_lower_tail(1.0)


def test_clayton_lower_tail_negative_theta_rejected():
    with pytest.raises(ValueError):
        clayton_lower_tail(-1.0)


# --- Estimate report -------------------------------------------


def test_estimate_returns_both_tails():
    a = [random.Random(1).gauss(0, 1) for _ in range(100)]
    b = list(a)
    report = estimate_tail_dependence(a, b)
    assert report.lower_tail == 1.0
    assert report.upper_tail == 1.0


def test_estimate_records_n_and_threshold():
    a = list(range(100))
    b = list(range(100))
    report = estimate_tail_dependence(a, b, q=0.10)
    assert report.n_observations == 100
    assert report.quantile_threshold == 0.10


# --- Render ----------------------------------------------------


def test_render_includes_summary():
    report = TailDependenceReport(
        lower_tail=0.5, upper_tail=0.4, n_observations=200, quantile_threshold=0.05
    )
    out = render_report(report)
    assert "Tail dependence" in out
    assert "lower=0.500" in out
    assert "upper=0.400" in out


def test_render_no_secret_leak():
    report = TailDependenceReport(
        lower_tail=0.5, upper_tail=0.4, n_observations=200, quantile_threshold=0.05
    )
    out = render_report(report)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E -----------------------------------------------------


def test_e2e_perfectly_correlated_pair_full_tail_dep():
    a = [random.Random(7).gauss(0, 1) for _ in range(300)]
    b = list(a)
    report = estimate_tail_dependence(a, b)
    assert report.lower_tail == 1.0
    assert report.upper_tail == 1.0


def test_replay_consistency():
    a = [random.Random(1).gauss(0, 1) for _ in range(100)]
    b = list(a)
    r1 = empirical_tail_dependence(a, b, q=0.05)
    r2 = empirical_tail_dependence(a, b, q=0.05)
    assert r1 == r2
