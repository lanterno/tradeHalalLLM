"""Tests for ml/halal_alpha_beta.py — Round-5 Wave 7.F."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.ml.halal_alpha_beta import (
    HalalReplicationPlan,
    UnreliableBetaError,
    attribute,
    decompose,
    plan_replication,
    render_attribution,
    render_decomposition,
    render_plan,
)

# --- decompose ----------------------------------------------------------


def test_decompose_pure_beta_one():
    """Pin: y = 1.0 × x → β=1, α=0, R²=1."""
    bench = [0.01, -0.02, 0.005, 0.015, -0.01]
    port = list(bench)
    d = decompose(port, bench)
    assert d.beta == pytest.approx(1.0, abs=1e-9)
    assert d.alpha_per_period == pytest.approx(0.0, abs=1e-9)
    assert d.r_squared == pytest.approx(1.0, abs=1e-9)


def test_decompose_pure_alpha_zero_beta():
    """Constant excess return + zero correlation → β=0, α=constant."""
    bench = [0.01, -0.02, 0.005, 0.015, -0.01]
    port = [0.005] * 5  # constant
    d = decompose(port, bench)
    assert d.beta == pytest.approx(0.0, abs=1e-9)
    assert d.alpha_per_period == pytest.approx(0.005, abs=1e-9)


def test_decompose_beta_two():
    """y = 2x → β=2, α=0."""
    bench = [0.01, -0.02, 0.005, 0.015, -0.01]
    port = [2 * x for x in bench]
    d = decompose(port, bench)
    assert d.beta == pytest.approx(2.0, abs=1e-9)


def test_decompose_too_few_periods_rejected():
    with pytest.raises(ValueError):
        decompose([0.01, 0.02], [0.01, 0.02])


def test_decompose_length_mismatch_rejected():
    with pytest.raises(ValueError):
        decompose([0.01, 0.02, 0.03, 0.04, 0.05], [0.01, 0.02])


def test_decompose_zero_benchmark_variance_rejected():
    with pytest.raises(ValueError):
        decompose([0.01, 0.02, 0.03, 0.04, 0.05], [0.01] * 5)


def test_decompose_unreliable_beta_raises():
    """Pin: β > 3 → UnreliableBetaError."""
    bench = [0.01, -0.02, 0.005, 0.015, -0.01]
    port = [5 * x for x in bench]  # β=5
    with pytest.raises(UnreliableBetaError):
        decompose(port, bench)


def test_decompose_annualised_alpha_helper():
    bench = [0.01, -0.02, 0.005, 0.015, -0.01]
    port = [0.005] * 5
    d = decompose(port, bench)
    assert d.annualised_alpha(252) == pytest.approx(0.005 * 252)


def test_decompose_imperfect_fit_r_squared_below_one():
    bench = [0.01, -0.02, 0.005, 0.015, -0.01]
    port = [0.01, 0.05, 0.005, 0.015, -0.01]  # one outlier
    d = decompose(port, bench)
    assert d.r_squared < 1.0


# --- plan_replication ----------------------------------------------------


def test_plan_replication_basic_long_beta():
    plan = plan_replication(
        portfolio_value=1_000_000.0,
        target_beta=0.8,
        benchmark_symbol="SPX_HALAL",
        waad_promisor_id="bob",
        issue_date=date(2026, 6, 1),
        expiry=date(2026, 12, 1),
    )
    assert plan.waad_notional == pytest.approx(800_000.0)
    assert plan.cash_sleeve_value == pytest.approx(200_000.0)


def test_plan_replication_beta_above_one_no_cash_sleeve():
    plan = plan_replication(
        portfolio_value=1_000_000.0,
        target_beta=1.5,
        benchmark_symbol="SPX_HALAL",
        waad_promisor_id="bob",
        issue_date=date(2026, 6, 1),
        expiry=date(2026, 12, 1),
    )
    assert plan.waad_notional == pytest.approx(1_500_000.0)
    assert plan.cash_sleeve_value == 0.0


def test_plan_replication_negative_beta_keeps_full_cash():
    """Pin: short-benchmark via reverse-Wa'd; cash sleeve full."""
    plan = plan_replication(
        portfolio_value=1_000_000.0,
        target_beta=-0.5,
        benchmark_symbol="SPX_HALAL",
        waad_promisor_id="bob",
        issue_date=date(2026, 6, 1),
        expiry=date(2026, 12, 1),
    )
    assert plan.waad_notional == pytest.approx(500_000.0)
    assert plan.cash_sleeve_value == pytest.approx(1_000_000.0)


def test_plan_replication_invalid_portfolio_rejected():
    with pytest.raises(ValueError):
        plan_replication(
            portfolio_value=-1.0,
            target_beta=1.0,
            benchmark_symbol="SPX",
            waad_promisor_id="bob",
            issue_date=date(2026, 6, 1),
            expiry=date(2026, 12, 1),
        )


def test_plan_replication_expiry_before_issue_rejected():
    with pytest.raises(ValueError):
        HalalReplicationPlan(
            portfolio_value=1_000_000.0,
            target_beta=1.0,
            benchmark_symbol="SPX",
            waad_promisor_id="bob",
            waad_notional=1_000_000.0,
            cash_sleeve_value=0.0,
            issue_date=date(2026, 12, 1),
            expiry=date(2026, 6, 1),
        )


def test_plan_replication_beta_out_of_band_rejected():
    with pytest.raises(ValueError):
        HalalReplicationPlan(
            portfolio_value=1_000_000.0,
            target_beta=10.0,
            benchmark_symbol="SPX",
            waad_promisor_id="bob",
            waad_notional=10_000_000.0,
            cash_sleeve_value=0.0,
            issue_date=date(2026, 6, 1),
            expiry=date(2026, 12, 1),
        )


def test_plan_replication_immutable():
    plan = plan_replication(
        portfolio_value=1_000_000.0,
        target_beta=1.0,
        benchmark_symbol="SPX",
        waad_promisor_id="bob",
        issue_date=date(2026, 6, 1),
        expiry=date(2026, 12, 1),
    )
    with pytest.raises(AttributeError):
        plan.target_beta = 0.5  # type: ignore[misc]


# --- attribute -----------------------------------------------------------


def test_attribute_pure_beta_pnl():
    plan = plan_replication(
        portfolio_value=1_000_000.0,
        target_beta=1.0,
        benchmark_symbol="SPX",
        waad_promisor_id="bob",
        issue_date=date(2026, 6, 1),
        expiry=date(2026, 12, 1),
    )
    rep = attribute(
        plan,
        realised_portfolio_return=0.05,
        realised_benchmark_return=0.05,
    )
    # β=1, port and bench moved 5%, so β_pnl = 1 × 0.05 × 1M = 50,000.
    # α_pnl = 50,000 - 50,000 - 0 = 0.
    assert rep.beta_pnl == pytest.approx(50_000.0)
    assert rep.alpha_pnl == pytest.approx(0.0)


def test_attribute_alpha_when_excess_return():
    plan = plan_replication(
        portfolio_value=1_000_000.0,
        target_beta=1.0,
        benchmark_symbol="SPX",
        waad_promisor_id="bob",
        issue_date=date(2026, 6, 1),
        expiry=date(2026, 12, 1),
    )
    # Portfolio outperformed benchmark by 2% → α_pnl = 20,000.
    rep = attribute(
        plan,
        realised_portfolio_return=0.07,
        realised_benchmark_return=0.05,
    )
    assert rep.alpha_pnl == pytest.approx(20_000.0)


def test_attribute_with_cash_drag():
    plan = plan_replication(
        portfolio_value=1_000_000.0,
        target_beta=0.5,
        benchmark_symbol="SPX",
        waad_promisor_id="bob",
        issue_date=date(2026, 6, 1),
        expiry=date(2026, 12, 1),
        expected_alpha_drag_per_period=-0.001,  # -10bps drag
    )
    rep = attribute(
        plan,
        realised_portfolio_return=0.03,
        realised_benchmark_return=0.05,
    )
    # cash_sleeve = 500k; drag = -0.001 × 500k = -500.
    assert rep.cash_drag == pytest.approx(-500.0)


def test_attribute_invalid_returns_rejected():
    plan = plan_replication(
        portfolio_value=1_000_000.0,
        target_beta=1.0,
        benchmark_symbol="SPX",
        waad_promisor_id="bob",
        issue_date=date(2026, 6, 1),
        expiry=date(2026, 12, 1),
    )
    with pytest.raises(ValueError):
        attribute(plan, realised_portfolio_return=10.0, realised_benchmark_return=0.05)
    with pytest.raises(ValueError):
        attribute(plan, realised_portfolio_return=0.05, realised_benchmark_return=-2.0)


# --- Render --------------------------------------------------------------


def test_render_decomposition_format():
    bench = [0.01, -0.02, 0.005, 0.015, -0.01]
    port = list(bench)
    d = decompose(port, bench)
    out = render_decomposition(d)
    assert "α=" in out
    assert "β=" in out
    assert "R²" in out


def test_render_plan_no_secret_leak():
    plan = plan_replication(
        portfolio_value=1_000_000.0,
        target_beta=1.0,
        benchmark_symbol="SPX",
        waad_promisor_id="bob@example.com",
        issue_date=date(2026, 6, 1),
        expiry=date(2026, 12, 1),
    )
    out = render_plan(plan)
    assert "@example.com" not in out
    assert "bob@example.com" not in out


def test_render_attribution():
    plan = plan_replication(
        portfolio_value=1_000_000.0,
        target_beta=1.0,
        benchmark_symbol="SPX",
        waad_promisor_id="bob",
        issue_date=date(2026, 6, 1),
        expiry=date(2026, 12, 1),
    )
    rep = attribute(
        plan,
        realised_portfolio_return=0.05,
        realised_benchmark_return=0.05,
    )
    out = render_attribution(rep)
    assert "Attribution" in out
    assert "β leg" in out
    assert "α leg" in out
