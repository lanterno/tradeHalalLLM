"""Tests for ml/monte_carlo.py — Round-5 Wave 14.B."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from halal_trader.ml.monte_carlo import (
    SimulationInputs,
    SimulationResult,
    render_result,
    simulate,
)


def _two_asset_inputs(**overrides) -> SimulationInputs:
    base = {
        "weights": (0.6, 0.4),
        "expected_daily_returns": (0.0005, 0.0003),
        "daily_covariance": (
            (0.0001, 0.00005),
            (0.00005, 0.00009),
        ),
        "initial_value": 100000.0,
        "horizon_days": 252,
    }
    base.update(overrides)
    return SimulationInputs(**base)


# --- Validation -------------------------------------------------------------


def test_inputs_empty_weights_rejected():
    with pytest.raises(ValueError):
        SimulationInputs(
            weights=(),
            expected_daily_returns=(),
            daily_covariance=(),
            initial_value=1.0,
            horizon_days=1,
        )


def test_inputs_weights_dont_sum_to_one_rejected():
    with pytest.raises(ValueError):
        _two_asset_inputs(weights=(0.5, 0.4))


def test_inputs_negative_weight_rejected():
    with pytest.raises(ValueError):
        _two_asset_inputs(weights=(1.5, -0.5))


def test_inputs_returns_length_mismatch_rejected():
    with pytest.raises(ValueError):
        _two_asset_inputs(expected_daily_returns=(0.0005,))


def test_inputs_non_square_cov_rejected():
    with pytest.raises(ValueError):
        _two_asset_inputs(daily_covariance=((0.0001, 0.00005),))


def test_inputs_non_square_inner_rejected():
    with pytest.raises(ValueError):
        _two_asset_inputs(
            daily_covariance=(
                (0.0001, 0.00005, 0.00001),  # wrong inner length
                (0.00005, 0.00009),
            )
        )


def test_inputs_zero_initial_value_rejected():
    with pytest.raises(ValueError):
        _two_asset_inputs(initial_value=0.0)


def test_inputs_zero_horizon_rejected():
    with pytest.raises(ValueError):
        _two_asset_inputs(horizon_days=0)


def test_inputs_immutable():
    inp = _two_asset_inputs()
    with pytest.raises(AttributeError):
        inp.initial_value = 0.0  # type: ignore[misc]


# --- Simulation -------------------------------------------------------------


def test_simulate_returns_correct_shape():
    result = simulate(_two_asset_inputs(), n_paths=200, seed=42)
    assert result.n_paths == 200
    assert len(result.terminal_values) == 200


def test_simulate_seed_replay_consistent():
    a = simulate(_two_asset_inputs(), n_paths=100, seed=12345)
    b = simulate(_two_asset_inputs(), n_paths=100, seed=12345)
    assert a == b


def test_simulate_different_seed_different_paths():
    a = simulate(_two_asset_inputs(), n_paths=100, seed=1)
    b = simulate(_two_asset_inputs(), n_paths=100, seed=2)
    assert a.terminal_values != b.terminal_values


def test_simulate_zero_paths_rejected():
    with pytest.raises(ValueError):
        simulate(_two_asset_inputs(), n_paths=0, seed=1)


def test_simulate_p10_le_p50_le_p90():
    result = simulate(_two_asset_inputs(), n_paths=500, seed=42)
    assert result.p10_terminal <= result.p50_terminal <= result.p90_terminal


def test_simulate_var_non_negative():
    result = simulate(_two_asset_inputs(), n_paths=500, seed=42)
    assert result.var_95 >= 0


def test_simulate_cvar_at_least_var():
    """CVaR (expected loss in tail) >= VaR (5th percentile loss)."""
    result = simulate(_two_asset_inputs(), n_paths=500, seed=42)
    assert result.cvar_95 >= result.var_95 - 1e-6  # tolerance


def test_simulate_zero_volatility_terminal_close_to_drift():
    """With zero covariance, terminal is deterministic at e^{drift * horizon}."""
    inp = SimulationInputs(
        weights=(1.0,),
        expected_daily_returns=(0.001,),
        daily_covariance=((0.0,),),
        initial_value=100.0,
        horizon_days=100,
    )
    result = simulate(inp, n_paths=10, seed=1)
    expected = 100.0 * pow(1.001, 100)
    # All paths near identical
    for v in result.terminal_values:
        assert v == pytest.approx(expected, rel=0.01)


def test_simulate_higher_horizon_wider_spread():
    short = simulate(_two_asset_inputs(horizon_days=10), n_paths=500, seed=1)
    long_run = simulate(_two_asset_inputs(horizon_days=500), n_paths=500, seed=1)
    short_spread = short.p90_terminal - short.p10_terminal
    long_spread = long_run.p90_terminal - long_run.p10_terminal
    assert long_spread > short_spread


# --- Result validation -----------------------------------------------------


def test_simulation_result_zero_n_paths_rejected():
    with pytest.raises(ValueError):
        SimulationResult(
            n_paths=0,
            horizon_days=1,
            initial_value=1.0,
            terminal_values=(),
            var_95=0.0,
            cvar_95=0.0,
            p50_terminal=1.0,
            p10_terminal=1.0,
            p90_terminal=1.0,
            mean_terminal=1.0,
        )


# --- Render ----------------------------------------------------------------


def test_render_includes_summary():
    result = simulate(_two_asset_inputs(), n_paths=200, seed=42)
    out = render_result(result)
    assert "Monte Carlo" in out
    assert "200 paths" in out
    assert "VaR" in out
    assert "CVaR" in out


def test_render_no_secret_leak():
    result = simulate(_two_asset_inputs(), n_paths=200, seed=42)
    out = render_result(result)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E -------------------------------------------------------------------


def test_e2e_balanced_60_40_portfolio_one_year():
    result = simulate(_two_asset_inputs(), n_paths=1000, seed=2026)
    # Mean should be positive (drift) and reasonable
    assert result.mean_terminal > result.initial_value * 0.8
    assert result.mean_terminal < result.initial_value * 2.0
