"""Tests for `ml/hrp.py` (Hierarchical Risk Parity allocator).

Covers the algorithmic invariants — non-negative weights, no
leverage, lower variance → higher weight — plus the input-validation
edges. We use deterministic returns (NumPy RNG with a fixed seed)
so the test reasons about the *shape* of the answer rather than
brittle exact values.
"""

from __future__ import annotations

import numpy as np
import pytest

from halal_trader.ml.hrp import (
    HRPAllocation,
    _cluster_variance,
    _correlation_to_distance,
    _inverse_variance_weights,
    _recursive_bisection,
    _single_linkage_order,
    allocate,
)


def _synth_returns(
    n_periods: int = 200,
    *,
    seed: int = 42,
    n_assets: int = 4,
    vol: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04),
    rho: float = 0.0,
) -> np.ndarray:
    """Generate Gaussian returns with given per-asset vol + a single
    correlation knob (block-correlated across all assets).

    Lets each test build a controlled scenario without dragging in
    real market data.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_periods, n_assets))
    if rho > 0:
        common = rng.standard_normal((n_periods, 1))
        base = np.sqrt(1 - rho) * base + np.sqrt(rho) * common
    if len(vol) != n_assets:
        raise AssertionError("vol tuple must match n_assets")
    return base * np.array(vol)


# ── invariants ────────────────────────────────────────────


def test_weights_are_non_negative():
    """Halal constraint: no shorts. Pin so a refactor that swaps the
    bisection branch can't sneak a negative weight in."""
    returns = _synth_returns()
    result = allocate(returns, ["A", "B", "C", "D"])
    for sym, w in result.weights.items():
        assert w >= 0, f"{sym} got negative weight {w}"


def test_weights_sum_to_one_when_no_buffer():
    returns = _synth_returns()
    result = allocate(returns, ["A", "B", "C", "D"])
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-9)


def test_weights_sum_to_one_minus_buffer_when_buffer_set():
    returns = _synth_returns()
    result = allocate(returns, ["A", "B", "C", "D"], cash_buffer_pct=0.1)
    assert sum(result.weights.values()) == pytest.approx(0.9, abs=1e-9)


def test_lower_vol_asset_gets_more_weight_than_higher_vol():
    """Core HRP semantic: with uncorrelated assets, the inverse-
    variance preference inside each cluster bubbles up. The lowest-
    vol asset should never be the lightest-weighted."""
    # vol = 1%, 2%, 3%, 4% — strictly increasing.
    returns = _synth_returns(vol=(0.01, 0.02, 0.03, 0.04), rho=0.0)
    result = allocate(returns, ["A", "B", "C", "D"])
    # The 1%-vol asset should outweigh the 4%-vol one.
    assert result.weights["A"] > result.weights["D"]


def test_equal_vol_uncorrelated_assets_get_equal_weights():
    """Sanity: with identical risk and zero co-movement, HRP
    degenerates to equal-weight."""
    returns = _synth_returns(vol=(0.02, 0.02, 0.02, 0.02), rho=0.0)
    result = allocate(returns, ["A", "B", "C", "D"])
    weights = list(result.weights.values())
    # Allow some sampling noise — the cluster-tree structure can
    # introduce small skews even with iid Gaussian returns.
    assert max(weights) - min(weights) < 0.06


def test_cluster_order_lists_every_symbol_exactly_once():
    """The cluster ordering is the leaf traversal — must be a
    permutation of the kept symbols, no dupes / drops."""
    returns = _synth_returns()
    result = allocate(returns, ["A", "B", "C", "D"])
    assert sorted(result.cluster_order) == ["A", "B", "C", "D"]


# ── edge cases ────────────────────────────────────────────


def test_single_asset_universe_returns_full_weight():
    returns = _synth_returns(n_assets=1, vol=(0.02,))
    result = allocate(returns, ["SOLO"])
    assert result.weights == {"SOLO": pytest.approx(1.0)}
    assert result.cluster_order == ["SOLO"]


def test_single_asset_with_buffer_returns_buffer_short():
    returns = _synth_returns(n_assets=1, vol=(0.02,))
    result = allocate(returns, ["SOLO"], cash_buffer_pct=0.2)
    assert result.weights["SOLO"] == pytest.approx(0.8)


def test_empty_universe_returns_empty_allocation():
    returns = np.zeros((50, 0))
    result = allocate(returns, [])
    assert result == HRPAllocation(weights={}, cluster_order=[])


def test_zero_variance_asset_is_dropped_silently():
    """A frozen-price asset has no usable covariance signal —
    silently exclude rather than erroring out, since the caller's
    universe can change cycle-to-cycle."""
    returns = _synth_returns(vol=(0.01, 0.02, 0.03, 0.04))
    # Force asset C (index 2) to have zero variance.
    returns[:, 2] = 0.0
    result = allocate(returns, ["A", "B", "C", "D"])
    assert "C" not in result.weights
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_all_assets_zero_variance_returns_empty_alloc():
    returns = np.zeros((100, 3))
    result = allocate(returns, ["A", "B", "C"])
    assert result.weights == {}


def test_only_one_non_zero_variance_asset_routes_full_weight():
    returns = np.zeros((100, 3))
    rng = np.random.default_rng(0)
    returns[:, 1] = rng.standard_normal(100) * 0.02
    result = allocate(returns, ["A", "B", "C"])
    assert result.weights == {"B": pytest.approx(1.0)}


# ── input validation ──────────────────────────────────────


def test_rejects_non_2d_returns():
    with pytest.raises(ValueError, match="2D"):
        allocate(np.zeros(10), ["A"])


def test_rejects_mismatched_symbol_count():
    returns = _synth_returns(n_assets=4)
    with pytest.raises(ValueError, match="columns"):
        allocate(returns, ["A", "B", "C"])


def test_rejects_too_short_history():
    returns = _synth_returns(n_periods=10)
    with pytest.raises(ValueError, match="at least"):
        allocate(returns, ["A", "B", "C", "D"])


def test_rejects_invalid_cash_buffer():
    returns = _synth_returns()
    with pytest.raises(ValueError, match="cash_buffer_pct"):
        allocate(returns, ["A", "B", "C", "D"], cash_buffer_pct=1.0)
    with pytest.raises(ValueError, match="cash_buffer_pct"):
        allocate(returns, ["A", "B", "C", "D"], cash_buffer_pct=-0.1)


def test_min_history_can_be_overridden():
    """Default is 30 — operator can lower for sandbox testing.
    Pinned so a future refactor doesn't drop the kwarg."""
    returns = _synth_returns(n_periods=15)
    result = allocate(returns, ["A", "B", "C", "D"], min_history=10)
    assert sum(result.weights.values()) == pytest.approx(1.0)


# ── helper coverage ───────────────────────────────────────


def test_correlation_to_distance_clamps_floating_drift():
    """A correlation slightly outside [-1, 1] from numerical drift
    should not produce NaN. Pin the clamp."""
    corr = np.array([[1.0000001, 0.5], [0.5, 1.0]])
    d = _correlation_to_distance(corr)
    assert not np.any(np.isnan(d))
    # Diagonal must be 0 (asset to itself).
    assert d[0, 0] == pytest.approx(0.0)


def test_correlation_to_distance_perfect_correlation_is_zero():
    corr = np.eye(3)
    d = _correlation_to_distance(corr)
    np.testing.assert_array_almost_equal(np.diag(d), np.zeros(3))


def test_inverse_variance_weights_normalise_to_one():
    cov = np.diag([1.0, 2.0, 4.0])
    w = _inverse_variance_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    # Lower variance → higher weight.
    assert w[0] > w[1] > w[2]


def test_cluster_variance_matches_known_diagonal_case():
    """For a diagonal cov, the IV-weighted cluster variance has a
    closed form — pin it to detect a regression in the helper."""
    cov = np.diag([1.0, 4.0])
    # IV weights: 1/1, 1/4 → normalized [4/5, 1/5].
    # Variance: (4/5)²·1 + (1/5)²·4 = 16/25 + 4/25 = 20/25 = 0.8
    assert _cluster_variance(cov, [0, 1]) == pytest.approx(0.8)


def test_single_linkage_order_handles_two_assets():
    d = np.array([[0.0, 0.5], [0.5, 0.0]])
    order = _single_linkage_order(d)
    assert sorted(order) == [0, 1]


def test_single_linkage_order_handles_empty():
    assert _single_linkage_order(np.zeros((0, 0))) == []


def test_single_linkage_order_handles_single():
    assert _single_linkage_order(np.zeros((1, 1))) == [0]


def test_recursive_bisection_assigns_full_weight_to_singleton():
    cov = np.eye(3)
    weights = _recursive_bisection(cov, [1])
    # Only index 1 should get weight 1.0; others 0.
    assert weights[1] == pytest.approx(1.0)
    assert weights[0] == 0.0 and weights[2] == 0.0


# ── integration: highly-correlated cluster shouldn't dominate ─


def test_correlated_cluster_is_treated_as_one_unit():
    """Build a 4-asset universe where A & B are near-clones (high
    rho) and C & D are independent. HRP should *not* give A + B a
    disproportionate combined weight just because they look like 2
    rows in the matrix — the diversification penalty should bring
    their pair allocation closer to a single-asset slice.

    Pin: under naive equal-weight, A+B gets 50% and {C,D} get 25%
    each. Under HRP with rho≈0.95 between A,B, the {A,B} cluster
    receives less than equal-weight.
    """
    rng = np.random.default_rng(7)
    n = 500
    common_ab = rng.standard_normal(n) * 0.02
    a = common_ab + rng.standard_normal(n) * 0.005
    b = common_ab + rng.standard_normal(n) * 0.005
    c = rng.standard_normal(n) * 0.02
    d = rng.standard_normal(n) * 0.02
    returns = np.column_stack([a, b, c, d])
    result = allocate(returns, ["A", "B", "C", "D"])
    ab = result.weights["A"] + result.weights["B"]
    cd = result.weights["C"] + result.weights["D"]
    # Diversification at work: AB cluster < CD cluster.
    assert ab < cd
