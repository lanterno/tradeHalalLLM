"""Tests for `core/ab_compare.py` (strategy A/B comparator).

Covers the per-cohort metric formulas (Sharpe, drawdown, profit
factor, total compound return), the Welch's t-test path (precise
p-value via scipy when available + the normal-approximation fallback
+ the small-df-without-scipy → None contract), and the input
robustness edges (empty cohort, NaN entries, zero-variance, single
observation).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from halal_trader.core.ab_compare import (
    ABComparison,
    CohortStats,
    _max_drawdown,
    _normal_cdf,
    _profit_factor,
    _safe_array,
    _two_tailed_p,
    _welch_t,
    cohort_stats,
    compare,
)

# ── _safe_array ───────────────────────────────────────────


def test_safe_array_drops_nan_and_inf():
    """Closed-trade rows occasionally have a missing return_pct;
    one bad row mustn't poison the whole stat. Pin the drop."""
    arr = _safe_array([0.01, np.nan, 0.02, np.inf, -np.inf, -0.005])
    np.testing.assert_array_equal(arr, [0.01, 0.02, -0.005])


def test_safe_array_handles_empty():
    assert _safe_array([]).size == 0


def test_safe_array_accepts_ndarray_input():
    arr = _safe_array(np.array([0.01, 0.02]))
    np.testing.assert_array_equal(arr, [0.01, 0.02])


# ── cohort_stats helpers ──────────────────────────────────


def test_max_drawdown_zero_for_all_winners():
    """No equity-curve trough → 0.0 drawdown by convention."""
    assert _max_drawdown(np.array([0.01, 0.02, 0.005])) == 0.0


def test_max_drawdown_zero_for_empty():
    assert _max_drawdown(np.array([])) == 0.0


def test_max_drawdown_negative_for_loss_streak():
    """A loss right after a peak should drive drawdown negative.
    1.0 → 1.05 → 1.05*0.9 = 0.945 → drawdown = (0.945 − 1.05)/1.05."""
    dd = _max_drawdown(np.array([0.05, -0.10]))
    assert dd == pytest.approx((0.945 - 1.05) / 1.05, abs=1e-9)


def test_profit_factor_inf_when_no_losses():
    assert _profit_factor(np.array([0.01, 0.02])) == math.inf


def test_profit_factor_zero_when_no_wins_no_losses():
    assert _profit_factor(np.array([0.0, 0.0])) == 0.0


def test_profit_factor_typical_mix():
    """0.10 + 0.05 wins / |−0.05| loss = 3.0."""
    pf = _profit_factor(np.array([0.10, 0.05, -0.05]))
    assert pf == pytest.approx(3.0)


# ── cohort_stats integration ─────────────────────────────


def test_cohort_stats_empty_input_returns_zero_block():
    """Caller shouldn't have to guard against empty cohorts — the
    stats block must be safe to render."""
    stats = cohort_stats([])
    assert stats == CohortStats(
        n_trades=0,
        win_rate=0.0,
        mean_return=0.0,
        median_return=0.0,
        std_return=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
        total_return=0.0,
        profit_factor=0.0,
    )


def test_cohort_stats_single_trade_has_zero_std():
    """N=1 → no sample std defined; pin the safe-zero fallback."""
    stats = cohort_stats([0.05])
    assert stats.n_trades == 1
    assert stats.std_return == 0.0
    assert stats.sharpe == 0.0


def test_cohort_stats_win_rate_counts_strictly_positive():
    """Zero-return trades are not wins. Pin so a refactor doesn't
    flip the inequality."""
    stats = cohort_stats([0.01, 0.0, -0.005, 0.02])
    # 2 wins out of 4 → 0.5
    assert stats.win_rate == 0.5


def test_cohort_stats_mean_and_total_return_match_formula():
    rs = [0.01, 0.02, -0.005, 0.015]
    stats = cohort_stats(rs)
    assert stats.mean_return == pytest.approx(np.mean(rs))
    expected_total = math.prod(1.0 + r for r in rs) - 1.0
    assert stats.total_return == pytest.approx(expected_total)


def test_cohort_stats_sharpe_is_mean_over_std():
    rs = [0.01, 0.02, -0.005, 0.015]
    stats = cohort_stats(rs)
    expected = float(np.mean(rs)) / float(np.std(rs, ddof=1))
    assert stats.sharpe == pytest.approx(expected)


def test_cohort_stats_handles_all_negative_returns():
    """Pure-loss cohort: win_rate 0, total_return < 0, drawdown is
    the full equity erosion. Ensures we don't divide-by-zero."""
    stats = cohort_stats([-0.01, -0.02, -0.005])
    assert stats.win_rate == 0.0
    assert stats.total_return < 0
    assert stats.max_drawdown < 0
    assert stats.profit_factor == 0.0


# ── Welch's t-test ────────────────────────────────────────


def test_welch_t_zero_for_empty_cohort():
    t, df = _welch_t(np.array([]), np.array([0.01, 0.02, 0.03]))
    assert (t, df) == (0.0, 0.0)


def test_welch_t_zero_for_single_observation_each():
    t, df = _welch_t(np.array([0.01]), np.array([0.02]))
    assert (t, df) == (0.0, 0.0)


def test_welch_t_zero_when_both_zero_variance():
    t, df = _welch_t(np.array([0.01, 0.01]), np.array([0.02, 0.02]))
    assert (t, df) == (0.0, 0.0)


def test_welch_t_positive_when_a_mean_exceeds_b():
    rng = np.random.default_rng(0)
    a = rng.normal(0.02, 0.01, 100)
    b = rng.normal(0.005, 0.01, 100)
    t, df = _welch_t(a, b)
    assert t > 0
    # Welch-Satterthwaite df ≈ n_a + n_b - 2 for similar-variance
    # samples; allow a wide tolerance.
    assert 100 < df <= 198


def test_welch_t_negative_when_b_mean_exceeds_a():
    rng = np.random.default_rng(1)
    a = rng.normal(0.005, 0.01, 100)
    b = rng.normal(0.02, 0.01, 100)
    t, _ = _welch_t(a, b)
    assert t < 0


# ── p-value helper ────────────────────────────────────────


def test_two_tailed_p_returns_none_for_zero_df():
    assert _two_tailed_p(2.0, 0.0) is None


def test_two_tailed_p_uses_scipy_when_available():
    """Pin: when scipy is importable, the precise t-distribution sf
    is used — different (smaller) than the normal approximation for
    small df."""
    p = _two_tailed_p(2.0, df=5)
    assert p is not None
    # scipy.stats.t.sf(2, 5)*2 ≈ 0.10199; normal-approx ≈ 0.04550.
    # Either way, the precise p must be > the normal-approx p.
    assert 0.08 < p < 0.12


def test_two_tailed_p_normal_approximation_branch_when_scipy_missing(monkeypatch):
    """Force the scipy import to fail; pin the large-df fallback to
    the standard normal SF."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("simulated missing scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = _two_tailed_p(1.96, df=100)
    # Standard normal two-tailed p for |z|=1.96 ≈ 0.05.
    assert p is not None
    assert 0.04 < p < 0.06


def test_two_tailed_p_returns_none_for_small_df_without_scipy(monkeypatch):
    """Tiny-df + no scipy: don't lie with a normal approximation;
    return None so the caller renders 'unknown'."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("simulated missing scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _two_tailed_p(1.96, df=10) is None


def test_normal_cdf_matches_known_values():
    """Φ(0)=0.5, Φ(1.96)≈0.975."""
    assert _normal_cdf(0.0) == pytest.approx(0.5)
    assert _normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)


# ── compare() integration ─────────────────────────────────


def test_compare_detects_significant_winner_with_enough_data():
    """A clearly-better strategy A vs strategy B over 200 trades
    each should land below α=0.05. Uses fixed seeds so the test
    is deterministic."""
    rng_a = np.random.default_rng(101)
    rng_b = np.random.default_rng(202)
    a = rng_a.normal(0.015, 0.01, 200)
    b = rng_b.normal(0.005, 0.01, 200)
    result = compare(a, b)
    assert result.mean_diff > 0
    assert result.t_statistic > 0
    assert result.p_value is not None
    assert result.p_value < 0.05
    assert result.significant_at_05 is True


def test_compare_does_not_flag_significance_for_indistinguishable_cohorts():
    """Same distribution → t ≈ 0 and p > 0.05; the operator must
    not see a 'green' signal on noise."""
    # Seed picked so the iid sample lands well outside the α=0.05
    # tail — under iid sampling ~5% of seeds hit a false positive
    # by chance, which is the false-positive rate the test is
    # asserting we don't report on a normal middle-of-the-road run.
    rng = np.random.default_rng(1)
    a = rng.normal(0.01, 0.01, 100)
    b = rng.normal(0.01, 0.01, 100)
    result = compare(a, b)
    assert result.p_value is not None
    assert result.p_value > 0.05
    assert result.significant_at_05 is False


def test_compare_returns_zero_block_for_empty_cohorts():
    result = compare([], [])
    assert isinstance(result, ABComparison)
    assert result.a.n_trades == 0
    assert result.b.n_trades == 0
    assert result.mean_diff == 0.0
    assert result.t_statistic == 0.0
    assert result.p_value is None
    assert result.significant_at_05 is False


def test_compare_propagates_per_cohort_stats():
    """Spot-check that the per-cohort blocks aren't garbage / mixed
    up between A and B."""
    a = [0.10, 0.10, 0.10, 0.10]
    b = [-0.05, -0.05, -0.05, -0.05]
    result = compare(a, b)
    assert result.a.win_rate == 1.0
    assert result.b.win_rate == 0.0
    assert result.a.mean_return == pytest.approx(0.10)
    assert result.b.mean_return == pytest.approx(-0.05)
    assert result.mean_diff == pytest.approx(0.15)


def test_compare_with_nan_entries_still_produces_clean_stats():
    """NaN-injection drill: the cohort sizes should match the
    NaN-stripped lengths, not the input lengths."""
    a = [0.01, np.nan, 0.02, 0.03]
    b = [-0.01, -0.02, np.nan]
    result = compare(a, b)
    assert result.a.n_trades == 3
    assert result.b.n_trades == 2
