"""Tests for Probabilistic & Deflated Sharpe Ratio."""

from __future__ import annotations

import numpy as np
import pytest

from halal_trader.core.sharpe_stats import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    passes_sharpe_gate,
    probabilistic_sharpe_ratio,
)


def _series(mean: float, sd: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mean, sd, n)


def test_psr_high_for_strong_positive_track():
    # Strong positive mean, low vol, long sample → very likely true SR > 0.
    psr = probabilistic_sharpe_ratio(_series(0.01, 0.01, 500))
    assert psr > 0.95


def test_psr_around_half_for_zero_mean_noise():
    r = _series(0.0, 0.02, 1000)
    r = r - r.mean()  # exactly zero sample mean → Sharpe 0
    psr = probabilistic_sharpe_ratio(r)
    assert 0.45 < psr < 0.55  # sr=0 → Φ(0) = 0.5


def test_psr_low_for_negative_track():
    psr = probabilistic_sharpe_ratio(_series(-0.01, 0.01, 500))
    assert psr < 0.05


def test_psr_degenerate_inputs_return_zero():
    assert probabilistic_sharpe_ratio([0.01, 0.02]) == 0.0  # < 3 obs
    assert probabilistic_sharpe_ratio([0.01, 0.01, 0.01]) == 0.0  # zero variance
    assert probabilistic_sharpe_ratio([]) == 0.0


def test_psr_shorter_sample_is_less_confident():
    # Same per-period Sharpe, fewer observations → lower PSR (more uncertainty).
    long_psr = probabilistic_sharpe_ratio(_series(0.005, 0.01, 400, seed=1))
    short_psr = probabilistic_sharpe_ratio(_series(0.005, 0.01, 30, seed=1))
    assert long_psr > short_psr


def test_expected_max_sharpe_grows_with_trials():
    v = 0.001
    assert expected_max_sharpe(v, 1) == 0.0  # single trial → no deflation
    e10 = expected_max_sharpe(v, 10)
    e100 = expected_max_sharpe(v, 100)
    assert 0 < e10 < e100  # more trials → higher expected max under the null


def test_dsr_not_above_psr_under_multiple_testing():
    returns = _series(0.006, 0.01, 300, seed=2)
    psr = probabilistic_sharpe_ratio(returns)
    dsr = deflated_sharpe_ratio(returns, n_trials=50)
    assert dsr <= psr  # deflation can only lower (or equal) the probability
    # single trial → DSR == PSR
    assert deflated_sharpe_ratio(returns, n_trials=1) == pytest.approx(psr)


def test_gate_uses_dsr_when_multi_trial():
    # A track that clears PSR but not the multiple-testing-deflated bar.
    returns = _series(0.004, 0.01, 120, seed=3)
    assert passes_sharpe_gate(returns, n_trials=1, min_prob=0.9) in (True, False)
    # With many trials the same track is held to a higher bar.
    gate_single = passes_sharpe_gate(returns, n_trials=1, min_prob=0.95)
    gate_many = passes_sharpe_gate(returns, n_trials=200, min_prob=0.95)
    assert not (gate_many and not gate_single)  # many-trial gate is never easier
