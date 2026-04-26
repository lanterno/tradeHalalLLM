"""Historical VaR / ES tests."""

import numpy as np
import pytest

from halal_trader.core.var import (
    historical_var,
    klines_to_returns,
    portfolio_var_es,
)


def _normal_returns(n: int = 500, scale: float = 0.01, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.normal(0, scale, n).tolist()


def test_historical_var_empty_returns_zero():
    assert historical_var([]) == (0.0, 0.0)


def test_historical_var_too_few_samples_returns_zero():
    # Default 95% confidence needs at least 20/(1-0.95) = 400 samples.
    assert historical_var([0.01, -0.02, 0.005]) == (0.0, 0.0)


def test_historical_var_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        historical_var([0.01] * 500, confidence=0.0)
    with pytest.raises(ValueError):
        historical_var([0.01] * 500, confidence=1.0)


def test_historical_var_positive_for_loss_distribution():
    """A normal-ish loss distribution must produce a positive VaR."""
    var, es = historical_var(_normal_returns(), confidence=0.95)
    assert var > 0
    # ES is always at least as bad as VaR by definition (mean of the tail).
    assert es >= var


def test_historical_var_zero_when_no_losses():
    """All-positive returns → no left-tail loss → VaR = 0."""
    var, es = historical_var([0.01] * 500, confidence=0.95)
    assert var == 0.0
    assert es == 0.0


def test_portfolio_var_aggregates_weighted_returns():
    rets_a = _normal_returns(seed=1)
    rets_b = _normal_returns(seed=2)
    result = portfolio_var_es(
        weights={"A": 0.5, "B": 0.5},
        returns_by_symbol={"A": rets_a, "B": rets_b},
        confidence=0.95,
    )
    assert result.var > 0
    assert result.sample_size == 500
    assert result.confidence == 0.95


def test_portfolio_var_skips_unknown_symbols():
    rets = _normal_returns(seed=3)
    result = portfolio_var_es(
        weights={"A": 0.5, "B": 0.5, "C": 0.2},
        returns_by_symbol={"A": rets},  # only A has data
    )
    assert result.sample_size == 500


def test_portfolio_var_empty_weights_returns_zero():
    assert portfolio_var_es({}, {"A": _normal_returns()}).sample_size == 0


def test_klines_to_returns_basic():
    assert klines_to_returns([100, 110, 99]) == pytest.approx([0.10, -0.10])
    assert klines_to_returns([]) == []
    assert klines_to_returns([100]) == []


def test_var_es_relationship_holds_under_left_skew():
    """In a left-skewed distribution ES is meaningfully greater than VaR."""
    # Mostly small gains, occasional big losses — exactly the crypto pattern.
    rng = np.random.default_rng(11)
    base = rng.normal(0, 0.005, 1000)
    base[::50] -= 0.05  # inject big losses every 50 samples
    var, es = historical_var(base.tolist(), confidence=0.95)
    assert es > var * 1.2  # tail mean materially worse than the VaR cutoff
