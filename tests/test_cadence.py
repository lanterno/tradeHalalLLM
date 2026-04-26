"""Adaptive cycle-interval selector tests."""

import pytest

from halal_trader.crypto.cadence import CadenceDecision, select_interval


def test_high_vol_halves_interval():
    d = select_interval(
        indicators_cache={
            "BTCUSDT": {"atr_pct": 0.04},  # 2× baseline
            "ETHUSDT": {"atr_pct": 0.05},
        },
        base_interval=60,
        atr_baseline=0.02,
    )
    assert d.regime == "fast"
    assert d.interval_seconds == 30


def test_low_vol_doubles_interval():
    d = select_interval(
        indicators_cache={
            "BTCUSDT": {"atr_pct": 0.012},
            "ETHUSDT": {"atr_pct": 0.010},
        },
        base_interval=60,
        atr_baseline=0.02,
    )
    assert d.regime == "slow"
    assert d.interval_seconds == 120


def test_normal_vol_keeps_base_interval():
    d = select_interval(
        indicators_cache={"BTCUSDT": {"atr_pct": 0.020}},
        base_interval=60,
        atr_baseline=0.02,
    )
    assert d.regime == "normal"
    assert d.interval_seconds == 60


def test_clamp_to_min_interval():
    """Even at extreme vol the interval can't drop below the floor."""
    d = select_interval(
        indicators_cache={"BTCUSDT": {"atr_pct": 0.10}},
        base_interval=20,
        atr_baseline=0.02,
        min_interval=15,
    )
    # 20 // 2 = 10, but min is 15.
    assert d.interval_seconds == 15


def test_clamp_to_max_interval():
    d = select_interval(
        indicators_cache={"BTCUSDT": {"atr_pct": 0.005}},
        base_interval=200,
        atr_baseline=0.02,
        max_interval=300,
    )
    # 200 * 2 = 400, capped at 300.
    assert d.interval_seconds == 300


def test_no_indicator_data_returns_base():
    d = select_interval(
        indicators_cache={},
        base_interval=60,
        atr_baseline=0.02,
    )
    assert d.regime == "normal"
    assert d.interval_seconds == 60


def test_skips_error_indicators():
    """A pair flagged as error shouldn't poison the median."""
    d = select_interval(
        indicators_cache={
            "BAD": {"error": "no data"},
            "BTCUSDT": {"atr_pct": 0.04},
        },
        base_interval=60,
        atr_baseline=0.02,
    )
    assert d.regime == "fast"  # only BTC counted


def test_uses_median_not_mean():
    """One outlier shouldn't push the bot into fast mode."""
    d = select_interval(
        indicators_cache={
            "A": {"atr_pct": 0.020},
            "B": {"atr_pct": 0.020},
            "C": {"atr_pct": 0.020},
            "D": {"atr_pct": 0.020},
            "OUTLIER": {"atr_pct": 0.30},  # mean would be 0.07; median stays 0.02
        },
        base_interval=60,
        atr_baseline=0.02,
    )
    assert d.regime == "normal"


def test_negative_base_interval_raises():
    with pytest.raises(ValueError):
        select_interval(indicators_cache={}, base_interval=0, atr_baseline=0.02)


def test_returns_cadence_decision_dataclass():
    d = select_interval(indicators_cache={}, base_interval=60, atr_baseline=0.02)
    assert isinstance(d, CadenceDecision)
