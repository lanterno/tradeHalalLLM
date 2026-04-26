"""Extended feature builder tests."""

from __future__ import annotations

import pytest

from halal_trader.domain.models import Kline
from halal_trader.ml.extended_features import (
    EXTENDED_FEATURE_ORDER,
    assemble_features,
    base_indicator_features,
    derived_features,
    kline_window_features,
    to_vector,
)


def _kl(close, t):
    return Kline(
        open_time=t,
        open=close - 0.1,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=10.0,
        close_time=t + 60_000,
    )


def test_base_skips_missing_keys():
    out = base_indicator_features({"rsi_14": 60, "missing": None})
    assert "rsi_14" in out
    assert "missing" not in out


def test_derived_features_compute_ratios():
    out = derived_features({"ema_9": 105, "ema_21": 100, "current_price": 110, "vwap": 108})
    assert out["ema_9_minus_21"] == 5
    assert out["ema_9_over_21"] == pytest.approx(1.05, rel=1e-3)
    assert out["price_minus_vwap"] == 2
    assert out["price_over_vwap"] == pytest.approx(110 / 108, rel=1e-6)


def test_derived_features_safe_when_missing():
    """Missing inputs shouldn't raise — they just don't add the feature."""
    out = derived_features({"ema_9": 100})  # no ema_21
    assert "ema_9_minus_21" not in out


def test_kline_window_basic():
    klines = [_kl(100 + i * 0.1, i * 60_000) for i in range(30)]
    out = kline_window_features(klines, window=20)
    # All expected keys present and finite.
    for key in (
        "ret_window_mean",
        "ret_window_std",
        "ret_window_skew",
        "drawdown_window",
        "high_low_range_pct",
        "body_to_range_ratio",
        "volume_change_pct",
        "up_candle_ratio",
    ):
        assert key in out
        assert isinstance(out[key], float)


def test_kline_window_empty_returns_empty():
    assert kline_window_features([]) == {}


def test_kline_window_single_candle_returns_empty():
    """Diff-of-one is zero samples — no return distribution."""
    assert kline_window_features([_kl(100, 0)]) == {}


def test_assemble_combines_all_builders():
    klines = [_kl(100 + i, i * 60_000) for i in range(25)]
    out = assemble_features(
        {"rsi_14": 50, "ema_9": 100, "ema_21": 99, "current_price": 102, "vwap": 101},
        klines=klines,
    )
    # Has at least one key from each builder.
    assert "rsi_14" in out  # base
    assert "ema_9_minus_21" in out  # derived
    assert "ret_window_mean" in out  # window


def test_to_vector_uses_extended_order():
    feats = {k: float(i) for i, k in enumerate(EXTENDED_FEATURE_ORDER)}
    vec = to_vector(feats)
    assert len(vec) == len(EXTENDED_FEATURE_ORDER)
    assert vec[0] == 0.0
    assert vec[-1] == float(len(EXTENDED_FEATURE_ORDER) - 1)


def test_to_vector_default_for_missing():
    vec = to_vector({}, default=-1.0)
    assert vec == [-1.0] * len(EXTENDED_FEATURE_ORDER)
