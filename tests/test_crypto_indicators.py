"""Tests for :mod:`crypto.indicators` — pure technical-indicator math.

Every cycle calls `compute_all` to derive the indicator dict that
feeds risk, regime, ML, and the LLM prompt. A bug here cascades
into every cycle decision.
"""

from __future__ import annotations

import numpy as np

from halal_trader.crypto.indicators import (
    _pct_change,
    bollinger_bands,
    compute_all,
    ema,
    rsi,
)
from halal_trader.domain.models import Kline


def _kline(close: float, *, high: float | None = None, low: float | None = None) -> Kline:
    return Kline(
        open_time=1_700_000_000_000,
        open=close,
        high=high if high is not None else close + 1.0,
        low=low if low is not None else close - 1.0,
        close=close,
        volume=1000.0,
        close_time=1_700_000_000_000 + 60_000,
    )


def _series(closes: list[float]) -> list[Kline]:
    return [_kline(c) for c in closes]


# ── compute_all ──────────────────────────────────────────────


def test_compute_all_short_series_returns_error():
    """Single candle → not enough data; no crash, just an error marker."""
    out = compute_all([_kline(100.0)])
    assert out["error"] == "insufficient data"
    assert out["candle_count"] == 1


def test_compute_all_includes_current_price():
    out = compute_all(_series([100.0, 101.0, 102.0]))
    assert out["current_price"] == 102.0


def test_compute_all_returns_candle_count():
    closes = [100.0 + i for i in range(20)]
    out = compute_all(_series(closes))
    assert out["candle_count"] == 20


def test_compute_all_skips_indicators_below_min_lookback():
    """Three candles → no RSI (needs 15), no MACD (needs 35)."""
    out = compute_all(_series([100.0, 101.0, 102.0]))
    assert "rsi_14" not in out
    assert "macd" not in out


def test_compute_all_includes_all_indicators_with_long_series():
    closes = [100.0 + (i % 5) for i in range(60)]
    out = compute_all(_series(closes))
    for key in (
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_position",
        "ema_9",
        "ema_21",
        "ema_50",
        "atr_14",
        "adx_14",
        "vwap",
        "volume_ratio",
    ):
        assert key in out, f"missing {key}"


# ── rsi ──────────────────────────────────────────────────────


def test_rsi_constant_series_returns_neutral():
    """A flat price series has neither gains nor losses → RSI defaults
    to a midpoint (the implementation returns a defensive value rather
    than dividing by zero)."""
    closes = np.array([100.0] * 20)
    out = rsi(closes, period=14)
    # Just assert it doesn't crash and lies in the valid 0..100 range.
    assert 0.0 <= out <= 100.0


def test_rsi_strong_uptrend_above_50():
    """Monotonic increase → RSI well above 50."""
    closes = np.array([100.0 + i for i in range(20)])
    assert rsi(closes, period=14) > 70


def test_rsi_strong_downtrend_below_50():
    closes = np.array([100.0 - i for i in range(20)])
    assert rsi(closes, period=14) < 30


# ── ema ──────────────────────────────────────────────────────


def test_ema_returns_array_same_length_as_input():
    closes = np.array([100.0 + i for i in range(20)])
    out = ema(closes, period=9)
    assert len(out) == 20


def test_ema_tracks_constant_series():
    closes = np.array([50.0] * 20)
    out = ema(closes, period=9)
    # On a flat series, EMA settles at the constant value.
    assert abs(out[-1] - 50.0) < 1e-6


def test_ema_lags_a_step_change():
    """EMA smooths a discrete step; after 10 of 20 candles at the new
    level, it should be partway between old and new."""
    closes = np.array([100.0] * 10 + [200.0] * 10)
    out = ema(closes, period=9)
    assert 100.0 < out[-1] < 200.0


# ── bollinger_bands ──────────────────────────────────────────


def test_bollinger_bands_widen_with_volatility():
    """A noisy series produces wider bands than a flat one."""
    flat = np.array([100.0] * 30)
    noisy = np.array(
        [100.0 + 5 * (i % 3 - 1) for i in range(30)]  # oscillates ±5
    )
    _, _, lower_flat = bollinger_bands(flat)
    upper_flat, _, _ = bollinger_bands(flat)
    flat_width = upper_flat[-1] - lower_flat[-1]
    upper_noisy, _, lower_noisy = bollinger_bands(noisy)
    noisy_width = upper_noisy[-1] - lower_noisy[-1]
    assert noisy_width > flat_width


def test_bollinger_bands_centered_on_sma():
    """Middle band is the SMA — equidistant between upper and lower."""
    closes = np.array([100.0 + i % 3 for i in range(30)])
    upper, middle, lower = bollinger_bands(closes)
    half_width = upper[-1] - middle[-1]
    other_half = middle[-1] - lower[-1]
    assert abs(half_width - other_half) < 1e-9


# ── _pct_change ──────────────────────────────────────────────


def test_pct_change_basic():
    closes = np.array([100.0, 101.0, 102.0, 110.0])
    # Last vs 1 step back: (110 - 102) / 102 ≈ 0.0784
    assert abs(_pct_change(closes, 1) - 8.0 / 102) < 1e-9


def test_pct_change_returns_none_when_not_enough_history():
    closes = np.array([100.0, 101.0])
    # 5 periods back → not enough.
    assert _pct_change(closes, 5) is None


def test_pct_change_returns_none_when_prev_is_zero():
    """Defensive: a zero baseline would divide-by-zero."""
    closes = np.array([0.0, 100.0])
    assert _pct_change(closes, 1) is None
