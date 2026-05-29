"""Structural regime signal — efficiency ratio + Donchian breakout (rank 5)."""

from __future__ import annotations

from halabot.cognition.structure import (
    donchian_breakout,
    efficiency_ratio,
    sma_trend_state,
    structural_label,
)


# ── efficiency ratio ──
def test_efficiency_ratio_one_for_clean_trend():
    # A perfectly monotone climb: net change == path length → ER == 1.
    closes = [100.0 + i for i in range(21)]
    assert efficiency_ratio(closes, 20) == 1.0


def test_efficiency_ratio_near_zero_for_chop():
    # Pure zigzag that ends where it started: net change 0 → ER 0.
    closes = [100.0 + (i % 2) for i in range(21)]
    er = efficiency_ratio(closes, 20)
    assert er is not None and er < 0.2


def test_efficiency_ratio_none_when_too_few_bars():
    assert efficiency_ratio([100.0, 101.0], 20) is None


def test_efficiency_ratio_none_on_flat_line():
    assert efficiency_ratio([100.0] * 21, 20) is None  # zero path length


# ── donchian breakout ──
def test_donchian_up_breakout():
    # Latest close exceeds the prior 20-bar high.
    highs = [100.0] * 20 + [110.0]
    lows = [99.0] * 21
    closes = [99.5] * 20 + [110.0]
    assert donchian_breakout(highs, lows, closes, 20) == 1


def test_donchian_down_breakout():
    highs = [101.0] * 21
    lows = [100.0] * 20 + [90.0]
    closes = [100.5] * 20 + [90.0]
    assert donchian_breakout(highs, lows, closes, 20) == -1


def test_donchian_no_breakout_inside_channel():
    highs = [110.0] * 21
    lows = [90.0] * 21
    closes = [100.0] * 21
    assert donchian_breakout(highs, lows, closes, 20) == 0


# ── composite label ──
def test_label_breakout_on_efficient_new_high():
    # Monotone climb (ER=1) whose last bar is a fresh 20-bar high → breakout.
    closes = [100.0 + i for i in range(21)]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    assert structural_label(highs, lows, closes, window=20, er_trend=0.5) == "breakout"


def test_label_chop_when_inefficient():
    closes = [100.0 + (i % 2) for i in range(21)]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    assert structural_label(highs, lows, closes, window=20, er_trend=0.5) == "chop"


def test_label_trend_efficient_but_not_fresh_breakout():
    # Efficient climb, but the final bar pulls back below the prior high so it is
    # NOT a fresh Donchian breakout → "trend", not "breakout".
    closes = [100.0 + i for i in range(20)] + [118.0]  # dips under the 119 prior high
    highs = [c + 0.5 for c in closes[:-1]] + [118.5]
    lows = [c - 0.5 for c in closes]
    label = structural_label(highs, lows, closes, window=20, er_trend=0.4)
    assert label == "trend"


def test_label_unknown_when_too_few_bars():
    assert structural_label([100.0], [99.0], [99.5], window=20) == "unknown"


# ── SMA trend state (market regime) ──
def test_sma_trend_above_on_uptrend():
    assert sma_trend_state([100.0 + i for i in range(50)], 50) == "above"


def test_sma_trend_below_on_downtrend():
    assert sma_trend_state([100.0 - i for i in range(50)], 50) == "below"


def test_sma_trend_unknown_when_too_few_bars():
    assert sma_trend_state([100.0, 101.0], 50) == "unknown"
