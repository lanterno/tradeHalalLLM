"""Tests for :class:`RegimeDetector`'s rule-based path.

The ML branch is gated on a trained model the test env doesn't have,
so this file pins the rule fallback — which is what every cycle hits
in practice (the ML model is only retrained on real-trade outcomes).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from halal_trader.crypto.regime import MarketRegime, RegimeDetector


@pytest.fixture
def detector(tmp_path: Path) -> RegimeDetector:
    """Detector pointed at an empty models dir so no ML model loads."""
    return RegimeDetector(models_dir=tmp_path)


# ── High-volatility branch ───────────────────────────────────


def test_high_volatility_when_wide_bb_and_high_volume(detector):
    """BB width > 0.06 + volume_ratio > 1.5 → HIGH_VOLATILITY."""
    indicators = {
        "bb_upper": 110.0,
        "bb_lower": 90.0,
        "bb_middle": 100.0,
        "volume_ratio": 2.0,
    }
    regime, conf, _ = detector.detect(indicators)
    assert regime == MarketRegime.HIGH_VOLATILITY
    assert conf == 0.8


def test_no_high_vol_with_normal_volume(detector):
    """Wide BB but normal volume → not HIGH_VOLATILITY (could be range)."""
    indicators = {
        "bb_upper": 110.0,
        "bb_lower": 90.0,
        "bb_middle": 100.0,
        "volume_ratio": 1.0,  # not enough
    }
    regime, _, _ = detector.detect(indicators)
    assert regime != MarketRegime.HIGH_VOLATILITY


# ── Trending branches ────────────────────────────────────────


def test_strong_uptrend_via_high_adx(detector):
    """ADX > 25 + price above EMA50 → TRENDING_UP."""
    indicators = {
        "adx_14": 30,
        "current_price": 102.0,
        "ema_50": 100.0,
        "ema_9": 101.0,
    }
    regime, conf, _ = detector.detect(indicators)
    assert regime == MarketRegime.TRENDING_UP
    assert conf == pytest.approx(30 / 40)


def test_strong_downtrend_via_high_adx(detector):
    indicators = {
        "adx_14": 35,
        "current_price": 95.0,
        "ema_50": 100.0,
        "ema_9": 96.0,
    }
    regime, conf, _ = detector.detect(indicators)
    assert regime == MarketRegime.TRENDING_DOWN
    assert conf == pytest.approx(35 / 40)


def test_uptrend_via_ema_spread_when_no_adx(detector):
    """No ADX but EMA9 vs EMA50 spread > 0.5% → trending; price > EMA50 → up."""
    indicators = {
        "ema_9": 101.0,
        "ema_50": 100.0,  # spread 1% > 0.5%
        "current_price": 101.5,
    }
    regime, _, _ = detector.detect(indicators)
    assert regime == MarketRegime.TRENDING_UP


def test_adx_confidence_capped_at_0_95(detector):
    """Even an extreme ADX can't push confidence past 0.95 — keeps the
    LLM from treating any rule output as a hard certainty."""
    indicators = {
        "adx_14": 100,  # would give 100/40 = 2.5 if uncapped
        "current_price": 102.0,
        "ema_50": 100.0,
    }
    _, conf, _ = detector.detect(indicators)
    assert conf == 0.95


# ── Ranging fallback ─────────────────────────────────────────


def test_ranging_when_no_trend_signals(detector):
    """No trend, no high-vol → RANGING."""
    indicators = {
        "rsi_14": 50,
        "volume_ratio": 1.0,
    }
    regime, _, _ = detector.detect(indicators)
    assert regime == MarketRegime.RANGING


def test_ranging_higher_confidence_when_adx_low(detector):
    """ADX < 20 reinforces the no-trend conclusion."""
    indicators = {"rsi_14": 50, "volume_ratio": 1.0, "adx_14": 15}
    _, conf, _ = detector.detect(indicators)
    assert conf == 0.8  # higher than the 0.6 default


# ── Strategy instructions ────────────────────────────────────


def test_each_regime_returns_non_empty_strategy_instructions(detector):
    """Every regime branch must include actionable text for the LLM."""
    cases = [
        # high vol
        {
            "bb_upper": 110.0,
            "bb_lower": 90.0,
            "bb_middle": 100.0,
            "volume_ratio": 2.0,
        },
        # uptrend
        {"adx_14": 30, "current_price": 102.0, "ema_50": 100.0},
        # downtrend
        {"adx_14": 30, "current_price": 95.0, "ema_50": 100.0},
        # ranging
        {"rsi_14": 50, "volume_ratio": 1.0},
    ]
    for ind in cases:
        _, _, instructions = detector.detect(ind)
        assert isinstance(instructions, str)
        assert len(instructions) > 10  # not just whitespace


# ── _compute_bb_width helper ────────────────────────────────


def test_bb_width_returns_none_when_missing_bands(detector):
    assert detector._compute_bb_width({"bb_upper": 110, "bb_lower": 90}) is None


def test_bb_width_returns_none_when_middle_zero(detector):
    """Defensive: zero divisor would crash if not guarded."""
    assert detector._compute_bb_width({"bb_upper": 1, "bb_lower": -1, "bb_middle": 0}) is None


def test_bb_width_correct_value(detector):
    """(upper - lower) / middle."""
    out = detector._compute_bb_width(
        {"bb_upper": 110.0, "bb_lower": 90.0, "bb_middle": 100.0}
    )
    assert out == pytest.approx(0.20)
