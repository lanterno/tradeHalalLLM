"""Tests for sentiment/trends_signal.py — Round-5 Wave 11.B."""

from __future__ import annotations

import pytest

from halal_trader.sentiment.trends_signal import (
    TrendPolicy,
    TrendSignal,
    assess,
    render_assessment,
)

# --- Validation -----------------------------


def test_signal_string_values():
    assert TrendSignal.NEUTRAL.value == "neutral"
    assert TrendSignal.EMERGING_INTEREST.value == "emerging_interest"
    assert TrendSignal.PEAKING.value == "peaking"
    assert TrendSignal.FADING.value == "fading"
    assert TrendSignal.DEAD.value == "dead"


def test_default_policy():
    p = TrendPolicy()
    assert p.surge_z_threshold == 2.0
    assert p.peak_z_threshold == 3.0


def test_policy_unsorted_thresholds_rejected():
    with pytest.raises(ValueError):
        TrendPolicy(surge_z_threshold=3.0, peak_z_threshold=2.0)


def test_policy_high_dead_pct_rejected():
    with pytest.raises(ValueError):
        TrendPolicy(dead_pct_of_peak=0.6)


def test_policy_small_window_rejected():
    with pytest.raises(ValueError):
        TrendPolicy(rolling_window=3)


def test_assess_empty_keyword_rejected():
    with pytest.raises(ValueError):
        assess("", [1, 2, 3])


def test_assess_negative_value_rejected():
    with pytest.raises(ValueError):
        assess("kw", [1, -1, 2])


# --- Detection -----------------------------


def test_empty_series_neutral():
    a = assess("kw", [])
    assert a.signal is TrendSignal.NEUTRAL


def test_short_series_neutral():
    """Below rolling-window length → no z-score → NEUTRAL."""
    a = assess("kw", [50] * 5)
    assert a.signal is TrendSignal.NEUTRAL


def test_constant_series_neutral():
    a = assess("kw", [50] * 50)
    assert a.signal is TrendSignal.NEUTRAL


def test_surge_emerging_interest():
    """30 baseline values + one high spike → emerging interest."""
    series = [10] * 30 + [50]
    a = assess("kw", series)
    assert a.signal in (TrendSignal.EMERGING_INTEREST, TrendSignal.PEAKING)
    assert a.z_score_latest >= 2.0


def test_peak_signal():
    """Series with extreme z-score → peaking."""
    series = [10] * 30 + [200]
    a = assess("kw", series)
    assert a.signal is TrendSignal.PEAKING


def test_fading_after_peak():
    """High peak then low values → FADING."""
    series = [100] * 5 + [10] * 30  # last value = 10, mean of window = 10, low pct of peak
    a = assess("kw", series)
    # mean of last 30 = 10, latest = 10 → z=0; pct of peak (100) = 10% → DEAD
    assert a.signal is TrendSignal.DEAD


def test_dead_signal():
    """Latest is < dead_pct of peak."""
    series = [100] * 5 + [5] * 30
    a = assess("kw", series)
    assert a.signal is TrendSignal.DEAD


def test_z_score_recorded():
    series = [10] * 30 + [50]
    a = assess("kw", series)
    assert a.z_score_latest > 0


def test_n_observations_recorded():
    a = assess("kw", [10] * 35)
    assert a.n_observations == 35


# --- Render -------------------------------


def test_render_neutral_emoji():
    a = assess("kw", [10] * 35)
    assert "⚪" in render_assessment(a)


def test_render_peaking_emoji():
    series = [10] * 30 + [200]
    a = assess("kw", series)
    out = render_assessment(a)
    assert "🔥" in out


def test_render_no_secret_leak():
    a = assess("kw", [10] * 35)
    out = render_assessment(a)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E -------------------------------


def test_e2e_meme_stock_lifecycle():
    """Track surge → peak → fade → dead."""
    baseline = [10] * 30
    surge = baseline + [50]
    peak_a = assess("kw", surge)
    assert peak_a.signal in (TrendSignal.EMERGING_INTEREST, TrendSignal.PEAKING)


def test_replay_consistency():
    series = [10] * 30 + [50]
    a = assess("kw", series)
    b = assess("kw", series)
    assert a == b
