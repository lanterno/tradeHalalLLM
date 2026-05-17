"""Tests for `ml/equity_anomaly.py` (equity-curve anomaly detector).

Covers the two detector entry points (per-trade return z-score +
drawdown z-score), the severity / direction labelling, the cold-start
"don't false-alert" semantics, and the input-validation edges.
"""

from __future__ import annotations

import numpy as np
import pytest

from halal_trader.ml.equity_anomaly import (
    EquityAnomalyReport,
    _trim_to_finite,
    detect_drawdown_anomaly,
    detect_return_anomaly,
    equity_curve_from_returns,
)

# ── _trim_to_finite ───────────────────────────────────────


def test_trim_drops_nan_and_inf():
    arr = _trim_to_finite([0.01, np.nan, 0.02, np.inf, -np.inf, -0.005])
    np.testing.assert_array_equal(arr, [0.01, 0.02, -0.005])


def test_trim_handles_empty():
    assert _trim_to_finite([]).size == 0


# ── detect_return_anomaly cold-start ─────────────────────


def test_return_detector_says_normal_below_min_window():
    """Cold-start: with too few prior trades, we'd rather report
    'normal' than fire a false alert from a noisy 5-sample baseline.
    Pin so a refactor can't drop the cold-start guard."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.01, 0.01, 10)  # below default min_window=30
    report = detect_return_anomaly(returns)
    assert report.severity == "normal"
    assert report.direction == "normal"
    assert report.z_score == 0.0


def test_return_detector_min_window_can_be_overridden():
    rng = np.random.default_rng(0)
    returns = list(rng.normal(0.01, 0.01, 10)) + [0.5]  # last is huge
    report = detect_return_anomaly(returns, min_window=5)
    # Now it should have enough baseline; the +0.5 is way out.
    assert report.severity in ("warn", "alert")


# ── detect_return_anomaly happy path ─────────────────────


def test_return_detector_flags_drawdown_alert():
    """A clear loss tail should trip the alert path."""
    rng = np.random.default_rng(42)
    returns = list(rng.normal(0.01, 0.01, 50)) + [-0.20]
    report = detect_return_anomaly(returns)
    assert report.severity == "alert"
    assert report.direction == "drawdown"
    assert report.z_score < -3
    assert "halt" in report.recommendation.lower() or "edge" in report.recommendation.lower()


def test_return_detector_flags_hot_streak_alert():
    rng = np.random.default_rng(42)
    returns = list(rng.normal(0.01, 0.01, 50)) + [+0.20]
    report = detect_return_anomaly(returns)
    assert report.severity == "alert"
    assert report.direction == "hot"
    assert report.z_score > 3
    assert "hot" in report.recommendation.lower() or "raising" in report.recommendation.lower()


def test_return_detector_warn_band_between_2_and_3():
    """Z in the |2..3| band should land in 'warn', not 'alert'."""
    rng = np.random.default_rng(0)
    base = list(rng.normal(0.0, 0.01, 100))
    # Pick a tail value at ~2.5σ.
    base.append(0.025)
    report = detect_return_anomaly(base)
    assert 2 <= abs(report.z_score) < 3
    assert report.severity == "warn"


def test_return_detector_normal_when_last_within_one_sigma():
    rng = np.random.default_rng(7)
    base = list(rng.normal(0.0, 0.01, 100))
    base.append(0.005)  # well within 1σ
    report = detect_return_anomaly(base)
    assert report.severity == "normal"
    assert report.direction == "normal"


def test_return_detector_zero_std_baseline_yields_zero_z():
    """Degenerate baseline (every prior trade identical) — mustn't
    blow up to ±∞. Pin the safe-zero fallback."""
    returns = [0.01] * 50 + [0.5]
    report = detect_return_anomaly(returns)
    assert report.z_score == 0.0
    assert report.severity == "normal"


def test_return_detector_recommends_for_each_severity():
    """Operator-readable strings differ across severities so a
    notifier can route on text. Pin the contracts."""
    # Normal
    rng = np.random.default_rng(0)
    base = list(rng.normal(0.0, 0.01, 100))
    base.append(0.0)
    rep = detect_return_anomaly(base)
    assert "normal" in rep.recommendation.lower() or "within" in rep.recommendation.lower()
    # Alert drawdown
    base[-1] = -0.30
    rep = detect_return_anomaly(base)
    assert "z=" in rep.recommendation


# ── detect_return_anomaly validation ─────────────────────


def test_return_detector_rejects_negative_thresholds():
    with pytest.raises(ValueError, match="non-negative"):
        detect_return_anomaly([0.01] * 50 + [0.0], z_warn=-1)


def test_return_detector_rejects_threshold_below_warn():
    with pytest.raises(ValueError, match="z_threshold"):
        detect_return_anomaly([0.01] * 50 + [0.0], z_warn=3.0, z_threshold=2.0)


def test_return_detector_handles_empty_input():
    rep = detect_return_anomaly([])
    assert isinstance(rep, EquityAnomalyReport)
    assert rep.severity == "normal"
    assert rep.last_value == 0.0


# ── detect_drawdown_anomaly ──────────────────────────────


def test_drawdown_detector_normal_on_steady_growth():
    """An always-rising curve has zero drawdown everywhere — must
    not fire."""
    curve = np.cumprod(1.0 + np.full(100, 0.005))
    rep = detect_drawdown_anomaly(curve)
    assert rep.severity == "normal"
    assert rep.direction == "normal"


def test_drawdown_detector_flags_recent_deep_trough():
    """Build a curve that's been quietly rising, then crash 30%.
    The rolling drawdown z-score should land in alert territory."""
    rng = np.random.default_rng(0)
    rs = list(rng.normal(0.005, 0.01, 100))
    rs.append(-0.30)
    curve = np.cumprod(1.0 + np.array(rs))
    rep = detect_drawdown_anomaly(curve)
    assert rep.severity == "alert"
    assert rep.direction == "drawdown"
    assert rep.z_score < -3


def test_drawdown_detector_ignores_positive_tail():
    """Drawdown is bounded above by 0 — a 'positive' deviation isn't
    meaningful, must be reported as normal regardless of z magnitude."""
    rng = np.random.default_rng(0)
    rs = list(rng.normal(-0.005, 0.01, 100))  # slow bleed
    rs.append(0.50)  # massive recovery
    curve = np.cumprod(1.0 + np.array(rs))
    rep = detect_drawdown_anomaly(curve)
    # The last point sets a new peak → drawdown=0; z relative to
    # negative baseline is positive — must NOT be alert.
    assert rep.severity == "normal"


def test_drawdown_detector_handles_too_short_curve():
    rep = detect_drawdown_anomaly([1.0, 1.01, 0.99])
    assert rep.severity == "normal"


def test_drawdown_detector_handles_non_positive_equity():
    """If the curve dips to 0 or below (account blew up), the percent
    calc is undefined — fall back to 'normal' rather than NaN-out."""
    curve = np.array([1.0, 0.9, 0.0, -0.1])
    rep = detect_drawdown_anomaly(curve, min_window=2)
    assert rep.severity == "normal"


def test_drawdown_detector_rejects_threshold_below_warn():
    with pytest.raises(ValueError, match="z_threshold"):
        detect_drawdown_anomaly([1.0] * 100, z_warn=3.0, z_threshold=2.0)


def test_drawdown_detector_rejects_negative_thresholds():
    with pytest.raises(ValueError, match="non-negative"):
        detect_drawdown_anomaly([1.0] * 100, z_threshold=-1)


# ── equity_curve_from_returns helper ─────────────────────


def test_equity_curve_default_starting_is_one():
    curve = equity_curve_from_returns([0.10, -0.05])
    # 1.0 * 1.10 * 0.95 = 1.045
    np.testing.assert_array_almost_equal(curve, [1.10, 1.045])


def test_equity_curve_custom_starting():
    curve = equity_curve_from_returns([0.10], starting=1000.0)
    np.testing.assert_array_almost_equal(curve, [1100.0])


def test_equity_curve_empty_returns_starting_only():
    curve = equity_curve_from_returns([], starting=500.0)
    np.testing.assert_array_equal(curve, [500.0])


def test_equity_curve_rejects_non_positive_starting():
    with pytest.raises(ValueError, match="starting must"):
        equity_curve_from_returns([0.01], starting=0.0)


def test_equity_curve_skips_nan_inputs():
    curve = equity_curve_from_returns([0.1, np.nan, -0.05])
    # NaN dropped; effectively the same as [0.1, -0.05]
    np.testing.assert_array_almost_equal(curve, [1.10, 1.045])


# ── round-trip: returns → curve → drawdown detector ──────


def test_round_trip_detects_simulated_blowup():
    """End-to-end: simulate a quietly-positive bot that suddenly
    catches a regime shift and bleeds 25% over five trades. The
    drawdown detector should fire on the final state."""
    rng = np.random.default_rng(99)
    good = list(rng.normal(0.005, 0.005, 100))
    bad = [-0.05, -0.05, -0.05, -0.05, -0.05]
    curve = equity_curve_from_returns(good + bad)
    rep = detect_drawdown_anomaly(curve)
    assert rep.severity in ("warn", "alert")
    assert rep.direction == "drawdown"
