"""Tests for ml/crypto_equity_corr.py — Round-5 Wave 11.J."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from halal_trader.ml.crypto_equity_corr import (
    Leadership,
    LeadershipReport,
    LeadershipShift,
    TimeSeriesPoint,
    detect_leadership,
    detect_leadership_shifts,
    lag_correlation_grid,
    pearson_correlation,
    render_report,
    render_shift,
    rolling_correlation,
)


def _points(
    crypto: list[float | None],
    equity: list[float | None],
    start: date = date(2026, 1, 1),
) -> list[TimeSeriesPoint]:
    if len(crypto) != len(equity):
        raise ValueError("test fixture mismatch")
    return [
        TimeSeriesPoint(
            obs_date=start + timedelta(days=i),
            crypto_value=c,
            equity_value=e,
        )
        for i, (c, e) in enumerate(zip(crypto, equity))
    ]


# --- TimeSeriesPoint validation ----------------------------------------


def test_point_valid():
    p = TimeSeriesPoint(obs_date=date(2026, 5, 1), crypto_value=1.0, equity_value=2.0)
    assert p.crypto_value == 1.0


def test_point_nan_rejected():
    with pytest.raises(ValueError):
        TimeSeriesPoint(
            obs_date=date(2026, 5, 1),
            crypto_value=float("nan"),
            equity_value=2.0,
        )


def test_point_inf_rejected():
    with pytest.raises(ValueError):
        TimeSeriesPoint(
            obs_date=date(2026, 5, 1),
            crypto_value=1.0,
            equity_value=float("inf"),
        )


def test_point_none_allowed():
    p = TimeSeriesPoint(obs_date=date(2026, 5, 1), crypto_value=None, equity_value=2.0)
    assert p.crypto_value is None


# --- pearson_correlation ----------------------------------------------


def test_pearson_perfect_positive():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.0, 4.0, 6.0, 8.0]
    assert pearson_correlation(xs, ys) == pytest.approx(1.0)


def test_pearson_perfect_negative():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [4.0, 3.0, 2.0, 1.0]
    assert pearson_correlation(xs, ys) == pytest.approx(-1.0)


def test_pearson_zero_for_uncorrelated():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [3.0, 1.0, 4.0, 2.0]
    corr = pearson_correlation(xs, ys)
    assert corr is not None
    assert -0.5 < corr < 0.5


def test_pearson_handles_nones():
    xs: list[float | None] = [1.0, None, 3.0, 4.0, 5.0]
    ys: list[float | None] = [None, 4.0, 6.0, 8.0, 10.0]
    corr = pearson_correlation(xs, ys)
    # Valid pairs: (3,6), (4,8), (5,10) — perfect positive.
    assert corr == pytest.approx(1.0)


def test_pearson_too_few_returns_none():
    xs = [1.0, 2.0]
    ys = [3.0, 4.0]
    assert pearson_correlation(xs, ys) is None


def test_pearson_length_mismatch_rejected():
    with pytest.raises(ValueError):
        pearson_correlation([1.0, 2.0], [1.0])


def test_pearson_constant_returns_zero():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [5.0, 5.0, 5.0, 5.0]
    assert pearson_correlation(xs, ys) == 0.0


# --- rolling_correlation ----------------------------------------------


def test_rolling_first_window_minus_one_none():
    points = _points([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0])
    out = rolling_correlation(points, window=3)
    assert out[0] is None
    assert out[1] is None


def test_rolling_perfect_corr():
    points = _points([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0])
    out = rolling_correlation(points, window=3)
    for v in out[2:]:
        assert v == pytest.approx(1.0)


def test_rolling_invalid_window_rejected():
    points = _points([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        rolling_correlation(points, window=2)


# --- lag_correlation_grid ---------------------------------------------


def test_grid_length_around_zero():
    """Grid should include lags -max..+max."""
    points = _points([1.0] * 30, [1.0] * 30)
    grid = lag_correlation_grid(points, max_lag=5)
    # All 11 lags valid since the series has plenty of points; constants → 0 corr.
    lags = [g.lag for g in grid]
    assert sorted(lags) == lags
    assert -5 in lags and 5 in lags
    assert 0 in lags


def test_grid_invalid_max_lag_rejected():
    points = _points([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        lag_correlation_grid(points, max_lag=0)


def test_grid_crypto_leads_detection():
    """Crypto leads equity by 3 periods (using a non-monotone signal so
    only the correct lag yields full correlation).

    Pin: with crypto[t] = pattern[t] and equity[t] = pattern[t-3],
    lag=+3 aligns crypto[i] with equity[i+3] = pattern[i] → perfect
    correlation. positive lag → CRYPTO_LEADS by our pinned convention.
    """
    pattern = [math.sin(i * 0.7) for i in range(30)]
    crypto = pattern.copy()
    equity: list[float | None] = [None, None, None] + pattern[:27]
    points = _points(crypto, equity)
    grid = lag_correlation_grid(points, max_lag=5)
    best = max(grid, key=lambda r: abs(r.correlation))
    assert best.lag == 3
    assert best.correlation == pytest.approx(1.0, rel=1e-6)


# --- detect_leadership ------------------------------------------------


def test_detect_synchronous():
    """Use a non-monotone signal so only lag=0 gives full correlation."""
    pattern = [math.sin(i * 0.7) for i in range(30)]
    points = _points(pattern, pattern)
    report = detect_leadership(points, max_lag=5)
    # Aligned non-monotone series → best lag is 0 → synchronous.
    assert report.leadership is Leadership.SYNCHRONOUS


def test_detect_equity_leads():
    """Pin: equity leading crypto by 2 → best_lag = -2 → EQUITY_LEADS.

    Construct crypto[t] = pattern[t-2] (crypto lags equity). Then at
    lag=-2 we align xs = crypto[2:] = pattern[0..27] with ys =
    equity[:-2] = pattern[0..27] → perfect correlation.
    """
    pattern = [math.sin(i * 0.7) for i in range(30)]
    crypto_list: list[float | None] = [None, None] + pattern[:28]
    equity_list: list[float | None] = list(pattern)
    points = _points(crypto_list, equity_list)
    report = detect_leadership(points, max_lag=5)
    assert report.best_lag == -2
    assert report.leadership is Leadership.EQUITY_LEADS


def test_detect_crypto_leads():
    """Pin: crypto leading equity by 2 → best_lag = +2 → CRYPTO_LEADS."""
    pattern = [math.sin(i * 0.7) for i in range(30)]
    # equity[t] = crypto[t-2]; crypto leads.
    crypto_list: list[float | None] = list(pattern)
    equity_list: list[float | None] = [None, None] + pattern[:28]
    points = _points(crypto_list, equity_list)
    report = detect_leadership(points, max_lag=5)
    assert report.best_lag == 2
    assert report.leadership is Leadership.CRYPTO_LEADS


def test_detect_too_few_data_raises():
    points = _points([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        detect_leadership(points, max_lag=5)


def test_detect_synchronous_threshold_pinned():
    """Pin: |best_lag| ≤ threshold → synchronous."""
    pattern = [math.sin(i * 0.7) for i in range(30)]
    points = _points(pattern, pattern)
    # default synchronous_threshold = 1 → lag 0 should classify SYNC.
    report = detect_leadership(points, max_lag=5, synchronous_threshold=1)
    assert report.leadership is Leadership.SYNCHRONOUS


def test_detect_invalid_threshold_rejected():
    points = _points([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        detect_leadership(points, max_lag=5, synchronous_threshold=-1)


# --- detect_leadership_shifts -----------------------------------------


def test_shift_detected_when_leadership_flips():
    # First half: equity leads crypto. Second half: synchronous.
    base = list(range(50))
    crypto = [float(i) for i in base]
    equity = [float(i) for i in base]
    # Re-shuffle the first 25 so equity leads crypto.
    for i in range(25):
        crypto[i] = float(i - 2) if i >= 2 else 0.0
    points = _points(crypto, equity)
    shifts = detect_leadership_shifts(points, window=15, max_lag=5)
    # Expect at least one shift event.
    assert len(shifts) >= 1


def test_shift_window_too_small_rejected():
    points = _points([1.0] * 50, [1.0] * 50)
    with pytest.raises(ValueError):
        detect_leadership_shifts(points, window=5, max_lag=5)


def test_shift_no_change_no_shifts():
    """All-synchronous → no shifts."""
    points = _points(list(range(50)), list(range(50)))
    shifts = detect_leadership_shifts(points, window=15, max_lag=3)
    assert shifts == ()


# --- Render -----------------------------------------------------------


def test_render_report_synchronous_emoji():
    report = LeadershipReport(
        best_lag=0,
        best_correlation=0.95,
        leadership=Leadership.SYNCHRONOUS,
        grid=(),
    )
    out = render_report(report)
    assert "⚪" in out


def test_render_report_crypto_leads_emoji():
    report = LeadershipReport(
        best_lag=-3,
        best_correlation=0.85,
        leadership=Leadership.CRYPTO_LEADS,
        grid=(),
    )
    out = render_report(report)
    assert "🟠" in out


def test_render_shift():
    shift = LeadershipShift(
        window_index=20,
        prior_leadership=Leadership.SYNCHRONOUS,
        new_leadership=Leadership.CRYPTO_LEADS,
    )
    out = render_shift(shift)
    assert "🔄" in out
    assert "synchronous" in out
    assert "crypto_leads" in out
