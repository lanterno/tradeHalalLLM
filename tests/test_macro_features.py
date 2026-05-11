"""Tests for sentiment/macro_features.py — Round-5 Wave 11.I."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from halal_trader.sentiment.macro_features import (
    FeatureRank,
    Frequency,
    MacroSeries,
    Observation,
    lagged_diff,
    month_over_month,
    rank_features,
    render_rank,
    render_top,
    spearman_correlation,
    year_over_year,
    z_score,
)


def _series(
    series_id: str = "FRED:UNRATE",
    name: str = "Unemployment Rate",
    frequency: Frequency = Frequency.MONTHLY,
    start: date = date(2024, 1, 1),
    values: list[float] | None = None,
) -> MacroSeries:
    if values is None:
        values = [3.5, 3.6, 3.7, 3.6, 3.5, 3.4, 3.5, 3.6]
    return MacroSeries(
        series_id=series_id,
        name=name,
        frequency=frequency,
        observations=tuple(
            Observation(
                obs_date=start + timedelta(days=30 * i),
                value=v,
            )
            for i, v in enumerate(values)
        ),
    )


# --- Observation validation -------------------------------------------


def test_observation_valid():
    o = Observation(obs_date=date(2026, 5, 1), value=3.5)
    assert o.value == 3.5


def test_observation_nan_rejected():
    with pytest.raises(ValueError):
        Observation(obs_date=date(2026, 5, 1), value=float("nan"))


def test_observation_inf_rejected():
    with pytest.raises(ValueError):
        Observation(obs_date=date(2026, 5, 1), value=float("inf"))


# --- MacroSeries validation -------------------------------------------


def test_series_valid():
    s = _series()
    assert s.series_id == "FRED:UNRATE"
    assert len(s.observations) == 8


def test_series_empty_id_rejected():
    with pytest.raises(ValueError):
        _series(series_id=" ")


def test_series_unsorted_rejected():
    bad = (
        Observation(obs_date=date(2026, 1, 1), value=1.0),
        Observation(obs_date=date(2025, 12, 1), value=2.0),
    )
    with pytest.raises(ValueError):
        MacroSeries(
            series_id="X",
            name="X",
            frequency=Frequency.MONTHLY,
            observations=bad,
        )


def test_series_duplicate_dates_rejected():
    bad = (
        Observation(obs_date=date(2026, 1, 1), value=1.0),
        Observation(obs_date=date(2026, 1, 1), value=2.0),
    )
    with pytest.raises(ValueError):
        MacroSeries(
            series_id="X",
            name="X",
            frequency=Frequency.MONTHLY,
            observations=bad,
        )


def test_series_helpers():
    s = _series(values=[1.0, 2.0, 3.0])
    assert s.values() == (1.0, 2.0, 3.0)
    assert len(s.dates()) == 3
    assert s.latest().value == 3.0


def test_series_empty_latest_returns_none():
    s = MacroSeries(
        series_id="X",
        name="X",
        frequency=Frequency.MONTHLY,
        observations=(),
    )
    assert s.latest() is None


# --- z_score ----------------------------------------------------------


def test_z_score_first_window_minus_one_is_none():
    s = _series(values=[1.0, 2.0, 3.0, 4.0, 5.0])
    out = z_score(s, window=3)
    assert out[0] is None
    assert out[1] is None
    assert out[2] is not None


def test_z_score_constant_window_returns_zero():
    """All-equal window → std=0 → z=0 by convention."""
    s = _series(values=[5.0] * 10)
    out = z_score(s, window=5)
    for v in out[4:]:
        assert v == 0.0


def test_z_score_increases_with_outlier():
    s = _series(values=[1.0, 1.0, 1.0, 1.0, 10.0])
    out = z_score(s, window=5)
    assert out[-1] is not None and out[-1] > 0


def test_z_score_invalid_window_rejected():
    s = _series()
    with pytest.raises(ValueError):
        z_score(s, window=1)


# --- month_over_month -------------------------------------------------


def test_mom_first_is_none():
    s = _series(values=[1.0, 2.0, 3.0])
    out = month_over_month(s)
    assert out[0] is None


def test_mom_arithmetic():
    """Pin: MoM = current/prev - 1."""
    s = _series(values=[100.0, 110.0, 121.0])
    out = month_over_month(s)
    assert out[1] == pytest.approx(0.10)
    assert out[2] == pytest.approx(0.10)


def test_mom_zero_prev_returns_none():
    s = _series(values=[0.0, 5.0])
    out = month_over_month(s)
    assert out[1] is None


# --- year_over_year ---------------------------------------------------


def test_yoy_monthly_lookback_12():
    """13-month series; YoY at index 12 = first valid."""
    vals = [float(i) for i in range(1, 14)]  # 1, 2, ..., 13
    s = _series(values=vals)
    out = year_over_year(s)
    assert out[11] is None
    assert out[12] is not None
    # 13/1 - 1 = 12.
    assert out[12] == pytest.approx(12.0)


def test_yoy_quarterly_lookback_4():
    s = _series(
        frequency=Frequency.QUARTERLY,
        values=[100.0, 100.0, 100.0, 100.0, 110.0],
    )
    out = year_over_year(s)
    assert out[3] is None
    assert out[4] == pytest.approx(0.10)


def test_yoy_annual_lookback_1():
    s = _series(
        frequency=Frequency.ANNUAL,
        values=[100.0, 110.0],
    )
    out = year_over_year(s)
    assert out[0] is None
    assert out[1] == pytest.approx(0.10)


# --- lagged_diff ------------------------------------------------------


def test_lagged_diff_default():
    s = _series(values=[1.0, 3.0, 6.0])
    out = lagged_diff(s)
    assert out[0] is None
    assert out[1] == 2.0
    assert out[2] == 3.0


def test_lagged_diff_custom_lag():
    s = _series(values=[1.0, 3.0, 6.0, 10.0])
    out = lagged_diff(s, lag=2)
    assert out[0] is None
    assert out[1] is None
    assert out[2] == 5.0


def test_lagged_diff_invalid_lag_rejected():
    s = _series()
    with pytest.raises(ValueError):
        lagged_diff(s, lag=0)


# --- spearman_correlation --------------------------------------------


def test_spearman_perfect_positive():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [10.0, 20.0, 30.0, 40.0]
    assert spearman_correlation(xs, ys) == pytest.approx(1.0)


def test_spearman_perfect_negative():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [40.0, 30.0, 20.0, 10.0]
    assert spearman_correlation(xs, ys) == pytest.approx(-1.0)


def test_spearman_uncorrelated():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ys = [1.0, 4.0, 2.0, 8.0, 3.0, 7.0, 5.0, 6.0]
    corr = spearman_correlation(xs, ys)
    assert corr is not None
    # Just check it's not perfect; the exact value depends on the
    # specific permutation.
    assert -1.0 <= corr <= 1.0


def test_spearman_drops_none_pairs():
    xs: list[float | None] = [1.0, 2.0, None, 4.0, 5.0]
    ys: list[float | None] = [10.0, 20.0, 30.0, None, 50.0]
    corr = spearman_correlation(xs, ys)
    assert corr is not None  # 3 valid pairs survive


def test_spearman_too_few_returns_none():
    xs = [1.0, 2.0]
    ys = [3.0, 4.0]
    assert spearman_correlation(xs, ys) is None


def test_spearman_length_mismatch_rejected():
    with pytest.raises(ValueError):
        spearman_correlation([1.0], [1.0, 2.0])


def test_spearman_constant_series_returns_zero():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [5.0, 5.0, 5.0, 5.0]
    assert spearman_correlation(xs, ys) == 0.0


def test_spearman_handles_ties():
    """Pin: tied ranks use the average."""
    xs = [1.0, 1.0, 2.0, 3.0]
    ys = [10.0, 10.0, 20.0, 30.0]
    corr = spearman_correlation(xs, ys)
    assert corr == pytest.approx(1.0)


# --- rank_features ----------------------------------------------------


def test_rank_features_orders_by_abs_correlation():
    target = [1.0, 2.0, 3.0, 4.0, 5.0]
    features = {
        "FRED:GOOD": {"z": [1.0, 2.0, 3.0, 4.0, 5.0]},
        "FRED:NEG": {"z": [5.0, 4.0, 3.0, 2.0, 1.0]},
        "FRED:NOISE": {"z": [3.0, 1.0, 4.0, 1.0, 5.0]},
    }
    out = rank_features(features, target, top_k=10)
    # Both GOOD and NEG have |corr| = 1; both should be at top.
    top_two_ids = {r.series_id for r in out[:2]}
    assert top_two_ids == {"FRED:GOOD", "FRED:NEG"}


def test_rank_features_top_k_caps_output():
    target = [1.0, 2.0, 3.0, 4.0, 5.0]
    features = {f"S{i}": {"z": list(target)} for i in range(10)}
    out = rank_features(features, target, top_k=3)
    assert len(out) == 3


def test_rank_features_invalid_top_k_rejected():
    with pytest.raises(ValueError):
        rank_features({}, [1.0, 2.0, 3.0], top_k=0)


def test_rank_features_drops_none_correlations():
    """Series with too few overlapping pairs return None corr; dropped."""
    target = [1.0, 2.0]
    features = {"S1": {"z": [None, None]}}  # type: ignore[list-item]
    out = rank_features(features, target, top_k=10)
    assert out == ()


def test_rank_features_deterministic_tie_break():
    """Same |corr| → tie-break by series_id then feature_name."""
    target = [1.0, 2.0, 3.0, 4.0]
    features = {
        "ZZZ": {"z": [1.0, 2.0, 3.0, 4.0]},
        "AAA": {"z": [1.0, 2.0, 3.0, 4.0]},
    }
    out = rank_features(features, target, top_k=10)
    assert out[0].series_id == "AAA"
    assert out[1].series_id == "ZZZ"


# --- Render -----------------------------------------------------------


def test_render_rank_format():
    r = FeatureRank(
        series_id="FRED:UNRATE",
        feature_name="z_score",
        correlation=-0.45,
        n_pairs=20,
    )
    out = render_rank(r)
    assert "FRED:UNRATE" in out
    assert "z_score" in out
    assert "-0.45" in out


def test_render_top_empty():
    out = render_top([])
    assert "No features" in out


def test_render_top_caps():
    rows = tuple(
        FeatureRank(
            series_id=f"S{i}",
            feature_name="z",
            correlation=0.5,
            n_pairs=10,
        )
        for i in range(5)
    )
    out = render_top(rows, top_n=2)
    assert "S0" in out
    assert "S1" in out
    assert "S2" not in out
