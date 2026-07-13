"""Tests for quant/outlook.py — the per-symbol PriceOutlook composer."""

from __future__ import annotations

import numpy as np
import pytest

from halal_trader.quant.calibration import CalibrationArtifact, HorizonCalibration
from halal_trader.quant.outlook import DEFAULT_Z, build_outlook

ARTIFACT = CalibrationArtifact(
    version="zcal-test",
    created_at="2026-07-13T00:00:00+00:00",
    target_coverage=0.8,
    horizons={
        1: HorizonCalibration(z=1.9, n=100, target_coverage=0.8),
        5: HorizonCalibration(z=2.1, n=100, target_coverage=0.8),
    },
    symbols=("AAPL",),
)


def _gbm_ohlc(n: int, sigma: float = 0.02, seed: int = 0):
    """Synthetic daily OHLC with realistic intrabar range around a GBM close."""
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))
    opens = np.empty(n)
    opens[0] = 100.0
    opens[1:] = closes[:-1] * np.exp(rng.normal(0.0, sigma / 4, n - 1))
    spread = np.abs(rng.normal(0.0, sigma, n)) * closes
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread
    return opens, highs, lows, closes


class TestBuildOutlook:
    def test_long_series_uses_har_for_all_horizons(self):
        o, h, lo, c = _gbm_ohlc(250)
        out = build_outlook(o, h, lo, c)
        assert out is not None
        assert set(out.bands) == {1, 5}
        assert out.bands[1].sigma_source == "har_yz"
        assert out.bands[5].sigma_source == "har_yz"
        assert out.calibrated is False
        assert out.calibration_version is None
        assert out.bands[1].band.z == DEFAULT_Z

    def test_calibration_artifact_overrides_z(self):
        o, h, lo, c = _gbm_ohlc(250)
        out = build_outlook(o, h, lo, c, calibration=ARTIFACT)
        assert out.calibrated is True
        assert out.calibration_version == "zcal-test"
        assert out.bands[1].band.z == pytest.approx(1.9)
        assert out.bands[5].band.z == pytest.approx(2.1)

    def test_partial_artifact_demotes_to_uncalibrated(self):
        partial = CalibrationArtifact(
            version="zcal-partial",
            created_at="2026-07-13T00:00:00+00:00",
            target_coverage=0.8,
            horizons={1: HorizonCalibration(z=1.9, n=100, target_coverage=0.8)},
            symbols=(),
        )
        o, h, lo, c = _gbm_ohlc(250)
        out = build_outlook(o, h, lo, c, calibration=partial)
        # h=5 fell back to DEFAULT_Z → the outlook must not claim calibration.
        assert out.calibrated is False
        assert out.calibration_version is None
        assert out.bands[1].band.z == pytest.approx(1.9)
        assert out.bands[5].band.z == DEFAULT_Z

    def test_band_geometry_is_sane(self):
        o, h, lo, c = _gbm_ohlc(250)
        out = build_outlook(o, h, lo, c)
        close = float(c[-1])
        for hb in out.bands.values():
            assert hb.band.low < close < hb.band.high
            assert hb.band.expected_range > 0
        # 5-day band strictly contains the 1-day band (same sigma family, √h)
        assert out.bands[5].band.high > out.bands[1].band.high
        assert out.bands[5].band.low < out.bands[1].band.low

    def test_medium_series_falls_back_to_yz_current(self):
        # 60 bars → ~40 finite YZ points → HAR refuses → current YZ estimate.
        o, h, lo, c = _gbm_ohlc(60)
        out = build_outlook(o, h, lo, c)
        assert out is not None
        assert out.bands[5].sigma_source == "yz_current"

    def test_short_series_returns_none(self):
        o, h, lo, c = _gbm_ohlc(10)
        assert build_outlook(o, h, lo, c) is None

    def test_vol_percentile_needs_history(self):
        o, h, lo, c = _gbm_ohlc(60)  # ~40 finite YZ points < 60
        assert build_outlook(o, h, lo, c).vol_percentile is None
        o, h, lo, c = _gbm_ohlc(250)
        pctl = build_outlook(o, h, lo, c).vol_percentile
        assert pctl is not None and 0.0 <= pctl <= 1.0

    def test_atr_baseline_band(self):
        o, h, lo, c = _gbm_ohlc(250)
        out = build_outlook(o, h, lo, c, atr=2.0)
        assert out.atr_baseline_5d is not None
        assert out.atr_baseline_5d.low < float(c[-1]) < out.atr_baseline_5d.high
        assert build_outlook(o, h, lo, c, atr=None).atr_baseline_5d is None

    def test_malformed_input_raises(self):
        o, h, lo, c = _gbm_ohlc(100)
        with pytest.raises(ValueError):
            build_outlook(o[:-1], h, lo, c)  # length mismatch

    def test_forecast_tracks_true_sigma_roughly(self):
        o, h, lo, c = _gbm_ohlc(500, sigma=0.02, seed=7)
        out = build_outlook(o, h, lo, c)
        # The 1-day sigma forecast should be in the right ballpark of the
        # true GBM sigma (the fixture's intrabar spread inflates range
        # estimators somewhat — this is a sanity bound, not a bias test).
        assert 0.01 < out.bands[1].band.sigma_daily < 0.06
