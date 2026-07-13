"""Tests for quant/bands.py — HAR forecaster, band conversion, z calibration."""

from __future__ import annotations

import numpy as np
import pytest

from halal_trader.quant.bands import (
    CalibratedZ,
    HARModel,
    atr_band,
    calibrate_z,
    fit_har,
    price_bands,
)


def _noisy_const_vol(level: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return level * np.exp(rng.normal(0.0, 0.05, n))


class TestHAR:
    def test_constant_vol_forecasts_that_vol(self):
        vol = _noisy_const_vol(0.02, 300)
        model = fit_har(vol, horizon=5)
        fc = model.forecast(vol)
        assert fc == pytest.approx(0.02, rel=0.05)
        assert model.horizon == 5
        assert model.n >= 60

    def test_regime_shift_pulls_forecast_toward_recent_vol(self):
        rng = np.random.default_rng(1)
        calm = 0.01 * np.exp(rng.normal(0.0, 0.05, 400))
        wild = 0.04 * np.exp(rng.normal(0.0, 0.05, 60))
        vol = np.concatenate([calm, wild])
        model = fit_har(vol, horizon=1)
        fc = model.forecast(vol)
        # Persistence: with the trailing 22d in the wild regime the forecast
        # must sit far above the calm level.
        assert fc > 0.02

    def test_forecast_ignores_nan_warmup(self):
        vol = np.concatenate([[np.nan] * 21, _noisy_const_vol(0.02, 300)])
        model = fit_har(vol, horizon=1)
        assert model.forecast(vol) == pytest.approx(0.02, rel=0.05)

    def test_too_short_series_raises(self):
        with pytest.raises(ValueError, match="regression rows"):
            fit_har(_noisy_const_vol(0.02, 50), horizon=5)

    def test_nonpositive_vol_raises(self):
        vol = np.concatenate([_noisy_const_vol(0.02, 200), [0.0]])
        with pytest.raises(ValueError, match="strictly positive"):
            fit_har(vol, horizon=1)

    def test_forecast_needs_22_points(self):
        model = HARModel(coefs=(0.0, 1.0, 0.0, 0.0), resid_var=0.0, horizon=1, n=100)
        with pytest.raises(ValueError, match="22"):
            model.forecast(np.full(10, 0.02))


class TestPriceBands:
    def test_hand_computed_lognormal_band(self):
        b = price_bands(close=100.0, sigma_daily=0.02, horizon=4, z=1.0)
        # scale = 0.02 * sqrt(4) = 0.04
        assert b.high == pytest.approx(100.0 * np.exp(0.04))
        assert b.low == pytest.approx(100.0 * np.exp(-0.04))
        # E[range] = sqrt(8/pi) * 0.04 * 100 ≈ 6.3831
        assert b.expected_range == pytest.approx(100.0 * np.sqrt(8 / np.pi) * 0.04)
        assert b.z == 1.0

    def test_wider_z_widens_band(self):
        narrow = price_bands(100.0, 0.02, 1, z=1.0)
        wide = price_bands(100.0, 0.02, 1, z=2.0)
        assert wide.high > narrow.high
        assert wide.low < narrow.low

    def test_validation(self):
        for bad in (
            {"close": 0.0, "sigma_daily": 0.02, "horizon": 1, "z": 1.0},
            {"close": 100.0, "sigma_daily": -0.1, "horizon": 1, "z": 1.0},
            {"close": 100.0, "sigma_daily": 0.02, "horizon": 0, "z": 1.0},
            {"close": 100.0, "sigma_daily": 0.02, "horizon": 1, "z": 0.0},
        ):
            with pytest.raises(ValueError):
                price_bands(**bad)


class TestAtrBand:
    def test_arithmetic(self):
        b = atr_band(close=50.0, atr=2.0, horizon=4, multiple=1.5)
        # half-width = 1.5 * 2 * sqrt(4) = 6
        assert b.high == pytest.approx(56.0)
        assert b.low == pytest.approx(44.0)
        assert b.expected_range == pytest.approx(12.0)

    def test_low_floored_at_zero(self):
        b = atr_band(close=1.0, atr=5.0, horizon=1, multiple=1.0)
        assert b.low == 0.0

    def test_validation(self):
        with pytest.raises(ValueError):
            atr_band(close=50.0, atr=0.0, horizon=1)


class TestCalibrateZ:
    def _fixture(self, n: int = 200, sigma: float = 0.02, seed: int = 0):
        """Observations whose binding z values are known by construction."""
        rng = np.random.default_rng(seed)
        closes = np.full(n, 100.0)
        sigmas = np.full(n, sigma)
        z_up = np.abs(rng.normal(0.0, 1.0, n))
        z_dn = np.abs(rng.normal(0.0, 1.0, n))
        highs = closes * np.exp(z_up * sigma)
        lows = closes * np.exp(-z_dn * sigma)
        return closes, sigmas, highs, lows, np.maximum(z_up, z_dn)

    def test_recovers_empirical_quantile(self):
        closes, sigmas, highs, lows, z_binding = self._fixture()
        cal = calibrate_z(closes, sigmas, highs, lows, horizon=1, target_coverage=0.8)
        assert isinstance(cal, CalibratedZ)
        assert cal.z == pytest.approx(float(np.quantile(z_binding, 0.8)))
        assert cal.n == 200

    def test_calibrated_band_actually_covers_target(self):
        closes, sigmas, highs, lows, _ = self._fixture(n=500, seed=3)
        cal = calibrate_z(closes, sigmas, highs, lows, horizon=1, target_coverage=0.8)
        covered = 0
        for c, s, hi, lo in zip(closes, sigmas, highs, lows, strict=True):
            b = price_bands(float(c), float(s), 1, cal.z)
            covered += int(lo >= b.low and hi <= b.high)
        assert covered / len(closes) == pytest.approx(0.8, abs=0.02)

    def test_higher_target_gives_wider_z(self):
        closes, sigmas, highs, lows, _ = self._fixture()
        z80 = calibrate_z(closes, sigmas, highs, lows, 1, target_coverage=0.8).z
        z95 = calibrate_z(closes, sigmas, highs, lows, 1, target_coverage=0.95).z
        assert z95 > z80

    def test_nan_rows_dropped(self):
        closes, sigmas, highs, lows, _ = self._fixture()
        sigmas = sigmas.copy()
        sigmas[:10] = np.nan
        cal = calibrate_z(closes, sigmas, highs, lows, 1)
        assert cal.n == 190

    def test_too_few_samples_raises(self):
        closes, sigmas, highs, lows, _ = self._fixture(n=10)
        with pytest.raises(ValueError, match="calibration observations"):
            calibrate_z(closes, sigmas, highs, lows, 1)

    def test_length_mismatch_raises(self):
        closes, sigmas, highs, lows, _ = self._fixture()
        with pytest.raises(ValueError, match="length mismatch"):
            calibrate_z(closes[:-1], sigmas, highs, lows, 1)

    def test_bad_target_raises(self):
        closes, sigmas, highs, lows, _ = self._fixture()
        with pytest.raises(ValueError, match="target_coverage"):
            calibrate_z(closes, sigmas, highs, lows, 1, target_coverage=1.0)
