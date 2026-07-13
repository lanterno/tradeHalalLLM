"""Tests for quant/calibration.py — walk-forward z calibration + artifact."""

from __future__ import annotations

import numpy as np
import pytest

from halal_trader.quant.calibration import (
    CalibrationArtifact,
    HorizonCalibration,
    load_artifact,
    run_pooled_calibration,
    save_artifact,
    walk_forward_observations,
)


def _gbm_ohlc(n: int, sigma: float = 0.02, seed: int = 0):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))
    opens = np.empty(n)
    opens[0] = 100.0
    opens[1:] = closes[:-1] * np.exp(rng.normal(0.0, sigma / 4, n - 1))
    spread = np.abs(rng.normal(0.0, sigma, n)) * closes
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread
    return opens, highs, lows, closes


class TestWalkForward:
    def test_produces_observations_with_positive_sigmas(self):
        o, h, lo, c = _gbm_ohlc(260)
        closes, sigmas, highs, lows = walk_forward_observations(o, h, lo, c, horizon=1)
        assert closes.size > 50
        assert (sigmas > 0).all()
        assert (highs >= lows).all()

    def test_short_series_yields_nothing(self):
        o, h, lo, c = _gbm_ohlc(80)  # below min_history
        closes, *_ = walk_forward_observations(o, h, lo, c, horizon=1)
        assert closes.size == 0

    def test_no_lookahead_prefix_stability(self):
        # The first K observations must be identical whether or not the
        # series continues afterwards — the definition of walk-forward.
        o, h, lo, c = _gbm_ohlc(260)
        full = walk_forward_observations(o, h, lo, c, horizon=1)
        trunc = walk_forward_observations(o[:200], h[:200], lo[:200], c[:200], horizon=1)
        k = trunc[0].size
        assert k > 0
        for a, b in zip(full, trunc, strict=True):
            np.testing.assert_allclose(a[:k], b[:k])


class TestPooledCalibration:
    def test_calibrates_both_horizons_and_reports_per_symbol(self):
        ohlc = {
            "AAA": _gbm_ohlc(280, seed=1),
            "BBB": _gbm_ohlc(280, sigma=0.03, seed=2),
        }
        artifact, report = run_pooled_calibration(ohlc, target_coverage=0.8)
        assert set(artifact.horizons) == {1, 5}
        assert artifact.symbols == ("AAA", "BBB")
        assert artifact.version.startswith("zcal-")
        for cal in artifact.horizons.values():
            assert cal.z > 0
            assert cal.n >= 80
        # In-sample pooled coverage should sit near the target per symbol
        # (loose: per-symbol residuals are allowed to drift — that's the
        # number the report exists to expose).
        for sym in ohlc:
            for h in (1, 5):
                assert 0.5 <= report[sym][h]["coverage"] <= 1.0

    def test_pooled_insample_coverage_matches_target(self):
        ohlc = {"AAA": _gbm_ohlc(300, seed=4)}
        artifact, report = run_pooled_calibration(ohlc, horizons=(1,), target_coverage=0.8)
        assert report["AAA"][1]["coverage"] == pytest.approx(0.8, abs=0.03)

    def test_all_short_series_raises(self):
        with pytest.raises(ValueError, match="no calibration observations"):
            run_pooled_calibration({"AAA": _gbm_ohlc(60)})


class TestArtifactPersistence:
    def _artifact(self) -> CalibrationArtifact:
        return CalibrationArtifact(
            version="zcal-20260713-c80",
            created_at="2026-07-13T08:00:00+00:00",
            target_coverage=0.8,
            horizons={
                1: HorizonCalibration(z=1.85, n=2400, target_coverage=0.8),
                5: HorizonCalibration(z=2.05, n=2300, target_coverage=0.8),
            },
            symbols=("AAPL", "MSFT"),
        )

    def test_round_trip(self, tmp_path):
        path = tmp_path / "band_calibration.json"
        save_artifact(self._artifact(), path)
        loaded = load_artifact(path)
        assert loaded == self._artifact()
        assert loaded.z_for(5) == pytest.approx(2.05)
        assert loaded.z_for(20) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert load_artifact(tmp_path / "nope.json") is None

    def test_corrupt_file_returns_none(self, tmp_path):
        path = tmp_path / "band_calibration.json"
        path.write_text("{not json")
        assert load_artifact(path) is None
