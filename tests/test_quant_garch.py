"""Tests for quant/garch.py — GBM baseline + GARCH-FHS path extremes."""

from __future__ import annotations

import numpy as np
import pytest

from halal_trader.quant.garch import garch_fhs_path_extremes, gbm_path_extremes


class TestGbmBaseline:
    def test_band_geometry_and_determinism(self):
        a = gbm_path_extremes(100.0, 0.02, 5, seed=3)
        b = gbm_path_extremes(100.0, 0.02, 5, seed=3)
        lo, hi = a.band(0.8)
        assert lo < 100.0 < hi
        assert a.band(0.8) == b.band(0.8)  # seeded → reproducible
        assert a.model == "gbm"

    def test_joint_band_covers_target_on_own_sims(self):
        bands = gbm_path_extremes(100.0, 0.02, 5, n_sims=8000, seed=0)
        lo, hi = bands.band(0.8)
        joint = float(((bands._mins >= lo) & (bands._maxes <= hi)).mean())
        assert joint == pytest.approx(0.8, abs=0.02)

    def test_wider_coverage_widens_band(self):
        bands = gbm_path_extremes(100.0, 0.02, 5, seed=0)
        lo80, hi80 = bands.band(0.8)
        lo95, hi95 = bands.band(0.95)
        assert lo95 < lo80 and hi95 > hi80

    def test_marginal_quantiles_monotone(self):
        bands = gbm_path_extremes(100.0, 0.02, 5, seed=0)
        assert bands.high_q[0.5] < bands.high_q[0.9] < bands.high_q[0.95]
        # low_q[p] = price the min stays ABOVE with prob p: higher p → lower.
        assert bands.low_q[0.5] > bands.low_q[0.9] > bands.low_q[0.95]

    def test_path_extreme_exceeds_terminal_spread(self):
        # The horizon max is a running extreme: its median must exceed the
        # close (a driftless terminal median would sit ~at the close).
        bands = gbm_path_extremes(100.0, 0.02, 5, seed=0)
        assert bands.high_q[0.5] > 100.0
        assert bands.low_q[0.5] < 100.0

    def test_validation(self):
        with pytest.raises(ValueError):
            gbm_path_extremes(0.0, 0.02, 5)
        with pytest.raises(ValueError):
            gbm_path_extremes(100.0, 0.02, 5).band(1.0)


class TestGarchFhs:
    @pytest.fixture(autouse=True)
    def _need_arch(self):
        pytest.importorskip("arch")

    def _closes(self, n=500, sigma=0.02, seed=0):
        rng = np.random.default_rng(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, sigma, n)))

    def test_produces_sane_bands(self):
        closes = self._closes()
        bands = garch_fhs_path_extremes(closes, 5, n_sims=1000, seed=1)
        assert bands is not None
        assert bands.model == "garch_fhs"
        lo, hi = bands.band(0.8)
        assert lo < float(closes[-1]) < hi

    def test_short_series_returns_none(self):
        assert garch_fhs_path_extremes(self._closes(n=100), 5) is None

    def test_reacts_to_recent_vol_regime(self):
        rng = np.random.default_rng(2)
        calm = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
        wild_tail = np.concatenate(
            [calm[:400], calm[399] * np.exp(np.cumsum(rng.normal(0, 0.05, 100)))]
        )
        b_calm = garch_fhs_path_extremes(calm, 5, n_sims=1000, seed=3)
        b_wild = garch_fhs_path_extremes(wild_tail, 5, n_sims=1000, seed=3)
        assert b_calm is not None and b_wild is not None
        lo_c, hi_c = b_calm.band(0.8)
        lo_w, hi_w = b_wild.band(0.8)
        width_c = (hi_c - lo_c) / b_calm.close
        width_w = (hi_w - lo_w) / b_wild.close
        # Vol clustering: the recently-wild series must band far wider.
        assert width_w > 2.0 * width_c

    def test_validation(self):
        with pytest.raises(ValueError):
            garch_fhs_path_extremes(self._closes(), 0)
