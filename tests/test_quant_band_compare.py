"""Tests for quant/band_compare.py — the disjoint-OOS band A/B."""

from __future__ import annotations

import numpy as np
import pytest

from halal_trader.quant.band_compare import (
    SourceScore,
    _Row,
    _score,
    build_rows,
    compare_band_sources,
    garch_verdict,
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
    dates = [f"2024-{1 + i // 500:02d}-{i:04d}" for i in range(n)]  # sortable
    return dates, opens, highs, lows, closes


def _mk_row(
    i: int,
    *,
    close=100.0,
    hi=104.0,
    lo=97.0,
    sigma=0.02,
    atr=2.0,
    g_lo=95.0,
    g_hi=106.0,
    symbol="AAA",
    features=(),
) -> _Row:
    return _Row(
        date=f"2024-01-{i:04d}",
        symbol=symbol,
        close=close,
        realized_high=hi,
        realized_low=lo,
        sigma_har=sigma,
        atr=atr,
        garch_low=g_lo,
        garch_high=g_hi,
        features=features,
    )


class TestScore:
    def test_hand_computed_coverage_and_winkler(self):
        rows = [
            _mk_row(1, hi=104.0, lo=97.0),  # inside (96, 105)
            _mk_row(2, hi=106.0, lo=97.0),  # high breaches by 1
        ]
        bands = [(96.0, 105.0), (96.0, 105.0)]
        sc = _score(bands, rows, target=0.8)
        assert sc.n == 2
        assert sc.coverage == pytest.approx(0.5)
        assert sc.coverage_error == pytest.approx(0.3)
        # widths 9% each; row 2 breach 1/100 → + (2/0.2)*0.01 = +0.10 → 10%.
        assert sc.winkler == pytest.approx((9.0 + 19.0) / 2)


class TestBuildRows:
    @pytest.fixture(autouse=True)
    def _need_arch(self):
        pytest.importorskip("arch")

    def test_rows_have_all_sources_and_no_lookahead_window(self):
        dates, o, h, lo, c = _gbm_ohlc(320)
        rows = build_rows(
            dates,
            o,
            h,
            lo,
            c,
            symbol="TST",
            horizon=5,
            garch_sims=200,
            garch_min_returns=150,
        )
        assert len(rows) >= 5
        for r in rows:
            assert r.garch_low < r.close < r.garch_high
            assert r.realized_low <= r.realized_high
            assert r.sigma_har > 0 and r.atr > 0
            assert r.symbol == "TST"
            assert len(r.features) == 8
            assert all(np.isfinite(f) for f in r.features)


class TestCompare:
    def _rows(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n):
            sigma = 0.02
            close = 100.0
            z_up = abs(rng.normal(0, 1.0))
            z_dn = abs(rng.normal(0, 1.0))
            rows.append(
                _mk_row(
                    i,
                    close=close,
                    hi=close * float(np.exp(z_up * sigma * np.sqrt(5))),
                    lo=close * float(np.exp(-z_dn * sigma * np.sqrt(5))),
                    sigma=sigma,
                    atr=2.0,
                    g_lo=close * 0.94,
                    g_hi=close * 1.06,
                )
            )
        return rows

    def test_window_zero_never_scored_and_aggregate_pools(self):
        results = compare_band_sources({"AAA": self._rows()}, n_windows=3)
        assert "window_0" not in results
        assert set(results) == {"window_1", "window_2", "aggregate"}
        agg = results["aggregate"]
        assert set(agg) == {"atr", "har_cal", "garch_fhs"}
        assert agg["har_cal"].n == agg["atr"].n == agg["garch_fhs"].n
        # har z was calibrated walk-forward → coverage lands near target.
        assert abs(agg["har_cal"].coverage - 0.8) < 0.12

    def test_too_few_rows_raises(self):
        with pytest.raises(ValueError, match="pooled rows"):
            compare_band_sources({"AAA": self._rows(n=30)}, n_windows=3)


class TestVerdict:
    def _results(self, g_wink, h_wink, a_wink, g_cov=0.78, h_cov=0.75):
        def s(w, cov):
            return SourceScore(coverage=cov, coverage_error=abs(cov - 0.8), winkler=w, n=100)

        window = {
            "garch_fhs": s(g_wink, g_cov),
            "har_cal": s(h_wink, h_cov),
            "atr": s(a_wink, 0.65),
        }
        return {"window_1": window, "window_2": window, "aggregate": window}

    def test_pass_when_beats_har_and_atr(self):
        assert garch_verdict(self._results(8.0, 9.0, 12.0)) == "pass"

    def test_fail_when_worse_than_naive_atr(self):
        assert garch_verdict(self._results(13.0, 9.0, 12.0)) == "fail"

    def test_inconclusive_when_between(self):
        assert garch_verdict(self._results(9.5, 9.0, 12.0)) == "inconclusive"

    def test_inconclusive_when_coverage_error_worse(self):
        # garch coverage 0.71 (err 0.09) vs har 0.78 (err 0.02).
        r = self._results(8.0, 9.0, 12.0, g_cov=0.71, h_cov=0.78)
        assert garch_verdict(r) == "inconclusive"


class TestEmbargoAndQgbm:
    def test_embargo_drops_each_symbols_latest_row(self):
        from halal_trader.quant.band_compare import _embargo_last_per_symbol

        rows = [
            _mk_row(1, symbol="AAA"),
            _mk_row(2, symbol="AAA"),
            _mk_row(1, symbol="BBB"),
        ]
        kept = _embargo_last_per_symbol(rows)
        assert len(kept) == 1
        assert kept[0].symbol == "AAA" and kept[0].date.endswith("0001")

    def test_qgbm_fit_predict_learns_conditional_width(self):
        pytest.importorskip("sklearn")
        from halal_trader.quant.qgbm import fit_qgbm, predict_bands

        rng = np.random.default_rng(0)
        rows = []
        for i in range(600):
            # Feature 0 carries the true (log) vol; extremes scale with it.
            vol = 0.01 if i % 2 == 0 else 0.04
            feats = (float(np.log(vol)), 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.5)
            hi = 100.0 * float(np.exp(abs(rng.normal(0, vol)) * 2))
            lo = 100.0 * float(np.exp(-abs(rng.normal(0, vol)) * 2))
            rows.append(_mk_row(i, hi=hi, lo=lo, features=feats))
        models = fit_qgbm(rows)
        assert models is not None
        calm = [_mk_row(0, features=(float(np.log(0.01)), 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.5))]
        wild = [_mk_row(1, features=(float(np.log(0.04)), 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.5))]
        (c_lo, c_hi) = predict_bands(models, calm)[0]
        (w_lo, w_hi) = predict_bands(models, wild)[0]
        assert (w_hi - w_lo) > 1.5 * (c_hi - c_lo)  # conditional width learned

    def test_qgbm_refuses_thin_training(self):
        pytest.importorskip("sklearn")
        from halal_trader.quant.qgbm import fit_qgbm

        rows = [_mk_row(i, features=(0.0,) * 8) for i in range(50)]
        assert fit_qgbm(rows) is None

    def test_compare_includes_qgbm_when_trainable(self):
        pytest.importorskip("sklearn")
        rng = np.random.default_rng(1)
        rows = []
        for i in range(900):
            sigma = 0.02
            feats = (float(np.log(sigma)), 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.5)
            z_up = abs(rng.normal(0, 1.0))
            z_dn = abs(rng.normal(0, 1.0))
            rows.append(
                _mk_row(
                    i,
                    hi=100.0 * float(np.exp(z_up * sigma * np.sqrt(5))),
                    lo=100.0 * float(np.exp(-z_dn * sigma * np.sqrt(5))),
                    sigma=sigma,
                    features=feats,
                )
            )
        results = compare_band_sources({"AAA": rows}, n_windows=3)
        # 300 prior rows in window_1's train set (minus embargo) → qgbm fits.
        assert "qgbm" in results["window_1"]
        assert "qgbm" in results["aggregate"]

    def test_ship_verdict_on_shared_windows_only(self):
        from halal_trader.quant.band_compare import SourceScore, ship_verdict

        def sc(w, err=0.02):
            return SourceScore(coverage=0.8, coverage_error=err, winkler=w, n=100)

        results = {
            "window_1": {"har_cal": sc(9.0), "atr": sc(12.0)},  # no qgbm here
            "window_2": {"har_cal": sc(9.0), "atr": sc(12.0), "qgbm": sc(8.0)},
            "aggregate": {"har_cal": sc(9.0), "atr": sc(12.0)},
        }
        # qgbm judged only on window_2, where it beats har_cal.
        assert ship_verdict(results, "qgbm") == "pass"
        assert ship_verdict(results, "missing") == "inconclusive"
