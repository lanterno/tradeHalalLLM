"""Known-value and property tests for the OHLC vol estimators in quant/volatility.py.

Hand-computed expectations pin each estimator to its published formula
(Parkinson 1980, Garman-Klass 1980, Rogers-Satchell 1991, Yang-Zhang 2000,
RiskMetrics EWMA); the property tests (constant series → 0, scale invariance,
GBM recovery of the true sigma) pin the family-wide contract.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from halal_trader.quant.volatility import (
    close_to_close,
    ewma_vol,
    garman_klass,
    parkinson,
    rogers_satchell,
    yang_zhang,
)


def _ohlc_fixture(n=40, seed=7):
    """Deterministic, internally consistent OHLC bars (H >= max(O,C) >= min(O,C) >= L)."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, 0.005, n - 1))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.01, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.01, n)))
    return open_, high, low, close


def _all_estimates(open_, high, low, close, window):
    """(name, output, documented NaN-prefix length) for every estimator."""
    return [
        ("close_to_close", close_to_close(close, window), window),
        ("parkinson", parkinson(high, low, window), window - 1),
        ("garman_klass", garman_klass(open_, high, low, close, window), window - 1),
        ("rogers_satchell", rogers_satchell(open_, high, low, close, window), window - 1),
        ("yang_zhang", yang_zhang(open_, high, low, close, window), window),
        ("ewma_vol", ewma_vol(close), 1),
    ]


# ---------------------------------------------------------------------------
# property: constant prices → zero vol everywhere it's defined
# ---------------------------------------------------------------------------


def test_constant_prices_give_zero_vol():
    n, w = 30, 5
    p = np.full(n, 100.0)
    for name, out, prefix in _all_estimates(p, p, p, p, w):
        assert np.isnan(out[:prefix]).all(), name
        assert out[prefix:] == pytest.approx(np.zeros(n - prefix), abs=1e-12), name


# ---------------------------------------------------------------------------
# property: scale invariance (all estimators are log-based)
# ---------------------------------------------------------------------------


def test_scale_invariance_under_price_rescaling():
    o, h, low, c = _ohlc_fixture()
    w = 10
    base = _all_estimates(o, h, low, c, w)
    scaled = _all_estimates(10.0 * o, 10.0 * h, 10.0 * low, 10.0 * c, w)
    for (name, out_base, _), (_, out_scaled, _) in zip(base, scaled):
        np.testing.assert_allclose(out_scaled, out_base, rtol=1e-9, equal_nan=True, err_msg=name)


# ---------------------------------------------------------------------------
# known values: Parkinson
# ---------------------------------------------------------------------------


def test_parkinson_hand_computed():
    # ln(H/L) per bar: 1, 2, 3 → squared: 1, 4, 9 → window-3 mean = 14/3.
    # sigma = sqrt((14/3) / (4·ln 2)).
    low = np.array([1.0, 1.0, 1.0])
    high = np.array([math.e, math.e**2, math.e**3])
    out = parkinson(high, low, 3)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(math.sqrt(14.0 / (3.0 * 4.0 * math.log(2.0))))


# ---------------------------------------------------------------------------
# known values: close-to-close
# ---------------------------------------------------------------------------


def test_close_to_close_hand_computed():
    # close = [1, e, 1, e] → log returns [1, -1, 1]. Window 3 at t=3:
    # mean = 1/3, squared deviations (2/3)² + (4/3)² + (2/3)² = 24/9,
    # ddof=1 variance = (24/9)/2 = 4/3 → sigma = 2/sqrt(3).
    close = np.array([1.0, math.e, 1.0, math.e])
    out = close_to_close(close, 3)
    assert np.isnan(out[:3]).all()
    assert out[3] == pytest.approx(2.0 / math.sqrt(3.0))


# ---------------------------------------------------------------------------
# known values: Yang-Zhang k constant and reduction to Rogers-Satchell
# ---------------------------------------------------------------------------


def test_yang_zhang_reduces_to_scaled_rogers_satchell():
    # Craft bars with O_t == C_t == C_{t-1} == 100 while H/L vary:
    # overnight returns ln(O_t/C_{t-1}) ≡ 0 kill the overnight variance and
    # open-to-close returns ln(C_t/O_t) ≡ 0 kill the k-weighted term, so
    # YZ² = (1-k)·RS² exactly, with k = 0.34/(1.34 + (n+1)/(n-1)).
    n_bars, w = 40, 20
    idx = np.arange(n_bars)
    flat = np.full(n_bars, 100.0)
    high = flat * np.exp(0.005 * (1.0 + idx % 5))
    low = flat * np.exp(-0.004 * (1.0 + (3 * idx) % 7))

    yz = yang_zhang(flat, high, low, flat, w)
    rs = rogers_satchell(flat, high, low, flat, w)
    k = 0.34 / (1.34 + 21.0 / 19.0)  # 0.34/(1.34 + (n+1)/(n-1)) at n=20

    valid = slice(w, None)  # YZ warm-up is `w` slots; RS is valid there too
    assert (rs[valid] > 0.0).all()  # identity must be tested on non-trivial vol
    np.testing.assert_allclose(yz[valid], math.sqrt(1.0 - k) * rs[valid], rtol=1e-12)

    # The implied k recovered from the outputs matches the published constant.
    implied_k = 1.0 - (yz[w] / rs[w]) ** 2
    assert implied_k == pytest.approx(k, rel=1e-9)


# ---------------------------------------------------------------------------
# property: synthetic GBM recovers the true daily sigma
# ---------------------------------------------------------------------------


def _gbm_ohlc(seed=0, sigma=0.02, n_bars=2000, n_steps=128):
    """Driftless GBM bars: each bar is a 128-step intrabar walk; O_t = C_{t-1}."""
    rng = np.random.default_rng(seed)
    inc = rng.standard_normal((n_bars, n_steps)) * (sigma / math.sqrt(n_steps))
    bar_ret = inc.sum(axis=1)
    log_open = math.log(100.0) + np.concatenate(([0.0], np.cumsum(bar_ret[:-1])))
    # Intrabar cumulative path, including the open itself as a path point.
    path = np.concatenate((np.zeros((n_bars, 1)), np.cumsum(inc, axis=1)), axis=1)
    open_ = np.exp(log_open)
    high = np.exp(log_open + path.max(axis=1))
    low = np.exp(log_open + path.min(axis=1))
    close = np.exp(log_open + bar_ret)
    return open_, high, low, close


def test_gbm_all_estimators_near_true_sigma():
    sigma = 0.02
    o, h, low, c = _gbm_ohlc(seed=0, sigma=sigma)
    w = 1000
    estimates = {
        "close_to_close": close_to_close(c, w)[-1],
        "parkinson": parkinson(h, low, w)[-1],
        "garman_klass": garman_klass(o, h, low, c, w)[-1],
        "rogers_satchell": rogers_satchell(o, h, low, c, w)[-1],
        "yang_zhang": yang_zhang(o, h, low, c, w)[-1],
        # EWMA has ~17-bar memory → noisy point values; average the series.
        "ewma_vol": float(np.nanmean(ewma_vol(c)[100:])),
    }
    for name, est in estimates.items():
        assert est == pytest.approx(sigma, rel=0.30), f"{name}: {est}"


def test_gbm_close_to_close_is_noisiest_estimator():
    # The efficiency claim behind the range family: across rolling windows,
    # close-to-close estimates scatter far more than any range estimator's.
    # (Comparing |estimate - sigma| instead is NOT stable on this fixture:
    # with 128 discrete intrabar steps the observed H/L under-samples the
    # true extremes, biasing every range estimator ~6-9 % low, which swamps
    # close-to-close's small sampling error. Dispersion isolates the
    # sampling-efficiency property the estimators are actually sold on.)
    o, h, low, c = _gbm_ohlc(seed=0)
    w = 125
    c2c_disp = float(np.nanstd(close_to_close(c, w)))
    range_disps = {
        "parkinson": float(np.nanstd(parkinson(h, low, w))),
        "garman_klass": float(np.nanstd(garman_klass(o, h, low, c, w))),
        "rogers_satchell": float(np.nanstd(rogers_satchell(o, h, low, c, w))),
        "yang_zhang": float(np.nanstd(yang_zhang(o, h, low, c, w))),
    }
    for name, disp in range_disps.items():
        assert c2c_disp > 1.5 * disp, f"{name}: c2c={c2c_disp}, {name}={disp}"


# ---------------------------------------------------------------------------
# warm-up NaN prefixes
# ---------------------------------------------------------------------------


def test_warmup_nan_prefix_lengths():
    o, h, low, c = _ohlc_fixture(n=15)
    w = 4
    for name, out, prefix in _all_estimates(o, h, low, c, w):
        assert out.shape == c.shape, name
        assert np.isnan(out[:prefix]).all(), name
        assert np.isfinite(out[prefix:]).all(), name


# ---------------------------------------------------------------------------
# ValueError cases
# ---------------------------------------------------------------------------

_EMPTY = np.array([])
_OK = np.array([1.0, 2.0, 3.0, 4.0])
_SHORT = np.array([1.0, 2.0])


def _call_all(open_, high, low, close, window=2):
    return [
        lambda: close_to_close(close, window),
        lambda: parkinson(high, low, window),
        lambda: garman_klass(open_, high, low, close, window),
        lambda: rogers_satchell(open_, high, low, close, window),
        lambda: yang_zhang(open_, high, low, close, window),
        lambda: ewma_vol(close),
    ]


def test_empty_input_raises():
    for call in _call_all(_EMPTY, _EMPTY, _EMPTY, _EMPTY):
        with pytest.raises(ValueError):
            call()


def test_length_mismatch_raises():
    # ewma_vol takes a single series, so only the multi-array estimators apply.
    for call in _call_all(_OK, _SHORT, _OK, _OK)[1:5]:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_non_positive_price_raises(bad):
    corrupt = _OK.copy()
    corrupt[1] = bad
    for call in _call_all(corrupt, corrupt, corrupt, corrupt):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("bad_window", [1, 0, -3])
def test_window_below_two_raises(bad_window):
    # ewma_vol has no window parameter; check the five windowed estimators.
    for call in _call_all(_OK, _OK, _OK, _OK, window=bad_window)[:5]:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("bad_lam", [0.0, 1.0, -0.1, 1.5])
def test_ewma_bad_lam_raises(bad_lam):
    with pytest.raises(ValueError):
        ewma_vol(_OK, lam=bad_lam)
