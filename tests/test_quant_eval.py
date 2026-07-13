"""Known-value tests for the forecast-evaluation primitives in quant/eval.py.

Every expected number below is hand-computed from the published formula
(pinball loss, PICP, Winkler score, Kupiec POF LR, Christoffersen LR), so
these tests pin the implementation to the literature, not to itself.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from halal_trader.quant.eval import (
    BucketCoverage,
    LRTestResult,
    _chi2_sf,
    christoffersen_conditional,
    christoffersen_independence,
    coverage_by_bucket,
    interval_coverage,
    kupiec_pof,
    pinball_loss,
    winkler_score,
)

# ---------------------------------------------------------------------------
# pinball_loss
# ---------------------------------------------------------------------------


def test_pinball_underprediction_high_quantile():
    # y=10, ŷ=8, q=0.9: error e=2 >= 0 → q·e = 0.9·2 = 1.8.
    assert pinball_loss([10.0], [8.0], 0.9) == pytest.approx(1.8)


def test_pinball_underprediction_low_quantile():
    # Same error, q=0.1 → 0.1·2 = 0.2: a 10 %-quantile forecast is barely
    # penalized for being below the outcome.
    assert pinball_loss([10.0], [8.0], 0.1) == pytest.approx(0.2)


def test_pinball_overprediction():
    # y=8, ŷ=10, q=0.9: e=-2 < 0 → (q-1)·e = (-0.1)·(-2) = 0.2.
    assert pinball_loss([8.0], [10.0], 0.9) == pytest.approx(0.2)


def test_pinball_median_is_half_mae():
    # q=0.5 → 0.5·|e| per point. Errors: |1-2|=1, |2-2|=0, |3-5|=2 → MAE=1.
    y_true = [1.0, 2.0, 3.0]
    y_pred = [2.0, 2.0, 5.0]
    assert pinball_loss(y_true, y_pred, 0.5) == pytest.approx(0.5)


def test_pinball_perfect_forecast_is_zero():
    assert pinball_loss([1.0, 2.0], [1.0, 2.0], 0.25) == pytest.approx(0.0)


@pytest.mark.parametrize("bad_q", [0.0, 1.0, -0.1, 1.5])
def test_pinball_bad_quantile_raises(bad_q):
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], bad_q)


def test_pinball_empty_raises():
    with pytest.raises(ValueError):
        pinball_loss([], [], 0.5)


def test_pinball_length_mismatch_raises():
    with pytest.raises(ValueError):
        pinball_loss([1.0, 2.0], [1.0], 0.5)


# ---------------------------------------------------------------------------
# interval_coverage
# ---------------------------------------------------------------------------


def test_coverage_simple_fraction():
    # Covered: 1∈[0,2] ✓, 2∈[0,2] ✓, 3∈[4,5] ✗, 4∈[4,5] ✓ → 3/4.
    y = [1.0, 2.0, 3.0, 4.0]
    lo = [0.0, 0.0, 4.0, 4.0]
    hi = [2.0, 2.0, 5.0, 5.0]
    assert interval_coverage(y, lo, hi) == pytest.approx(0.75)


def test_coverage_boundaries_inclusive():
    # Exactly on lower and exactly on upper both count as covered.
    assert interval_coverage([5.0, 7.0], [5.0, 5.0], [7.0, 7.0]) == pytest.approx(1.0)


def test_coverage_none_covered():
    assert interval_coverage([10.0], [0.0], [1.0]) == pytest.approx(0.0)


def test_coverage_empty_raises():
    with pytest.raises(ValueError):
        interval_coverage([], [], [])


def test_coverage_length_mismatch_raises():
    with pytest.raises(ValueError):
        interval_coverage([1.0, 2.0], [0.0], [3.0, 3.0])


# ---------------------------------------------------------------------------
# winkler_score
# ---------------------------------------------------------------------------


def test_winkler_inside_interval_is_width():
    # y=5 ∈ [4,6] → score = width = 2.
    assert winkler_score([5.0], [4.0], [6.0], 0.1) == pytest.approx(2.0)


def test_winkler_below_lower():
    # y=3 < lower=4 → width 2 + (2/0.1)·(4-3) = 2 + 20 = 22.
    assert winkler_score([3.0], [4.0], [6.0], 0.1) == pytest.approx(22.0)


def test_winkler_above_upper():
    # y=7 > upper=6 → width 2 + (2/0.2)·(7-6) = 2 + 10 = 12.
    assert winkler_score([7.0], [4.0], [6.0], 0.2) == pytest.approx(12.0)


def test_winkler_mean_over_observations():
    # Inside (2.0) and below-lower (22.0) cases from above → mean 12.0.
    y = [5.0, 3.0]
    lo = [4.0, 4.0]
    hi = [6.0, 6.0]
    assert winkler_score(y, lo, hi, 0.1) == pytest.approx(12.0)


def test_winkler_boundary_no_penalty():
    # On the bound: covered inclusively, no penalty term.
    assert winkler_score([4.0], [4.0], [6.0], 0.1) == pytest.approx(2.0)


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.5, 2.0])
def test_winkler_bad_alpha_raises(bad_alpha):
    with pytest.raises(ValueError):
        winkler_score([1.0], [0.0], [2.0], bad_alpha)


def test_winkler_empty_raises():
    with pytest.raises(ValueError):
        winkler_score([], [], [], 0.1)


def test_winkler_length_mismatch_raises():
    with pytest.raises(ValueError):
        winkler_score([1.0], [0.0, 0.0], [2.0], 0.1)


# ---------------------------------------------------------------------------
# kupiec_pof
# ---------------------------------------------------------------------------


def test_kupiec_observed_equals_expected():
    # 5 breaches of 100 at 5 %: π̂ = p → LR = 0, p-value = 1.
    result = kupiec_pof(5, 100, 0.05)
    assert result.lr_stat == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)


def test_kupiec_gross_excess_breaches():
    # 30 of 100 at 5 %: LR = 2·[30·ln(0.30/0.05) + 70·ln(0.70/0.95)]
    #                      = 2·[53.752784 - 21.376715] = 64.752137.
    result = kupiec_pof(30, 100, 0.05)
    assert result.lr_stat == pytest.approx(64.752137, rel=1e-6)
    assert result.p_value < 0.001


def test_kupiec_zero_breaches_is_significant():
    # 0 of 250 at 5 %: LR = 2·250·ln(1/0.95) = 25.646647 → suspiciously few
    # breaches (over-wide bands) is also a calibration failure.
    result = kupiec_pof(0, 250, 0.05)
    assert result.lr_stat == pytest.approx(25.646647, rel=1e-6)
    assert result.p_value < 0.05


def test_kupiec_all_breaches_finite():
    # n1 = n edge: LR = 2·100·ln(1/0.05) = 200·ln(20) = 599.146455.
    result = kupiec_pof(100, 100, 0.05)
    assert result.lr_stat == pytest.approx(200.0 * math.log(20.0), rel=1e-9)
    assert result.p_value < 1e-12


@pytest.mark.parametrize("bad_rate", [0.0, 1.0, -0.1, 1.1])
def test_kupiec_bad_expected_rate_raises(bad_rate):
    with pytest.raises(ValueError):
        kupiec_pof(1, 10, bad_rate)


def test_kupiec_zero_obs_raises():
    with pytest.raises(ValueError):
        kupiec_pof(0, 0, 0.05)


def test_kupiec_breaches_exceed_obs_raises():
    with pytest.raises(ValueError):
        kupiec_pof(11, 10, 0.05)


def test_kupiec_negative_breaches_raises():
    with pytest.raises(ValueError):
        kupiec_pof(-1, 10, 0.05)


# ---------------------------------------------------------------------------
# christoffersen_independence
# ---------------------------------------------------------------------------


def test_independence_evenly_spread_breaches_high_p():
    # A breach every 20th observation (5 of 100, never adjacent):
    # n00=90, n01=5, n10=4, n11=0 → LR = 0.423443 → p ≈ 0.515.
    series = ([0] * 19 + [1]) * 5
    result = christoffersen_independence(series)
    assert result.lr_stat == pytest.approx(0.423443, rel=1e-5)
    assert result.p_value > 0.4


def test_independence_clustered_breaches_low_p():
    # One long run of breaches then calm: n11=9, n10=1, n01=0, n00=89.
    # LR = 2·[ln(0.1) + 9·ln(0.9) + 90·ln(11/10) + 9·ln(11)] = 53.816288.
    series = [1] * 10 + [0] * 90
    result = christoffersen_independence(series)
    assert result.lr_stat == pytest.approx(53.816288, rel=1e-6)
    assert result.p_value < 0.001


def test_independence_all_zeros_degenerate():
    # Never in the breach state → π11 inestimable → LR=0, p=1 by convention.
    result = christoffersen_independence([0] * 50)
    assert result.lr_stat == 0.0
    assert result.p_value == 1.0


def test_independence_all_ones_degenerate():
    result = christoffersen_independence([1] * 50)
    assert result.lr_stat == 0.0
    assert result.p_value == 1.0


def test_independence_single_observation_degenerate():
    result = christoffersen_independence([1])
    assert result.lr_stat == 0.0
    assert result.p_value == 1.0


def test_independence_accepts_bools():
    as_bool = [True] * 10 + [False] * 90
    as_int = [1] * 10 + [0] * 90
    assert christoffersen_independence(as_bool) == christoffersen_independence(as_int)


def test_independence_empty_raises():
    with pytest.raises(ValueError):
        christoffersen_independence([])


def test_independence_non_binary_raises():
    with pytest.raises(ValueError):
        christoffersen_independence([0, 1, 2])


# ---------------------------------------------------------------------------
# christoffersen_conditional
# ---------------------------------------------------------------------------


def test_conditional_is_pof_plus_independence():
    series = [1] * 10 + [0] * 90
    rate = 0.05
    pof = kupiec_pof(10, 100, rate)
    ind = christoffersen_independence(series)
    cc = christoffersen_conditional(series, rate)
    assert cc.lr_stat == pytest.approx(pof.lr_stat + ind.lr_stat, rel=1e-12)
    # χ²(2) survival function is exp(-x/2).
    assert cc.p_value == pytest.approx(math.exp(-cc.lr_stat / 2.0), rel=1e-12)
    assert cc.p_value < 0.001


def test_conditional_well_calibrated_series_high_p():
    # Exactly the nominal rate (5 of 100 at 5 % → LR_pof = 0) and evenly
    # spread (LR_ind = 0.423443) → LR_cc = 0.423443, p = exp(-0.211722) ≈ 0.81.
    series = ([0] * 19 + [1]) * 5
    result = christoffersen_conditional(series, 0.05)
    assert result.lr_stat == pytest.approx(0.423443, rel=1e-5)
    assert result.p_value == pytest.approx(math.exp(-0.423443 / 2.0), rel=1e-5)
    assert result.p_value > 0.5


def test_conditional_degenerate_independence_reduces_to_pof():
    # All-zeros: LR_ind = 0 by convention → LR_cc = LR_pof, judged on 2 dof.
    series = [0] * 250
    pof = kupiec_pof(0, 250, 0.05)
    result = christoffersen_conditional(series, 0.05)
    assert result.lr_stat == pytest.approx(pof.lr_stat, rel=1e-12)
    assert result.p_value == pytest.approx(math.exp(-pof.lr_stat / 2.0), rel=1e-12)


def test_conditional_bad_rate_raises():
    with pytest.raises(ValueError):
        christoffersen_conditional([0, 1, 0], 0.0)


def test_conditional_empty_raises():
    with pytest.raises(ValueError):
        christoffersen_conditional([], 0.05)


# ---------------------------------------------------------------------------
# coverage_by_bucket
# ---------------------------------------------------------------------------


def test_coverage_by_bucket_two_buckets():
    # "low" bucket: both inside [0,5] → 1.0; "high": 10 and 20 vs [0,5] → 0.0.
    y = [1.0, 2.0, 10.0, 20.0]
    lo = [0.0, 0.0, 0.0, 0.0]
    hi = [5.0, 5.0, 5.0, 5.0]
    buckets = ["low", "low", "high", "high"]
    result = coverage_by_bucket(y, lo, hi, buckets)
    assert set(result) == {"low", "high"}
    assert result["low"] == BucketCoverage(n=2, coverage=1.0)
    assert result["high"] == BucketCoverage(n=2, coverage=0.0)


def test_coverage_by_bucket_partial_coverage_and_counts():
    # "calm": 3 obs, 2 covered → 2/3; "storm": 1 obs, covered → 1.0.
    y = [1.0, 9.0, 3.0, 4.0]
    lo = [0.0, 0.0, 0.0, 4.0]
    hi = [5.0, 5.0, 5.0, 6.0]
    buckets = ["calm", "calm", "calm", "storm"]
    result = coverage_by_bucket(y, lo, hi, buckets)
    assert result["calm"].n == 3
    assert result["calm"].coverage == pytest.approx(2.0 / 3.0)
    assert result["storm"] == BucketCoverage(n=1, coverage=1.0)


def test_coverage_by_bucket_matches_unconditional_when_single_bucket():
    y = [1.0, 2.0, 3.0, 4.0]
    lo = [0.0, 0.0, 4.0, 4.0]
    hi = [2.0, 2.0, 5.0, 5.0]
    result = coverage_by_bucket(y, lo, hi, ["all"] * 4)
    assert result["all"].n == 4
    assert result["all"].coverage == pytest.approx(interval_coverage(y, lo, hi))


def test_coverage_by_bucket_length_mismatch_raises():
    with pytest.raises(ValueError):
        coverage_by_bucket([1.0, 2.0], [0.0, 0.0], [3.0, 3.0], ["a"])


def test_coverage_by_bucket_empty_raises():
    with pytest.raises(ValueError):
        coverage_by_bucket([], [], [], [])


# ---------------------------------------------------------------------------
# _chi2_sf and result dataclasses
# ---------------------------------------------------------------------------


def test_chi2_sf_known_critical_values():
    # 95 % critical values: χ²(1) = 3.841459, χ²(2) = 5.991465 → sf ≈ 0.05.
    assert _chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, rel=1e-6)
    assert _chi2_sf(5.991464547107979, 2) == pytest.approx(0.05, rel=1e-9)


def test_chi2_sf_at_zero_is_one():
    assert _chi2_sf(0.0, 1) == 1.0
    assert _chi2_sf(0.0, 2) == 1.0
    assert _chi2_sf(-1.0, 1) == 1.0


def test_chi2_sf_unsupported_dof_raises():
    with pytest.raises(ValueError):
        _chi2_sf(1.0, 3)


def test_result_dataclasses_are_frozen():
    lr = LRTestResult(lr_stat=1.0, p_value=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        lr.p_value = 0.1  # type: ignore[misc]
    bc = BucketCoverage(n=3, coverage=0.9)
    with pytest.raises(dataclasses.FrozenInstanceError):
        bc.n = 4  # type: ignore[misc]
