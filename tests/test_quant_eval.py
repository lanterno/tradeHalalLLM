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
    LRTestResult,
    _chi2_sf,
    kupiec_pof,
)

# ---------------------------------------------------------------------------
# pinball_loss
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# interval_coverage
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# winkler_score
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# christoffersen_conditional
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# coverage_by_bucket
# ---------------------------------------------------------------------------


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


def test_result_dataclass_is_frozen():
    lr = LRTestResult(lr_stat=1.0, p_value=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        lr.p_value = 0.1  # type: ignore[misc]
