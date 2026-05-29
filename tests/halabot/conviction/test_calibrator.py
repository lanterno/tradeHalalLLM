"""FittedCalibrator — identity cold-start, Platt fit, monotonicity, no NaNs."""

from __future__ import annotations

import math

import pytest

from halabot.conviction.calibrator import CalibrationSample, FittedCalibrator, platt_fit


def _samples(pairs):
    return [CalibrationSample(raw=r, won=w) for r, w in pairs]


@pytest.mark.asyncio
async def test_identity_below_min_samples():
    cal = FittedCalibrator(min_samples=50)
    assert await cal.calibrate("NVDA", 0.4, features={}) == 0.4  # identity
    fit = cal.fit(_samples([(0.1, False), (0.9, True)]))  # only 2 < 50
    assert fit is False
    assert cal.fitted is False
    assert await cal.calibrate("NVDA", 0.4, features={}) == 0.4  # still identity


@pytest.mark.asyncio
async def test_fit_makes_calibration_monotonic_in_raw():
    # High raw wins, low raw loses → the fit should rank high above low.
    pairs = [(0.8, True)] * 40 + [(0.2, False)] * 40
    cal = FittedCalibrator(min_samples=20)
    assert cal.fit(_samples(pairs)) is True
    lo = await cal.calibrate("X", 0.2, features={})
    hi = await cal.calibrate("X", 0.8, features={})
    assert hi > lo  # higher conviction → higher P(win)
    # Monotone across the whole range.
    vals = [await cal.calibrate("X", r / 10, features={}) for r in range(11)]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


@pytest.mark.asyncio
async def test_degenerate_all_wins_no_nan():
    pairs = [(r / 100, True) for r in range(60)]  # every sample a win
    cal = FittedCalibrator(min_samples=20)
    cal.fit(_samples(pairs))
    for r in (0.0, 0.5, 1.0):
        v = await cal.calibrate("X", r, features={})
        assert math.isfinite(v) and 0.0 <= v <= 1.0


def test_platt_fit_returns_none_on_no_feature_variance():
    assert platt_fit(_samples([(0.5, True), (0.5, False), (0.5, True)])) is None


def test_platt_fit_slope_nonnegative():
    # Even with inverted data, slope is clamped >= 0 (never flips the ranking).
    pairs = [(0.9, False)] * 30 + [(0.1, True)] * 30
    model = platt_fit(_samples(pairs))
    assert model is not None and model[0] >= 0.0


@pytest.mark.asyncio
async def test_flat_or_inverted_fit_is_rejected_keeps_ranking():
    # Regression: inverted data (high raw loses) → Platt slope clamps to ~0, which
    # would flatten every conviction to a constant and destroy the policy's ranking.
    # The slope guard must REJECT it and keep identity (raw passes through, ranking
    # preserved).
    pairs = [(0.9, False)] * 40 + [(0.1, True)] * 40
    cal = FittedCalibrator(min_samples=20, min_slope=0.05)
    assert cal.fit(_samples(pairs)) is False
    assert cal.fitted is False
    hi = await cal.calibrate("X", 0.9, features={})
    lo = await cal.calibrate("X", 0.1, features={})
    assert hi == 0.9 and lo == 0.1  # identity → ranking intact


@pytest.mark.asyncio
async def test_failed_refit_keeps_prior_model():
    cal = FittedCalibrator(min_samples=20)
    cal.fit(_samples([(0.8, True)] * 30 + [(0.2, False)] * 30))
    good = await cal.calibrate("X", 0.8, features={})
    # A too-small refit must not regress the working model (INV-1).
    assert cal.fit(_samples([(0.5, True)])) is False
    assert await cal.calibrate("X", 0.8, features={}) == good
