"""Calibrator retrain loop — walk-forward log-loss + DB-backed refit (L8)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa

from halabot.conviction.calibrator import CalibrationSample, FittedCalibrator
from halabot.learning.retrain import (
    CalibratorRetrainer,
    load_calibration_samples,
    walk_forward_logloss,
)
from halabot.platform.db import outcome as _outcome

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


def _samples(pairs):
    return [CalibrationSample(raw=r, won=w) for r, w in pairs]


def test_walk_forward_none_when_undersized():
    assert walk_forward_logloss(_samples([(0.5, True)] * 10)) is None


def test_walk_forward_fitted_beats_identity_on_separable_data():
    # High raw → win, low raw → loss: a fit should predict the held-out half
    # better than feeding raw straight through as a probability.
    pairs = [(0.9, True), (0.1, False)] * 40
    wf = walk_forward_logloss(_samples(pairs), min_train=20)
    assert wf is not None
    fitted_ll, identity_ll = wf
    assert fitted_ll <= identity_ll  # calibration helps (or at least doesn't hurt)


async def _insert_outcome(engine, *, raw: float, label: int, exit_ts: datetime):
    async with engine.begin() as conn:
        await conn.execute(
            sa.insert(_outcome).values(
                asset="NVDA",
                entry_ts=exit_ts - timedelta(hours=1),
                exit_ts=exit_ts,
                entry_price=100.0,
                exit_price=101.0,
                closed_weight=0.1,
                return_pct=0.01,
                hold_seconds=3600,
                belief_version=1,
                entry_belief={"conviction_raw": raw, "regime": "trending_up"},
                label=label,
                reason="test",
                created_at=exit_ts,
            )
        )


@pytest.mark.asyncio
async def test_load_calibration_samples_reads_entry_raw(halabot_engine):
    await _insert_outcome(halabot_engine, raw=0.8, label=1, exit_ts=T0)
    await _insert_outcome(halabot_engine, raw=0.2, label=0, exit_ts=T0 + timedelta(minutes=1))
    samples = await load_calibration_samples(halabot_engine)
    assert len(samples) == 2
    assert samples[0].raw == pytest.approx(0.8) and samples[0].won is True
    assert samples[1].raw == pytest.approx(0.2) and samples[1].won is False


@pytest.mark.asyncio
async def test_retrainer_fits_after_enough_closes(halabot_engine):
    cal = FittedCalibrator(min_samples=20)
    retrainer = CalibratorRetrainer(engine=halabot_engine, calibrator=cal, retrain_every=40)
    # Seed 40 separable outcomes.
    for i in range(40):
        raw, label = (0.8, 1) if i % 2 == 0 else (0.2, 0)
        await _insert_outcome(
            halabot_engine, raw=raw, label=label, exit_ts=T0 + timedelta(minutes=i)
        )
    # 40 on_outcome_closed calls → one refit at the threshold.
    for _ in range(40):
        await retrainer.on_outcome_closed()
    assert retrainer.refits == 1
    assert cal.fitted is True


@pytest.mark.asyncio
async def test_retrainer_does_not_activate_on_nonpredictive_data(halabot_engine):
    # Regression for the live collapse: when raw does NOT predict wins (here
    # inverted — high raw loses), the calibrator must stay identity rather than
    # activate a flat/constant model that destroys conviction ranking.
    cal = FittedCalibrator(min_samples=20)
    retrainer = CalibratorRetrainer(engine=halabot_engine, calibrator=cal, retrain_every=60)
    for i in range(60):
        raw, label = (0.9, 0) if i % 2 == 0 else (0.1, 1)  # inverted
        await _insert_outcome(
            halabot_engine, raw=raw, label=label, exit_ts=T0 + timedelta(minutes=i)
        )
    for _ in range(60):
        await retrainer.on_outcome_closed()
    assert cal.fitted is False  # never activated — identity preserved
    assert retrainer.refits == 0


@pytest.mark.asyncio
async def test_retrainer_noop_below_min_samples(halabot_engine):
    cal = FittedCalibrator(min_samples=100)  # higher than what we seed
    retrainer = CalibratorRetrainer(engine=halabot_engine, calibrator=cal, retrain_every=5)
    for i in range(10):
        await _insert_outcome(
            halabot_engine, raw=0.5, label=i % 2, exit_ts=T0 + timedelta(minutes=i)
        )
    for _ in range(5):
        await retrainer.on_outcome_closed()
    assert cal.fitted is False  # never reached min_samples
