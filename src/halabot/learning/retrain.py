"""Calibrator retraining off closed outcomes (REARCHITECTURE L8, fix R leakage).

Reads ``hb_outcome`` rows and refits the conviction calibrator on **entry-only**
features (``entry_belief.conviction_raw`` → ``label``) — never the mid-trade
``conviction_score`` telemetry, which would leak information correlated with the
result. Held-out quality is measured **walk-forward by close date** (fit on the
earlier half, score log-loss on the later half).

Triggered after every ``retrain_every`` closed outcomes (the ShadowOutcomeTracker
calls :meth:`on_outcome_closed`); a refit that doesn't clear ``min_samples`` or
doesn't fit is a no-op that keeps the prior model (INV-1).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.conviction.calibrator import CalibrationSample, FittedCalibrator, platt_fit
from halabot.platform.db import outcome as _outcome

logger = logging.getLogger(__name__)


async def load_calibration_samples(engine: AsyncEngine) -> list[CalibrationSample]:
    """Entry-only (raw, won) samples ordered by close date (for walk-forward)."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                sa.select(_outcome.c.entry_belief, _outcome.c.label).order_by(_outcome.c.exit_ts)
            )
        ).all()
    samples: list[CalibrationSample] = []
    for entry_belief, label in rows:
        if not entry_belief:
            continue
        raw = entry_belief.get("conviction_raw")
        if raw is None:
            continue
        samples.append(CalibrationSample(raw=float(raw), won=bool(label)))
    return samples


def log_loss(samples: list[CalibrationSample], probs: list[float]) -> float:
    """Mean binary cross-entropy (clamped to avoid log(0))."""
    eps = 1e-12
    total = 0.0
    for s, p in zip(samples, probs):
        p = min(1.0 - eps, max(eps, p))
        total += -(math.log(p) if s.won else math.log(1.0 - p))
    return total / len(samples) if samples else 0.0


def walk_forward_logloss(
    samples: list[CalibrationSample], *, min_train: int = 20
) -> tuple[float, float] | None:
    """Fit Platt on the earlier half, score log-loss on the later half, vs the
    identity baseline. Returns ``(fitted_logloss, identity_logloss)`` or None if
    too few samples. A good calibrator has ``fitted_logloss < identity_logloss``.
    """
    n = len(samples)
    if n < 2 * min_train:
        return None
    cut = n // 2
    train, test = samples[:cut], samples[cut:]
    model = platt_fit(train)
    base_rate = sum(1 for s in train if s.won) / len(train)

    def _sig(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))

    if model is None:
        fitted_probs = [base_rate for _ in test]
    else:
        a, b = model
        fitted_probs = [min(1.0, max(0.0, _sig(a * s.raw + b))) for s in test]
    # Identity baseline maps raw (already in [0,1]) straight to a probability.
    identity_probs = [min(1.0, max(0.0, s.raw)) for s in test]
    return log_loss(test, fitted_probs), log_loss(test, identity_probs)


@dataclass
class CalibratorRetrainer:
    """Refits a :class:`FittedCalibrator` from accumulated outcomes."""

    engine: AsyncEngine
    calibrator: FittedCalibrator
    retrain_every: int = 20
    _since_last: int = 0
    refits: int = 0

    async def on_outcome_closed(self) -> None:
        """Hook the outcome tracker calls on each close; refits every N closes."""
        self._since_last += 1
        if self._since_last >= self.retrain_every:
            self._since_last = 0
            await self.retrain()

    async def retrain(self) -> bool:
        samples = await load_calibration_samples(self.engine)
        wf = walk_forward_logloss(samples)
        if wf is not None:
            fitted_ll, identity_ll = wf
            logger.info(
                "walk-forward log-loss: fitted=%.4f identity=%.4f (%s)",
                fitted_ll,
                identity_ll,
                "improved" if fitted_ll < identity_ll else "no improvement",
            )
        ok = self.calibrator.fit(samples)
        if ok:
            self.refits += 1
        return ok
