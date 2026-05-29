"""Fitted conviction calibrator (REARCHITECTURE L4, L8).

Maps raw conviction → P(win | raw) by Platt scaling — a 1-D logistic regression
``sigmoid(a·raw + b)`` fit on closed-outcome labels. Cold-start safe: with no
fitted model (or below ``min_samples``) it falls back to identity, so the engine
runs from day one and self-activates once the learning loop (L8) accumulates
enough leakage-free outcomes and calls :meth:`fit`.

Two guarantees from the spec's calibrator tests:
* **Monotonic in raw** — the slope ``a`` is clamped ≥ 0, so a higher raw score
  never produces a *lower* calibrated probability (calibration re-weights; it
  never flips the ranking).
* **No NaNs on degenerate data** — a ridge penalty keeps ``(a, b)`` finite even
  on a fully-separable (all-win / all-loss) training set.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationSample:
    raw: float  # raw conviction at entry (the feature)
    won: bool  # label: net_return_pct > win_threshold


def _sigmoid(z: float) -> float:
    # Numerically stable logistic.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def platt_fit(
    samples: list[CalibrationSample],
    *,
    iters: int = 800,
    lr: float = 0.5,
    ridge: float = 1e-3,
) -> tuple[float, float] | None:
    """Fit ``sigmoid(a·raw + b)`` by ridge-regularized logistic GD.

    Returns ``(a, b)`` with ``a ≥ 0`` (monotone), or ``None`` if the data is
    degenerate (no variance in the feature). Ridge keeps weights finite on
    separable data, so the output never NaNs."""
    n = len(samples)
    if n < 2:
        return None
    xs = [s.raw for s in samples]
    if max(xs) - min(xs) < 1e-9:
        return None  # no feature variance → nothing to calibrate against
    ys = [1.0 if s.won else 0.0 for s in samples]
    a, b = 0.0, 0.0
    for _ in range(iters):
        ga, gb = 0.0, 0.0
        for x, y in zip(xs, ys):
            p = _sigmoid(a * x + b)
            err = p - y
            ga += err * x
            gb += err
        ga = ga / n + ridge * a
        gb = gb / n + ridge * b
        a -= lr * ga
        b -= lr * gb
        if not (math.isfinite(a) and math.isfinite(b)):
            return None
    # Enforce monotonicity: higher raw must never lower P(win).
    return max(0.0, a), b


class FittedCalibrator:
    """Platt-scaling calibrator with an identity cold-start fallback.

    Conforms to the :class:`~halabot.conviction.raw.Calibrator` protocol
    (``calibrate``). Holds a single global model (per-asset models are a later
    refinement once per-asset outcomes are dense enough)."""

    def __init__(self, *, min_samples: int = 50, min_slope: float = 0.05) -> None:
        self._min_samples = min_samples
        # A near-flat slope maps every raw score to ~the same probability, which
        # DESTROYS the ranking the long-only policy sizes on (worse than identity).
        # Reject it and keep the prior/identity model.
        self._min_slope = min_slope
        self._model: tuple[float, float] | None = None
        self.fitted = False

    async def calibrate(self, asset: str, raw: float, *, features: dict[str, Any]) -> float:
        raw = max(0.0, min(1.0, raw))
        if self._model is None:
            return raw  # identity cold-start
        a, b = self._model
        return max(0.0, min(1.0, _sigmoid(a * raw + b)))

    def fit(self, samples: list[CalibrationSample]) -> bool:
        """Fit on closed-outcome samples. Below ``min_samples`` or on degenerate
        data the prior model is kept and ``False`` is returned (INV-1: a failed
        refit never regresses a working calibrator)."""
        if len(samples) < self._min_samples:
            return False
        model = platt_fit(samples)
        if model is None:
            logger.warning("calibrator fit produced no model (degenerate data); keeping prior")
            return False
        if model[0] < self._min_slope:
            logger.warning(
                "calibrator fit slope a=%.3f < %.3f (non-discriminating, would flatten "
                "conviction); keeping prior/identity",
                model[0],
                self._min_slope,
            )
            return False
        self._model = model
        self.fitted = True
        logger.info(
            "calibrator fit on %d samples: a=%.3f b=%.3f", len(samples), model[0], model[1]
        )
        return True
