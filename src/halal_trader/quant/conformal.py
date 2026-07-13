"""Adaptive-conformal band maintenance + coverage-drift detection.

Phase 2's "cheapest quality win": the static z-calibration artifact goes
stale as regimes shift; this module keeps it honest ONLINE, and — unlike
the repo's fitted-but-unconsumed ``DriftRiskPolicy`` cautionary tale — it
ships with its action hook wired, not aspirational.

* **ACI update** (Gibbs–Candès adaptive conformal inference, hand-rolled in
  house style — no MAPIE dependency): every matured band outcome from the
  candidate-universe labeling nudges the miscoverage rate ``alpha`` by
  ``gamma`` — a breach widens the next band (lower alpha → higher binding-z
  quantile), a covered outcome narrows it fractionally. Equilibrium sits
  where the realized breach rate equals the target. The alpha → z mapping
  interpolates the ``z_grid`` stored in the calibration artifact.
* **Coverage-drift check** — a Kupiec proportion-of-failures test on the
  trailing outcomes (``quant/eval.py``): a significant breach-rate
  distortion fires the scheduler's AlertSink (``band.coverage_drift``
  event) while the ACI is already correcting. ±5 pp point thresholds
  false-alarm at these sample sizes; a likelihood-ratio test does not.

``gamma`` defaults low (0.005) because outcomes arrive in daily batches of
~20 candidates, not one at a time — the classic per-period ACI step would
overshoot by an order of magnitude.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from halal_trader.quant.calibration import (
    DEFAULT_ARTIFACT_PATH,
    AciState,
    CalibrationArtifact,
    load_artifact,
    save_artifact,
)
from halal_trader.quant.eval import kupiec_pof

logger = logging.getLogger(__name__)

GRID_LEVELS = tuple(0.50 + 0.01 * i for i in range(50))
DEFAULT_GAMMA = 0.005
_ALPHA_MIN, _ALPHA_MAX = 0.01, 0.5
_DRIFT_WINDOW = 60  # trailing outcomes for the Kupiec drift test
_DRIFT_P = 0.05
_MIN_DRIFT_OBS = 30
# A rec's candidates are fully matured once its 5-session window has
# certainly closed (5 trading days ≈ 7 calendar, +2 safety).
_MATURITY_CALENDAR_DAYS = 9


def z_from_grid(z_grid: tuple[float, ...], alpha: float) -> float:
    """Interpolate the binding-z grid at coverage level ``1 - alpha``.

    The grid spans levels 0.50..0.99; alpha is clamped so the lookup stays
    on the grid (never extrapolates into the unsampled tail).
    """
    level = min(max(1.0 - alpha, GRID_LEVELS[0]), GRID_LEVELS[-1])
    return float(np.interp(level, GRID_LEVELS, z_grid))


def aci_step(
    alpha: float, target_alpha: float, breached: bool, gamma: float = DEFAULT_GAMMA
) -> float:
    """One Gibbs–Candès update: ``alpha += gamma·(target_alpha − err)``.

    ``err = 1`` on a breach (band missed the realized path) pushes alpha
    down → a higher grid level → a wider next band; a covered outcome
    pushes it up by ``gamma·target_alpha`` (much smaller). Clamped to
    ``[0.01, 0.5]`` so the band can neither explode nor invert.
    """
    err = 1.0 if breached else 0.0
    return float(np.clip(alpha + gamma * (target_alpha - err), _ALPHA_MIN, _ALPHA_MAX))


def _matured_band_outcomes(
    rows: list[dict[str, Any]], *, after_rec_id: int, now: datetime
) -> tuple[list[bool], int, list[bool]]:
    """(new outcomes in id order, max consumed rec id, trailing outcomes).

    New outcomes come from recs with ``id > after_rec_id`` whose date is
    old enough that every candidate's 5-session window has closed —
    partial-maturity recs are deferred whole so nothing is half-consumed.
    The trailing list (for the stateless drift test) spans ALL recent recs
    regardless of consumption state.
    """
    cutoff = (now - timedelta(days=_MATURITY_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    new_flags: list[bool] = []
    trailing: list[bool] = []
    max_id = after_rec_id
    for rec in sorted(rows, key=lambda r: r.get("id") or 0):
        rec_id = rec.get("id") or 0
        cands = rec.get("candidates")
        if not isinstance(cands, dict):
            continue
        flags = [
            bool(d["outcome"]["band_covered_5d"] is False)
            for d in cands.values()
            if isinstance(d, dict)
            and isinstance(d.get("outcome"), dict)
            and d["outcome"].get("band_covered_5d") is not None
        ]
        if not flags:
            continue
        trailing.extend(flags)
        if rec_id > after_rec_id and str(rec.get("date", "")) <= cutoff:
            new_flags.extend(flags)
            max_id = max(max_id, rec_id)
    return new_flags, max_id, trailing[-_DRIFT_WINDOW:]


async def update_band_conformal(
    repo: Any,
    *,
    horizon: int = 5,
    gamma: float = DEFAULT_GAMMA,
    path: Path = DEFAULT_ARTIFACT_PATH,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Feed matured band outcomes into the ACI state and check for drift.

    Idempotent daily maintenance (runs after the scorecard backfill in the
    09:05 job): consumes each rec's candidate outcomes exactly once via
    ``last_rec_id``, saves the adapted state back into the artifact, and
    returns a summary — including ``drift=True`` when the trailing breach
    rate fails the Kupiec test, which the caller must surface via
    AlertSink (the action hook is the caller's contract, not optional).
    """
    artifact = load_artifact(path)
    if artifact is None:
        return {"updated": False, "reason": "no calibration artifact"}
    cal = artifact.horizons.get(horizon)
    if cal is None or cal.z_grid is None:
        return {"updated": False, "reason": "artifact has no z_grid (re-run quant calibrate)"}
    now = now or datetime.now(UTC)
    target_alpha = 1.0 - artifact.target_coverage
    state = (artifact.aci or {}).get(horizon) or AciState(
        alpha=target_alpha, n_obs=0, last_rec_id=0, updated_at=""
    )
    rows = await repo.get_recent_recommendations(limit=200)
    breaches, max_id, trailing = _matured_band_outcomes(
        rows, after_rec_id=state.last_rec_id, now=now
    )
    alpha = state.alpha
    for breached in breaches:
        alpha = aci_step(alpha, target_alpha, breached, gamma)
    drift = False
    drift_p: float | None = None
    if len(trailing) >= _MIN_DRIFT_OBS:
        test = kupiec_pof(sum(trailing), len(trailing), target_alpha)
        drift_p = test.p_value
        drift = test.p_value < _DRIFT_P
    new_state = AciState(
        alpha=alpha,
        n_obs=state.n_obs + len(breaches),
        last_rec_id=max_id,
        updated_at=now.isoformat(),
    )
    updated_artifact = CalibrationArtifact(
        version=artifact.version,
        created_at=artifact.created_at,
        target_coverage=artifact.target_coverage,
        horizons=artifact.horizons,
        symbols=artifact.symbols,
        aci={**(artifact.aci or {}), horizon: new_state},
    )
    save_artifact(updated_artifact, path)
    effective = updated_artifact.effective_z(horizon)
    if breaches:
        logger.info(
            "band conformal: consumed %d outcomes, alpha %.4f -> %.4f, z_eff %.3f",
            len(breaches),
            state.alpha,
            alpha,
            effective or 0.0,
        )
    return {
        "updated": True,
        "consumed": len(breaches),
        "alpha": round(alpha, 4),
        "effective_z": round(effective, 4) if effective is not None else None,
        "trailing_n": len(trailing),
        "drift": drift,
        "drift_p": drift_p,
    }
