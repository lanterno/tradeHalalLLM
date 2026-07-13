"""Walk-forward z-calibration of the price bands, with a versioned artifact.

Closes the loop the roadmap warns about (fit-and-forget is the repo's known
failure mode): this module both *produces* the calibrated band multiplier
and *serves* it back to `quant/outlook.py`, so a successful calibration run
immediately changes what the daily recommendation prints — from an
UNCALIBRATED ±zσ√h approximation to a band with measured path coverage.

Method (roadmap Phase 1 "band conversion" item): for every historical day
``t`` with enough history, forecast the daily vol using ONLY bars ≤ t
(expanding-window HAR-on-Yang-Zhang refit each step — genuinely
walk-forward, no lookahead), record the realized max-high/min-low over the
following ``h`` days, and hand the pooled observations to
``bands.calibrate_z`` — the empirical binding-z quantile. Pooling across
the ~20-symbol universe is a deliberate thin-data compromise (the research
says per-symbol; ~20 symbols × ~130 observations each can't support that
yet), so the runner also reports per-symbol coverage residuals — the
number to watch for when pooling starts costing accuracy.

The artifact is a small JSON file (`data/analytics/band_calibration.json`
by default) with a version stamp that travels into every PriceOutlook and
the `candidates` JSONB, so any surfaced band is traceable to the exact
calibration that produced its multiplier.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from halal_trader.quant.bands import calibrate_z, fit_har
from halal_trader.quant.volatility import FloatArray, yang_zhang

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_PATH = Path("data/analytics/band_calibration.json")
_YZ_WINDOW = 20
# First forecastable index: enough bars that the expanding HAR fit has its
# minimum regression rows (22-day lookback + 60 rows + warm-up).
_MIN_HISTORY = 110


@dataclass(frozen=True, slots=True)
class HorizonCalibration:
    """Calibrated multiplier for one horizon: z, sample size, target.

    ``z_grid`` holds the binding-z quantiles at levels 0.50..0.99 (1 %
    steps) from the pooled calibration sample — the lookup table the ACI
    conformal layer (`quant/conformal.py`) interpolates to widen/narrow
    the band online as live coverage evidence arrives.
    """

    z: float
    n: int
    target_coverage: float
    z_grid: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class AciState:
    """Adaptive-conformal state for one horizon (runtime band maintenance).

    ``alpha`` is the ADAPTED miscoverage rate (Gibbs–Candès ACI): each
    matured band outcome nudges it by ``gamma`` toward the rate that
    delivers the target coverage under the current regime. ``last_rec_id``
    marks consumption so an observation is never fed twice.
    """

    alpha: float
    n_obs: int
    last_rec_id: int
    updated_at: str


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    """A persisted, versioned band calibration.

    ``horizons`` maps horizon (trading days) → :class:`HorizonCalibration`;
    ``version`` is stamped into every band built from this artifact.
    """

    version: str
    created_at: str
    target_coverage: float
    horizons: dict[int, HorizonCalibration]
    symbols: tuple[str, ...]
    aci: dict[int, AciState] | None = None

    def z_for(self, horizon: int) -> float | None:
        """The BASE calibrated z (ignores any ACI adaptation)."""
        cal = self.horizons.get(horizon)
        return cal.z if cal is not None else None

    def effective_z(self, horizon: int) -> float | None:
        """The z to actually band with: ACI-adapted when state exists.

        Falls back to the base z when there is no ACI state or no z_grid
        for the horizon (e.g. a pre-grid artifact).
        """
        base = self.z_for(horizon)
        if base is None:
            return None
        state = (self.aci or {}).get(horizon)
        cal = self.horizons[horizon]
        if state is None or cal.z_grid is None:
            return base
        from halal_trader.quant.conformal import z_from_grid

        return z_from_grid(cal.z_grid, state.alpha)


def save_artifact(artifact: CalibrationArtifact, path: Path = DEFAULT_ARTIFACT_PATH) -> Path:
    """Write the artifact JSON (parents created); returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": artifact.version,
        "created_at": artifact.created_at,
        "target_coverage": artifact.target_coverage,
        "horizons": {
            str(h): {
                "z": c.z,
                "n": c.n,
                "target_coverage": c.target_coverage,
                "z_grid": list(c.z_grid) if c.z_grid is not None else None,
            }
            for h, c in artifact.horizons.items()
        },
        "symbols": list(artifact.symbols),
        "aci": {
            str(h): {
                "alpha": a.alpha,
                "n_obs": a.n_obs,
                "last_rec_id": a.last_rec_id,
                "updated_at": a.updated_at,
            }
            for h, a in (artifact.aci or {}).items()
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("band calibration saved: %s -> %s", artifact.version, path)
    return path


def load_artifact(path: Path = DEFAULT_ARTIFACT_PATH) -> CalibrationArtifact | None:
    """Read an artifact; ``None`` when absent or unparseable (logged)."""
    try:
        raw = json.loads(path.read_text())
        return CalibrationArtifact(
            version=str(raw["version"]),
            created_at=str(raw["created_at"]),
            target_coverage=float(raw["target_coverage"]),
            horizons={
                int(h): HorizonCalibration(
                    z=float(c["z"]),
                    n=int(c["n"]),
                    target_coverage=float(c["target_coverage"]),
                    z_grid=(tuple(float(x) for x in c["z_grid"]) if c.get("z_grid") else None),
                )
                for h, c in raw["horizons"].items()
            },
            symbols=tuple(raw.get("symbols", ())),
            aci=(
                {
                    int(h): AciState(
                        alpha=float(a["alpha"]),
                        n_obs=int(a["n_obs"]),
                        last_rec_id=int(a["last_rec_id"]),
                        updated_at=str(a["updated_at"]),
                    )
                    for h, a in raw["aci"].items()
                }
                if raw.get("aci")
                else None
            ),
        )
    except FileNotFoundError:
        return None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("band calibration at %s unreadable: %s", path, exc)
        return None


_cache: tuple[Path, float, CalibrationArtifact | None] | None = None


def load_default_artifact() -> CalibrationArtifact | None:
    """Mtime-aware cached read of the default artifact (hot-path friendly)."""
    global _cache
    path = DEFAULT_ARTIFACT_PATH
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _cache = None
        return None
    if _cache is not None and _cache[0] == path and _cache[1] == mtime:
        return _cache[2]
    artifact = load_artifact(path)
    _cache = (path, mtime, artifact)
    return artifact


def walk_forward_observations(
    opens: FloatArray,
    highs: FloatArray,
    lows: FloatArray,
    closes: FloatArray,
    horizon: int,
    *,
    min_history: int = _MIN_HISTORY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Walk-forward (close, σ-forecast, realized high, realized low) tuples.

    At each ``t >= min_history - 1`` (and with a full ``horizon`` of bars
    after it), the σ forecast is an expanding-window HAR fit on the
    Yang-Zhang series **up to and including t only** — the same estimator
    the live outlook uses, refit each step so no observation ever sees its
    own future. Days where the HAR refuses (thin expanding window) are
    skipped rather than filled with a different estimator: the calibration
    must measure the σ source it will actually be applied to.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    h_arr = np.asarray(highs, dtype=np.float64)
    l_arr = np.asarray(lows, dtype=np.float64)
    c_arr = np.asarray(closes, dtype=np.float64)
    yz = yang_zhang(opens, highs, lows, closes, window=_YZ_WINDOW)
    n = c_arr.size
    obs_c: list[float] = []
    obs_s: list[float] = []
    obs_hi: list[float] = []
    obs_lo: list[float] = []
    for t in range(max(min_history - 1, 0), n - horizon):
        window_vol = yz[: t + 1]
        try:
            sigma = fit_har(window_vol, horizon).forecast(window_vol)
        except ValueError:
            continue
        obs_c.append(float(c_arr[t]))
        obs_s.append(sigma)
        obs_hi.append(float(h_arr[t + 1 : t + 1 + horizon].max()))
        obs_lo.append(float(l_arr[t + 1 : t + 1 + horizon].min()))
    return (
        np.asarray(obs_c),
        np.asarray(obs_s),
        np.asarray(obs_hi),
        np.asarray(obs_lo),
    )


def run_pooled_calibration(
    ohlc_by_symbol: dict[str, tuple[FloatArray, FloatArray, FloatArray, FloatArray]],
    *,
    horizons: tuple[int, ...] = (1, 5),
    target_coverage: float = 0.8,
    min_history: int = _MIN_HISTORY,
) -> tuple[CalibrationArtifact, dict[str, dict[int, dict[str, float]]]]:
    """Pooled walk-forward calibration across the universe.

    Returns the artifact plus a per-symbol coverage report:
    ``{symbol: {horizon: {"coverage": float, "n": int}}}`` — the realized
    two-sided path coverage of the *pooled* z inside each symbol's own
    observations. Symbols drifting far from ``target_coverage`` are the
    signal that pooling needs per-symbol shrinkage (roadmap note).
    """
    per_symbol: dict[str, dict[int, tuple[np.ndarray, ...]]] = {}
    horizons_cal: dict[int, HorizonCalibration] = {}
    report: dict[str, dict[int, dict[str, float]]] = {}
    for h in horizons:
        pooled: list[tuple[np.ndarray, ...]] = []
        for sym, (o, hi, lo, c) in ohlc_by_symbol.items():
            obs = walk_forward_observations(o, hi, lo, c, h, min_history=min_history)
            per_symbol.setdefault(sym, {})[h] = obs
            if obs[0].size:
                pooled.append(obs)
        if not pooled:
            continue
        closes = np.concatenate([p[0] for p in pooled])
        sigmas = np.concatenate([p[1] for p in pooled])
        highs = np.concatenate([p[2] for p in pooled])
        lows = np.concatenate([p[3] for p in pooled])
        cal = calibrate_z(closes, sigmas, highs, lows, h, target_coverage=target_coverage)
        # Binding-z quantile grid (levels 0.50..0.99) — the ACI layer's
        # alpha → z lookup table (quant/conformal.py).
        scale_all = sigmas * np.sqrt(h)
        z_binding = np.maximum(
            np.log(highs / closes) / scale_all, -np.log(lows / closes) / scale_all
        )
        grid = tuple(float(np.quantile(z_binding, 0.50 + 0.01 * i)) for i in range(50))
        horizons_cal[h] = HorizonCalibration(
            z=cal.z, n=cal.n, target_coverage=target_coverage, z_grid=grid
        )
        # Per-symbol residual coverage of the pooled z.
        for sym, by_h in per_symbol.items():
            sym_obs = by_h.get(h)
            if sym_obs is None or sym_obs[0].size == 0:
                continue
            c_s, s_s, hi_s, lo_s = sym_obs
            scale = s_s * np.sqrt(h)
            z_binding = np.maximum(np.log(hi_s / c_s) / scale, -np.log(lo_s / c_s) / scale)
            report.setdefault(sym, {})[h] = {
                "coverage": float((z_binding <= cal.z).mean()),
                "n": int(c_s.size),
            }
    if not horizons_cal:
        raise ValueError(
            "no calibration observations produced — series too short "
            f"(need ~{_MIN_HISTORY}+ bars per symbol)"
        )
    now = datetime.now(UTC)
    artifact = CalibrationArtifact(
        version=f"zcal-{now.strftime('%Y%m%d')}-c{int(target_coverage * 100)}",
        created_at=now.isoformat(),
        target_coverage=target_coverage,
        horizons=horizons_cal,
        symbols=tuple(sorted(ohlc_by_symbol)),
    )
    return artifact, report
