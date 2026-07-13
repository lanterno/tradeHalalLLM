"""Per-symbol quantitative PriceOutlook — the "how high / how low" summary.

Phase 1 advisory core of ``docs/QUANT_PREDICTION_ROADMAP.md``: composes the
volatility estimators (``quant/volatility.py``) and the band layer
(``quant/bands.py``) into one per-symbol object the recommendation engine
(and later the live cycle) can consume. Deliberately *honest about its own
maturity*: until the walk-forward z-calibration artifact exists, every band
is stamped ``calibrated=False`` and uses a fixed default multiplier —
consumers must present it as an approximate statistical band, never as a
measured-coverage interval.

Sigma source per horizon, best-first:

1. ``har_yz`` — direct HAR(1, 5, 22) forecast on the Yang-Zhang series
   (needs ~110+ bars; the roadmap's preferred forecaster).
2. ``yz_current`` — the latest Yang-Zhang estimate, √h-scaled by the band
   layer (no mean reversion — acceptable fallback at 1–5 day horizons).
3. ``ewma`` — RiskMetrics EWMA on closes when the OHLC series is too short
   for a stable Yang-Zhang window.

Pure numpy; returns ``None`` instead of raising when a symbol simply lacks
data (the engine skips it), but raises on malformed inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from halal_trader.quant.bands import PriceBands, atr_band, fit_har, price_bands
from halal_trader.quant.calibration import CalibrationArtifact
from halal_trader.quant.volatility import FloatArray, ewma_vol, yang_zhang

DEFAULT_HORIZONS = (1, 5)
# Uncalibrated default band multiplier. Deliberately conservative-ish but
# NOT a coverage claim: the calibrate_z artifact replaces it (roadmap Phase 1
# "band conversion" item), at which point calibrated=True and the measured
# target coverage travels with the band.
DEFAULT_Z = 1.28
_YZ_WINDOW = 20
_MIN_BARS = 25  # below this a symbol gets no outlook at all
_MIN_PCTL_POINTS = 60  # vol percentile needs history to mean anything


@dataclass(frozen=True, slots=True)
class HorizonBand:
    """One horizon's band plus the provenance of its vol forecast."""

    band: PriceBands
    sigma_source: str  # har_yz | yz_current | ewma


@dataclass(frozen=True, slots=True)
class PriceOutlook:
    """Quantitative range outlook for one symbol at one moment.

    ``bands`` maps horizon (trading days) → :class:`HorizonBand`;
    ``atr_baseline_5d`` is the naive ATR band every model must beat (None
    when the caller has no ATR); ``vol_percentile`` ranks the current
    Yang-Zhang vol against the symbol's own available history (None below
    ``60`` finite points — a thin percentile is worse than none).
    ``calibrated`` is True only when EVERY band's multiplier came from a
    walk-forward :class:`~halal_trader.quant.calibration.CalibrationArtifact`
    (whose version is then in ``calibration_version``); otherwise the bands
    use ``DEFAULT_Z`` and must be presented as uncalibrated approximations.
    """

    close: float
    n_bars: int
    bands: dict[int, HorizonBand]
    atr_baseline_5d: PriceBands | None
    vol_percentile: float | None
    calibrated: bool
    calibration_version: str | None


def build_outlook(
    opens: FloatArray,
    highs: FloatArray,
    lows: FloatArray,
    closes: FloatArray,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    atr: float | None = None,
    calibration: "CalibrationArtifact | None" = None,
) -> PriceOutlook | None:
    """Build the outlook from ascending daily OHLC arrays.

    Returns ``None`` when there is not enough data for even the fallback
    estimators (< 25 bars) — callers treat that as "no quantitative view",
    not an error. Raises ``ValueError`` on malformed inputs (mismatched
    lengths, non-positive prices), which indicates a broken feed rather
    than a thin one. When ``calibration`` is provided, each horizon's z
    comes from the artifact (falling back to ``DEFAULT_Z`` for horizons it
    doesn't cover — which also demotes the outlook to uncalibrated).
    """
    c_arr = np.asarray(closes, dtype=np.float64)
    if c_arr.ndim != 1:
        raise ValueError(f"closes must be 1-dimensional, got ndim={c_arr.ndim}")
    n = int(c_arr.size)
    if n < _MIN_BARS:
        return None
    close = float(c_arr[-1])

    yz = yang_zhang(opens, highs, lows, closes, window=_YZ_WINDOW)
    yz_finite = yz[np.isfinite(yz) & (yz > 0)]

    bands: dict[int, HorizonBand] = {}
    all_calibrated = True
    for h in horizons:
        sigma, source = _sigma_for_horizon(yz, yz_finite, c_arr, h)
        if sigma is None or source is None:
            continue
        z_h = calibration.effective_z(h) if calibration is not None else None
        if z_h is None:
            z_h = DEFAULT_Z
            all_calibrated = False
        bands[h] = HorizonBand(band=price_bands(close, sigma, h, z_h), sigma_source=source)
    if not bands:
        return None

    vol_percentile: float | None = None
    if yz_finite.size >= _MIN_PCTL_POINTS:
        current = float(yz_finite[-1])
        vol_percentile = float((yz_finite < current).mean())

    baseline: PriceBands | None = None
    if atr is not None and atr > 0:
        baseline = atr_band(close, float(atr), horizon=5, multiple=1.0)

    version: str | None = None
    if calibration is not None and all_calibrated:
        version = calibration.version
    return PriceOutlook(
        close=close,
        n_bars=n,
        bands=bands,
        atr_baseline_5d=baseline,
        vol_percentile=vol_percentile,
        calibrated=version is not None,
        calibration_version=version,
    )


def _sigma_for_horizon(
    yz: np.ndarray,
    yz_finite: np.ndarray,
    closes: np.ndarray,
    horizon: int,
) -> tuple[float | None, str | None]:
    """Best available daily-vol forecast for one horizon (see module doc)."""
    if yz_finite.size > 0:
        try:
            model = fit_har(yz, horizon)
            return model.forecast(yz), "har_yz"
        except ValueError:
            return float(yz_finite[-1]), "yz_current"
    ew = ewma_vol(closes, lam=0.94)
    ew_finite = ew[np.isfinite(ew) & (ew > 0)]
    if ew_finite.size == 0:
        return None, None
    return float(ew_finite[-1]), "ewma"
