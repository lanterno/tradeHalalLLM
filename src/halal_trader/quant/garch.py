"""GARCH-FHS Monte Carlo of horizon path extremes (Phase 2, ``[ml]``-gated).

The one model family that answers "how high / how low will it go" *natively*:
simulate return paths, take each path's running max and min over the horizon,
and read empirical quantiles of those extremes — no reflection-principle
correction needed because the extreme is computed, not approximated from a
terminal quantile. Two tiers:

* :func:`gbm_path_extremes` — constant-vol Gaussian Monte Carlo. Pure numpy,
  no dependencies; ~20 lines of simulation. This is the SANITY BASELINE the
  fitted model must beat (roadmap validation gate 1), never the product.
* :func:`garch_fhs_path_extremes` — GJR-GARCH(1,1,1) with skew-t innovations
  fit on a rolling window, forecast by filtered-historical-simulation
  bootstrap (the industry-standard bank VaR engine: captures vol clustering,
  leverage asymmetry and fat tails that GBM misses). Lazy-imports ``arch``
  and degrades to ``None`` without the ``[ml]`` extra — callers fall back to
  the deterministic HAR bands.

Honesty notes carried from the research: drift is pinned to zero (5-day
drift is unforecastable; estimating it adds noise); returns are scaled ×100
before fitting (the classic ``arch`` optimizer gotcha); refit on a rolling
window, never once; and daily-step simulation sees only daily-close
extremes — the true intraday high/low is systematically more extreme, which
is exactly the bias the downstream empirical coverage measurement exists to
absorb. NOT wired into the engine: per the roadmap it ships only after
beating the ATR/HAR baseline on pinball + Winkler + coverage on disjoint
OOS windows (`quant compare-bands`, trials ledger).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from halal_trader.quant.volatility import FloatArray

logger = logging.getLogger(__name__)

_DEFAULT_SIMS = 4000
_GARCH_WINDOW = 750  # rolling fit window (≈3 trading years)
_MIN_RETURNS = 250  # below this a GARCH fit is noise — refuse
_arch_missing_logged = False


@dataclass(frozen=True, slots=True, eq=False)
class PathExtremeBands:
    """Empirical quantiles of the horizon path max/min, as prices.

    ``high_q[p]`` is the price the path max stays UNDER with probability
    ``p``; ``low_q[p]`` the price the path min stays ABOVE with probability
    ``p`` (both marginal). ``band(coverage)`` returns the jointly-calibrated
    two-sided band — marginal quantile pairs under-cover jointly, so the
    pair is searched on the simulations themselves.
    """

    horizon: int
    n_sims: int
    model: str  # "gbm" | "garch_fhs"
    close: float
    high_q: dict[float, float]
    low_q: dict[float, float]
    _maxes: np.ndarray
    _mins: np.ndarray

    def band(self, coverage: float = 0.8) -> tuple[float, float]:
        """Two-sided band with ``coverage`` JOINT path containment (on sims)."""
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")
        lo_a, hi_a = 0.0, 0.5
        for _ in range(30):
            alpha = (lo_a + hi_a) / 2
            low = float(np.quantile(self._mins, alpha))
            high = float(np.quantile(self._maxes, 1.0 - alpha))
            joint = float(((self._mins >= low) & (self._maxes <= high)).mean())
            if joint > coverage:
                lo_a = alpha  # band too wide → trim
            else:
                hi_a = alpha
        alpha = lo_a
        return (
            float(np.quantile(self._mins, alpha)),
            float(np.quantile(self._maxes, 1.0 - alpha)),
        )


_MARGINAL_LEVELS = (0.5, 0.8, 0.9, 0.95)


def _bands_from_paths(
    price_paths: np.ndarray, close: float, horizon: int, model: str
) -> PathExtremeBands:
    """Reduce simulated price paths (n_sims × horizon) to extreme bands."""
    maxes = price_paths.max(axis=1)
    mins = price_paths.min(axis=1)
    return PathExtremeBands(
        horizon=horizon,
        n_sims=int(price_paths.shape[0]),
        model=model,
        close=close,
        high_q={p: float(np.quantile(maxes, p)) for p in _MARGINAL_LEVELS},
        low_q={p: float(np.quantile(mins, 1.0 - p)) for p in _MARGINAL_LEVELS},
        _maxes=maxes,
        _mins=mins,
    )


def gbm_path_extremes(
    close: float,
    sigma_daily: float,
    horizon: int,
    *,
    n_sims: int = _DEFAULT_SIMS,
    seed: int = 0,
) -> PathExtremeBands:
    """Constant-vol, zero-drift Gaussian Monte Carlo of path extremes.

    The 20-line no-dependency baseline: understates tails (no clustering,
    no leverage, thin tails) — any fitted model that can't beat it on
    coverage/Winkler does not ship.
    """
    if close <= 0 or sigma_daily <= 0 or horizon < 1 or n_sims < 100:
        raise ValueError(
            f"need close>0, sigma>0, horizon>=1, n_sims>=100; got "
            f"{close=}, {sigma_daily=}, {horizon=}, {n_sims=}"
        )
    rng = np.random.default_rng(seed)
    log_steps = rng.normal(0.0, sigma_daily, size=(n_sims, horizon))
    price_paths = close * np.exp(np.cumsum(log_steps, axis=1))
    return _bands_from_paths(price_paths, close, horizon, "gbm")


def garch_fhs_path_extremes(
    closes: FloatArray,
    horizon: int,
    *,
    n_sims: int = _DEFAULT_SIMS,
    window: int = _GARCH_WINDOW,
    seed: int = 0,
) -> PathExtremeBands | None:
    """GJR-GARCH(1,1,1)-skew-t filtered-historical-simulation path extremes.

    Fit on the trailing ``window`` daily log returns (×100 for optimizer
    stability), forecast ``horizon`` steps by bootstrap simulation of the
    standardized residuals, reconstruct price paths, and reduce to extreme
    bands. Returns ``None`` when ``arch`` is unavailable, the series is too
    short (< 250 returns), or the fit fails — callers fall back to the
    deterministic HAR bands.
    """
    global _arch_missing_logged
    if horizon < 1 or n_sims < 100:
        raise ValueError(f"need horizon>=1, n_sims>=100; got {horizon=}, {n_sims=}")
    try:
        from arch import arch_model
    except ImportError:
        if not _arch_missing_logged:
            logger.info("arch not installed ([ml] extra) — GARCH bands disabled")
            _arch_missing_logged = True
        return None
    c = np.asarray(closes, dtype=np.float64)
    c = c[np.isfinite(c) & (c > 0)]
    if c.size < _MIN_RETURNS + 1:
        return None
    close = float(c[-1])
    returns = 100.0 * np.diff(np.log(c))[-window:]
    try:
        am = arch_model(returns, p=1, o=1, q=1, dist="skewt", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        # Zero-drift: overwrite mu with 0 — 5-day drift is unforecastable
        # and a fitted mean only injects estimation noise into the band.
        params = res.params.copy()
        if "mu" in params.index:
            params["mu"] = 0.0
        fc = res.forecast(
            params=params,
            horizon=horizon,
            method="bootstrap",
            simulations=n_sims,
            reindex=False,
            random_state=np.random.RandomState(seed),
        )
        sims = fc.simulations.values  # (1, n_sims, horizon), log-returns ×100
    except Exception as exc:  # noqa: BLE001 — fits explode on gappy series
        logger.debug("GARCH fit/forecast failed: %s", exc)
        return None
    if sims is None:
        return None
    log_paths = np.cumsum(np.asarray(sims)[0] / 100.0, axis=1)
    price_paths = close * np.exp(log_paths)
    return _bands_from_paths(price_paths, close, horizon, "garch_fhs")
