"""Every research backtest is a trial; a result is judged against all of them.

The question each strategy answers is the same -- does it beat holding
SPUS, the halal index fund? -- so every recorded backtest counts against
every other one, whatever strategy it came from. A run is recorded in
``quant_trials`` under ``research.<strategy>`` with its active returns
(strategy minus SPUS) summarised, and judged by the Deflated Sharpe Ratio
of those active returns:

* the trial count is the number of distinct configurations ever recorded
  (re-running one configuration on newer data is not a new trial);
* the Sharpe variance across trials is the larger of the variance across
  their recorded Sharpes and this run's own estimator variance, so a
  handful of near-identical trials cannot shrink the deflation to nothing.

A pass means: after allowing for how many things were tried, the
probability that the strategy's true active Sharpe beats the best a
no-skill search would have found is at least 95%. Only point-in-time runs
are recorded; a backtest with look-ahead in it is not evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.sharpe_stats import (
    _sharpe_and_moments,
    _sharpe_estimator_variance,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl, config_hash

FloatArray = NDArray[np.float64]

PREFIX = "research."
BENCHMARK = "SPUS"
MIN_DSR = 0.95
CRITERION = f"DSR of active returns vs {BENCHMARK} >= {MIN_DSR} across all research trials"
_TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class Assessment:
    trial_id: int
    n_trials: int
    active_sharpe: float  # annualised Sharpe of strategy minus benchmark
    hurdle_sharpe: float  # annualised expected-max Sharpe of a no-skill search
    dsr: float
    verdict: str  # pass | fail
    cagr: float
    benchmark_cagr: float
    tracking_error: float  # annualised standard deviation of active returns


async def _trial_sharpes(engine: AsyncEngine) -> dict[str, float]:
    """config_hash -> per-period active Sharpe of its latest recorded run."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT DISTINCT ON (config_hash) config_hash, metrics "
                "FROM quant_trials WHERE name LIKE :p "
                "ORDER BY config_hash, id DESC"
            ),
            {"p": PREFIX + "%"},
        )
        return {
            r.config_hash: float(r.metrics["active_sr_period"])
            for r in rows
            if r.metrics and r.metrics.get("active_sr_period") is not None
        }


async def record_backtest(
    engine: AsyncEngine,
    *,
    strategy: str,
    config: dict[str, Any],
    days: list[date],
    returns: FloatArray,
    benchmark: FloatArray,
    extra: dict[str, Any] | None = None,
) -> Assessment | None:
    """Record one run and judge it; None when the active series is degenerate."""
    active = np.asarray(returns, dtype=float) - np.asarray(benchmark, dtype=float)
    moments = _sharpe_and_moments(active)
    if moments is None:
        return None
    n, sr, skew, kurt = moments
    own_var = _sharpe_estimator_variance(n, sr, skew, kurt)

    others = await _trial_sharpes(engine)
    others[config_hash(config)] = sr
    n_trials = len(others)
    spread = float(np.var(list(others.values()), ddof=1)) if n_trials > 1 else 0.0
    sr_variance = max(spread, own_var)
    dsr = deflated_sharpe_ratio(active, n_trials, sr_variance)
    hurdle = expected_max_sharpe(sr_variance, n_trials)
    verdict = "pass" if dsr >= MIN_DSR else "fail"

    growth = float(np.prod(1.0 + np.asarray(returns)))
    bench_growth = float(np.prod(1.0 + np.asarray(benchmark)))
    years = n / _TRADING_DAYS
    cagr = growth ** (1 / years) - 1
    bench_cagr = bench_growth ** (1 / years) - 1
    tracking = float(np.std(active, ddof=1)) * math.sqrt(_TRADING_DAYS)
    metrics = {
        "sessions": n,
        "cagr": cagr,
        "benchmark_cagr": bench_cagr,
        "tracking_error": tracking,
        "active_sr_period": sr,
        "active_sharpe": sr * math.sqrt(_TRADING_DAYS),
        "skew": skew,
        "kurtosis": kurt,
        "n_trials": n_trials,
        "sr_variance": sr_variance,
        "dsr": dsr,
        **(extra or {}),
    }
    trial_id = await QuantTrialRepoImpl(engine).record_trial(
        name=PREFIX + strategy,
        kind="backtest",
        config=config,
        window=f"{days[0]}..{days[-1]} vs {BENCHMARK}",
        metrics=metrics,
        criterion=CRITERION,
        verdict=verdict,
    )
    return Assessment(
        trial_id=trial_id,
        n_trials=n_trials,
        active_sharpe=sr * math.sqrt(_TRADING_DAYS),
        hurdle_sharpe=hurdle * math.sqrt(_TRADING_DAYS),
        dsr=dsr,
        verdict=verdict,
        cagr=cagr,
        benchmark_cagr=bench_cagr,
        tracking_error=tracking,
    )
