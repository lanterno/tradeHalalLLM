"""First look at strategy S1: a long-only halal multi-factor portfolio.

Each month-end, every eligible name is scored on two classic long-only
factors, z-scored across the universe and summed:

* momentum 12-1: return over the last ~12 months excluding the last month
  (the most recent month mean-reverts);
* low volatility: minus the annualised volatility of the last 63 sessions.

The top ``n`` are bought equal-weight and held to the next month-end with
weights drifting; each rebalance pays ``cost_bps`` per side on turnover.
Results are compared with SPY, SPUS and HLAL over the same sessions.

Biases -- all three flatter the result, and the report prints them:
* survivorship: the universe is names listed today;
* halal look-ahead: today's screen decides eligibility in every past month;
* liquidity look-ahead: the universe was chosen by today's dollar volume.
So this is a first look at whether there is anything worth building the
full event-driven backtester for (plan 3.2), not a capital gate.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.sharpe_stats import probabilistic_sharpe_ratio

FloatArray = NDArray[np.float64]

_MOM_LOOKBACK = 252
_MOM_SKIP = 21
_VOL_LOOKBACK = 63
_TRADING_DAYS = 252
_REUSE_RATIO = 10.0

BIASES = (
    "survivorship: universe = names listed today",
    "halal look-ahead: today's screen used for every past month",
    "liquidity look-ahead: universe chosen by today's dollar volume",
)


@dataclass(frozen=True, slots=True)
class Prices:
    days: list[date]
    symbols: list[str]
    close: FloatArray  # (n_days, n_symbols), NaN where no bar


@dataclass(frozen=True, slots=True)
class Stats:
    total_return: float
    cagr: float
    volatility: float
    sharpe: float | None
    psr: float | None
    max_drawdown: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    days: list[date]
    returns: FloatArray  # daily portfolio returns, net of costs
    stats: Stats
    yearly: dict[int, float]
    avg_turnover: float
    holdings: dict[date, list[str]] = field(default_factory=dict)


async def load_prices(engine: AsyncEngine, symbols: list[str], *, since: date) -> Prices:
    """Adjusted closes ('all') for ``symbols`` from ``since``, as a day x symbol matrix."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT symbol, day, close FROM daily_bars "
                    "WHERE adjustment = 'all' AND symbol = ANY(:s) AND day >= :since"
                ),
                {"s": symbols, "since": since},
            )
        ).all()
    days = sorted({r.day for r in rows})
    syms = sorted({r.symbol for r in rows})
    di = {d: i for i, d in enumerate(days)}
    si = {s: i for i, s in enumerate(syms)}
    close = np.full((len(days), len(syms)), np.nan)
    for r in rows:
        close[di[r.day], si[r.symbol]] = r.close
    return Prices(days, syms, close)


def split_reused_tickers(
    prices: Prices, *, max_ratio: float = _REUSE_RATIO
) -> tuple[Prices, list[tuple[str, date]]]:
    """Drop each symbol's history before a one-day move of ``max_ratio`` or more.

    Alpaca keys bars by symbol, so a recycled ticker stitches two companies
    into one series with no gap between them: SN was Sanchez Energy at cents
    until SharkNinja listed on 2023-07-31, which reads as a single +11,600%
    day; relisted bankruptcies (VAL, CORZ, BTU, WOLF) look the same. The
    largest genuine one-day moves in the 2016-2026 store are under 7x, the
    stitches 12x-155x. Only the latest segment is kept, so the earlier
    company can neither be scored nor held. Returns the breaks it applied.
    """
    close = prices.close.copy()
    breaks: list[tuple[str, date]] = []
    for j, symbol in enumerate(prices.symbols):
        col = close[:, j]
        seen = np.flatnonzero(~np.isnan(col))
        if len(seen) < 2:
            continue
        ratio = col[seen[1:]] / col[seen[:-1]]
        jumps = np.flatnonzero((ratio >= max_ratio) | (ratio <= 1.0 / max_ratio))
        if len(jumps):
            first_kept = int(seen[jumps[-1] + 1])
            col[:first_kept] = np.nan
            breaks.append((symbol, prices.days[first_kept]))
    return Prices(prices.days, prices.symbols, close), breaks


def month_ends(days: list[date]) -> list[int]:
    """Indices of the last session of each calendar month."""
    return [i for i in range(len(days)) if i == len(days) - 1 or days[i + 1].month != days[i].month]


def _zscore(x: FloatArray) -> FloatArray:
    # A spread at float-noise level carries no information; z-scoring it would
    # turn 1e-17 rounding differences into full-strength "signal".
    sd = float(np.nanstd(x))
    scale = max(1.0, float(np.nanmax(np.abs(x))))
    return (x - np.nanmean(x)) / sd if sd > 1e-12 * scale else np.zeros_like(x)


def scores_at(close: FloatArray, t: int) -> FloatArray:
    """Composite factor score per symbol at session ``t`` (NaN = not eligible)."""
    if t < _MOM_LOOKBACK:
        return np.full(close.shape[1], np.nan)
    p_now = close[t - _MOM_SKIP]
    p_then = close[t - _MOM_LOOKBACK]
    momentum = p_now / p_then - 1.0
    window = close[t - _VOL_LOOKBACK : t + 1]
    daily = window[1:] / window[:-1] - 1.0
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns: not eligible anyway
        vol = np.nanstd(daily, axis=0) * math.sqrt(_TRADING_DAYS)
    complete = ~np.isnan(window).any(axis=0) & ~np.isnan(momentum) & ~np.isnan(close[t])
    score = np.full(close.shape[1], np.nan)
    if complete.sum() < 2:
        return score
    score[complete] = _zscore(momentum[complete]) + _zscore(-vol[complete])
    return score


def stats(returns: FloatArray) -> Stats:
    growth = np.cumprod(1.0 + returns)
    total = float(growth[-1] - 1.0)
    years = len(returns) / _TRADING_DAYS
    sd = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    peak = np.maximum.accumulate(growth)
    return Stats(
        total_return=total,
        cagr=(1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else 0.0,
        volatility=sd * math.sqrt(_TRADING_DAYS),
        sharpe=float(np.mean(returns)) / sd * math.sqrt(_TRADING_DAYS) if sd > 0 else None,
        psr=float(probabilistic_sharpe_ratio(returns)) if len(returns) > 30 else None,
        max_drawdown=float(np.min(growth / peak - 1.0)),
    )


def _yearly(days: list[date], returns: FloatArray) -> dict[int, float]:
    out: dict[int, float] = {}
    for d, r in zip(days, returns, strict=True):
        out[d.year] = (1.0 + out.get(d.year, 0.0)) * (1.0 + float(r)) - 1.0
    return out


def backtest(
    prices: Prices,
    *,
    eligible: set[str],
    top_n: int = 30,
    cost_bps: float = 10.0,
    start: date | None = None,
) -> BacktestResult:
    """Monthly-rebalanced top-``top_n`` equal-weight portfolio of ``eligible`` names."""
    close = prices.close
    allowed = np.array([s in eligible for s in prices.symbols])
    # Start at the first month-end that can be scored (a year of history):
    # earlier months would sit in cash and dilute every statistic.
    ends = [
        t
        for t in month_ends(prices.days)
        if t >= _MOM_LOOKBACK and (start is None or prices.days[t] >= start)
    ]
    n_days = len(prices.days)
    returns = np.zeros(n_days)
    weights = np.zeros(close.shape[1])  # current (drifting) weights
    turnovers: list[float] = []
    holdings: dict[date, list[str]] = {}
    first = ends[0] if ends else n_days
    for t in range(first, n_days):
        if t > first:
            prev, now = close[t - 1], close[t]
            day_ret = np.where(weights > 0, now / prev - 1.0, 0.0)
            day_ret = np.nan_to_num(day_ret)  # a missing bar holds its value
            gross = float(weights @ day_ret)
            returns[t] = gross
            weights = weights * (1.0 + day_ret)
            total = weights.sum()
            if total > 0:
                weights = weights / total
        if t in ends:
            score = scores_at(close, t)
            score[~allowed] = np.nan
            ranked = np.argsort(np.where(np.isnan(score), -np.inf, score))[::-1]
            picks = [i for i in ranked[:top_n] if not np.isnan(score[i])]
            target = np.zeros_like(weights)
            if picks:
                target[picks] = 1.0 / len(picks)
            turnover = float(np.abs(target - weights).sum())
            turnovers.append(turnover)
            returns[t] -= turnover * cost_bps / 10_000.0
            weights = target
            holdings[prices.days[t]] = [prices.symbols[i] for i in picks]
    # From the first rebalance day itself: that day carries the entry cost
    # (and no market return, since the book was empty the day before).
    window = slice(first, n_days)
    days = prices.days[window]
    r = returns[window]
    return BacktestResult(
        days=days,
        returns=r,
        stats=stats(r),
        yearly=_yearly(days, r),
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        holdings=holdings,
    )


def benchmark_returns(prices: Prices, symbol: str, days: list[date]) -> FloatArray | None:
    """Daily returns of one stored symbol over exactly ``days`` (None if not stored)."""
    if symbol not in prices.symbols:
        return None
    col = prices.close[:, prices.symbols.index(symbol)]
    by_day = dict(zip(prices.days, col, strict=True))
    series = np.array([by_day.get(d, np.nan) for d in days])
    prev_idx = prices.days.index(days[0]) - 1
    prev = by_day.get(prices.days[prev_idx], np.nan) if prev_idx >= 0 else np.nan
    full = np.concatenate([[prev], series])
    out = full[1:] / full[:-1] - 1.0
    return None if np.isnan(out).any() else out
