"""Walk-forward + Monte Carlo wrappers around the backtest engine.

A single full-history backtest tells you whether a strategy *would have*
made money on the precise trade order it found — which is dangerous,
because that ordering is one realisation out of many. Two upgrades:

* **Walk-forward** — split the history into rolling train/test windows
  and aggregate the *out-of-sample* performance. Catches strategies
  that look good only because they overfit recent regimes.
* **Monte Carlo** — given a closed-trade list, shuffle the trade order
  many times and recompute the equity curve to surface the
  drawdown-distribution rather than just the realised one. Catches
  strategies whose Sharpe is dominated by a single lucky run.

Both are pure functions over the existing :class:`BacktestResult`
shape, so they apply to crypto and stock backtests interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

import numpy as np

from halal_trader.crypto.backtest import BacktestResult, SimulatedTrade
from halal_trader.domain.models import Kline

BacktestFn = Callable[[str, list[Kline]], Awaitable[BacktestResult]]


@dataclass(frozen=True)
class WalkForwardWindow:
    """One train/test fold of a walk-forward run."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass
class WalkForwardReport:
    """Aggregate result of all out-of-sample fold runs."""

    folds: list[BacktestResult]
    avg_return_pct: float
    avg_sharpe: float
    win_rate: float
    fold_count: int
    # Mean out-of-sample Probabilistic Sharpe across folds (0..1). Low values
    # mean the per-fold edge is statistically indistinguishable from noise.
    avg_psr: float = 0.0
    # Mean out-of-sample 5% Conditional VaR across folds (negative = loss).
    avg_cvar_5pct: float = 0.0


def split_walk_forward(
    n_klines: int,
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[WalkForwardWindow]:
    """Generate non-overlapping (or stepped) train/test windows.

    ``step`` defaults to ``test_size`` (i.e. every fold's test window
    starts where the previous fold's test window ended — non-overlapping
    out-of-sample sets, the standard walk-forward layout).
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step or test_size
    out: list[WalkForwardWindow] = []
    cursor = 0
    while cursor + train_size + test_size <= n_klines:
        out.append(
            WalkForwardWindow(
                train_start=cursor,
                train_end=cursor + train_size,
                test_start=cursor + train_size,
                test_end=cursor + train_size + test_size,
            )
        )
        cursor += step
    return out


async def run_walk_forward(
    pair: str,
    klines: list[Kline],
    *,
    backtest_fn: BacktestFn,
    train_size: int,
    test_size: int,
    step: int | None = None,
    warmup: int = 100,
) -> WalkForwardReport:
    """Run ``backtest_fn`` over each test window and aggregate the results.

    Each fold feeds ``backtest_fn`` exactly ``warmup`` bars *before* the test
    window (for indicator context only) followed by the test window itself.
    ``BacktestEngine.run`` starts trading at index ``window_size`` of its input,
    so when ``warmup`` equals the engine's ``window_size`` the engine begins
    trading exactly at ``test_start`` and the fold result is **pure
    out-of-sample**.

    This fixes a leakage bug: the previous ``klines[train_start:test_end]``
    slice handed the engine the whole train+test span, so it traded the last
    ``window_size`` *training* bars too and folded their in-sample performance
    into the "out-of-sample" metrics. ``warmup`` MUST equal the backtest
    engine's ``window_size`` — pass them together (see research_jobs).
    """
    folds = split_walk_forward(len(klines), train_size=train_size, test_size=test_size, step=step)
    fold_results: list[BacktestResult] = []
    for w in folds:
        # Exactly ``warmup`` context bars before the test window, then the test
        # window — so the engine trades only [test_start, test_end). Skip a fold
        # that lacks a full warmup prefix (can't be cleanly evaluated).
        start = w.test_start - warmup
        if start < 0:
            continue
        slice_ = klines[start : w.test_end]
        if len(slice_) < 2:
            continue
        result = await backtest_fn(pair, slice_)
        fold_results.append(result)

    if not fold_results:
        return WalkForwardReport(
            folds=[], avg_return_pct=0.0, avg_sharpe=0.0, win_rate=0.0, fold_count=0
        )

    avg_ret = float(np.mean([r.total_return_pct for r in fold_results]))
    avg_sharpe = float(np.mean([r.sharpe_ratio for r in fold_results]))
    avg_psr = float(np.mean([r.psr for r in fold_results]))
    avg_cvar = float(np.mean([r.cvar_5pct for r in fold_results]))
    winning_folds = sum(1 for r in fold_results if r.total_return_pct > 0)
    win_rate = winning_folds / len(fold_results)
    return WalkForwardReport(
        folds=fold_results,
        avg_return_pct=avg_ret,
        avg_sharpe=avg_sharpe,
        avg_psr=avg_psr,
        avg_cvar_5pct=avg_cvar,
        win_rate=win_rate,
        fold_count=len(fold_results),
    )


# ── Monte Carlo ─────────────────────────────────────────────────


@dataclass(frozen=True)
class MonteCarloReport:
    """Distribution of outcomes over shuffled trade orderings."""

    runs: int
    final_return_pct_mean: float
    final_return_pct_p5: float  # 5th percentile (worst-case proxy)
    final_return_pct_p95: float
    max_drawdown_pct_mean: float
    max_drawdown_pct_p95: float  # bigger is worse → 95th percentile is the bad tail


def monte_carlo_resample(
    trades: Sequence[SimulatedTrade],
    *,
    initial_balance: float,
    runs: int = 500,
    seed: int | None = None,
) -> MonteCarloReport:
    """Shuffle trade order ``runs`` times and report the distribution.

    Each trade's P&L is added in the shuffled order; equity peaks/troughs
    are recomputed per shuffle. This breaks the time-correlation
    assumption built into the original equity curve, which surfaces
    *path*-dependent risk that point-in-time Sharpe hides.

    Returns mean + tail percentiles for both final return and max
    drawdown — the operator should care about the bad-tail more than
    the mean.
    """
    if runs <= 0:
        raise ValueError("runs must be positive")
    if not trades:
        return MonteCarloReport(
            runs=0,
            final_return_pct_mean=0.0,
            final_return_pct_p5=0.0,
            final_return_pct_p95=0.0,
            max_drawdown_pct_mean=0.0,
            max_drawdown_pct_p95=0.0,
        )

    pnls = np.array([t.pnl for t in trades], dtype=float)
    rng = np.random.default_rng(seed)
    final_returns: list[float] = []
    max_dds: list[float] = []
    for _ in range(runs):
        order = rng.permutation(len(pnls))
        equity = initial_balance + np.cumsum(pnls[order])
        equity = np.insert(equity, 0, initial_balance)
        peak = equity[0]
        max_dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        final_returns.append((equity[-1] - initial_balance) / initial_balance)
        max_dds.append(max_dd)

    return MonteCarloReport(
        runs=runs,
        final_return_pct_mean=float(np.mean(final_returns)),
        final_return_pct_p5=float(np.percentile(final_returns, 5)),
        final_return_pct_p95=float(np.percentile(final_returns, 95)),
        max_drawdown_pct_mean=float(np.mean(max_dds)),
        max_drawdown_pct_p95=float(np.percentile(max_dds, 95)),
    )
