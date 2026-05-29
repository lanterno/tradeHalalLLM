"""Backtester — deterministic replay of synthetic bar series → metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halabot.analysis.backtest import Backtester
from halabot.cognition.bars import Bar
from halabot.policy.sizing import PolicyConfig

T0 = datetime(2026, 5, 1, 14, 0, tzinfo=UTC)  # a weekday, RTH


def _bars(prices: list[float], *, start: datetime = T0, step_min: int = 1) -> list[Bar]:
    return [
        Bar(o=c, h=c + 0.5, low=c - 0.5, c=c, v=1000.0, ts=start + timedelta(minutes=i * step_min))
        for i, c in enumerate(prices)
    ]


# Cold-start bands tuned to the raw-conviction scale (as the live shadow uses).
CFG = PolicyConfig(
    conviction_entry_band=0.25, conviction_exit_band=0.15,
    max_weight_per_asset=0.20, max_gross_exposure=1.0, target_rebalance_threshold=0.03,
)


@pytest.mark.asyncio
async def test_backtest_empty_is_noop():
    res = await Backtester(policy_config=CFG, trading_hours=False).run({})
    assert res.proposals == 0 and res.closed == 0


@pytest.mark.asyncio
async def test_backtest_rides_a_clean_uptrend_positive():
    # A steady uptrend → the engine should go long and the marked book is positive.
    up = _bars([100.0 + i for i in range(80)])
    res = await Backtester(policy_config=CFG, trading_hours=False).run({"NVDA": up})
    assert res.proposals >= 1
    assert res.closed >= 1
    assert res.total_return > 0.0  # rode the trend up
    assert res.max_drawdown >= 0.0


@pytest.mark.asyncio
async def test_backtest_downtrend_stays_mostly_flat():
    # Long-only: a pure downtrend should not accumulate a winning long book.
    down = _bars([180.0 - i for i in range(80)])
    res = await Backtester(policy_config=CFG, trading_hours=False).run({"DOWN": down})
    assert res.total_return <= 0.0001  # no positive edge from shorting (we can't)


@pytest.mark.asyncio
async def test_transaction_costs_reduce_returns():
    # The same run with a high cost should net a lower total return (churn penalty).
    up = _bars([100.0 + i for i in range(80)])
    free = await Backtester(policy_config=CFG, trading_hours=False, cost_bps=0.0).run({"NVDA": up})
    costly = await Backtester(
        policy_config=CFG, trading_hours=False, cost_bps=50.0
    ).run({"NVDA": up})
    assert costly.total_return < free.total_return  # costs bite each round-trip


@pytest.mark.asyncio
async def test_backtest_result_summary_renders():
    up = _bars([100.0 + i for i in range(60)])
    res = await Backtester(policy_config=CFG, trading_hours=False).run({"NVDA": up})
    s = res.summary()
    assert "proposals=" in s and "profit_factor=" in s
