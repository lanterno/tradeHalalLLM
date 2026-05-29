"""Backtester — deterministic replay of synthetic bar series → metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from halabot.analysis.backtest import Backtester, _Book
from halabot.cognition.bars import Bar, BarBuffer, BufferPriceSource
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


# ── exit ladder (Appendix-H rungs 5–6 wired into the book) ──
def _book(**kw) -> _Book:
    return _Book(win_threshold_pct=0.002, prices=BufferPriceSource(BarBuffer()), **kw)


async def _buy(book: _Book, asset: str, price: float, weight: float = 0.2) -> None:
    await book.on_proposal(
        SimpleNamespace(asset=asset, ts=T0, payload={"price": price, "weight_delta": weight})
    )


@pytest.mark.asyncio
async def test_exit_ladder_trend_break_cuts_winner():
    # A winner (price > entry) closing BELOW its SMA → trend-break exit (rung 5).
    book = _book(exit_ladder=True, trailing_pct=0.05)
    await _buy(book, "NVDA", 100.0)
    book.tick("NVDA", 110.0, 115.0)  # 110 > entry 100 (winner), 110 < SMA 115 → exit
    assert len(book.returns) == 1
    assert book.returns[0] == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_exit_ladder_trailing_stop_ratchets_then_exits():
    # Ratchet a trailing stop up on a new high (rung 6), then exit when price
    # falls back through it (rung 4, stop_loss on the ratcheted stop).
    book = _book(exit_ladder=True, trailing_pct=0.05)
    await _buy(book, "NVDA", 100.0)
    book.tick("NVDA", 120.0, 90.0)  # SMA below price → no trend-break; ratchet stop→114
    assert not book.returns  # still open
    book.tick("NVDA", 113.0, 90.0)  # 113 <= 114 trailing stop → exit
    assert len(book.returns) == 1
    assert book.returns[0] == pytest.approx(0.13)


@pytest.mark.asyncio
async def test_exit_ladder_off_is_a_noop():
    # With the ladder disabled, tick() never touches an open position.
    book = _book(exit_ladder=False)
    await _buy(book, "NVDA", 100.0)
    book.tick("NVDA", 50.0, 200.0)  # would trend-break if the ladder were on
    assert not book.returns


@pytest.mark.asyncio
async def test_exit_ladder_locks_gains_on_a_reversal_vs_off():
    # Up then a sharp reversal: the slow-out ladder cuts the winner on the
    # trend-break/trailing stop, so it never gives back as much as the
    # conviction-decay-only path. Ladder ON should not net WORSE here.
    bars = _bars([100.0 + i for i in range(60)] + [160.0 - 3.0 * i for i in range(30)])
    off = await Backtester(
        policy_config=CFG, trading_hours=False, exit_ladder=False
    ).run({"NVDA": bars})
    on = await Backtester(
        policy_config=CFG, trading_hours=False, exit_ladder=True, trailing_pct=0.05
    ).run({"NVDA": bars})
    assert on.closed >= 1
    assert on.total_return >= off.total_return - 1e-9
