"""Tests for :class:`PortfolioTracker`'s daily P&L methods.

The broker/DB-backed path is exercised in
``tests/invariants/test_daily_loss_limit_survives_restarts.py``. The P&L
math, halt threshold and day-end summary shape are pure once the broker
and repo accessors are stubbed.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from halal_trader.trading.portfolio import PortfolioTracker


class _StubPortfolio(PortfolioTracker):
    """Tracker whose broker/DB accessors record what they saw."""

    def __init__(
        self,
        *,
        equity: float,
        trades: list[dict] | None = None,
        daily_loss_limit: float = 0.05,
    ) -> None:
        super().__init__(MagicMock(), MagicMock(), daily_loss_limit=daily_loss_limit)
        self._equity = equity
        self._trades = trades or []
        self.persisted_start: float | None = None
        self.persisted_end: tuple[float, float, int] | None = None

    async def _get_equity(self) -> float:
        return self._equity

    async def _get_today_trades(self) -> list[dict[str, Any]]:
        return list(self._trades)

    async def _persist_day_start(self, equity: float) -> None:
        self.persisted_start = equity

    async def _persist_day_end(self, equity: float, pnl: float, count: int) -> None:
        self.persisted_end = (equity, pnl, count)


@pytest.mark.asyncio
async def test_record_day_start_persists_and_remembers_starting_equity():
    p = _StubPortfolio(equity=100_000.0)
    out = await p.record_day_start()
    assert out == 100_000.0
    assert p.persisted_start == 100_000.0
    assert p._starting_equity == 100_000.0


@pytest.mark.asyncio
async def test_record_day_end_returns_summary_with_realized_pnl():
    """Summary includes the canonical keys the Telegram notifier reads."""
    p = _StubPortfolio(equity=100_000.0, trades=[{}, {}, {}])
    await p.record_day_start()
    p._equity = 102_500.0  # day moved up $2500
    summary = await p.record_day_end()
    assert summary["starting_equity"] == 100_000.0
    assert summary["ending_equity"] == 102_500.0
    assert summary["realized_pnl"] == 2_500.0
    assert summary["trades_count"] == 3
    assert abs(summary["return_pct"] - 0.025) < 1e-9


@pytest.mark.asyncio
async def test_record_day_end_persists_to_subclass_hook():
    p = _StubPortfolio(equity=100_000.0)
    await p.record_day_start()
    p._equity = 99_000.0
    await p.record_day_end()
    assert p.persisted_end == (99_000.0, -1_000.0, 0)


@pytest.mark.asyncio
async def test_record_day_end_handles_no_day_start():
    """If `record_day_start` was never called, P&L is zero (not NaN)."""
    p = _StubPortfolio(equity=100_000.0)
    summary = await p.record_day_end()
    assert summary["starting_equity"] == 100_000.0
    assert summary["ending_equity"] == 100_000.0
    assert summary["realized_pnl"] == 0.0
    assert summary["return_pct"] == 0.0


@pytest.mark.asyncio
async def test_get_current_pnl_uses_starting_equity():
    p = _StubPortfolio(equity=100_000.0)
    await p.record_day_start()
    p._equity = 100_500.0
    assert await p.get_current_pnl() == 500.0


@pytest.mark.asyncio
async def test_should_halt_trading_trips_at_loss_limit():
    p = _StubPortfolio(equity=100_000.0, daily_loss_limit=0.05)
    await p.record_day_start()
    # Down 5.5% — past the 5% limit.
    p._equity = 94_500.0
    assert await p.should_halt_trading() is True


@pytest.mark.asyncio
async def test_should_halt_trading_clear_at_smaller_loss():
    p = _StubPortfolio(equity=100_000.0, daily_loss_limit=0.05)
    await p.record_day_start()
    p._equity = 97_000.0  # -3%
    assert await p.should_halt_trading() is False


@pytest.mark.asyncio
async def test_should_halt_trading_ignores_gains():
    """A profitable day must never halt trading, regardless of magnitude."""
    p = _StubPortfolio(equity=100_000.0, daily_loss_limit=0.05)
    await p.record_day_start()
    p._equity = 110_000.0  # +10%
    assert await p.should_halt_trading() is False


@pytest.mark.asyncio
async def test_should_halt_trading_sets_the_baseline_before_day_start():
    """Before record_day_start ran, the first check records today's baseline
    (the equity it reads) and measures against it."""
    p = _StubPortfolio(equity=99_000.0, daily_loss_limit=0.05)
    assert await p.should_halt_trading() is False
    assert p.persisted_start == 99_000.0


@pytest.mark.asyncio
async def test_equity_the_broker_cannot_report_is_never_a_baseline():
    """An all-zero account (a refused read) raises; it never becomes a $100k baseline."""
    from unittest.mock import AsyncMock

    from halal_trader.domain.models import Account

    broker = MagicMock(get_account_info=AsyncMock(return_value=Account(status="")))
    repo = MagicMock(start_day=AsyncMock(return_value=None))
    p = PortfolioTracker(broker, repo, daily_loss_limit=0.02)
    with pytest.raises(RuntimeError, match="no equity"):
        await p.record_day_start()
    with pytest.raises(RuntimeError, match="no equity"):
        await p.should_halt_trading()
    repo.start_day.assert_not_awaited()
    assert p._starting_equity is None
