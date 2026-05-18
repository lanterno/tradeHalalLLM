"""Tests for :class:`BasePortfolioTracker`'s template methods.

The crypto and stock portfolio trackers both inherit from this class
but only the broker-specific subclasses are exercised in DB-backed
tests. The template methods themselves (P&L math, halt threshold,
day-end summary shape) are pure once the abstract hooks are mocked.
"""

from typing import Any

import pytest

from halal_trader.core.portfolio import BasePortfolioTracker


class _StubPortfolio(BasePortfolioTracker):
    """Minimal subclass that records what each hook saw."""

    def __init__(
        self,
        *,
        equity: float,
        trades: list[dict] | None = None,
        daily_loss_limit: float = 0.05,
    ) -> None:
        super().__init__(daily_loss_limit=daily_loss_limit)
        self._equity = equity
        self._trades = trades or []
        self.persisted_start: float | None = None
        self.persisted_end: tuple[float, float, int] | None = None

    async def _get_equity(self, **_kwargs: Any) -> float:
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
async def test_should_halt_trading_falls_back_to_default_equity_before_day_start():
    """Before record_day_start is called, the loss percentage is
    computed against the safe `_DEFAULT_EQUITY` constant rather than
    raising on the `None` starting equity."""
    p = _StubPortfolio(equity=99_000.0, daily_loss_limit=0.05)
    # No record_day_start — _starting_equity is None.
    # Loss = 99_000 - None? get_current_pnl returns 0 (uses equity for both).
    assert await p.should_halt_trading() is False
