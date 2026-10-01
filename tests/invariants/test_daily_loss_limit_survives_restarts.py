"""INVARIANT: the daily loss limit is anchored to the day's FIRST equity.

It used to be re-based on whatever equity the process saw when it started,
so a restart late in a losing day reset the 2% limit to zero; and a failed
pre-market left no baseline at all, so the limit could never trip. Real
Postgres: the anchor lives in daily_pnl, which is the point.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.db.repository import Repository
from halal_trader.domain.models import Account
from halal_trader.trading.portfolio import PortfolioTracker


def _broker(equity: float) -> MagicMock:
    b = MagicMock()
    b.get_account_info = AsyncMock(
        return_value=Account(
            equity=equity,
            buying_power=equity,
            cash=equity,
            portfolio_value=equity,
            status="ACTIVE",
        )
    )
    return b


def _tracker(engine: AsyncEngine, equity: float) -> PortfolioTracker:
    return PortfolioTracker(_broker(equity), Repository(engine), daily_loss_limit=0.02)


async def test_a_restart_keeps_the_mornings_baseline(engine: AsyncEngine) -> None:
    morning = _tracker(engine, 100_000.0)
    await morning.record_day_start()

    # The process restarts at 15:00 after losing 1.9% ...
    afternoon = _tracker(engine, 98_100.0)
    baseline = await afternoon.record_day_start()

    assert baseline == 100_000.0  # ... and keeps the morning's anchor
    assert await afternoon.get_current_pnl() == pytest.approx(-1_900.0)


async def test_the_limit_trips_on_the_days_loss_not_the_loss_since_restart(
    engine: AsyncEngine,
) -> None:
    await _tracker(engine, 100_000.0).record_day_start()
    restarted = _tracker(engine, 98_100.0)
    await restarted.record_day_start()

    restarted._broker = _broker(97_900.0)  # another 0.2% down after the restart

    assert await restarted.should_halt_trading() is True  # 2.1% on the day


async def test_a_missed_pre_market_still_gets_a_baseline(engine: AsyncEngine) -> None:
    tracker = _tracker(engine, 100_000.0)  # record_day_start never called

    assert await tracker.get_current_pnl() == 0.0
    tracker._broker = _broker(97_000.0)
    assert await tracker.should_halt_trading() is True


async def test_a_baseline_from_yesterday_is_never_reused(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    import halal_trader.core.portfolio as core_portfolio

    tracker = _tracker(engine, 100_000.0)
    await tracker.record_day_start()
    tracker._starting_date = date(2000, 1, 3)  # pretend it was set on an old day
    tracker._broker = _broker(90_000.0)

    # A fresh day re-anchors (to today's row, which already holds 100k here);
    # the point is that the stale in-memory date forces a re-read.
    calls: list[str] = []
    real = tracker.record_day_start

    async def spy() -> float:
        calls.append("record_day_start")
        return await real()

    monkeypatch.setattr(tracker, "record_day_start", spy)
    await tracker.get_current_pnl()

    assert calls == ["record_day_start"]
    assert tracker._starting_date == core_portfolio.today_eastern()
