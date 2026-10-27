"""Each buy records the risk meter's level the cycle read (HAL-13)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader import market_hours
from halal_trader.core import risk_meter
from halal_trader.domain.models import Account, TradeAction, TradeDecision
from halal_trader.trading.cycle import TradingCycleService
from halal_trader.trading.executor import TradeExecutor


@pytest.fixture(autouse=True)
def _mid_session_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """10:30 ET on a session day: clear of the last half hour's buy lockout."""
    pinned = datetime(2026, 9, 30, 10, 30, tzinfo=market_hours.MARKET_TZ)
    monkeypatch.setattr(market_hours, "now_eastern", lambda: pinned)


def _executor(place_order_return: object) -> tuple[TradeExecutor, MagicMock]:
    broker = MagicMock()
    broker.get_account_info = AsyncMock(
        return_value=Account(
            equity=100_000, buying_power=100_000, cash=100_000, portfolio_value=100_000
        )
    )
    broker.get_stock_snapshot = AsyncMock(return_value={"latest_trade": {"price": 200.0}})
    broker.place_order = AsyncMock(return_value=place_order_return)
    broker.get_order_by_id = AsyncMock(
        return_value={
            "id": "ord-1",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "200",
            "filled_at": "2026-09-30T14:30:00Z",
        }
    )
    repo = MagicMock()
    repo.get_open_trades = AsyncMock(return_value=[])
    repo.record_trade = AsyncMock(return_value=1)
    executor = TradeExecutor(
        broker, repo, max_position_pct=1.0, max_simultaneous_positions=10, max_sector_pct=0
    )
    return executor, repo


def _buy() -> TradeDecision:
    return TradeDecision(
        action=TradeAction.BUY, symbol="NVDA", quantity=10, confidence=0.8, reasoning="t"
    )


@pytest.mark.parametrize(
    ("response", "status"),
    [({"id": "ord-1", "status": "filled"}, "filled"), ("broker said no", "rejected")],
)
async def test_a_buy_row_carries_the_level_filled_or_refused(response, status) -> None:
    executor, repo = _executor(response)
    executor.risk_level = 2

    await executor._execute_buy(_buy(), positions=[])

    kwargs = repo.record_trade.await_args.kwargs
    assert (kwargs["status"], kwargs["risk_level"]) == (status, 2)


async def test_a_buy_before_any_cycle_has_no_level() -> None:
    executor, repo = _executor({"id": "ord-1", "status": "filled"})

    await executor._execute_buy(_buy(), positions=[])

    assert repo.record_trade.await_args.kwargs["risk_level"] is None


def _cycle(engine: object) -> TradingCycleService:
    broker = MagicMock()
    broker.get_account_info = AsyncMock(side_effect=RuntimeError("stop after the read"))
    return TradingCycleService(
        broker=broker,
        screener=MagicMock(),
        strategy=MagicMock(),
        executor=MagicMock(),
        portfolio=MagicMock(),
        engine=engine,  # type: ignore[arg-type]
    )


async def test_the_cycle_hands_the_meters_level_to_the_executor(engine, monkeypatch) -> None:
    monkeypatch.setattr(risk_meter, "SETTABLE", frozenset(risk_meter.LEVELS))
    await risk_meter.set_level(engine, 2, reason="r", set_by="op")
    cycle = _cycle(engine)

    with pytest.raises(RuntimeError, match="stop after the read"):
        await cycle._run_cycle_impl()

    assert cycle._executor.risk_level == 2


async def test_an_unreadable_meter_leaves_the_level_empty(engine, monkeypatch) -> None:
    monkeypatch.setattr(risk_meter, "get_status", AsyncMock(side_effect=OSError("db down")))

    assert await _cycle(engine)._read_risk_level() is None
