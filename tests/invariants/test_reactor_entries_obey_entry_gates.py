"""INVARIANT: news-reactor ("fast in") entries obey the cycle's entry gates.

The reactor enters between 15-minute cycles. It used to skip the daily loss
limit, the risk engine's halt and the max-positions cap, and a failed
positions read became an empty book. Every gate here fails closed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.domain.models import Account, Position
from halal_trader.trading.executor import TradeExecutor
from halal_trader.trading.scheduler import TradingBot


@pytest.fixture(autouse=True)
def _kill_switch_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import halal_trader.core.halt as halt

    monkeypatch.setattr(halt, "is_halted", AsyncMock(return_value=False))


def _event(symbol: str = "NVDA") -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        title="beat and raise",
        classification=SimpleNamespace(score=0.95, tag="earnings", rationale="strong"),
    )


def _bot(
    *,
    loss_halt: bool | Exception = False,
    risk_halt: str | None = None,
    positions: list[Position] | Exception | None = None,
) -> tuple[TradingBot, MagicMock]:
    bot = TradingBot.__new__(TradingBot)
    bot._engine = object()
    bot.broker = MagicMock()
    bot.broker.get_clock = AsyncMock(return_value=SimpleNamespace(is_open=True))
    if isinstance(positions, Exception):
        bot.broker.get_all_positions = AsyncMock(side_effect=positions)
    else:
        bot.broker.get_all_positions = AsyncMock(return_value=positions or [])
    bot.portfolio = MagicMock()
    if isinstance(loss_halt, Exception):
        bot.portfolio.should_halt_trading = AsyncMock(side_effect=loss_halt)
    else:
        bot.portfolio.should_halt_trading = AsyncMock(return_value=loss_halt)
    bot.cycle_service = SimpleNamespace(last_risk_halt=risk_halt)
    executor = MagicMock()
    executor.execute_reactor_entry = AsyncMock(return_value={"status": "filled", "quantity": 5})
    bot.executor = executor
    return bot, executor


async def test_entry_proceeds_when_every_gate_is_clear() -> None:
    bot, executor = _bot()

    result, _ = await bot._maybe_execute_reactor_entry(_event())

    assert result is not None
    executor.execute_reactor_entry.assert_awaited_once()


@pytest.mark.parametrize(
    ("kwargs", "note"),
    [
        ({"loss_halt": True}, "daily loss limit"),
        ({"loss_halt": ConnectionError("broker down")}, "daily P&L unknown"),
        ({"risk_halt": "drawdown 9% > 8%"}, "risk engine halt"),
        ({"positions": ConnectionError("broker down")}, "positions unknown"),
    ],
)
async def test_a_tripped_or_unknown_gate_blocks_the_entry(kwargs: dict, note: str) -> None:
    bot, executor = _bot(**kwargs)

    result, status = await bot._maybe_execute_reactor_entry(_event())

    assert result is None
    assert note in status
    executor.execute_reactor_entry.assert_not_awaited()


def _position(symbol: str) -> Position:
    return Position(symbol=symbol, qty=10, avg_entry_price=100.0, current_price=101.0)


def _executor(max_positions: int) -> tuple[TradeExecutor, MagicMock]:
    broker = MagicMock()
    broker.get_account_info = AsyncMock(
        return_value=Account(
            equity=100_000,
            buying_power=100_000,
            cash=100_000,
            portfolio_value=100_000,
            status="ACTIVE",
        )
    )
    broker.get_stock_snapshot = AsyncMock(return_value={})
    return (
        TradeExecutor(
            broker, MagicMock(), max_position_pct=0.2, max_simultaneous_positions=max_positions
        ),
        broker,
    )


async def test_a_new_name_is_refused_at_the_max_positions_cap() -> None:
    executor, broker = _executor(max_positions=2)

    result = await executor.execute_reactor_entry(
        "NVDA", score=0.95, reasoning="x", positions=[_position("AAPL"), _position("MSFT")]
    )

    assert result["status"] == "rejected"
    assert "max simultaneous positions" in result["reason"]
    broker.get_stock_snapshot.assert_not_awaited()


async def test_adding_to_a_held_name_is_not_a_new_position() -> None:
    executor, broker = _executor(max_positions=2)

    result = await executor.execute_reactor_entry(
        "AAPL", score=0.95, reasoning="x", positions=[_position("AAPL"), _position("MSFT")]
    )

    assert "max simultaneous positions" not in result.get("reason", "")
    broker.get_stock_snapshot.assert_awaited_once()  # went on to price it
