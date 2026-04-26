"""Stock executor sector-rotation cap integration test."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from halal_trader.domain.models import Account, Position, TradeAction, TradeDecision
from halal_trader.trading.executor import TradeExecutor


def _account(buying_power=10_000, portfolio_value=10_000) -> Account:
    return Account(
        equity=portfolio_value,
        buying_power=buying_power,
        cash=buying_power,
        portfolio_value=portfolio_value,
        status="ACTIVE",
    )


def _position(symbol: str, qty: float, price: float) -> Position:
    return Position(symbol=symbol, qty=qty, avg_entry_price=price, current_price=price)


def _broker(account: Account, *, snapshot_price: float = 100.0):
    b = MagicMock()
    b.get_account_info = AsyncMock(return_value=account)
    b.get_stock_snapshot = AsyncMock(return_value={"latest_trade": {"price": snapshot_price}})
    b.place_order = AsyncMock(return_value={"id": "ord-1", "status": "filled"})
    # Return an immediately-filled poll response so _confirm_fill doesn't
    # spend its full timeout.
    b.get_order_by_id = AsyncMock(
        return_value={
            "id": "ord-1",
            "status": "filled",
            "filled_qty": "30",
            "filled_avg_price": "100",
            "filled_at": "2026-04-26T15:00:00Z",
        }
    )
    return b


def _decision(symbol: str = "MSFT", quantity: int = 30) -> TradeDecision:
    return TradeDecision(
        action=TradeAction.BUY,
        symbol=symbol,
        quantity=quantity,
        confidence=0.8,
        reasoning="t",
    )


async def test_sector_cap_blocks_buy_that_concentrates_tech():
    """7k AAPL position + 3k MSFT buy = 100% Tech → reject under 40% cap."""
    account = _account(buying_power=10_000, portfolio_value=10_000)
    repo = MagicMock()
    broker = _broker(account, snapshot_price=100.0)
    executor = TradeExecutor(
        broker,
        repo,
        max_position_pct=1.0,  # disable per-position cap for this test
        max_simultaneous_positions=10,
        max_sector_pct=0.40,
    )
    positions = [_position("AAPL", qty=70, price=100)]  # 7000 in Tech
    result = await executor._execute_buy(_decision("MSFT", quantity=30), positions=positions)

    assert result["status"] == "rejected"
    assert "sector" in result["reason"].lower()
    broker.place_order.assert_not_called()


async def test_sector_cap_allows_buy_in_a_different_sector():
    account = _account(buying_power=10_000, portfolio_value=10_000)
    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=42)
    broker = _broker(account, snapshot_price=100.0)
    executor = TradeExecutor(
        broker,
        repo,
        max_position_pct=1.0,
        max_simultaneous_positions=10,
        max_sector_pct=0.40,
    )
    positions = [_position("AAPL", qty=70, price=100)]
    # JNJ is Healthcare, separate bucket from Tech.
    result = await executor._execute_buy(_decision("JNJ", quantity=10), positions=positions)
    assert result["status"] != "rejected"


async def test_sector_cap_disabled_when_max_set_to_zero():
    """An operator can disable the check by setting max_sector_pct=0."""
    account = _account(buying_power=10_000, portfolio_value=10_000)
    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=42)
    broker = _broker(account, snapshot_price=100.0)
    executor = TradeExecutor(
        broker,
        repo,
        max_position_pct=1.0,
        max_simultaneous_positions=10,
        max_sector_pct=0.0,  # disabled
    )
    positions = [_position("AAPL", qty=99, price=100)]  # massively concentrated
    result = await executor._execute_buy(_decision("MSFT", quantity=10), positions=positions)
    assert result["status"] != "rejected"


async def test_sector_cap_no_op_with_zero_equity():
    """Cold start (no portfolio value) shouldn't block trades."""
    account = _account(buying_power=10_000, portfolio_value=0)
    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=42)
    broker = _broker(account, snapshot_price=100.0)
    executor = TradeExecutor(
        broker,
        repo,
        max_position_pct=1.0,
        max_simultaneous_positions=10,
        max_sector_pct=0.40,
    )
    result = await executor._execute_buy(_decision(), positions=[])
    assert result["status"] != "rejected"
