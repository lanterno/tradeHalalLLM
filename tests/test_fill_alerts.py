"""Per-fill Telegram alerts carry what filled, and are actually wired."""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader import market_hours
from halal_trader.domain.models import Account, TradeAction, TradeDecision
from halal_trader.trading.executor import TradeExecutor


@pytest.fixture(autouse=True)
def _mid_session_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = datetime(2026, 9, 30, 10, 30, tzinfo=market_hours.MARKET_TZ)
    monkeypatch.setattr(market_hours, "now_eastern", lambda: pinned)


async def test_a_filled_buy_reports_its_fill_price_and_order_id() -> None:
    broker = MagicMock()
    broker.get_account_info = AsyncMock(
        return_value=Account(
            equity=1e5, buying_power=1e5, cash=1e5, portfolio_value=1e5, status="ACTIVE"
        )
    )
    broker.get_stock_snapshot = AsyncMock(return_value={"NVDA": {"latestTrade": {"p": 200.0}}})
    broker.place_order = AsyncMock(return_value={"id": "ord-9", "status": "filled"})
    broker.get_order_by_id = AsyncMock(
        return_value={
            "id": "ord-9",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "201.5",
            "filled_at": "2026-09-30T14:30:00Z",
        }
    )
    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=1)
    executor = TradeExecutor(broker, repo, max_position_pct=1.0, max_simultaneous_positions=5)
    decision = TradeDecision(
        action=TradeAction.BUY, symbol="NVDA", quantity=10, confidence=0.8, reasoning="t"
    )

    result = await executor._execute_buy(decision, positions=[])

    assert (result["price"], result["order_id"], result["filled_quantity"]) == (
        201.5,
        "ord-9",
        10.0,
    )


def test_the_live_composition_root_passes_the_notifier_to_the_cycle() -> None:
    import halal_trader.trading.scheduler as scheduler

    tree = ast.parse(Path(scheduler.__file__).read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "TradingCycleService"
    ]
    assert calls and all("notifier" in {k.arg for k in c.keywords} for c in calls)
