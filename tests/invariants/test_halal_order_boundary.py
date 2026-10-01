"""INVARIANT: no BUY reaches the broker for a symbol not screened halal.

Halal compliance is non-negotiable, so the rule is enforced where orders are
placed (TradeExecutor._execute_buy, which every BUY passes: the LLM cycle and
the news reactor alike), not only in the symbol list the prompt is shown.
It fails CLOSED: a screen that cannot be read refuses the order.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader import market_hours
from halal_trader.domain.models import Account, TradeAction, TradeDecision
from halal_trader.trading.executor import TradeExecutor


class FakeScreener:
    """ComplianceScreener with a fixed halal set; optionally broken."""

    def __init__(self, halal: set[str], *, broken: bool = False) -> None:
        self.halal = halal
        self.broken = broken
        self.asked: list[str] = []

    async def is_halal(self, symbol: str) -> bool:
        self.asked.append(symbol)
        if self.broken:
            raise ConnectionError("halal cache unreachable")
        return symbol in self.halal

    async def ensure_cache(self, symbols: list[str] | None = None) -> None: ...

    async def get_halal_symbols(self) -> list[str]:
        return sorted(self.halal)

    async def filter_halal(self, symbols: list[str]) -> list[str]:
        return [s for s in symbols if s in self.halal]


@pytest.fixture(autouse=True)
def _mid_session_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = datetime(2026, 9, 30, 10, 30, tzinfo=market_hours.MARKET_TZ)
    monkeypatch.setattr(market_hours, "now_eastern", lambda: pinned)


def _broker() -> MagicMock:
    account = Account(
        equity=100_000, buying_power=100_000, cash=100_000, portfolio_value=100_000, status="ACTIVE"
    )
    b = MagicMock()
    b.get_account_info = AsyncMock(return_value=account)
    b.get_stock_snapshot = AsyncMock(
        side_effect=lambda sym: {
            sym: {
                "latestTrade": {"p": 201.0},
                "prevDailyBar": {"c": 200.0},
                "dailyBar": {"o": 200.0, "c": 201.0},
            }
        }
    )
    b.place_order = AsyncMock(return_value={"id": "ord-1", "status": "filled"})
    b.get_order_by_id = AsyncMock(
        return_value={
            "id": "ord-1",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "201",
            "filled_at": "2026-09-30T14:30:00Z",
        }
    )
    return b


def _executor(broker: MagicMock, screener: FakeScreener) -> TradeExecutor:
    repo = MagicMock()
    repo.record_trade = AsyncMock(return_value=1)
    return TradeExecutor(
        broker,
        repo,
        max_position_pct=1.0,
        max_simultaneous_positions=10,
        max_sector_pct=0,
        recent_close_cooldown_minutes=0,
        screener=screener,
    )


def _buy(symbol: str) -> TradeDecision:
    return TradeDecision(
        action=TradeAction.BUY, symbol=symbol, quantity=10, confidence=0.8, reasoning="t"
    )


async def test_llm_buy_outside_the_halal_universe_never_reaches_the_broker() -> None:
    broker = _broker()
    executor = _executor(broker, FakeScreener({"AAPL", "NVDA"}))

    result = await executor._execute_buy(_buy("XYZQ"), positions=[])  # hallucinated ticker

    assert result["status"] == "rejected"
    assert "halal" in result["reason"]
    broker.place_order.assert_not_awaited()
    broker.get_account_info.assert_not_awaited()  # refused before any broker call


async def test_halal_buy_still_goes_through() -> None:
    broker = _broker()
    screener = FakeScreener({"NVDA"})
    executor = _executor(broker, screener)

    result = await executor._execute_buy(_buy("NVDA"), positions=[])

    assert result["status"] == "filled"
    assert screener.asked == ["NVDA"]
    broker.place_order.assert_awaited_once()


async def test_unreadable_screen_fails_closed() -> None:
    broker = _broker()
    executor = _executor(broker, FakeScreener({"NVDA"}, broken=True))

    result = await executor._execute_buy(_buy("NVDA"), positions=[])

    assert result["status"] == "rejected"
    assert "failing closed" in result["reason"]
    broker.place_order.assert_not_awaited()


async def test_news_reactor_entry_is_gated_too() -> None:
    """The reactor's watchlist is read once at startup; a name the screen
    has since dropped must still be refused at order time."""
    broker = _broker()
    executor = _executor(broker, FakeScreener({"AAPL"}))  # NVDA dropped since startup

    result = await executor.execute_reactor_entry("NVDA", score=0.95, reasoning="beat")

    assert result["status"] == "rejected"
    assert "halal" in result["reason"]
    broker.place_order.assert_not_awaited()


def test_the_live_composition_root_wires_the_screener_into_the_executor() -> None:
    """A gate that is never wired protects nothing. TradingBot builds the only
    production TradeExecutor; it must hand it the screener."""
    import ast
    from pathlib import Path

    import halal_trader.trading.scheduler as scheduler

    tree = ast.parse(Path(scheduler.__file__).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TradeExecutor"
    ]
    assert calls, "TradeExecutor is no longer built in trading/scheduler.py -- update this test"
    for call in calls:
        assert "screener" in {kw.arg for kw in call.keywords}, (
            f"TradeExecutor(...) at scheduler.py:{call.lineno} is built without screener="
        )
