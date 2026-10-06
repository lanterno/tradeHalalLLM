"""Tests for :meth:`TradeExecutor.execute_plan`'s sells-first-then-buys flow.

Sell every sell decision first, then run buys until
``max_simultaneous_positions`` is hit. The per-order paths are exercised
elsewhere; this file locks the orchestration contract with the order
methods stubbed out.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.domain.models import TradeAction, TradeDecision, TradingPlan
from halal_trader.trading.executor import TradeExecutor


def _decision(symbol: str, action: str) -> TradeDecision:
    return TradeDecision(
        action=TradeAction.BUY if action == "buy" else TradeAction.SELL,
        symbol=symbol,
        quantity=1,
        confidence=0.8,
        reasoning="t",
    )


def _plan(*decisions: TradeDecision) -> TradingPlan:
    return TradingPlan(decisions=list(decisions))


class _StubExecutor(TradeExecutor):
    """Records call order so tests can assert sells-before-buys."""

    def __init__(
        self,
        *,
        open_positions: int = 0,
        max_simultaneous_positions: int = 5,
        buy_status: str = "filled",
    ) -> None:
        broker = MagicMock()
        broker.get_all_positions = AsyncMock(return_value=[object()] * open_positions)
        super().__init__(
            broker,
            MagicMock(),
            max_position_pct=0.1,
            max_simultaneous_positions=max_simultaneous_positions,
        )
        self._buy_status = buy_status
        self.call_log: list[str] = []

    async def _execute_buy(self, decision: Any, **_kwargs: Any) -> dict[str, Any]:
        self.call_log.append(f"buy:{decision.symbol}")
        return {
            "symbol": decision.symbol,
            "action": "buy",
            "status": self._buy_status,
        }

    async def _execute_sell(self, decision: Any, **_kwargs: Any) -> dict[str, Any]:
        self.call_log.append(f"sell:{decision.symbol}")
        return {
            "symbol": decision.symbol,
            "action": "sell",
            "status": "filled",
        }


@pytest.mark.asyncio
async def test_executes_sells_before_buys_regardless_of_plan_order():
    """Sells always run first — frees up cash and a position slot."""
    plan = _plan(
        _decision("AAPL", "buy"),
        _decision("MSFT", "sell"),
        _decision("GOOG", "buy"),
        _decision("NVDA", "sell"),
    )
    e = _StubExecutor()
    await e.execute_plan(plan)
    # Both sells must come before either buy.
    sells_at = [i for i, c in enumerate(e.call_log) if c.startswith("sell:")]
    buys_at = [i for i, c in enumerate(e.call_log) if c.startswith("buy:")]
    assert max(sells_at) < min(buys_at)


@pytest.mark.asyncio
async def test_buys_stop_at_max_simultaneous_positions():
    """When the position cap is hit mid-plan, remaining buys are
    rejected with a 'max positions reached' reason — they don't
    silently drop, so the operator can see what was skipped."""
    plan = _plan(*(_decision(f"S{i}", "buy") for i in range(5)))
    e = _StubExecutor(open_positions=3, max_simultaneous_positions=4)
    results = await e.execute_plan(plan)
    # Only one buy should have actually executed (3 → 4 = cap hit).
    executed = [r for r in results if r["status"] == "filled"]
    rejected = [r for r in results if r["status"] == "rejected"]
    assert len(executed) == 1
    assert len(rejected) == 4
    assert all("Max simultaneous positions" in r["reason"] for r in rejected)


@pytest.mark.asyncio
async def test_open_count_increments_only_on_filled_or_submitted_buys():
    """A rejected buy doesn't take a position slot."""
    plan = _plan(_decision("S1", "buy"), _decision("S2", "buy"))
    # Buys "succeed" but with an unrecognised status — should NOT count
    # toward the cap.
    e = _StubExecutor(
        open_positions=4,
        max_simultaneous_positions=5,
        buy_status="rejected",  # not in {submitted, filled}
    )
    results = await e.execute_plan(plan)
    # Both buys executed (cap was 5, started at 4, neither incremented)
    assert all(r["status"] == "rejected" for r in results)
    assert len(e.call_log) == 2  # both buys ran


@pytest.mark.asyncio
async def test_empty_plan_returns_empty_results():
    e = _StubExecutor()
    results = await e.execute_plan(_plan())
    assert results == []
    assert e.call_log == []


@pytest.mark.asyncio
async def test_sell_only_plan_still_reads_position_count():
    """The flow reads the open-position count before the buys loop
    *unconditionally*, even for a pure sell-off (e.g. EOD). If we want to
    optimise it later we'd need to skip when buys is empty; for now lock
    the current behaviour."""
    plan = _plan(_decision("AAPL", "sell"), _decision("MSFT", "sell"))
    e = _StubExecutor()
    e._broker.get_all_positions = AsyncMock(side_effect=AssertionError("count read"))  # type: ignore[method-assign]

    with pytest.raises(AssertionError):
        await e.execute_plan(plan)
    assert e.call_log == ["sell:AAPL", "sell:MSFT"]
