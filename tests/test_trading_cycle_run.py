"""TradingCycleService._run_cycle_impl: what one cycle shows the strategy, and what it executes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import Trade
from halal_trader.domain.models import (
    Account,
    Position,
    TradeAction,
    TradeDecision,
    TradingPlan,
)
from halal_trader.trading.cycle import TradingCycleService


class _Broker:
    def __init__(self, positions: list[Position]) -> None:
        self.positions = positions
        self.position_reads = 0

    async def get_account_info(self) -> Account:
        return Account(equity=100_000, buying_power=50_000, cash=50_000, portfolio_value=100_000)

    async def get_all_positions(self) -> list[Position]:
        self.position_reads += 1
        return self.positions

    async def get_stock_snapshot(self, symbol: str) -> dict[str, Any]:
        return {"symbol": symbol}

    async def get_stock_bars(self, symbol: str, *, days: int, timeframe: str) -> dict[str, Any]:
        return {"bars": {symbol: []}}


class _Screener:
    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols

    async def get_halal_symbols(self) -> list[str]:
        return list(self.symbols)


class _Strategy:
    def __init__(self, plan: TradingPlan) -> None:
        self.plan = plan
        self.calls: list[dict[str, Any]] = []

    async def analyze(self, **kwargs: Any) -> TradingPlan:
        self.calls.append(kwargs)
        return self.plan


class _Executor:
    # The cycle reads the executor's gate windows to know which symbols it would refuse.
    _recent_close_cooldown_minutes = 30
    _stop_loss_reentry_cooldown_minutes = 120

    def __init__(self) -> None:
        self.plans: list[tuple[TradingPlan, list[Position]]] = []

    async def execute_plan(self, plan: TradingPlan, *, positions: list[Position]) -> list[Any]:
        self.plans.append((plan, positions))
        return []


def _buy(symbol: str) -> TradingPlan:
    decision = TradeDecision(
        action=TradeAction.BUY, symbol=symbol, quantity=1, confidence=0.7, reasoning="test"
    )
    return TradingPlan(decisions=[decision], market_outlook="test")


def _service(
    *,
    symbols: list[str],
    positions: list[Position] | None = None,
    plan: TradingPlan | None = None,
    engine: AsyncEngine | None = None,
    live_mode_checker: Any = None,
) -> tuple[TradingCycleService, _Broker, _Strategy, _Executor]:
    broker = _Broker(positions or [])
    strategy = _Strategy(plan if plan is not None else TradingPlan())
    executor = _Executor()
    portfolio = AsyncMock()
    portfolio.get_current_pnl.return_value = 12.5
    svc = TradingCycleService(
        broker=broker,  # type: ignore[arg-type]
        screener=_Screener(symbols),  # type: ignore[arg-type]
        strategy=strategy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        portfolio=portfolio,
        engine=engine,
        live_mode_checker=live_mode_checker,
    )
    return svc, broker, strategy, executor


async def test_a_cycle_shows_the_strategy_the_screen_and_executes_its_plan() -> None:
    held = [Position(symbol="AAPL", qty=3, avg_entry_price=200, current_price=201)]
    plan = _buy("MSFT")
    svc, _, strategy, executor = _service(symbols=["AAPL", "MSFT"], positions=held, plan=plan)

    await svc._run_cycle_impl()

    [call] = strategy.calls
    assert call["halal_symbols"] == ["AAPL", "MSFT"]
    assert set(call["snapshots"]) == set(call["bars"]) == {"AAPL", "MSFT"}
    assert call["positions"] == held
    assert call["today_pnl"] == 12.5
    assert executor.plans == [(plan, held)]


async def test_an_empty_plan_executes_nothing() -> None:
    svc, _, strategy, executor = _service(symbols=["AAPL"])

    await svc._run_cycle_impl()

    assert len(strategy.calls) == 1
    assert executor.plans == []


async def test_no_halal_symbols_means_no_llm_call() -> None:
    svc, _, strategy, executor = _service(symbols=[], plan=_buy("AAPL"))

    await svc._run_cycle_impl()

    assert strategy.calls == [] and executor.plans == []


async def test_a_tripped_live_mode_safeguard_stops_the_cycle_before_it_reads_positions() -> None:
    checker = SimpleNamespace(active=True, assert_safe=AsyncMock(return_value=False))
    svc, broker, strategy, executor = _service(
        symbols=["AAPL"], plan=_buy("AAPL"), live_mode_checker=checker
    )

    await svc._run_cycle_impl()

    checker.assert_safe.assert_awaited_once()
    assert broker.position_reads == 0
    assert strategy.calls == [] and executor.plans == []


async def test_a_risk_engine_halt_stops_the_cycle_before_the_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from halal_trader.trading import risk

    halted = SimpleNamespace(
        is_halted=True,
        halt_reason="drawdown 9% over the 8% limit",
        portfolio_heat_pct=4.0,
        drawdown_pct=9.0,
        avg_correlation=0.3,
    )
    monkeypatch.setattr(
        risk,
        "evaluate_stock_risk",
        lambda **_: SimpleNamespace(risk_text="halted", indicators_by_symbol={}, state=halted),
    )
    svc, _, strategy, executor = _service(symbols=["AAPL"], plan=_buy("AAPL"))

    await svc._run_cycle_impl()

    assert strategy.calls == [] and executor.plans == []
    assert svc.last_risk_halt == "drawdown 9% over the 8% limit"
    assert svc.last_risk_snapshot is not None and svc.last_risk_snapshot["is_halted"] is True


async def _store(engine: AsyncEngine, *trades: Trade) -> None:
    async with AsyncSession(engine) as session:
        session.add_all(trades)
        await session.commit()


def _closed(symbol: str, *, minutes_ago: float, exit_reason: str) -> Trade:
    closed_at = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    return Trade(
        symbol=symbol,
        side="buy",
        quantity=1,
        price=100,
        status="filled",
        timestamp=closed_at - timedelta(hours=1),
        exit_price=99,
        exit_reason=exit_reason,
        closed_at=closed_at,
    )


async def test_a_symbol_inside_a_buy_gate_leaves_the_universe_unless_it_is_held(
    engine: AsyncEngine,
) -> None:
    await _store(
        engine,
        _closed("ADBE", minutes_ago=90, exit_reason="stop_loss"),  # inside the 120-min re-entry
        _closed("INTU", minutes_ago=10, exit_reason="take_profit"),  # inside the 30-min cooldown
        _closed("MSFT", minutes_ago=45, exit_reason="take_profit"),  # past its 30 minutes
    )
    held = [Position(symbol="INTU", qty=1, avg_entry_price=500, current_price=505)]
    svc, _, strategy, _ = _service(
        symbols=["ADBE", "INTU", "MSFT", "AAPL"], positions=held, engine=engine
    )

    await svc._run_cycle_impl()

    [call] = strategy.calls
    # ADBE can't be bought, so it isn't shown as buyable; INTU is held, so it stays visible.
    assert call["halal_symbols"] == ["INTU", "MSFT", "AAPL"]
    assert "ADBE" not in call["snapshots"] and "ADBE" not in call["bars"]
    assert {"INTU", "MSFT", "AAPL"} <= set(call["snapshots"])
    # The exit itself stays in the prompt: it explains why ADBE is gone.
    assert "ADBE" in call["recent_closed_text"]


async def test_the_strategy_sees_how_long_each_holding_has_been_open(engine: AsyncEngine) -> None:
    now = datetime.now(UTC)
    await _store(
        engine,
        Trade(
            symbol="AAPL",
            side="buy",
            quantity=1,
            status="filled",
            timestamp=now - timedelta(minutes=45),
        ),
        Trade(
            symbol="AAPL",
            side="buy",
            quantity=1,
            status="filled",
            timestamp=now - timedelta(minutes=10),
        ),
        Trade(
            symbol="NVDA",
            side="buy",
            quantity=1,
            status="filled",
            timestamp=now - timedelta(minutes=200),
        ),
    )
    held = [Position(symbol="AAPL", qty=2), Position(symbol="NVDA", qty=1)]
    svc, _, strategy, _ = _service(symbols=["AAPL", "NVDA"], positions=held, engine=engine)

    await svc._run_cycle_impl()

    minutes = strategy.calls[0]["holding_minutes_by_symbol"]
    # The youngest open buy sets the clock: that is the one the min-hold gate checks.
    assert set(minutes) == {"AAPL", "NVDA"}
    assert 9 < minutes["AAPL"] < 12
    assert 199 < minutes["NVDA"] < 202
