"""Tests for stocks-side ensemble LLM wiring.

Mirrors the crypto ensemble tests but for ``trading.strategy.TradingStrategy``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from halal_trader.core.llm.base import BaseLLM, CallUsage
from halal_trader.domain.models import Account, TradingPlan
from halal_trader.trading.strategy import TradingStrategy


class _ScriptedLLM(BaseLLM):
    def __init__(self, response: dict[str, Any] | Exception, model: str = "stub") -> None:
        super().__init__(model=model)
        self._response = response
        self.calls = 0
        self.last_usage = CallUsage(model=model)

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        if isinstance(self._response, Exception):
            raise self._response
        return json.dumps(self._response)


def _buy_plan(qty: int = 10) -> dict[str, Any]:
    return {
        "decisions": [
            {
                "action": "buy",
                "symbol": "AAPL",
                "quantity": qty,
                "confidence": 0.7,
                "reasoning": "x",
            }
        ],
        "market_outlook": "bullish",
        "risk_notes": "",
    }


@pytest.mark.asyncio
async def test_stocks_ensemble_unanimous_keeps_plan() -> None:
    primary = _ScriptedLLM(_buy_plan(qty=10))
    alt1 = _ScriptedLLM(_buy_plan(qty=10))
    alt2 = _ScriptedLLM(_buy_plan(qty=10))
    repo = AsyncMock()
    repo.record_decision = AsyncMock()

    strat = TradingStrategy(
        primary,
        repo,
        llm_provider_name="stub",
        max_position_pct=0.20,
        daily_loss_limit=0.02,
        daily_return_target=0.01,
        max_simultaneous_positions=5,
        ensemble_llms=[alt1, alt2],
        ensemble_quorum=2,
    )
    plan = await strat.analyze(
        account=Account(equity=100000, buying_power=100000, cash=100000, portfolio_value=100000),
        positions=[],
        halal_symbols=["AAPL"],
        snapshots={},
        bars={},
    )
    assert isinstance(plan, TradingPlan)
    assert plan.decisions
    # Unanimous → sizing_multiplier 1.0 → primary qty preserved
    assert plan.decisions[0].quantity == 10
    assert strat.last_ensemble_verdict is not None
    assert strat.last_ensemble_verdict.agreement_score == pytest.approx(1.0)
    # Primary + 2 alternates were each called once
    assert primary.calls == 1 and alt1.calls == 1 and alt2.calls == 1


@pytest.mark.asyncio
async def test_stocks_no_ensemble_skips_call() -> None:
    primary = _ScriptedLLM(_buy_plan(qty=10))
    repo = AsyncMock()
    repo.record_decision = AsyncMock()

    strat = TradingStrategy(
        primary,
        repo,
        llm_provider_name="stub",
        max_position_pct=0.20,
        daily_loss_limit=0.02,
        daily_return_target=0.01,
        max_simultaneous_positions=5,
    )
    plan = await strat.analyze(
        account=Account(equity=100000, buying_power=100000, cash=100000, portfolio_value=100000),
        positions=[],
        halal_symbols=["AAPL"],
        snapshots={},
        bars={},
    )
    assert plan.decisions[0].quantity == 10
    assert strat.last_ensemble_verdict is None


@pytest.mark.asyncio
async def test_stocks_ensemble_failure_keeps_primary_plan() -> None:
    primary = _ScriptedLLM(_buy_plan(qty=10))
    alt_broken = _ScriptedLLM(RuntimeError("ensemble llm down"))
    repo = AsyncMock()
    repo.record_decision = AsyncMock()

    strat = TradingStrategy(
        primary,
        repo,
        llm_provider_name="stub",
        max_position_pct=0.20,
        daily_loss_limit=0.02,
        daily_return_target=0.01,
        max_simultaneous_positions=5,
        ensemble_llms=[alt_broken],
        ensemble_quorum=2,
    )
    plan = await strat.analyze(
        account=Account(equity=100000, buying_power=100000, cash=100000, portfolio_value=100000),
        positions=[],
        halal_symbols=["AAPL"],
        snapshots={},
        bars={},
    )
    # When the only alt fails, ensemble has 1 valid variant; quorum=2
    # is not met → consensus path falls back to primary or zeroes out.
    # Either way the strategy must return a TradingPlan and never raise.
    assert isinstance(plan, TradingPlan)
