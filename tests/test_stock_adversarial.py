"""Tests for stocks-side adversarial co-bot wiring.

Mirrors test_adversarial.py but for ``trading.strategy.TradingStrategy``.
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


@pytest.mark.asyncio
async def test_stocks_strategy_attacker_downsizes_buys() -> None:
    primary = _ScriptedLLM(
        {
            "decisions": [
                {
                    "action": "buy",
                    "symbol": "AAPL",
                    "quantity": 10,
                    "confidence": 0.7,
                    "reasoning": "earnings beat",
                }
            ],
            "market_outlook": "bullish",
            "risk_notes": "",
        }
    )
    attacker = _ScriptedLLM({"severity": 0.55, "counter_thesis": "iv too rich"})
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
        attacker_llm=attacker,
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
    # Attacker recommends downsize → 10 * 0.5 = 5
    assert plan.decisions[0].quantity == 5
    assert "adversarial" in plan.risk_notes
    assert strat.last_adversarial_review is not None
    assert strat.last_adversarial_review.recommendation == "downsize"


@pytest.mark.asyncio
async def test_stocks_strategy_no_attacker_unchanged() -> None:
    primary = _ScriptedLLM(
        {
            "decisions": [
                {
                    "action": "buy",
                    "symbol": "AAPL",
                    "quantity": 10,
                    "confidence": 0.7,
                    "reasoning": "x",
                }
            ],
            "market_outlook": "bullish",
            "risk_notes": "",
        }
    )
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
    assert strat.last_adversarial_review is None
