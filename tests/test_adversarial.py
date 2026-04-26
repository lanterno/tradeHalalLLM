"""Tests for the adversarial co-bot."""

from __future__ import annotations

import json
from typing import Any

import pytest

from halal_trader.core.llm.adversarial import (
    AdversarialReview,
    apply_review_to_buys,
    critique_plan,
)
from halal_trader.core.llm.base import BaseLLM, CallUsage
from halal_trader.domain.models import (
    CryptoTradeDecision,
    TradeAction,
    TradeDecision,
)


class _ScriptedLLM(BaseLLM):
    """Returns scripted JSON responses; raises on demand."""

    def __init__(self, response: dict[str, Any] | Exception, model: str = "stub") -> None:
        super().__init__(model=model)
        self._response = response
        self.calls = 0
        # Mimic a small cost so cost_usd flows through.
        self.last_usage = CallUsage(model=model, cost_usd=0)  # type: ignore[arg-type]

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        if isinstance(self._response, Exception):
            raise self._response
        return json.dumps(self._response)


def _crypto_buy(symbol: str = "BTCUSDT", qty: float = 0.1) -> CryptoTradeDecision:
    return CryptoTradeDecision(
        action=TradeAction.BUY,
        symbol=symbol,
        quantity=qty,
        confidence=0.7,
        reasoning="momentum + volume",
    )


def _crypto_sell(symbol: str = "ETHUSDT", qty: float = 0.5) -> CryptoTradeDecision:
    return CryptoTradeDecision(
        action=TradeAction.SELL,
        symbol=symbol,
        quantity=qty,
        confidence=0.6,
        reasoning="trailing stop hit",
    )


@pytest.mark.asyncio
async def test_proceed_when_severity_low() -> None:
    llm = _ScriptedLLM({"severity": 0.2, "counter_thesis": "fine"})
    review = await critique_plan(llm, decisions=[_crypto_buy()])
    assert review.recommendation == "proceed"
    assert review.severity == 0.2
    assert review.sizing_multiplier == 1.0
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_downsize_in_mid_band() -> None:
    llm = _ScriptedLLM({"severity": 0.55, "counter_thesis": "RSI extended"})
    review = await critique_plan(llm, decisions=[_crypto_buy()])
    assert review.recommendation == "downsize"
    assert review.sizing_multiplier == 0.5


@pytest.mark.asyncio
async def test_skip_when_severity_high() -> None:
    llm = _ScriptedLLM({"severity": 0.9, "counter_thesis": "blow-off top"})
    review = await critique_plan(llm, decisions=[_crypto_buy()])
    assert review.recommendation == "skip"
    assert review.sizing_multiplier == 0.0


@pytest.mark.asyncio
async def test_attacker_failure_degrades_to_proceed() -> None:
    llm = _ScriptedLLM(RuntimeError("network down"))
    review = await critique_plan(llm, decisions=[_crypto_buy()])
    assert review.recommendation == "proceed"
    assert "attacker-error" in review.counter_thesis


@pytest.mark.asyncio
async def test_no_call_when_no_buys() -> None:
    llm = _ScriptedLLM({"severity": 1.0, "counter_thesis": "shouldn't run"})
    review = await critique_plan(llm, decisions=[_crypto_sell()])
    assert llm.calls == 0
    assert review.recommendation == "proceed"


@pytest.mark.asyncio
async def test_severity_clamped() -> None:
    llm = _ScriptedLLM({"severity": 5.0, "counter_thesis": "out of range"})
    review = await critique_plan(llm, decisions=[_crypto_buy()])
    assert review.severity == 1.0


@pytest.mark.asyncio
async def test_severity_garbage_defaults_zero() -> None:
    llm = _ScriptedLLM({"severity": "not-a-number", "counter_thesis": "x"})
    review = await critique_plan(llm, decisions=[_crypto_buy()])
    assert review.severity == 0.0
    assert review.recommendation == "proceed"


def test_apply_review_proceed_returns_unchanged() -> None:
    decisions = [_crypto_buy(qty=1.0), _crypto_sell(qty=2.0)]
    review = AdversarialReview(severity=0.1, counter_thesis="x", recommendation="proceed")
    out = apply_review_to_buys(decisions, review)
    assert [d.quantity for d in out] == [1.0, 2.0]


def test_apply_review_downsize_halves_buys_only() -> None:
    decisions = [_crypto_buy(qty=1.0), _crypto_sell(qty=2.0), _crypto_buy(qty=0.4)]
    review = AdversarialReview(severity=0.5, counter_thesis="x", recommendation="downsize")
    out = apply_review_to_buys(decisions, review)
    assert [d.action for d in out] == [
        TradeAction.BUY,
        TradeAction.SELL,
        TradeAction.BUY,
    ]
    assert [d.quantity for d in out] == [0.5, 2.0, 0.2]


def test_apply_review_skip_drops_buys_keeps_sells() -> None:
    decisions = [_crypto_buy(qty=1.0), _crypto_sell(qty=2.0), _crypto_buy(qty=0.4)]
    review = AdversarialReview(severity=0.9, counter_thesis="x", recommendation="skip")
    out = apply_review_to_buys(decisions, review)
    assert len(out) == 1
    assert out[0].action == TradeAction.SELL
    assert out[0].quantity == 2.0


@pytest.mark.asyncio
async def test_strategy_attacker_downsizes_buys() -> None:
    """End-to-end: attacker hooked into CryptoTradingStrategy actually shrinks the plan."""
    from unittest.mock import AsyncMock

    from halal_trader.crypto.strategy import CryptoTradingStrategy
    from halal_trader.domain.models import CryptoAccount, CryptoTradingPlan

    primary = _ScriptedLLM(
        {
            "decisions": [
                {
                    "action": "buy",
                    "symbol": "BTCUSDT",
                    "quantity": 1.0,
                    "confidence": 0.7,
                    "reasoning": "vol surge",
                }
            ],
            "market_outlook": "bullish",
            "risk_notes": "",
        }
    )
    attacker = _ScriptedLLM({"severity": 0.55, "counter_thesis": "rsi extended"})
    repo = AsyncMock()
    repo.record_decision = AsyncMock()

    strat = CryptoTradingStrategy(
        primary,
        repo,
        llm_provider_name="stub",
        max_position_pct=0.25,
        daily_loss_limit=0.05,
        daily_return_target=0.01,
        max_simultaneous_positions=3,
        attacker_llm=attacker,
    )
    plan = await strat.analyze(
        account=CryptoAccount(total_balance_usdt=1000),
        positions_text="",
        halal_pairs=["BTCUSDT"],
        klines_by_symbol={},
        orderbooks={},
    )
    assert isinstance(plan, CryptoTradingPlan)
    assert plan.decisions
    # Attacker recommended downsize — quantity should be halved (1.0 -> 0.5)
    assert plan.decisions[0].quantity == 0.5
    assert "adversarial" in plan.risk_notes
    assert strat.last_adversarial_review is not None
    assert strat.last_adversarial_review.recommendation == "downsize"


def test_apply_review_works_on_stock_decisions() -> None:
    """Same downsize logic for the stocks ``TradeDecision`` flavor."""
    decisions = [
        TradeDecision(
            action=TradeAction.BUY,
            symbol="AAPL",
            quantity=10,
            confidence=0.6,
            reasoning="x",
        ),
        TradeDecision(
            action=TradeAction.SELL,
            symbol="MSFT",
            quantity=5,
            confidence=0.7,
            reasoning="x",
        ),
    ]
    review = AdversarialReview(severity=0.5, counter_thesis="x", recommendation="downsize")
    out = apply_review_to_buys(decisions, review)
    assert out[0].quantity == 5  # 10 * 0.5
    assert out[1].quantity == 5  # untouched
