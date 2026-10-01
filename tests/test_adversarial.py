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
from halal_trader.domain.models import TradeAction, TradeDecision


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


def _buy(symbol: str = "AAPL", qty: int = 10) -> TradeDecision:
    return TradeDecision(
        action=TradeAction.BUY,
        symbol=symbol,
        quantity=qty,
        confidence=0.7,
        reasoning="momentum + volume",
    )


def _sell(symbol: str = "MSFT", qty: int = 5) -> TradeDecision:
    return TradeDecision(
        action=TradeAction.SELL,
        symbol=symbol,
        quantity=qty,
        confidence=0.6,
        reasoning="trailing stop hit",
    )


@pytest.mark.asyncio
async def test_proceed_when_severity_low() -> None:
    llm = _ScriptedLLM({"severity": 0.2, "counter_thesis": "fine"})
    review = await critique_plan(llm, decisions=[_buy()])
    assert review.recommendation == "proceed"
    assert review.severity == 0.2
    assert review.sizing_multiplier == 1.0
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_downsize_in_mid_band() -> None:
    llm = _ScriptedLLM({"severity": 0.55, "counter_thesis": "RSI extended"})
    review = await critique_plan(llm, decisions=[_buy()])
    assert review.recommendation == "downsize"
    assert review.sizing_multiplier == 0.5


@pytest.mark.asyncio
async def test_skip_when_severity_high() -> None:
    llm = _ScriptedLLM({"severity": 0.9, "counter_thesis": "blow-off top"})
    review = await critique_plan(llm, decisions=[_buy()])
    assert review.recommendation == "skip"
    assert review.sizing_multiplier == 0.0


@pytest.mark.asyncio
async def test_attacker_failure_degrades_to_proceed() -> None:
    llm = _ScriptedLLM(RuntimeError("network down"))
    review = await critique_plan(llm, decisions=[_buy()])
    assert review.recommendation == "proceed"
    assert "attacker-error" in review.counter_thesis


@pytest.mark.asyncio
async def test_no_call_when_no_buys() -> None:
    llm = _ScriptedLLM({"severity": 1.0, "counter_thesis": "shouldn't run"})
    review = await critique_plan(llm, decisions=[_sell()])
    assert llm.calls == 0
    assert review.recommendation == "proceed"


@pytest.mark.asyncio
async def test_severity_clamped() -> None:
    llm = _ScriptedLLM({"severity": 5.0, "counter_thesis": "out of range"})
    review = await critique_plan(llm, decisions=[_buy()])
    assert review.severity == 1.0


@pytest.mark.asyncio
async def test_severity_garbage_defaults_zero() -> None:
    llm = _ScriptedLLM({"severity": "not-a-number", "counter_thesis": "x"})
    review = await critique_plan(llm, decisions=[_buy()])
    assert review.severity == 0.0
    assert review.recommendation == "proceed"


def test_apply_review_proceed_returns_unchanged() -> None:
    decisions = [_buy(qty=10), _sell(qty=20)]
    review = AdversarialReview(severity=0.1, counter_thesis="x", recommendation="proceed")
    out = apply_review_to_buys(decisions, review)
    assert [d.quantity for d in out] == [10, 20]


def test_apply_review_downsize_halves_buys_only() -> None:
    decisions = [_buy(qty=10), _sell(qty=20), _buy(qty=4)]
    review = AdversarialReview(severity=0.5, counter_thesis="x", recommendation="downsize")
    out = apply_review_to_buys(decisions, review)
    assert [d.action for d in out] == [
        TradeAction.BUY,
        TradeAction.SELL,
        TradeAction.BUY,
    ]
    assert [d.quantity for d in out] == [5, 20, 2]


def test_apply_review_skip_drops_buys_keeps_sells() -> None:
    decisions = [_buy(qty=10), _sell(qty=20), _buy(qty=4)]
    review = AdversarialReview(severity=0.9, counter_thesis="x", recommendation="skip")
    out = apply_review_to_buys(decisions, review)
    assert len(out) == 1
    assert out[0].action == TradeAction.SELL
    assert out[0].quantity == 20


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
