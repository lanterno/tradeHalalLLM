"""Tests for the decision models (structured LLM output parsing)."""

import json

import pytest
from pydantic import ValidationError

from halal_trader.agent.decision import TradeAction, TradeDecision, TradingPlan


class TestTradeDecision:
    def test_valid_buy_decision(self):
        d = TradeDecision(
            action=TradeAction.BUY,
            symbol="AAPL",
            quantity=10,
            confidence=0.85,
            reasoning="Strong momentum, high volume breakout",
            target_price=195.0,
            stop_loss=188.0,
        )
        assert d.action == TradeAction.BUY
        assert d.symbol == "AAPL"
        assert d.quantity == 10
        assert d.confidence == 0.85

    def test_valid_sell_decision(self):
        d = TradeDecision(
            action=TradeAction.SELL,
            symbol="NVDA",
            quantity=5,
            confidence=0.7,
            reasoning="Reached target price",
        )
        assert d.action == TradeAction.SELL
        assert d.target_price is None
        assert d.stop_loss is None

    def test_hold_decision(self):
        d = TradeDecision(
            action=TradeAction.HOLD,
            symbol="MSFT",
            quantity=0,
            confidence=0.5,
            reasoning="No clear signal",
        )
        assert d.action == TradeAction.HOLD

    def test_invalid_confidence_too_high(self):
        with pytest.raises(ValidationError):
            TradeDecision(
                action=TradeAction.BUY,
                symbol="AAPL",
                quantity=10,
                confidence=1.5,  # exceeds max
                reasoning="test",
            )

    def test_invalid_negative_quantity(self):
        with pytest.raises(ValidationError):
            TradeDecision(
                action=TradeAction.BUY,
                symbol="AAPL",
                quantity=-1,
                confidence=0.5,
                reasoning="test",
            )


class TestTradingPlan:
    def test_empty_plan(self):
        plan = TradingPlan()
        assert plan.decisions == []
        assert plan.buys == []
        assert plan.sells == []
        assert plan.holds == []

    def test_plan_with_mixed_decisions(self):
        plan = TradingPlan(
            decisions=[
                TradeDecision(
                    action=TradeAction.BUY,
                    symbol="AAPL",
                    quantity=10,
                    confidence=0.8,
                    reasoning="buy reason",
                ),
                TradeDecision(
                    action=TradeAction.SELL,
                    symbol="NVDA",
                    quantity=5,
                    confidence=0.7,
                    reasoning="sell reason",
                ),
                TradeDecision(
                    action=TradeAction.HOLD,
                    symbol="MSFT",
                    quantity=0,
                    confidence=0.5,
                    reasoning="hold reason",
                ),
            ],
            market_outlook="Bullish tech sector",
            risk_notes="Earnings season volatility",
        )
        assert len(plan.buys) == 1
        assert len(plan.sells) == 1
        assert len(plan.holds) == 1
        assert plan.buys[0].symbol == "AAPL"
        assert plan.sells[0].symbol == "NVDA"

    def test_plan_from_json(self):
        """Simulate parsing LLM JSON output into a TradingPlan."""
        llm_output = {
            "decisions": [
                {
                    "action": "buy",
                    "symbol": "GOOG",
                    "quantity": 3,
                    "confidence": 0.9,
                    "reasoning": "Pre-market gap up with volume",
                    "target_price": 180.0,
                    "stop_loss": 172.0,
                },
                {
                    "action": "sell",
                    "symbol": "AMZN",
                    "quantity": 2,
                    "confidence": 0.75,
                    "reasoning": "Hit resistance level",
                    "target_price": None,
                    "stop_loss": None,
                },
            ],
            "market_outlook": "Mixed signals, sector rotation ongoing",
            "risk_notes": "Fed meeting this week",
        }

        plan = TradingPlan.model_validate(llm_output)
        assert len(plan.decisions) == 2
        assert plan.buys[0].symbol == "GOOG"
        assert plan.sells[0].symbol == "AMZN"
        assert plan.market_outlook == "Mixed signals, sector rotation ongoing"

    def test_plan_from_json_string(self):
        """Full roundtrip: JSON string -> dict -> TradingPlan."""
        raw = (
            '{"decisions": [{"action": "buy", "symbol": "AAPL",'
            ' "quantity": 5, "confidence": 0.8, "reasoning": "test"}],'
            ' "market_outlook": "ok", "risk_notes": ""}'
        )
        data = json.loads(raw)
        plan = TradingPlan.model_validate(data)
        assert len(plan.buys) == 1
        assert plan.buys[0].symbol == "AAPL"
