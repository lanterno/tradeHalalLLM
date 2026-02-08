"""Structured decision models for LLM trading outputs."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class TradeDecision(BaseModel):
    """A single trading decision produced by the LLM."""

    action: TradeAction
    symbol: str
    quantity: int = Field(ge=0, description="Number of shares")
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 confidence score")
    reasoning: str = Field(description="Brief explanation of the decision")
    target_price: Optional[float] = Field(default=None, description="Expected target price")
    stop_loss: Optional[float] = Field(default=None, description="Suggested stop-loss price")


class TradingPlan(BaseModel):
    """The complete trading plan returned by the LLM for one analysis cycle."""

    decisions: list[TradeDecision] = Field(default_factory=list)
    market_outlook: str = Field(default="", description="Overall market assessment")
    risk_notes: str = Field(default="", description="Risk factors identified")

    @property
    def buys(self) -> list[TradeDecision]:
        return [d for d in self.decisions if d.action == TradeAction.BUY]

    @property
    def sells(self) -> list[TradeDecision]:
        return [d for d in self.decisions if d.action == TradeAction.SELL]

    @property
    def holds(self) -> list[TradeDecision]:
        return [d for d in self.decisions if d.action == TradeAction.HOLD]
