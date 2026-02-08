"""Domain models — trading decisions, market data value objects, and portfolio state."""

from enum import Enum

from pydantic import BaseModel, Field

# ── Broker Data (Stocks) ──────────────────────────────────────


class Account(BaseModel):
    """Brokerage account snapshot."""

    equity: float = 0.0
    buying_power: float = 0.0
    cash: float = 0.0
    portfolio_value: float = 0.0
    status: str = ""

    @property
    def effective_equity(self) -> float:
        """Best available equity figure, preferring ``equity`` over ``portfolio_value``."""
        return self.equity or self.portfolio_value


class Position(BaseModel):
    """A single open position."""

    symbol: str
    qty: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pl: float = 0.0
    unrealized_plpc: float = 0.0


class MarketClock(BaseModel):
    """Market clock status."""

    is_open: bool = False
    next_open: str = ""
    next_close: str = ""


# ── Crypto Broker Data ─────────────────────────────────────────


class CryptoAccount(BaseModel):
    """Crypto exchange account snapshot."""

    total_balance_usdt: float = 0.0
    available_balance_usdt: float = 0.0
    in_order_usdt: float = 0.0


class CryptoBalance(BaseModel):
    """Balance for a single crypto asset."""

    asset: str
    free: float = 0.0
    locked: float = 0.0


class Kline(BaseModel):
    """A single candlestick (kline) bar."""

    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


# ── Trading Decisions ───────────────────────────────────────────


class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class TradeDecision(BaseModel):
    """A single trading decision produced by the LLM (stocks)."""

    action: TradeAction
    symbol: str
    quantity: int = Field(ge=0, description="Number of shares")
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 confidence score")
    reasoning: str = Field(description="Brief explanation of the decision")
    target_price: float | None = Field(default=None, description="Expected target price")
    stop_loss: float | None = Field(default=None, description="Suggested stop-loss price")


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


# ── Crypto Trading Decisions ──────────────────────────────────


class CryptoTradeDecision(BaseModel):
    """A single trading decision produced by the LLM (crypto)."""

    action: TradeAction
    symbol: str = Field(description="Trading pair, e.g. BTCUSDT")
    quantity: float = Field(ge=0.0, description="Fractional quantity to trade")
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 confidence score")
    reasoning: str = Field(description="Brief explanation of the decision")
    entry_price: float | None = Field(default=None, description="Suggested entry price")
    target_price: float | None = Field(default=None, description="Expected target price")
    stop_loss: float | None = Field(default=None, description="Suggested stop-loss price")


class CryptoTradingPlan(BaseModel):
    """The complete crypto trading plan returned by the LLM for one cycle."""

    decisions: list[CryptoTradeDecision] = Field(default_factory=list)
    market_outlook: str = Field(default="", description="Overall crypto market assessment")
    risk_notes: str = Field(default="", description="Risk factors identified")

    @property
    def buys(self) -> list[CryptoTradeDecision]:
        return [d for d in self.decisions if d.action == TradeAction.BUY]

    @property
    def sells(self) -> list[CryptoTradeDecision]:
        return [d for d in self.decisions if d.action == TradeAction.SELL]

    @property
    def holds(self) -> list[CryptoTradeDecision]:
        return [d for d in self.decisions if d.action == TradeAction.HOLD]
