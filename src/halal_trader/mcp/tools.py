"""Typed wrappers around Alpaca MCP tool results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AccountInfo:
    buying_power: float = 0.0
    cash: float = 0.0
    equity: float = 0.0
    portfolio_value: float = 0.0
    status: str = ""

    @classmethod
    def from_raw(cls, data: Any) -> AccountInfo:
        if isinstance(data, dict):
            return cls(
                buying_power=float(data.get("buying_power", 0)),
                cash=float(data.get("cash", 0)),
                equity=float(data.get("equity", 0)),
                portfolio_value=float(data.get("portfolio_value", 0)),
                status=data.get("status", ""),
            )
        # If the MCP server returns formatted text, parse what we can
        return cls()


@dataclass
class Position:
    symbol: str = ""
    qty: float = 0.0
    side: str = ""
    market_value: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pl: float = 0.0
    unrealized_plpc: float = 0.0

    @classmethod
    def from_raw(cls, data: dict[str, Any]) -> Position:
        return cls(
            symbol=str(data.get("symbol", "")),
            qty=float(data.get("qty", 0)),
            side=str(data.get("side", "")),
            market_value=float(data.get("market_value", 0)),
            avg_entry_price=float(data.get("avg_entry_price", 0)),
            current_price=float(data.get("current_price", 0)),
            unrealized_pl=float(data.get("unrealized_pl", 0)),
            unrealized_plpc=float(data.get("unrealized_plpc", 0)),
        )


@dataclass
class MarketClock:
    is_open: bool = False
    next_open: str = ""
    next_close: str = ""
    timestamp: str = ""

    @classmethod
    def from_raw(cls, data: Any) -> MarketClock:
        if isinstance(data, dict):
            return cls(
                is_open=bool(data.get("is_open", False)),
                next_open=str(data.get("next_open", "")),
                next_close=str(data.get("next_close", "")),
                timestamp=str(data.get("timestamp", "")),
            )
        return cls()


@dataclass
class StockSnapshot:
    symbol: str = ""
    latest_trade_price: float = 0.0
    latest_bid: float = 0.0
    latest_ask: float = 0.0
    daily_bar_open: float = 0.0
    daily_bar_high: float = 0.0
    daily_bar_low: float = 0.0
    daily_bar_close: float = 0.0
    daily_bar_volume: int = 0
    prev_daily_close: float = 0.0

    @classmethod
    def from_raw(cls, symbol: str, data: Any) -> StockSnapshot:
        if isinstance(data, dict):
            latest_trade = data.get("latest_trade", {})
            latest_quote = data.get("latest_quote", {})
            daily_bar = data.get("daily_bar", {})
            prev_bar = data.get("prev_daily_bar", {})
            return cls(
                symbol=symbol,
                latest_trade_price=float(latest_trade.get("price", 0)),
                latest_bid=float(latest_quote.get("bid_price", 0)),
                latest_ask=float(latest_quote.get("ask_price", 0)),
                daily_bar_open=float(daily_bar.get("open", 0)),
                daily_bar_high=float(daily_bar.get("high", 0)),
                daily_bar_low=float(daily_bar.get("low", 0)),
                daily_bar_close=float(daily_bar.get("close", 0)),
                daily_bar_volume=int(daily_bar.get("volume", 0)),
                prev_daily_close=float(prev_bar.get("close", 0)),
            )
        return cls(symbol=symbol)

    @property
    def change_pct(self) -> float:
        if self.prev_daily_close > 0:
            return (self.latest_trade_price - self.prev_daily_close) / self.prev_daily_close
        return 0.0
