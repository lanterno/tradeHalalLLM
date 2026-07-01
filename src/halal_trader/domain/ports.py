"""Port interfaces (Protocols) that the domain depends on.

Infrastructure adapters implement these protocols so that application-layer
code never couples to a specific broker, LLM provider, or database.
"""

from typing import Any, Protocol

from halal_trader.domain.models import (
    Account,
    CryptoAccount,
    CryptoBalance,
    Kline,
    MarketClock,
    Position,
)

# ── Broker (Stocks) ─────────────────────────────────────────────


class Broker(Protocol):
    """Abstraction over a stock-trading brokerage (e.g. Alpaca via MCP)."""

    async def get_account_info(self) -> Account: ...
    async def get_clock(self) -> MarketClock: ...
    async def get_calendar(self, start: str | None = None, end: str | None = None) -> Any: ...
    async def get_all_positions(self) -> list[Position]: ...
    async def get_stock_snapshot(self, symbols: str) -> Any: ...
    async def get_stock_bars(self, symbol: str, days: int = 5, timeframe: str = "1Day") -> Any: ...
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> Any: ...
    async def get_order_by_id(self, order_id: str) -> dict[str, Any]: ...
    async def close_position(self, symbol: str) -> Any: ...
    async def close_all_positions(self) -> Any: ...


# ── Crypto Broker ────────────────────────────────────────────────


class CryptoBroker(Protocol):
    """Abstraction over a crypto exchange (e.g. Binance)."""

    async def get_account(self) -> CryptoAccount: ...
    async def get_balances(self) -> list[CryptoBalance]: ...
    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]: ...
    async def get_klines(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> list[Kline]: ...
    async def get_order_book(self, symbol: str, limit: int = 10) -> dict[str, Any]: ...
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
    ) -> dict[str, Any]: ...
    async def cancel_order(self, symbol: str, order_id: str) -> dict[str, Any]: ...
    async def get_ticker_price(self, symbol: str) -> float: ...


# ── LLM Backend ────────────────────────────────────────────────


class LLMBackend(Protocol):
    """Abstraction over an LLM backend (GLM over any OpenAI-compatible endpoint)."""

    model: str

    async def generate(self, prompt: str, system: str | None = None) -> str: ...
    async def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]: ...
    async def generate_tool_call(
        self,
        prompt: str,
        *,
        tools: list[Any],
        system: str | None = None,
        force_tool: str | None = None,
    ) -> list[Any]: ...


# ── Compliance Screener (Stocks) ────────────────────────────────


class ComplianceScreener(Protocol):
    """Abstraction over a Shariah-compliance screening service (stocks)."""

    async def ensure_cache(self, symbols: list[str] | None = None) -> None: ...
    async def is_halal(self, symbol: str) -> bool: ...
    async def get_halal_symbols(self) -> list[str]: ...
    async def filter_halal(self, symbols: list[str]) -> list[str]: ...


# ── Crypto Compliance Screener ──────────────────────────────────


class CryptoComplianceScreener(Protocol):
    """Abstraction over a Shariah-compliance screening service for crypto."""

    async def refresh_screening(self, symbols: list[str] | None = None) -> None: ...
    async def is_halal(self, symbol: str) -> bool: ...
    async def get_halal_pairs(self) -> list[str]: ...
    async def filter_halal(self, symbols: list[str]) -> list[str]: ...


# NOTE: ``TradeRepository`` previously lived here as a 120-line structural
# Protocol shadowing every method on ``db.repository.Repository``. It had
# exactly one implementation, no test seam value, and added drift hazard.
# Code now annotates ``Repository`` directly.
