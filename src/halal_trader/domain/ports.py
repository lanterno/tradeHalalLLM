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
    """Abstraction over an LLM backend (Ollama, OpenAI, Anthropic, etc.)."""

    model: str

    async def generate(self, prompt: str, system: str | None = None) -> str: ...
    async def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]: ...


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


# ── Trade Repository ────────────────────────────────────────────


class TradeRepository(Protocol):
    """Persistence port for trades, P&L, halal cache, and LLM audit log."""

    # Trades (stocks)
    async def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_id: str | None = None,
        status: str = "pending",
        llm_reasoning: str | None = None,
    ) -> int: ...

    async def get_today_trades(self) -> list[dict[str, Any]]: ...
    async def get_recent_trades(self, limit: int = 50) -> list[dict[str, Any]]: ...

    # Daily P&L (stocks)
    async def start_day(self, starting_equity: float) -> None: ...
    async def end_day(
        self, ending_equity: float, realized_pnl: float, trades_count: int
    ) -> None: ...
    async def get_pnl_history(self, limit: int = 30) -> list[dict[str, Any]]: ...

    # Halal cache (stocks)
    async def cache_halal_status(
        self, symbol: str, compliance: str, detail: str | None = None
    ) -> None: ...
    async def get_halal_status(self, symbol: str) -> str | None: ...
    async def get_halal_symbols(self) -> list[str]: ...
    async def is_cache_fresh(self, max_age_hours: int = 24) -> bool: ...

    # LLM decisions (shared)
    async def record_decision(
        self,
        provider: str,
        model: str,
        prompt_summary: str | None = None,
        raw_response: str | None = None,
        parsed_action: dict[str, Any] | None = None,
        symbols: list[str] | None = None,
        execution_ms: int | None = None,
        thinking: str | None = None,
    ) -> int: ...

    # Crypto trades
    async def record_crypto_trade(
        self,
        pair: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_id: str | None = None,
        exchange: str = "binance",
        status: str = "pending",
        llm_reasoning: str | None = None,
        entry_price: float | None = None,
        stop_loss: float | None = None,
        target_price: float | None = None,
    ) -> int: ...

    async def update_crypto_trade_stop_loss(self, trade_id: int, new_stop_loss: float) -> None: ...

    async def close_crypto_trade(
        self, trade_id: int, exit_price: float, exit_reason: str
    ) -> None: ...

    async def get_open_crypto_trades(self) -> list[Any]: ...
    async def get_open_crypto_trades_for_pair(self, pair: str) -> list[Any]: ...
    async def close_open_crypto_trades_for_pair(
        self,
        pair: str,
        exit_price: float,
        exit_reason: str,
        exclude_id: int | None = None,
    ) -> int: ...

    async def get_today_crypto_trades(self) -> list[dict[str, Any]]: ...
    async def get_recent_crypto_trades(self, limit: int = 50) -> list[dict[str, Any]]: ...
    async def get_completed_round_trips(
        self, limit: int = 100, lookback_days: int | None = None
    ) -> list[dict[str, Any]]: ...

    # Crypto daily P&L
    async def start_crypto_day(self, starting_equity: float) -> None: ...
    async def end_crypto_day(
        self, ending_equity: float, realized_pnl: float, trades_count: int
    ) -> None: ...
    async def get_crypto_pnl_history(self, limit: int = 30) -> list[dict[str, Any]]: ...

    # Crypto halal cache
    async def cache_crypto_halal_status(
        self,
        symbol: str,
        compliance: str,
        category: str | None = None,
        market_cap: float | None = None,
        screening_criteria: str | None = None,
    ) -> None: ...
    async def get_crypto_halal_status(self, symbol: str) -> str | None: ...
    async def get_crypto_halal_symbols(self) -> list[str]: ...
    async def is_crypto_cache_fresh(self, max_age_hours: int = 24) -> bool: ...

    # Indicator snapshots (ML training)
    async def record_indicator_snapshot(
        self, trade_id: int, pair: str, indicators: dict[str, float]
    ) -> int: ...
    async def label_indicator_snapshot(
        self, trade_id: int, label: int, return_pct: float
    ) -> None: ...
    async def get_labeled_snapshots(self, min_samples: int = 50) -> list[dict[str, Any]]: ...

    # Strategy adjustments
    async def record_strategy_adjustment(
        self,
        parameter: str,
        old_value: float | None,
        new_value: float,
        reasoning: str | None = None,
    ) -> int: ...

    async def get_latest_strategy_adjustments(self) -> dict[str, float]: ...
    async def get_recent_adjustments(self, limit: int = 20) -> list[dict[str, Any]]: ...
    async def get_recent_decisions(self, limit: int = 50) -> list[dict[str, Any]]: ...
