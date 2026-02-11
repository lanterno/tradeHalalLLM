"""MCP client that spawns and connects to the Alpaca MCP server."""

import json
import logging
import re
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from halal_trader.config import get_settings
from halal_trader.domain.models import Account, MarketClock, Position

logger = logging.getLogger(__name__)


def _flex_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Look up a value trying multiple key variants (snake_case, camelCase, etc.).

    The Alpaca MCP server may return keys in different casings depending on version.
    """
    for key in keys:
        if key in d:
            return d[key]
    return default


class AlpacaMCPClient:
    """Programmatic MCP client for the Alpaca trading server.

    Spawns ``alpaca-mcp-server`` as a subprocess and communicates
    via the stdio transport.
    """

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._tools: dict[str, Any] = {}

    # ── Lifecycle ───────────────────────────────────────────────

    async def connect(self) -> None:
        """Spawn the Alpaca MCP server and establish a session."""
        settings = get_settings()

        server_params = StdioServerParameters(
            command="uvx",
            args=["alpaca-mcp-server", "serve"],
            env={
                "ALPACA_API_KEY": settings.alpaca_api_key,
                "ALPACA_SECRET_KEY": settings.alpaca_secret_key,
                "ALPACA_PAPER_TRADE": str(settings.alpaca_paper_trade),
            },
        )

        transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = transport

        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()

        # Cache available tools
        response = await self.session.list_tools()
        self._tools = {tool.name: tool for tool in response.tools}
        logger.info(
            "Connected to Alpaca MCP server with %d tools: %s",
            len(self._tools),
            list(self._tools.keys()),
        )

    async def disconnect(self) -> None:
        """Cleanly shut down the MCP session and subprocess."""
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logger.debug("MCP exit stack cleanup error (safe to ignore): %s", e)
        self.session = None
        self._tools = {}
        logger.info("Disconnected from Alpaca MCP server")

    # ── Tool execution ──────────────────────────────────────────

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Execute a tool on the Alpaca MCP server and return the result."""
        if self.session is None:
            raise RuntimeError("Not connected to Alpaca MCP server. Call connect() first.")
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}. Available: {list(self._tools.keys())}")

        logger.info("Calling MCP tool: %s(%s)", name, arguments or {})
        result = await self.session.call_tool(name, arguments or {})

        # Extract text content from the result
        contents = []
        for item in result.content:
            if hasattr(item, "text"):
                contents.append(item.text)
            elif hasattr(item, "data"):
                contents.append(item.data)

        # Try to parse JSON responses
        combined = "\n".join(str(c) for c in contents)
        logger.debug("Raw MCP response for %s: %s", name, combined[:500])
        try:
            return json.loads(combined)
        except (json.JSONDecodeError, TypeError):
            return combined

    # ── Convenience wrappers ────────────────────────────────────

    async def get_account_info(self) -> Account:
        raw = await self.call_tool("get_account_info")
        logger.info("get_account_info() raw response type=%s keys=%s",
                     type(raw).__name__,
                     list(raw.keys()) if isinstance(raw, dict) else repr(raw)[:200])
        if isinstance(raw, dict):
            return Account(
                equity=float(_flex_get(raw, "equity", default=0) or 0),
                buying_power=float(
                    _flex_get(raw, "buying_power", "buyingPower", "buying_power", default=0) or 0
                ),
                cash=float(_flex_get(raw, "cash", default=0) or 0),
                portfolio_value=float(
                    _flex_get(raw, "portfolio_value", "portfolioValue", default=0) or 0
                ),
                status=str(_flex_get(raw, "status", default="")),
            )
        # Fallback: try to extract numbers from text response
        if isinstance(raw, str) and raw.strip():
            logger.warning(
                "get_account_info() received text instead of JSON — parsing: %s",
                raw[:200],
            )
            account = Account()
            # Try to find equity value in text (e.g. "equity: $100,000.00")
            equity_match = re.search(r"equity[:\s]*\$?([\d,]+\.?\d*)", raw, re.IGNORECASE)
            if equity_match:
                account.equity = float(equity_match.group(1).replace(",", ""))
            buying_power_match = re.search(
                r"buying.power[:\s]*\$?([\d,]+\.?\d*)", raw, re.IGNORECASE
            )
            if buying_power_match:
                account.buying_power = float(buying_power_match.group(1).replace(",", ""))
            cash_match = re.search(r"cash[:\s]*\$?([\d,]+\.?\d*)", raw, re.IGNORECASE)
            if cash_match:
                account.cash = float(cash_match.group(1).replace(",", ""))
            return account
        logger.error("get_account_info() returned unexpected response: %r", raw)
        return Account()

    async def get_clock(self) -> MarketClock:
        raw = await self.call_tool("get_clock")
        logger.info("get_clock() raw response type=%s keys=%s",
                     type(raw).__name__,
                     list(raw.keys()) if isinstance(raw, dict) else repr(raw)[:200])
        if isinstance(raw, dict):
            is_open_val = _flex_get(raw, "is_open", "isOpen", "is_market_open", default=False)
            # Handle string booleans like "true"/"false"
            if isinstance(is_open_val, str):
                is_open_val = is_open_val.lower() in ("true", "1", "yes")
            next_open = _flex_get(
                raw, "next_open", "nextOpen", "next_open_time", default=""
            )
            next_close = _flex_get(
                raw, "next_close", "nextClose", "next_close_time", default=""
            )
            return MarketClock(
                is_open=bool(is_open_val),
                next_open=str(next_open),
                next_close=str(next_close),
            )
        # Fallback: parse text response for market open/closed status.
        # The Alpaca MCP server may return human-readable text instead of JSON.
        if isinstance(raw, str) and raw.strip():
            logger.warning(
                "get_clock() received text instead of JSON — parsing: %s", raw[:200]
            )
            text_lower = raw.lower()
            is_open = "is open" in text_lower or "market is currently open" in text_lower
            return MarketClock(is_open=is_open)
        logger.error("get_clock() returned unexpected response: %r", raw)
        return MarketClock()

    async def get_calendar(self, start: str | None = None, end: str | None = None) -> Any:
        """Get the market calendar (trading days and hours).

        Args:
            start: Start date (YYYY-MM-DD). Defaults to today.
            end: End date (YYYY-MM-DD). Defaults to 30 days from start.
        """
        args: dict[str, Any] = {}
        if start:
            args["start"] = start
        if end:
            args["end"] = end
        return await self.call_tool("get_calendar", args or None)

    async def get_all_positions(self) -> list[Position]:
        raw = await self.call_tool("get_all_positions")
        logger.info("get_all_positions() raw response type=%s len=%s",
                     type(raw).__name__,
                     len(raw) if isinstance(raw, (list, dict)) else repr(raw)[:200])
        if isinstance(raw, list):
            positions = []
            for p in raw:
                if isinstance(p, dict):
                    positions.append(
                        Position(
                            symbol=str(_flex_get(p, "symbol", default="")),
                            qty=float(_flex_get(p, "qty", "quantity", default=0) or 0),
                            avg_entry_price=float(
                                _flex_get(p, "avg_entry_price", "avgEntryPrice", default=0) or 0
                            ),
                            current_price=float(
                                _flex_get(p, "current_price", "currentPrice", default=0) or 0
                            ),
                            unrealized_pl=float(
                                _flex_get(
                                    p, "unrealized_pl", "unrealizedPl",
                                    "unrealized_pnl", default=0,
                                ) or 0
                            ),
                            unrealized_plpc=float(
                                _flex_get(
                                    p, "unrealized_plpc", "unrealizedPlpc",
                                    "unrealized_pnl_pct", default=0,
                                ) or 0
                            ),
                        )
                    )
            return positions
        if isinstance(raw, str) and raw.strip():
            # Text responses like "no positions" are acceptable — just means empty
            logger.info("get_all_positions() text response: %s", raw[:200])
        return []

    async def get_stock_snapshot(self, symbols: str) -> Any:
        """Get a comprehensive snapshot for one or more symbols (comma-separated)."""
        return await self.call_tool("get_stock_snapshot", {"symbol_or_symbols": symbols})

    async def get_stock_bars(
        self,
        symbol: str,
        days: int = 5,
        timeframe: str = "1Day",
    ) -> Any:
        return await self.call_tool(
            "get_stock_bars",
            {"symbol": symbol, "days": days, "timeframe": timeframe},
        )

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> Any:
        return await self.call_tool(
            "place_stock_order",
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "time_in_force": time_in_force,
            },
        )

    async def close_position(self, symbol: str) -> Any:
        return await self.call_tool("close_position", {"symbol": symbol})

    async def close_all_positions(self) -> Any:
        return await self.call_tool("close_all_positions", {"cancel_orders": True})
