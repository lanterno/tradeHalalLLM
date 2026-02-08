"""MCP client that spawns and connects to the Alpaca MCP server."""

import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from halal_trader.config import get_settings
from halal_trader.domain.models import Account, MarketClock, Position

logger = logging.getLogger(__name__)


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
        await self._exit_stack.aclose()
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
        try:
            return json.loads(combined)
        except json.JSONDecodeError, TypeError:
            return combined

    # ── Convenience wrappers ────────────────────────────────────

    async def get_account_info(self) -> Account:
        raw = await self.call_tool("get_account_info")
        if isinstance(raw, dict):
            return Account(
                equity=float(raw.get("equity", 0) or 0),
                buying_power=float(raw.get("buying_power", 0) or 0),
                cash=float(raw.get("cash", 0) or 0),
                portfolio_value=float(raw.get("portfolio_value", 0) or 0),
                status=str(raw.get("status", "")),
            )
        return Account()

    async def get_clock(self) -> MarketClock:
        raw = await self.call_tool("get_clock")
        if isinstance(raw, dict):
            return MarketClock(
                is_open=bool(raw.get("is_open", False)),
                next_open=str(raw.get("next_open", "")),
                next_close=str(raw.get("next_close", "")),
            )
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
        if isinstance(raw, list):
            positions = []
            for p in raw:
                if isinstance(p, dict):
                    positions.append(
                        Position(
                            symbol=str(p.get("symbol", "")),
                            qty=float(p.get("qty", 0) or 0),
                            avg_entry_price=float(p.get("avg_entry_price", 0) or 0),
                            current_price=float(p.get("current_price", 0) or 0),
                            unrealized_pl=float(p.get("unrealized_pl", 0) or 0),
                            unrealized_plpc=float(p.get("unrealized_plpc", 0) or 0),
                        )
                    )
            return positions
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
