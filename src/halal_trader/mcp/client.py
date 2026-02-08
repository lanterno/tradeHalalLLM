"""MCP client that spawns and connects to the Alpaca MCP server."""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from halal_trader.config import get_settings

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

    # ── Tool discovery ──────────────────────────────────────────

    def list_tools(self) -> list[dict[str, Any]]:
        """Return tool descriptions suitable for passing to an LLM."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in self._tools.values()
        ]

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
        except (json.JSONDecodeError, TypeError):
            return combined

    # ── Convenience wrappers ────────────────────────────────────

    async def get_account_info(self) -> dict[str, Any]:
        return await self.call_tool("get_account_info")

    async def get_clock(self) -> dict[str, Any]:
        return await self.call_tool("get_clock")

    async def get_all_positions(self) -> Any:
        return await self.call_tool("get_all_positions")

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

    async def get_orders(self, status: str | None = None) -> Any:
        args: dict[str, Any] = {}
        if status:
            args["status"] = status
        return await self.call_tool("get_orders", args)
