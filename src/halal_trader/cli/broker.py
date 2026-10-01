"""`halal-trader broker`: the Alpaca adapters, compared (execution/broker_compare.py)."""

from __future__ import annotations

import asyncio
import sys

import click

from halal_trader.logging import console


@click.group("broker")
def broker() -> None:
    """The Alpaca adapters: the MCP server and direct REST."""


@broker.command("compare")
@click.option("--symbols", default="AAPL,MSFT", show_default=True, help="Comma-separated.")
def compare_cmd(symbols: str) -> None:
    """Read account, clock, positions and prices through both adapters; exit 1 on a mismatch."""

    async def _run() -> bool:
        from halal_trader.config import get_settings
        from halal_trader.execution.alpaca_broker import AlpacaRestBroker
        from halal_trader.execution.broker_compare import compare
        from halal_trader.mcp.client import AlpacaMCPClient

        settings = get_settings().alpaca
        mcp = AlpacaMCPClient()
        rest = AlpacaRestBroker(settings.api_key, settings.secret_key, paper=settings.paper_trade)
        await mcp.connect()
        try:
            wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            checks = await compare(mcp, rest, wanted)
        finally:
            await mcp.disconnect()
            await rest.disconnect()
        for c in checks:
            mark = "[green]ok[/green]  " if c.ok else "[red]DIFF[/red]"
            console.print(f"{mark} {c.name:<22} mcp | rest: {c.detail}")
        return all(c.ok for c in checks)

    if not asyncio.run(_run()):
        sys.exit(1)
