"""Stock-side CLI commands: start, status, history, config."""

from __future__ import annotations

import asyncio

import click
from rich.panel import Panel

from halal_trader.cli._display import (
    print_account,
    print_clock,
    print_config,
    print_pnl,
    print_positions,
    print_trades,
)
from halal_trader.logging import console


@click.command()
@click.option("--once", is_flag=True, help="Run a single trading cycle then exit")
def start(once: bool) -> None:
    """Start the stock trading bot."""
    from halal_trader.trading.scheduler import TradingBot

    console.print(
        Panel(
            "[bold green]Halal Trader[/bold green]\n"
            "LLM-powered halal day-trading bot\n"
            "[dim]Paper trading mode - simulated funds[/dim]",
            title="Starting",
            border_style="green",
        )
    )

    bot = TradingBot()
    print_config()

    if once:
        console.print("[yellow]Running a single trading cycle...[/yellow]")
        asyncio.run(bot.run_once())
    else:
        console.print("[green]Starting scheduled trading bot (Ctrl+C to stop)...[/green]")
        asyncio.run(bot.run())


@click.command()
def status() -> None:
    """Show current portfolio status and positions."""

    async def _status() -> None:
        from halal_trader.mcp.client import AlpacaMCPClient

        mcp = AlpacaMCPClient()
        try:
            await mcp.connect()
            print_account(await mcp.get_account_info())
            print_positions(await mcp.get_all_positions())
            print_clock(await mcp.get_clock())
        finally:
            await mcp.disconnect()

    asyncio.run(_status())


@click.command()
@click.option("--limit", default=50, help="Number of recent trades to show")
def history(limit: int) -> None:
    """Show stock trade history and daily P&L."""

    async def _history() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.db.repository import Repository

        settings = get_settings()
        engine = await init_db(str(settings.resolve_db_path()))
        repo = Repository(engine)
        try:
            print_trades(await repo.get_recent_trades(limit))
            print_pnl(await repo.get_pnl_history(limit=14))
        finally:
            await engine.dispose()

    asyncio.run(_history())


@click.command()
def config() -> None:
    """Show current configuration."""
    print_config()
