"""Click CLI entrypoint with Rich terminal output."""

import asyncio
import logging

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def _setup_logging(level: str = "INFO") -> None:
    """Configure rich logging."""
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--log-level", default=None, help="Override log level (DEBUG, INFO, WARNING, ERROR)")
def cli(log_level: str | None) -> None:
    """Halal Trader - LLM-powered halal day-trading bot."""
    from halal_trader.config import get_settings

    settings = get_settings()
    level = log_level or settings.log_level
    _setup_logging(level)


@cli.command()
@click.option("--once", is_flag=True, help="Run a single trading cycle then exit")
def start(once: bool) -> None:
    """Start the trading bot."""
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

    _print_config()

    if once:
        console.print("[yellow]Running a single trading cycle...[/yellow]")
        asyncio.run(bot.run_once())
    else:
        console.print("[green]Starting scheduled trading bot (Ctrl+C to stop)...[/green]")
        asyncio.run(bot.run())


@cli.command()
def status() -> None:
    """Show current portfolio status and positions."""

    async def _status() -> None:
        from halal_trader.mcp.client import AlpacaMCPClient

        mcp = AlpacaMCPClient()

        try:
            await mcp.connect()

            # Account info
            account = await mcp.get_account_info()
            _print_account(account)

            # Positions
            positions = await mcp.get_all_positions()
            _print_positions(positions)

            # Market clock
            clock = await mcp.get_clock()
            _print_clock(clock)

        finally:
            await mcp.disconnect()

    asyncio.run(_status())


@cli.command()
@click.option("--limit", default=50, help="Number of recent trades to show")
def history(limit: int) -> None:
    """Show trade history and daily P&L."""

    async def _history() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.db.repository import Repository

        settings = get_settings()
        db = await init_db(str(settings.db_path))
        repo = Repository(db)

        try:
            # Recent trades
            trades = await repo.get_recent_trades(limit)
            _print_trades(trades)

            # Daily P&L
            pnl_history = await repo.get_pnl_history(limit=14)
            _print_pnl(pnl_history)
        finally:
            await db.close()

    asyncio.run(_history())


@cli.command()
def config() -> None:
    """Show current configuration."""
    _print_config()


# ── Display helpers ─────────────────────────────────────────────


def _print_config() -> None:
    from halal_trader.config import get_settings

    settings = get_settings()

    table = Table(title="Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="dim")
    table.add_column("Value")

    table.add_row("LLM Provider", settings.llm_provider.value)
    table.add_row("LLM Model", settings.llm_model)
    table.add_row("Trading Interval", f"{settings.trading_interval_minutes} min")
    table.add_row("Daily Return Target", f"{settings.daily_return_target:.1%}")
    table.add_row("Max Position Size", f"{settings.max_position_pct:.0%}")
    table.add_row("Daily Loss Limit", f"{settings.daily_loss_limit:.1%}")
    table.add_row("Max Positions", str(settings.max_simultaneous_positions))
    table.add_row("Paper Trading", str(settings.alpaca_paper_trade))
    table.add_row(
        "Alpaca API Key",
        settings.alpaca_api_key[:8] + "..." if settings.alpaca_api_key else "[red]NOT SET[/red]",
    )
    table.add_row(
        "Zoya API",
        "Configured" if settings.zoya_api_key else "[yellow]Not configured (defaults)[/yellow]",
    )
    table.add_row("Database", str(settings.db_path))

    console.print(table)


def _print_account(account: object) -> None:
    if isinstance(account, dict):
        table = Table(title="Account", show_header=True, header_style="bold cyan")
        table.add_column("Field", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Equity", f"${float(account.get('equity', 0)):,.2f}")
        table.add_row("Buying Power", f"${float(account.get('buying_power', 0)):,.2f}")
        table.add_row("Cash", f"${float(account.get('cash', 0)):,.2f}")
        table.add_row("Portfolio Value", f"${float(account.get('portfolio_value', 0)):,.2f}")
        table.add_row("Status", str(account.get("status", "")))

        console.print(table)
    else:
        console.print(Panel(str(account), title="Account Info"))


def _print_positions(positions: object) -> None:
    if isinstance(positions, list) and positions:
        table = Table(title="Open Positions", show_header=True, header_style="bold cyan")
        table.add_column("Symbol")
        table.add_column("Qty", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")

        for p in positions:
            if isinstance(p, dict):
                pnl = float(p.get("unrealized_pl", 0))
                pnl_pct = float(p.get("unrealized_plpc", 0))
                style = "green" if pnl >= 0 else "red"
                table.add_row(
                    str(p.get("symbol", "")),
                    str(p.get("qty", "")),
                    f"${float(p.get('avg_entry_price', 0)):,.2f}",
                    f"${float(p.get('current_price', 0)):,.2f}",
                    Text(f"${pnl:+,.2f}", style=style),
                    Text(f"{pnl_pct:+.2%}", style=style),
                )

        console.print(table)
    else:
        console.print("[dim]No open positions.[/dim]")


def _print_clock(clock: object) -> None:
    if isinstance(clock, dict):
        is_open = clock.get("is_open", False)
        status_text = "[bold green]OPEN[/bold green]" if is_open else "[bold red]CLOSED[/bold red]"
        console.print(f"\nMarket: {status_text}")
        if not is_open:
            console.print(f"  Next open: {clock.get('next_open', 'N/A')}")
        else:
            console.print(f"  Closes at: {clock.get('next_close', 'N/A')}")
    else:
        console.print(Panel(str(clock), title="Market Clock"))


def _print_trades(trades: list) -> None:
    if not trades:
        console.print("[dim]No trades recorded yet.[/dim]")
        return

    table = Table(title="Recent Trades", show_header=True, header_style="bold cyan")
    table.add_column("Time", style="dim")
    table.add_column("Symbol")
    table.add_column("Side")
    table.add_column("Qty", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Status")
    table.add_column("Reasoning", max_width=40)

    for t in trades:
        side_style = "green" if t.get("side") == "buy" else "red"
        table.add_row(
            str(t.get("timestamp", ""))[:19],
            str(t.get("symbol", "")),
            Text(str(t.get("side", "")), style=side_style),
            str(t.get("quantity", "")),
            f"${t.get('price', 0) or 0:,.2f}",
            str(t.get("status", "")),
            str(t.get("llm_reasoning", ""))[:40],
        )

    console.print(table)


def _print_pnl(pnl_history: list) -> None:
    if not pnl_history:
        console.print("[dim]No P&L history yet.[/dim]")
        return

    table = Table(title="Daily P&L", show_header=True, header_style="bold cyan")
    table.add_column("Date")
    table.add_column("Start Equity", justify="right")
    table.add_column("End Equity", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Trades", justify="right")

    for row in pnl_history:
        pnl = row.get("realized_pnl", 0) or 0
        ret = row.get("return_pct", 0) or 0
        style = "green" if pnl >= 0 else "red"
        table.add_row(
            str(row.get("date", "")),
            f"${row.get('starting_equity', 0) or 0:,.2f}",
            f"${row.get('ending_equity', 0) or 0:,.2f}",
            Text(f"${pnl:+,.2f}", style=style),
            Text(f"{ret:+.2%}", style=style),
            str(row.get("trades_count", 0)),
        )

    console.print(table)
