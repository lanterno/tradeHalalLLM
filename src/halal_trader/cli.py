"""Click CLI entrypoint with Rich terminal output."""

import asyncio

import click
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from halal_trader.logging import console


@click.group()
@click.option("--log-level", default=None, help="Override log level (DEBUG, INFO, WARNING, ERROR)")
def cli(log_level: str | None) -> None:
    """Halal Trader - LLM-powered halal day-trading bot."""
    from halal_trader.config import get_settings
    from halal_trader.logging import setup_logging

    settings = get_settings()
    setup_logging(settings, cli_log_level=log_level)


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
        engine = await init_db(str(settings.db_path))
        repo = Repository(engine)

        try:
            # Recent trades
            trades = await repo.get_recent_trades(limit)
            _print_trades(trades)

            # Daily P&L
            pnl_history = await repo.get_pnl_history(limit=14)
            _print_pnl(pnl_history)
        finally:
            await engine.dispose()

    asyncio.run(_history())


@cli.command()
def config() -> None:
    """Show current configuration."""
    _print_config()


# ── Crypto Command Group ───────────────────────────────────────


@cli.group()
def crypto() -> None:
    """Crypto trading commands (Binance-based, 24/7)."""


@crypto.command("start")
@click.option("--once", is_flag=True, help="Run a single crypto trading cycle then exit")
def crypto_start(once: bool) -> None:
    """Start the crypto trading bot."""
    from halal_trader.config import get_settings
    from halal_trader.crypto.scheduler import CryptoTradingBot

    settings = get_settings()
    mode = "TESTNET" if settings.binance_testnet else "PRODUCTION"

    console.print(
        Panel(
            f"[bold green]Halal Crypto Trader[/bold green]\n"
            f"LLM-powered crypto trading bot\n"
            f"[dim]{mode} mode — Binance[/dim]",
            title="Starting",
            border_style="green",
        )
    )

    _print_crypto_config()

    bot = CryptoTradingBot()

    if once:
        console.print("[yellow]Running a single crypto trading cycle...[/yellow]")
        asyncio.run(bot.run_once())
    else:
        console.print("[green]Starting 24/7 crypto trading bot (Ctrl+C to stop)...[/green]")
        asyncio.run(bot.run())


@crypto.command("status")
def crypto_status() -> None:
    """Show Binance account status and balances."""

    async def _status() -> None:
        from halal_trader.config import get_settings
        from halal_trader.crypto.exchange import BinanceClient

        settings = get_settings()
        client = BinanceClient(
            api_key=settings.binance_api_key,
            secret_key=settings.binance_secret_key,
            testnet=settings.binance_testnet,
        )

        try:
            await client.connect()

            # Account info
            account = await client.get_account()
            _print_crypto_account(account)

            # Balances
            balances = await client.get_balances()
            _print_crypto_balances(balances)

            # Ticker prices for configured pairs
            for pair in settings.crypto_pairs:
                try:
                    price = await client.get_ticker_price(pair)
                    console.print(f"  {pair}: [bold]${price:,.2f}[/bold]")
                except Exception:
                    console.print(f"  {pair}: [dim]N/A[/dim]")

        finally:
            await client.disconnect()

    asyncio.run(_status())


@crypto.command("history")
@click.option("--limit", default=50, help="Number of recent trades to show")
def crypto_history(limit: int) -> None:
    """Show crypto trade history and daily P&L."""

    async def _history() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.db.repository import Repository

        settings = get_settings()
        engine = await init_db(str(settings.db_path))
        repo = Repository(engine)

        try:
            # Recent crypto trades
            trades = await repo.get_recent_crypto_trades(limit)
            _print_crypto_trades(trades)

            # Daily P&L
            pnl_history = await repo.get_crypto_pnl_history(limit=14)
            _print_crypto_pnl(pnl_history)
        finally:
            await engine.dispose()

    asyncio.run(_history())


@crypto.command("screen")
def crypto_screen() -> None:
    """Show halal-screened crypto pairs."""

    async def _screen() -> None:
        from halal_trader.config import get_settings
        from halal_trader.crypto.screener import CryptoHalalScreener
        from halal_trader.db.models import init_db
        from halal_trader.db.repository import Repository

        settings = get_settings()
        engine = await init_db(str(settings.db_path))
        repo = Repository(engine)

        screener = CryptoHalalScreener(
            repo,
            coingecko_api_key=settings.coingecko_api_key,
            min_market_cap=settings.crypto_min_market_cap,
        )

        console.print("[yellow]Refreshing crypto halal screening...[/yellow]")
        await screener.refresh_screening()

        halal_symbols = await screener.get_halal_pairs()
        if halal_symbols:
            table = Table(title="Halal Crypto Tokens", show_header=True, header_style="bold cyan")
            table.add_column("#", style="dim", justify="right")
            table.add_column("Symbol")

            for i, sym in enumerate(sorted(halal_symbols), 1):
                table.add_row(str(i), sym)

            console.print(table)
            console.print(f"\n[green]{len(halal_symbols)} halal tokens found[/green]")
        else:
            console.print("[dim]No halal tokens in cache.[/dim]")

        await engine.dispose()

    asyncio.run(_screen())


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


def _print_crypto_config() -> None:
    from halal_trader.config import get_settings

    settings = get_settings()

    table = Table(title="Crypto Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="dim")
    table.add_column("Value")

    table.add_row("LLM Provider", settings.llm_provider.value)
    table.add_row("LLM Model", settings.llm_model)
    table.add_row("Trading Interval", f"{settings.crypto_trading_interval_seconds}s")
    table.add_row("Daily Return Target", f"{settings.crypto_daily_return_target:.1%}")
    table.add_row("Max Position Size", f"{settings.crypto_max_position_pct:.0%}")
    table.add_row("Daily Loss Limit", f"{settings.crypto_daily_loss_limit:.1%}")
    table.add_row("Max Positions", str(settings.crypto_max_simultaneous_positions))
    table.add_row("Trading Pairs", ", ".join(settings.crypto_pairs))
    table.add_row("Testnet", str(settings.binance_testnet))
    table.add_row(
        "Binance API Key",
        settings.binance_api_key[:8] + "..." if settings.binance_api_key else "[red]NOT SET[/red]",
    )
    table.add_row("Database", str(settings.db_path))

    console.print(table)


def _print_account(account: object) -> None:
    from halal_trader.domain.models import Account

    if isinstance(account, Account):
        table = Table(title="Account", show_header=True, header_style="bold cyan")
        table.add_column("Field", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Equity", f"${account.equity:,.2f}")
        table.add_row("Buying Power", f"${account.buying_power:,.2f}")
        table.add_row("Cash", f"${account.cash:,.2f}")
        table.add_row("Portfolio Value", f"${account.portfolio_value:,.2f}")
        table.add_row("Status", account.status)

        console.print(table)
    else:
        console.print(Panel(str(account), title="Account Info"))


def _print_positions(positions: object) -> None:
    from halal_trader.domain.models import Position

    if isinstance(positions, list) and positions:
        table = Table(title="Open Positions", show_header=True, header_style="bold cyan")
        table.add_column("Symbol")
        table.add_column("Qty", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")

        for p in positions:
            if isinstance(p, Position):
                style = "green" if p.unrealized_pl >= 0 else "red"
                table.add_row(
                    p.symbol,
                    str(p.qty),
                    f"${p.avg_entry_price:,.2f}",
                    f"${p.current_price:,.2f}",
                    Text(f"${p.unrealized_pl:+,.2f}", style=style),
                    Text(f"{p.unrealized_plpc:+.2%}", style=style),
                )

        console.print(table)
    else:
        console.print("[dim]No open positions.[/dim]")


def _print_clock(clock: object) -> None:
    from halal_trader.domain.models import MarketClock

    if isinstance(clock, MarketClock):
        status_text = (
            "[bold green]OPEN[/bold green]" if clock.is_open else "[bold red]CLOSED[/bold red]"
        )
        console.print(f"\nMarket: {status_text}")
        if not clock.is_open:
            console.print(f"  Next open: {clock.next_open or 'N/A'}")
        else:
            console.print(f"  Closes at: {clock.next_close or 'N/A'}")
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


# ── Crypto display helpers ─────────────────────────────────────


def _print_crypto_account(account: object) -> None:
    from halal_trader.domain.models import CryptoAccount

    if isinstance(account, CryptoAccount):
        table = Table(title="Crypto Account", show_header=True, header_style="bold cyan")
        table.add_column("Field", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Total Balance", f"${account.total_balance_usdt:,.2f} USDT")
        table.add_row("Available", f"${account.available_balance_usdt:,.2f} USDT")
        table.add_row("In Orders", f"${account.in_order_usdt:,.2f} USDT")

        console.print(table)
    else:
        console.print(Panel(str(account), title="Crypto Account"))


def _print_crypto_balances(balances: list) -> None:
    from halal_trader.domain.models import CryptoBalance

    if isinstance(balances, list) and balances:
        table = Table(title="Crypto Balances", show_header=True, header_style="bold cyan")
        table.add_column("Asset")
        table.add_column("Free", justify="right")
        table.add_column("Locked", justify="right")

        for b in balances:
            if isinstance(b, CryptoBalance):
                table.add_row(
                    b.asset,
                    f"{b.free:,.8f}",
                    f"{b.locked:,.8f}",
                )

        console.print(table)
    else:
        console.print("[dim]No crypto balances.[/dim]")


def _print_crypto_trades(trades: list) -> None:
    if not trades:
        console.print("[dim]No crypto trades recorded yet.[/dim]")
        return

    table = Table(title="Recent Crypto Trades", show_header=True, header_style="bold cyan")
    table.add_column("Time", style="dim")
    table.add_column("Pair")
    table.add_column("Side")
    table.add_column("Qty", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Exchange")
    table.add_column("Status")
    table.add_column("Reasoning", max_width=35)

    for t in trades:
        side_style = "green" if t.get("side") == "buy" else "red"
        table.add_row(
            str(t.get("timestamp", ""))[:19],
            str(t.get("pair", "")),
            Text(str(t.get("side", "")), style=side_style),
            f"{t.get('quantity', 0):.6f}",
            f"${t.get('price', 0) or 0:,.2f}",
            str(t.get("exchange", "binance")),
            str(t.get("status", "")),
            str(t.get("llm_reasoning", ""))[:35],
        )

    console.print(table)


def _print_crypto_pnl(pnl_history: list) -> None:
    if not pnl_history:
        console.print("[dim]No crypto P&L history yet.[/dim]")
        return

    table = Table(title="Crypto Daily P&L", show_header=True, header_style="bold cyan")
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
