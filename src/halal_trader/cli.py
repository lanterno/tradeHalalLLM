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
        engine = await init_db(str(settings.resolve_db_path()))
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


# ── Database Command Group ─────────────────────────────────────


@cli.group("db")
def db_group() -> None:
    """Database / Alembic schema management."""


@db_group.command("migrate")
@click.option("--revision", default="head", help="Target revision (default: head)")
def db_migrate(revision: str) -> None:
    """Apply Alembic migrations forward to the target revision."""
    from halal_trader.db import admin

    target = revision
    console.print(f"[yellow]Applying migrations up to {target}...[/yellow]")
    admin.upgrade(target)
    console.print(f"[green]Database migrated to {admin.current() or 'unknown'}[/green]")


@db_group.command("stamp")
@click.argument("revision", default="head")
def db_stamp(revision: str) -> None:
    """Mark the DB as being at REVISION without running migrations.

    Use this once when adopting a database that was previously managed by
    SQLModel.metadata.create_all (no alembic_version table).
    """
    from halal_trader.db import admin

    console.print(f"[yellow]Stamping database at revision {revision}...[/yellow]")
    admin.stamp(revision)
    console.print(f"[green]Database stamped at {admin.current() or 'unknown'}[/green]")


@db_group.command("current")
def db_current() -> None:
    """Show the current and head Alembic revisions."""
    from halal_trader.db import admin

    cur = admin.current()
    head = admin.head()
    cur_str = cur or "[red]uninitialized[/red]"
    style = "green" if cur == head else "yellow"
    console.print(f"Current: [{style}]{cur_str}[/{style}]")
    console.print(f"Head:    [bold]{head}[/bold]")


@db_group.command("revision")
@click.option("-m", "--message", required=True, help="Short description of the migration")
@click.option("--autogenerate", is_flag=True, help="Diff models against DB to autogenerate")
def db_revision(message: str, autogenerate: bool) -> None:
    """Create a new Alembic revision file under alembic/versions/."""
    from halal_trader.db import admin

    admin.revision(message=message, autogenerate=autogenerate)
    console.print(f"[green]Created revision: {message}[/green]")


# ── Kill-switch ────────────────────────────────────────────────


@cli.command("halt")
@click.option("--reason", required=True, help="Why are you halting? (audit trail)")
@click.option(
    "--close-all",
    is_flag=True,
    help="Also close all open positions immediately (panic button).",
)
def halt(reason: str, close_all: bool) -> None:
    """Engage the operator kill-switch — bots refuse new entries until resumed."""

    async def _halt() -> None:
        from halal_trader.config import get_settings
        from halal_trader.core import halt as halt_module
        from halal_trader.db.models import init_db

        settings = get_settings()
        engine = await init_db(str(settings.resolve_db_path()))
        try:
            status = await halt_module.set_halt(engine, reason=reason)
            console.print(
                f"[red]KILL-SWITCH ENGAGED[/red] "
                f"(by {status.set_by} at {status.set_at}): {status.reason}"
            )

            if close_all:
                console.print(
                    "[yellow]--close-all is set. To liquidate positions, "
                    "run `halal-trader status` / `crypto status` and use the "
                    "broker UI for now. Auto-liquidation lands in a follow-up.[/yellow]"
                )
        finally:
            await engine.dispose()

    asyncio.run(_halt())


@cli.command("resume")
def resume() -> None:
    """Disengage the operator kill-switch."""

    async def _resume() -> None:
        from halal_trader.config import get_settings
        from halal_trader.core import halt as halt_module
        from halal_trader.db.models import init_db

        settings = get_settings()
        engine = await init_db(str(settings.resolve_db_path()))
        try:
            status = await halt_module.clear_halt(engine)
            console.print(
                f"[green]Kill-switch cleared[/green] "
                f"(was set by {status.set_by} at {status.set_at} — reason: {status.reason})"
            )
        finally:
            await engine.dispose()

    asyncio.run(_resume())


@cli.command("halt-status")
def halt_status() -> None:
    """Show the current kill-switch state."""

    async def _status() -> None:
        from halal_trader.config import get_settings
        from halal_trader.core.halt import get_status
        from halal_trader.db.models import init_db

        settings = get_settings()
        engine = await init_db(str(settings.resolve_db_path()))
        try:
            s = await get_status(engine)
            if s.enabled:
                console.print(f"[red]HALTED[/red] (by {s.set_by} at {s.set_at}): {s.reason}")
            else:
                console.print("[green]Running[/green] — kill-switch is off.")
                if s.set_by:
                    console.print(f"[dim]Last set by {s.set_by} at {s.set_at}: {s.reason}[/dim]")
        finally:
            await engine.dispose()

    asyncio.run(_status())


# ── Backup ─────────────────────────────────────────────────────


@cli.command("backup")
@click.option(
    "--retention-days",
    default=None,
    type=int,
    help="Override BACKUP_RETENTION_DAYS for this run.",
)
@click.option(
    "--weekly-count",
    default=None,
    type=int,
    help="Override BACKUP_WEEKLY_COUNT for this run.",
)
def backup(retention_days: int | None, weekly_count: int | None) -> None:
    """Create a gzipped SQLite backup and prune old ones."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.backup import prune_backups, run_backup

        settings = get_settings()
        retention = retention_days if retention_days is not None else settings.backup_retention_days
        weekly = weekly_count if weekly_count is not None else settings.backup_weekly_count

        result = run_backup(
            db_path=settings.resolve_db_path(),
            backup_dir=settings.backup_dir,
        )
        console.print(
            f"[green]Backup written[/green]: {result.path} "
            f"([dim]{result.size_bytes / 1024:.1f} KB[/dim])"
        )
        deleted = prune_backups(
            backup_dir=settings.backup_dir,
            retention_days=retention,
            weekly_count=weekly,
        )
        if deleted:
            console.print(f"[dim]Pruned {len(deleted)} old backup file(s)[/dim]")

    asyncio.run(_run())


# ── Reconciliation ─────────────────────────────────────────────


@cli.command("reconcile")
@click.argument("market", type=click.Choice(["crypto", "stocks"]))
@click.option(
    "--threshold",
    default=0.01,
    show_default=True,
    help="Drift threshold (fractional, e.g. 0.01 = 1%).",
)
def reconcile(market: str, threshold: float) -> None:
    """Compare DB open trades to broker balances and surface drift."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.core import reconcile as recon
        from halal_trader.db.models import init_db

        settings = get_settings()
        engine = await init_db(str(settings.resolve_db_path()))
        try:
            if market == "crypto":
                from halal_trader.crypto.exchange import BinanceClient

                broker = BinanceClient(
                    api_key=settings.binance_api_key,
                    secret_key=settings.binance_secret_key,
                    testnet=settings.binance_testnet,
                    configured_pairs=settings.crypto_pairs,
                )
                await broker.connect()
                try:
                    report = await recon.reconcile_crypto(
                        engine=engine, broker=broker, threshold_pct=threshold
                    )
                finally:
                    await broker.disconnect()
            else:
                from halal_trader.mcp.client import AlpacaMCPClient

                broker = AlpacaMCPClient()
                await broker.connect()
                try:
                    report = await recon.reconcile_stocks(
                        engine=engine, broker=broker, threshold_pct=threshold
                    )
                finally:
                    await broker.disconnect()

            console.print(
                f"[dim]Checked {report.checked_symbols} symbol(s) at "
                f"threshold {threshold:.1%}[/dim]"
            )
            if not report.has_drift:
                console.print(f"[green]No drift detected for {market}.[/green]")
                return

            tbl = Table(title=f"Reconciliation Drift ({market})", header_style="bold cyan")
            tbl.add_column("Symbol")
            tbl.add_column("DB Qty", justify="right")
            tbl.add_column("Broker Qty", justify="right")
            tbl.add_column("Drift %", justify="right")
            tbl.add_column("Drift $", justify="right")
            tbl.add_column("Notes")

            for d in report.drifts:
                usd = f"${d.drift_usd:,.2f}" if d.drift_usd is not None else "-"
                tbl.add_row(
                    d.symbol,
                    f"{d.db_quantity:g}",
                    f"{d.broker_quantity:g}",
                    Text(f"{d.drift_pct * 100:.2f}%", style="red"),
                    usd,
                    d.notes or "",
                )
            console.print(tbl)
        finally:
            await engine.dispose()

    asyncio.run(_run())


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
            configured_pairs=settings.crypto_pairs,
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
        engine = await init_db(str(settings.resolve_db_path()))
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


@crypto.command("stats")
@click.option("--days", default=7, help="Lookback period in days")
def crypto_stats(days: int) -> None:
    """Show trading performance metrics and recent round-trips."""

    async def _stats() -> None:
        from halal_trader.config import get_settings
        from halal_trader.crypto.analytics import PerformanceAnalytics
        from halal_trader.db.models import init_db
        from halal_trader.db.repository import Repository

        settings = get_settings()
        engine = await init_db(str(settings.resolve_db_path()))
        repo = Repository(engine)
        analytics = PerformanceAnalytics(repo)

        stats = await analytics.compute_stats(lookback_days=days)

        if stats.total_trades == 0:
            console.print(f"[dim]No completed trades in the last {days} days.[/dim]")
            await engine.dispose()
            return

        # Summary table
        summary = Table(
            title=f"Performance Summary (last {days} days)",
            show_header=True,
            header_style="bold cyan",
        )
        summary.add_column("Metric", style="dim")
        summary.add_column("Value", justify="right")

        summary.add_row("Total Trades", str(stats.total_trades))
        summary.add_row("Wins / Losses", f"{stats.wins} / {stats.losses}")

        wr_style = "green" if stats.win_rate >= 0.5 else "red"
        summary.add_row("Win Rate", Text(f"{stats.win_rate:.0%}", style=wr_style))
        summary.add_row("Avg Win", Text(f"{stats.avg_win_pct:+.2%}", style="green"))
        summary.add_row("Avg Loss", Text(f"{stats.avg_loss_pct:+.2%}", style="red"))

        pf_style = "green" if stats.profit_factor >= 1.0 else "red"
        pf_str = f"{stats.profit_factor:.2f}" if stats.profit_factor < 100 else "∞"
        summary.add_row("Profit Factor", Text(pf_str, style=pf_style))

        pnl_style = "green" if stats.total_pnl >= 0 else "red"
        summary.add_row("Total P&L", Text(f"${stats.total_pnl:+,.2f}", style=pnl_style))
        summary.add_row("Max Drawdown", Text(f"{stats.max_drawdown_pct:.2%}", style="red"))

        hold_str = f"{stats.avg_hold_minutes:.0f} min"
        if stats.avg_hold_minutes >= 60:
            hold_str = f"{stats.avg_hold_minutes / 60:.1f} hrs"
        summary.add_row("Avg Hold Time", hold_str)

        streak_style = "green" if stats.streak_type == "wins" else "red"
        summary.add_row(
            "Current Streak",
            Text(f"{stats.streak} {stats.streak_type}", style=streak_style),
        )

        if stats.best_pair:
            summary.add_row(
                "Best Pair",
                Text(f"{stats.best_pair} (${stats.best_pair_pnl:+,.2f})", style="green"),
            )
            summary.add_row(
                "Worst Pair",
                Text(f"{stats.worst_pair} (${stats.worst_pair_pnl:+,.2f})", style="red"),
            )

        console.print(summary)

        # Exit reasons breakdown
        if stats.by_exit_reason:
            reasons_table = Table(title="Exit Reasons", show_header=True, header_style="bold cyan")
            reasons_table.add_column("Reason")
            reasons_table.add_column("Count", justify="right")
            for reason, count in sorted(
                stats.by_exit_reason.items(), key=lambda x: x[1], reverse=True
            ):
                reasons_table.add_row(reason, str(count))
            console.print(reasons_table)

        # Recent round-trips
        round_trips = await repo.get_completed_round_trips(limit=10, lookback_days=days)
        if round_trips:
            rt_table = Table(title="Recent Round-Trips", show_header=True, header_style="bold cyan")
            rt_table.add_column("Pair")
            rt_table.add_column("Entry", justify="right")
            rt_table.add_column("Exit", justify="right")
            rt_table.add_column("P&L", justify="right")
            rt_table.add_column("P&L %", justify="right")
            rt_table.add_column("Duration")
            rt_table.add_column("Reason")

            for rt in round_trips:
                pnl_style = "green" if rt["pnl"] >= 0 else "red"
                dur = rt["duration_minutes"]
                dur_str = f"{dur:.0f}m" if dur < 60 else f"{dur / 60:.1f}h"
                rt_table.add_row(
                    rt["pair"],
                    f"${rt['buy_price']:,.2f}",
                    f"${rt['sell_price']:,.2f}",
                    Text(f"${rt['pnl']:+,.2f}", style=pnl_style),
                    Text(f"{rt['pnl_pct']:+.2%}", style=pnl_style),
                    dur_str,
                    rt.get("exit_reason") or "",
                )
            console.print(rt_table)

        await engine.dispose()

    asyncio.run(_stats())


@crypto.command("backtest")
@click.option("--pair", default="BTCUSDT", help="Trading pair to backtest")
@click.option("--candles", default=1000, help="Number of historical candles")
@click.option("--balance", default=10000.0, help="Starting balance in USDT")
@click.option("--rsi-buy", default=35.0, help="RSI buy threshold")
@click.option("--rsi-sell", default=65.0, help="RSI sell threshold")
@click.option("--sl", default=0.01, help="Stop-loss percentage")
@click.option("--tp", default=0.015, help="Take-profit percentage")
@click.option("--llm", is_flag=True, help="Use LLM strategy instead of rule-based")
@click.option("--cycle-interval", default=5, help="LLM: run every N candles (reduces API calls)")
def crypto_backtest(
    pair: str,
    candles: int,
    balance: float,
    rsi_buy: float,
    rsi_sell: float,
    sl: float,
    tp: float,
    llm: bool,
    cycle_interval: int,
) -> None:
    """Run a backtest on historical data (rule-based or LLM-driven)."""

    async def _backtest() -> None:
        from halal_trader.config import get_settings
        from halal_trader.crypto.backtest import BacktestEngine, fetch_historical_klines
        from halal_trader.crypto.exchange import BinanceClient

        settings = get_settings()
        client = BinanceClient(
            api_key=settings.binance_api_key,
            secret_key=settings.binance_secret_key,
            testnet=settings.binance_testnet,
            configured_pairs=[pair],
        )

        try:
            await client.connect()
            console.print(f"[yellow]Fetching {candles} candles for {pair}...[/yellow]")
            klines = await fetch_historical_klines(client, pair, limit=candles)

            if len(klines) < 100:
                console.print(f"[red]Insufficient data: {len(klines)} candles (need 100+)[/red]")
                return

            if llm:
                from halal_trader.core.llm import create_llm
                from halal_trader.crypto.backtest import LLMBacktestEngine

                llm_backend = create_llm(settings)
                console.print(
                    f"[yellow]Running LLM backtest on {len(klines)} candles "
                    f"(every {cycle_interval} candles, SL={sl:.1%}, TP={tp:.1%})...[/yellow]"
                )

                engine = LLMBacktestEngine(
                    llm_backend,
                    initial_balance=balance,
                    sl_pct=sl,
                    tp_pct=tp,
                    cache_dir=str(settings.ml_models_dir),
                )
                result = await engine.run(pair, klines, cycle_interval=cycle_interval)
            else:
                console.print(
                    f"[yellow]Running rule-based backtest on {len(klines)} candles "
                    f"(RSI buy<{rsi_buy}, sell>{rsi_sell}, SL={sl:.1%}, TP={tp:.1%})...[/yellow]"
                )

                engine = BacktestEngine(
                    initial_balance=balance,
                    rsi_buy=rsi_buy,
                    rsi_sell=rsi_sell,
                    sl_pct=sl,
                    tp_pct=tp,
                )
                result = await engine.run(pair, klines)

            _print_backtest_results(result, pair, is_llm=llm)

        finally:
            await client.disconnect()

    asyncio.run(_backtest())


@crypto.command("screen")
def crypto_screen() -> None:
    """Show halal-screened crypto pairs."""

    async def _screen() -> None:
        from halal_trader.config import get_settings
        from halal_trader.crypto.screener import CryptoHalalScreener
        from halal_trader.db.models import init_db
        from halal_trader.db.repository import Repository

        settings = get_settings()
        engine = await init_db(str(settings.resolve_db_path()))
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
        console.print(f"\nMarket (US Eastern): {status_text}")
        if clock.timestamp:
            console.print(f"  As of: {clock.timestamp.strftime('%Y-%m-%d %H:%M:%S')} ET")
        if not clock.is_open:
            console.print(f"  Next open: {clock.next_open or 'N/A'} ET")
        else:
            console.print(f"  Closes at: {clock.next_close or 'N/A'} ET")
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


def _print_backtest_results(result, pair: str, *, is_llm: bool = False) -> None:
    mode = "LLM" if is_llm else "Rule-Based"
    table = Table(
        title=f"{mode} Backtest: {pair} ({result.start_date} to {result.end_date})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    ret_style = "green" if result.total_return_pct >= 0 else "red"
    table.add_row("Initial Balance", f"${result.initial_balance:,.2f}")
    table.add_row("Final Balance", Text(f"${result.final_balance:,.2f}", style=ret_style))
    table.add_row("Total Return", Text(f"{result.total_return_pct:+.2%}", style=ret_style))
    table.add_row("Total Trades", str(result.total_trades))
    table.add_row("Wins / Losses", f"{result.wins} / {result.losses}")

    wr_style = "green" if result.win_rate >= 0.5 else "red"
    table.add_row("Win Rate", Text(f"{result.win_rate:.0%}", style=wr_style))

    pf_str = f"{result.profit_factor:.2f}" if result.profit_factor < 100 else "inf"
    table.add_row("Profit Factor", pf_str)
    table.add_row("Max Drawdown", Text(f"{result.max_drawdown_pct:.2%}", style="red"))
    table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    table.add_row("Sortino Ratio", f"{result.sortino_ratio:.2f}")
    table.add_row("Avg Hold", f"{result.avg_hold_candles:.0f} candles")

    console.print(table)


@cli.command()
@click.option("--port", default=8082, help="Dashboard port")
@click.option("--host", default="0.0.0.0", help="Dashboard host")
def dashboard(port: int, host: str) -> None:
    """Launch the web dashboard."""
    try:
        import uvicorn

        from halal_trader.web.app import create_app

        console.print(
            Panel(
                f"[bold green]Halal Trader Dashboard[/bold green]\n[dim]http://{host}:{port}[/dim]",
                title="Starting",
                border_style="green",
            )
        )
        app = create_app()
        uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError:
        console.print(
            "[red]Dashboard requires fastapi and uvicorn. "
            "Install with: pip install fastapi uvicorn[/red]"
        )


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
