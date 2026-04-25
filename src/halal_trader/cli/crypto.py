"""Crypto trading commands (Binance-based, 24/7)."""

from __future__ import annotations

import asyncio

import click
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from halal_trader.cli._display import (
    print_backtest_results,
    print_crypto_account,
    print_crypto_balances,
    print_crypto_config,
    print_crypto_pnl,
    print_crypto_trades,
)
from halal_trader.logging import console


@click.group()
def crypto() -> None:
    """Crypto trading commands (Binance-based, 24/7)."""


@crypto.command("start")
@click.option("--once", is_flag=True, help="Run a single crypto trading cycle then exit")
def crypto_start(once: bool) -> None:
    """Start the crypto trading bot."""
    from halal_trader.config import get_settings
    from halal_trader.crypto.scheduler import CryptoTradingBot

    settings = get_settings()
    mode = "TESTNET" if settings.binance.testnet else "PRODUCTION"

    console.print(
        Panel(
            f"[bold green]Halal Crypto Trader[/bold green]\n"
            f"LLM-powered crypto trading bot\n"
            f"[dim]{mode} mode — Binance[/dim]",
            title="Starting",
            border_style="green",
        )
    )
    print_crypto_config()

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
            api_key=settings.binance.api_key,
            secret_key=settings.binance.secret_key,
            testnet=settings.binance.testnet,
            configured_pairs=settings.crypto.pairs,
        )
        try:
            await client.connect()
            print_crypto_account(await client.get_account())
            print_crypto_balances(await client.get_balances())
            for pair in settings.crypto.pairs:
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
            print_crypto_trades(await repo.get_recent_crypto_trades(limit))
            print_crypto_pnl(await repo.get_crypto_pnl_history(limit=14))
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

        if stats.by_exit_reason:
            reasons_table = Table(title="Exit Reasons", show_header=True, header_style="bold cyan")
            reasons_table.add_column("Reason")
            reasons_table.add_column("Count", justify="right")
            for reason, count in sorted(
                stats.by_exit_reason.items(), key=lambda x: x[1], reverse=True
            ):
                reasons_table.add_row(reason, str(count))
            console.print(reasons_table)

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
            api_key=settings.binance.api_key,
            secret_key=settings.binance.secret_key,
            testnet=settings.binance.testnet,
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
                    cache_dir=str(settings.ml.models_dir),
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

            print_backtest_results(result, pair, is_llm=llm)
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
            coingecko_api_key=settings.coingecko.api_key,
            min_market_cap=settings.crypto.min_market_cap,
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
