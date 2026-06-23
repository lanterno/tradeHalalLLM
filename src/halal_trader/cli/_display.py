"""Rich-based table/panel renderers shared by every CLI command."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from halal_trader.logging import console

_CLOUD_LLM_PROVIDERS = {"openai", "anthropic"}


def _warn_uncapped_cloud_llm() -> None:
    """Print a loud warning when a cloud LLM is configured with no spend cap.

    The combination ``LLM_PROVIDER ∈ {openai, anthropic}`` + ``LLM_DAILY_USD_CAP=0``
    means the bot will keep calling the cloud API regardless of cost — a
    runaway cycle loop can spend hundreds in an hour. Surface the risk at
    startup so the operator sees it before they walk away from the
    terminal.
    """
    from halal_trader.config import get_settings

    settings = get_settings()
    if settings.llm.provider.value not in _CLOUD_LLM_PROVIDERS:
        return
    if settings.llm.daily_usd_cap > 0:
        return
    console.print(
        Panel.fit(
            Text.from_markup(
                "[bold red]⚠ LLM_DAILY_USD_CAP=0[/bold red] with cloud provider "
                f"[bold]{settings.llm.provider.value}[/bold]\n"
                "No daily spend cap is enforced. A runaway cycle can spend\n"
                "hundreds in an hour. Set [cyan]LLM_DAILY_USD_CAP=10.0[/cyan] "
                "(or similar) in .env\n"
                "then restart — the kill-switch will engage if exceeded."
            ),
            border_style="red",
            title="Spend safety",
        )
    )


def print_config() -> None:
    """Print the stock-side configuration table."""
    from halal_trader.config import get_settings

    settings = get_settings()

    table = Table(title="Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="dim")
    table.add_column("Value")

    table.add_row("LLM Provider", settings.llm.provider.value)
    table.add_row("LLM Model", settings.llm.model)
    table.add_row("Trading Interval", f"{settings.stocks.trading_interval_minutes} min")
    table.add_row("Daily Return Target", f"{settings.stocks.daily_return_target:.1%}")
    table.add_row("Max Position Size", f"{settings.stocks.max_position_pct:.0%}")
    table.add_row("Daily Loss Limit", f"{settings.stocks.daily_loss_limit:.1%}")
    table.add_row("Max Positions", str(settings.stocks.max_simultaneous_positions))
    table.add_row("Paper Trading", str(settings.alpaca.paper_trade))
    table.add_row(
        "Alpaca API Key",
        settings.alpaca.api_key[:8] + "..." if settings.alpaca.api_key else "[red]NOT SET[/red]",
    )
    table.add_row(
        "Zoya API",
        "Configured" if settings.zoya.api_key else "[yellow]Not configured (defaults)[/yellow]",
    )
    table.add_row("Database", settings.database_url.split("@")[-1])

    console.print(table)
    _warn_uncapped_cloud_llm()


def print_crypto_config() -> None:
    """Print the crypto-side configuration table."""
    from halal_trader.config import get_settings

    settings = get_settings()

    table = Table(title="Crypto Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="dim")
    table.add_column("Value")

    table.add_row("LLM Provider", settings.llm.provider.value)
    table.add_row("LLM Model", settings.llm.model)
    table.add_row("Trading Interval", f"{settings.crypto.trading_interval_seconds}s")
    table.add_row("Daily Return Target", f"{settings.crypto.daily_return_target:.1%}")
    table.add_row("Max Position Size", f"{settings.crypto.max_position_pct:.0%}")
    table.add_row("Daily Loss Limit", f"{settings.crypto.daily_loss_limit:.1%}")
    table.add_row("Max Positions", str(settings.crypto.max_simultaneous_positions))
    table.add_row("Trading Pairs", ", ".join(settings.crypto.pairs))
    table.add_row("Testnet", str(settings.binance.testnet))
    table.add_row(
        "Binance API Key",
        settings.binance.api_key[:8] + "..." if settings.binance.api_key else "[red]NOT SET[/red]",
    )
    table.add_row("Database", settings.database_url.split("@")[-1])

    console.print(table)
    _warn_uncapped_cloud_llm()


def print_account(account: object) -> None:
    """Print an Alpaca Account snapshot."""
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


def print_positions(positions: object) -> None:
    """Print open Alpaca positions."""
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


def print_clock(clock: object) -> None:
    """Print the market clock."""
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


def print_trades(trades: list) -> None:
    """Print stock trade history."""
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


def print_pnl(pnl_history: list) -> None:
    """Print stock daily P&L history."""
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


def print_crypto_account(account: object) -> None:
    """Print Binance account snapshot."""
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


def print_crypto_balances(balances: list) -> None:
    """Print Binance per-asset balances."""
    from halal_trader.domain.models import CryptoBalance

    if isinstance(balances, list) and balances:
        table = Table(title="Crypto Balances", show_header=True, header_style="bold cyan")
        table.add_column("Asset")
        table.add_column("Free", justify="right")
        table.add_column("Locked", justify="right")
        for b in balances:
            if isinstance(b, CryptoBalance):
                table.add_row(b.asset, f"{b.free:,.8f}", f"{b.locked:,.8f}")
        console.print(table)
    else:
        console.print("[dim]No crypto balances.[/dim]")


def print_crypto_trades(trades: list) -> None:
    """Print crypto trade history."""
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


def print_crypto_pnl(pnl_history: list) -> None:
    """Print crypto daily P&L history."""
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


def print_backtest_results(result, pair: str, *, is_llm: bool = False) -> None:
    """Print a BacktestResult."""
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
    psr_style = "green" if result.psr >= 0.95 else "yellow" if result.psr >= 0.8 else "red"
    table.add_row("Prob. Sharpe (PSR>0)", Text(f"{result.psr:.1%}", style=psr_style))
    table.add_row("CVaR 5% (tail loss)", Text(f"{result.cvar_5pct:.3%}", style="red"))
    table.add_row("Avg Hold", f"{result.avg_hold_candles:.0f} candles")
    console.print(table)


def print_liquidation(results: list) -> None:
    """Print a liquidation result table from `halt --close-all`."""
    if not results:
        console.print("[dim]No positions to close.[/dim]")
        return
    tbl = Table(title="Liquidation", header_style="bold cyan")
    tbl.add_column("Market")
    tbl.add_column("Symbol")
    tbl.add_column("Qty", justify="right")
    tbl.add_column("Status")
    tbl.add_column("Detail")
    for r in results:
        style = "green" if r.status == "closed" else "yellow" if r.status == "skipped" else "red"
        tbl.add_row(
            r.market,
            r.symbol,
            f"{r.quantity:g}",
            Text(r.status, style=style),
            r.detail or "",
        )
    console.print(tbl)
