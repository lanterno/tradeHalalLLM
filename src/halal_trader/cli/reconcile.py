"""DB-vs-broker drift reconciliation commands.

``halal-trader reconcile check {crypto|stocks}`` — one-shot drift pass.
``halal-trader reconcile fix-orphans``           — backfill stale pending
Trade rows so the reconciler stops flagging them as phantom positions.
"""

from __future__ import annotations

import asyncio

import click
from rich.table import Table
from rich.text import Text

from halal_trader.logging import console


@click.group("reconcile")
def reconcile() -> None:
    """DB-vs-broker drift commands."""


@reconcile.command("check")
@click.argument("market", type=click.Choice(["crypto", "stocks"]))
@click.option(
    "--threshold",
    default=0.01,
    show_default=True,
    help="Drift threshold (fractional, e.g. 0.01 = 1%).",
)
def reconcile_check(market: str, threshold: float) -> None:
    """Compare DB open trades to broker balances and surface drift."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.core import reconcile as recon
        from halal_trader.db.models import init_db

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            if market == "crypto":
                from halal_trader.crypto.exchange import BinanceClient

                broker = BinanceClient(
                    api_key=settings.binance.api_key,
                    secret_key=settings.binance.secret_key,
                    testnet=settings.binance.testnet,
                    configured_pairs=settings.crypto.pairs,
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
                    d.notes or ("settling" if d.is_settling else ""),
                )
            console.print(tbl)
        finally:
            await engine.dispose()

    asyncio.run(_run())


@reconcile.command("fix-orphans")
@click.option(
    "--dry-run/--apply",
    default=True,
    show_default=True,
    help="By default, print what would change without touching the DB.",
)
@click.option(
    "--min-age-minutes",
    default=5,
    show_default=True,
    help="Only consider Trade rows older than this many minutes "
    "(gives in-flight orders a chance to confirm).",
)
@click.option(
    "--no-broker",
    is_flag=True,
    default=False,
    help="Skip broker lookups; mark every orphan with no order_id as rejected. "
    "Use when Alpaca is unreachable.",
)
def reconcile_fix_orphans(dry_run: bool, min_age_minutes: int, no_broker: bool) -> None:
    """Clean up stale 'pending' Trade rows that never filled.

    These rows are the source of long-running 'phantom position' drift
    warnings — the executor recorded a Trade for an order that never
    actually placed (or that the broker silently rejected). For each
    candidate, we either ask the broker for the order's real terminal
    state, or — if no order id was ever assigned — mark it 'rejected'
    so the reconciler stops counting it.
    """

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.core import reconcile as recon
        from halal_trader.db.models import init_db

        settings = get_settings()
        engine = await init_db(settings.database_url)
        broker = None
        try:
            if not no_broker:
                from halal_trader.mcp.client import AlpacaMCPClient

                broker = AlpacaMCPClient()
                try:
                    await broker.connect()
                except Exception as exc:  # noqa: BLE001
                    console.print(
                        f"[yellow]Broker unreachable ({exc!r}); falling back to "
                        f"--no-broker semantics for this run.[/yellow]"
                    )
                    broker = None

            report = await recon.fix_stocks_orphans(
                engine=engine,
                broker=broker,
                min_age_minutes=min_age_minutes,
                dry_run=dry_run,
            )

            mode = "DRY RUN" if dry_run else "APPLIED"
            console.print(
                f"[bold]{mode}[/bold] · scanned {report.candidates} orphan candidate(s)"
                + (f" · {report.updated} row(s) updated" if not dry_run else "")
            )
            if not report.fixes:
                console.print("[green]No orphan trades found.[/green]")
                return

            tbl = Table(
                title=f"Orphan fix-up ({mode})",
                header_style="bold cyan",
            )
            tbl.add_column("Trade #", justify="right")
            tbl.add_column("Symbol")
            tbl.add_column("Side")
            tbl.add_column("Qty", justify="right")
            tbl.add_column("Order id")
            tbl.add_column("Old → New")
            tbl.add_column("Source")
            tbl.add_column("Notes")

            for f in report.fixes:
                arrow = (
                    Text(f"{f.old_status} → {f.new_status}", style="green")
                    if f.old_status != f.new_status
                    else Text(f"{f.old_status} (unchanged)", style="dim")
                )
                tbl.add_row(
                    str(f.trade_id),
                    f.symbol,
                    f.side,
                    f"{f.quantity:g}",
                    f.order_id[:12] + ("…" if len(f.order_id) > 12 else ""),
                    arrow,
                    f.source,
                    f.notes or "",
                )
            console.print(tbl)

            if dry_run and any(f.old_status != f.new_status for f in report.fixes):
                console.print(
                    "[dim]Re-run with --apply to persist these changes.[/dim]"
                )
        finally:
            if broker is not None:
                try:
                    await broker.disconnect()
                except Exception:  # noqa: BLE001
                    pass
            await engine.dispose()

    asyncio.run(_run())
