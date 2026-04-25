"""DB-vs-broker drift reconciliation command."""

from __future__ import annotations

import asyncio

import click
from rich.table import Table
from rich.text import Text

from halal_trader.logging import console


@click.command("reconcile")
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
                    d.notes or "",
                )
            console.print(tbl)
        finally:
            await engine.dispose()

    asyncio.run(_run())
