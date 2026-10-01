"""`halal-trader ledger`: the broker's record of fills and equity (execution/ledger.py)."""

from __future__ import annotations

import asyncio
from datetime import date

import click

from halal_trader.logging import console


@click.group("ledger")
def ledger() -> None:
    """Broker-truth ledger: Alpaca's fills and equity, and our books against them."""


@ledger.command("sync")
def sync_cmd() -> None:
    """Copy new Alpaca activities and recent daily equity into the ledger."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.execution.alpaca_rest import AlpacaRestClient
        from halal_trader.execution.ledger import sync_broker_ledger

        settings = get_settings()
        engine = await init_db(settings.database_url)
        client = AlpacaRestClient(
            settings.alpaca.api_key,
            settings.alpaca.secret_key,
            paper=settings.alpaca.paper_trade,
        )
        try:
            r = await sync_broker_ledger(engine, client)
        finally:
            await client.aclose()
            await engine.dispose()
        console.print(
            f"activities: {r.activities_fetched} fetched, {r.activities_new} new; "
            f"equity days: {r.equity_days}"
        )

    asyncio.run(_run())


@ledger.command("reconcile")
@click.option(
    "--day", type=click.DateTime(["%Y-%m-%d"]), default=None, help="ET date (default today)."
)
def reconcile_cmd(day: object) -> None:
    """Compare one day's broker fills with the fills the bot recorded."""

    async def _run() -> int:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.execution.ledger import reconcile_fills
        from halal_trader.market_hours import today_eastern

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            when: date = day.date() if day is not None else today_eastern()  # type: ignore[attr-defined]
            rec = await reconcile_fills(engine, when)
        finally:
            await engine.dispose()
        if rec.clean:
            console.print(f"[green]{rec.day}: {rec.broker_fills} broker fills, books agree[/green]")
            return 0
        console.print(f"[red]{rec.day}: {len(rec.drifts)} difference(s)[/red]")
        for d in rec.drifts:
            console.print(
                f"  {d.symbol:6} {d.side:4}  broker {d.broker_qty:>10g}  "
                f"recorded {d.recorded_qty:>10g}  diff {d.difference:+g}"
            )
        return 1

    raise SystemExit(asyncio.run(_run()))


@ledger.command("performance")
@click.option("--since", type=click.DateTime(["%Y-%m-%d"]), default=None, help="First day.")
def performance_cmd(since: object) -> None:
    """Account performance from broker equity alone (net of deposits)."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.execution.ledger import performance

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            start = since.date() if since is not None else None  # type: ignore[attr-defined]
            p = await performance(engine, start=start)
        finally:
            await engine.dispose()
        if p is None:
            console.print(
                "[yellow]Fewer than two equity days on record; run `ledger sync`.[/yellow]"
            )
            return
        sharpe = f"{p.sharpe:.2f}" if p.sharpe is not None else "n/a"
        console.print(
            f"{p.first_day} -> {p.last_day}  ({p.days} trading days)\n"
            f"  equity        ${p.start_equity:,.2f} -> ${p.end_equity:,.2f}\n"
            f"  total return  {p.total_return:+.2%}   annualized {p.annualized_return:+.2%}\n"
            f"  volatility    {p.annualized_volatility:.2%} annualized   Sharpe {sharpe}\n"
            f"  max drawdown  {p.max_drawdown:.2%}\n"
            f"  best / worst  {p.best_day:+.2%} / {p.worst_day:+.2%}"
        )

    asyncio.run(_run())
