"""`halal-trader core`: the strict-halal core portfolio's own account."""

from __future__ import annotations

import asyncio
from typing import Any

import click

from halal_trader.logging import console


async def _run(execute: bool, monthly: bool | None) -> tuple[Any, list[Any]]:
    from halal_trader.config import get_settings
    from halal_trader.db.models import init_db
    from halal_trader.execution.alpaca_broker import AlpacaRestBroker
    from halal_trader.market_hours import today_eastern
    from halal_trader.portfolio import core_executor as ce

    settings = get_settings()
    core = settings.core
    if not (core.alpaca_api_key and core.alpaca_secret_key):
        raise click.ClickException(
            "set CORE_ALPACA_API_KEY and CORE_ALPACA_SECRET_KEY (the core's own paper account)"
        )
    if execute and not core.enabled:
        raise click.ClickException("CORE_ENABLED is false: preview with `halal-trader core plan`")
    engine = await init_db(settings.database_url)
    broker = AlpacaRestBroker(core.alpaca_api_key, core.alpaca_secret_key, paper=True)
    today = today_eastern()
    try:
        is_monthly = (await ce.monthly_due(engine, today)) if monthly is None else monthly
        p = await ce.plan(engine, broker, today=today, monthly=is_monthly, top_n=core.top_n)
        results = await ce.execute(engine, broker, p, today=today) if execute else []
        await ce.record_run(engine, p, today=today, executed=execute)
        return p, results
    finally:
        await broker.disconnect()
        await engine.dispose()


def _show(p: Any) -> None:
    kind = "monthly rebalance" if p.monthly else "forced sales only"
    console.print(
        f"[bold]core[/bold] {kind}; equity ${p.equity:,.2f}, cash ${p.cash:,.2f}, "
        f"screen {p.screen_as_of}"
    )
    if p.halted:
        console.print(f"  [red]halted:[/red] {p.halted}")
    for note in p.notes:
        console.print(f"  note: {note}")
    for o in p.orders:
        console.print(
            f"  {o.side:4} {o.symbol:6} {o.qty:12.6f} @ ~${o.price:,.2f} "
            f"= ${o.notional:,.2f}  ({o.reason})"
        )
    if not p.orders and not p.halted:
        console.print("  nothing to trade")


@click.group("core")
def core() -> None:
    """The strict-halal core portfolio, on its own Alpaca account."""


@core.command("plan")
@click.option(
    "--monthly/--forced-only", default=None, help="Default: monthly if not yet run this month."
)
def plan_cmd(monthly: bool | None) -> None:
    """Show the orders the core would place now, without placing any."""
    p, _ = asyncio.run(_run(False, monthly))
    _show(p)


@core.command("run")
@click.option(
    "--monthly/--forced-only", default=None, help="Default: monthly if not yet run this month."
)
def run_cmd(monthly: bool | None) -> None:
    """Place the core's orders now (needs CORE_ENABLED=true)."""
    p, results = asyncio.run(_run(True, monthly))
    _show(p)
    console.print(
        f"  {sum(1 for r in results if r['st'] == 'submitted')} submitted, "
        f"{sum(1 for r in results if r['st'] != 'submitted')} refused"
    )
