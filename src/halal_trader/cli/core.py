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
    broker = AlpacaRestBroker(core.alpaca_api_key, core.alpaca_secret_key, paper=core.paper)
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
    from halal_trader.config import get_settings

    kind = "monthly rebalance" if p.monthly else "forced sales only"
    where = "paper account" if get_settings().core.paper else "[bold red]LIVE account[/bold red]"
    console.print(
        f"[bold]core[/bold] on its {where}: {kind}; equity ${p.equity:,.2f}, "
        f"cash ${p.cash:,.2f}, screen {p.screen_as_of}"
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


@core.command("readiness")
def readiness_cmd() -> None:
    """Has the core earned real money? Its live-money gate, criterion by criterion."""

    async def _run() -> Any:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.market_hours import today_eastern
        from halal_trader.portfolio.readiness import check

        engine = await init_db(get_settings().database_url)
        try:
            return await check(engine, today=today_eastern())
        finally:
            await engine.dispose()

    r = asyncio.run(_run())
    te = f"{r.tracking_error:.2%}" if r.tracking_error is not None else "n/a"
    gap = f"{r.gap:+.2%}" if r.gap is not None else "n/a"
    console.print(
        f"core readiness: {'[green]READY[/green]' if r.ready else '[yellow]not yet[/yellow]'} — "
        f"{r.days} days, {r.monthly_runs} monthly run(s), tracking error {te}, gap {gap}, "
        f"{r.refused} refused, {r.halted} halted"
    )
    for failure in r.failures:
        console.print(f"  - {failure}")


@core.command("slippage")
@click.option("--days", default=30, show_default=True, help="How far back, in calendar days.")
def slippage_cmd(days: int) -> None:
    """How the core's orders filled: against the arrival price and against the close."""

    async def _run() -> Any:
        from datetime import timedelta

        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.market_hours import today_eastern
        from halal_trader.portfolio.execution_quality import report

        engine = await init_db(get_settings().database_url)
        try:
            today = today_eastern()
            return await report(engine, today - timedelta(days=days), today)
        finally:
            await engine.dispose()

    r = asyncio.run(_run())
    if not r.orders:
        console.print(f"no core orders since {r.start}")
        return

    def bps(x: float | None) -> str:
        return "n/a" if x is None else f"{x:+.1f} bps"

    console.print(
        f"core execution {r.start} to {r.end}: {len(r.orders)} orders, {r.count('filled')} filled, "
        f"{r.count('partial')} partial, {r.count('unfilled')} unfilled; "
        f"${r.filled_notional:,.2f} traded"
    )
    console.print(
        f"  vs arrival {bps(r.vs_arrival_bps)} · vs close {bps(r.vs_close_bps)} "
        f"(the book assumes +5.0 bps)"
        + (
            f" · ${r.cost_vs_close_usd:,.2f} against trading at the close"
            if r.cost_vs_close_usd is not None
            else ""
        )
    )
    for o in r.worst():
        console.print(
            f"  {o.side} {o.symbol}: {bps(o.vs_arrival_bps)} vs arrival, "
            f"{bps(o.vs_close_bps)} vs close"
        )


@core.command("digest")
@click.option("--send", is_flag=True, help="Send it to Telegram as well as printing it.")
def digest_cmd(send: bool) -> None:
    """The weekly digest, now (it is sent automatically every Friday 17:15 New York)."""

    async def _run() -> str:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.market_hours import today_eastern
        from halal_trader.notifications.digest import build

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            message = await build(engine, settings, today=today_eastern())
            if send:
                from halal_trader.notifications.telegram import TelegramNotifier

                notifier = TelegramNotifier(settings.telegram.bot_token, settings.telegram.chat_id)
                await notifier.send(message)
                await notifier.close()
            return message
        finally:
            await engine.dispose()

    console.print(asyncio.run(_run()), markup=False)
