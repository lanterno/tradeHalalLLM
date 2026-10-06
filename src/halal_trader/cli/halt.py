"""Operator kill-switch commands."""

from __future__ import annotations

import asyncio
from typing import Any

import click

from halal_trader.cli._display import print_liquidation
from halal_trader.logging import console


@click.command("halt")
@click.option("--reason", required=True, help="Why are you halting? (audit trail)")
@click.option(
    "--close-all",
    type=click.Choice(["stocks", "core"]),
    default=None,
    help=(
        "Also liquidate, once the halt is set: 'stocks' the day-trader's account, "
        "'core' the core portfolio's own account."
    ),
)
def halt(reason: str, close_all: str | None) -> None:
    """Engage the operator kill-switch — bots refuse new entries until resumed.

    The kill-switch is engaged FIRST, so nothing can open while the rest
    runs. With ``--close-all stocks``, every open position is then
    liquidated, best-effort: a broker that cannot be reached is reported,
    but the halt is already in place. (It used to liquidate first, and a
    broker error aborted the command before the halt was ever set.)

    ``--close-all core`` does the same to the core portfolio's account, with
    its own keys (CORE_ALPACA_*): open orders cancelled, every position
    closed. Without it the halt leaves the core's holdings in place and lets
    only its forced (screen) sales through.
    """

    async def _halt() -> None:
        from halal_trader.config import get_settings
        from halal_trader.core import halt as halt_module
        from halal_trader.core.liquidate import liquidate_stocks
        from halal_trader.db.models import init_db

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            status = await halt_module.set_halt(engine, reason=reason)
            console.print(
                f"[red]KILL-SWITCH ENGAGED[/red] "
                f"(by {status.set_by} at {status.set_at}): {status.reason}"
            )
            if close_all == "stocks":
                from halal_trader.mcp.client import AlpacaMCPClient

                mcp = AlpacaMCPClient()
                try:
                    await mcp.connect()
                    print_liquidation(await liquidate_stocks(mcp))
                except Exception as exc:  # noqa: BLE001 -- the halt already holds
                    console.print(
                        f"[red]Liquidation failed ({type(exc).__name__}: {exc}).[/red] "
                        "The kill-switch IS engaged; close positions by hand or retry."
                    )
                    raise SystemExit(1) from exc
                finally:
                    await mcp.disconnect()
            elif close_all == "core":
                await _close_core(settings)
        finally:
            await engine.dispose()

    asyncio.run(_halt())


async def _close_core(settings: Any) -> None:
    """Liquidate the core's account (the halt is already engaged)."""
    from halal_trader.core.liquidate import LiquidationResult, liquidate_stocks
    from halal_trader.execution.alpaca_broker import AlpacaRestBroker

    core = settings.core
    if not (core.alpaca_api_key and core.alpaca_secret_key):
        console.print("[red]No core keys (CORE_ALPACA_API_KEY/SECRET): nothing closed.[/red]")
        raise SystemExit(1)
    where = "paper" if core.paper else "LIVE"
    console.print(f"Closing every position of the core's {where} account...")
    broker = AlpacaRestBroker(core.alpaca_api_key, core.alpaca_secret_key, paper=core.paper)
    try:
        held = await broker.get_all_positions()
        results = await liquidate_stocks(broker)
    except Exception as exc:  # noqa: BLE001 -- the halt already holds
        console.print(
            f"[red]Core liquidation failed ({type(exc).__name__}: {exc}).[/red] "
            "The kill-switch IS engaged; close the core's positions by hand or retry."
        )
        raise SystemExit(1) from exc
    finally:
        await broker.disconnect()
    if any(r.status == "error" for r in results):
        print_liquidation(results)
        console.print("[red]The kill-switch IS engaged; the core's positions may be open.[/red]")
        raise SystemExit(1)
    # What was held when the close went out; the broker fills the sells.
    print_liquidation(
        [
            LiquidationResult("core", p.symbol, float(p.qty), "closed", "close_all_positions")
            for p in held
        ]
    )


@click.command("resume")
def resume() -> None:
    """Disengage the operator kill-switch."""

    async def _resume() -> None:
        from halal_trader.config import get_settings
        from halal_trader.core import halt as halt_module
        from halal_trader.db.models import init_db

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            status = await halt_module.clear_halt(engine)
            console.print(
                f"[green]Kill-switch cleared[/green] "
                f"(was set by {status.set_by} at {status.set_at} — "
                f"reason: {status.reason})"
            )
        finally:
            await engine.dispose()

    asyncio.run(_resume())


@click.command("halt-status")
def halt_status() -> None:
    """Show the current kill-switch state."""

    async def _status() -> None:
        from halal_trader.config import get_settings
        from halal_trader.core.halt import get_status
        from halal_trader.db.models import init_db

        settings = get_settings()
        engine = await init_db(settings.database_url)
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
