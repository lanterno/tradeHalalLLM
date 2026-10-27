"""The day trader's risk meter (core/risk_meter.py)."""

from __future__ import annotations

from typing import Any

import click

from halal_trader.cli._run import fail, run_db
from halal_trader.logging import console


@click.group("risk")
def risk() -> None:
    """The day trader's risk meter: one level, 1 (Cautious) to 5 (Max)."""


@risk.command("status")
def risk_status() -> None:
    """Show the level in effect and what each level sets."""

    async def _status(engine: Any, settings: Any) -> None:
        from rich.table import Table

        from halal_trader.core.risk_meter import LEVELS, describe, get_status, refusal

        s = await get_status(engine)
        console.print(describe(s))
        table = Table(title="Levels")
        for column in (
            "Level",
            "Largest position",
            "Typical",
            "Positions",
            "Daily loss",
            "Drawdown",
            "Open risk",
            "Stop",
            "Can be set",
        ):
            table.add_column(column)
        for n, lim in LEVELS.items():
            low, high = lim.typical_position_pct
            stop = (
                f"required, ≤ {lim.max_stop_distance_pct:.0%} below"
                if lim.max_stop_distance_pct is not None
                else "model's choice"
            )
            table.add_row(
                f"{'▶ ' if n == s.level else ''}{n} {lim.name}",
                f"{lim.max_position_pct:.0%}",
                f"{low:.0%}–{high:.0%}",
                str(lim.max_positions),
                f"{lim.daily_loss_limit:.1%}",
                f"{lim.max_drawdown_pct:.0%}",
                f"{lim.max_portfolio_heat_pct:.1%}",
                stop,
                refusal(n) or "yes",
            )
        console.print(table)

    run_db(_status)


@risk.command("set")
@click.argument("level", type=int)
@click.option("--reason", required=True, help="Why the level changes (audit trail).")
def risk_set(level: int, reason: str) -> None:
    """Set the level: a cut applies at once, a raise from the next trading day."""

    async def _set(engine: Any, settings: Any) -> None:
        from halal_trader.core.risk_meter import MeterRefused, announce, describe, set_level

        try:
            s = await set_level(engine, level, reason=reason)
        except MeterRefused as e:
            fail(str(e))
        console.print(describe(s))
        if not await announce(settings, s):
            console.print("[dim]Telegram isn't configured, so the change wasn't announced.[/dim]")

    run_db(_set)
