"""`halal-trader events`: the event store's labels and what the scores knew."""

from __future__ import annotations

import asyncio
from typing import Any

import click

from halal_trader.logging import console


@click.group("events")
def events() -> None:
    """News and filings as dated, scored, labelled events (S2 research)."""


@events.command("report")
@click.option("--threshold", default=0.85, show_default=True, help="The reactor's trigger.")
def report_cmd(threshold: float) -> None:
    """Label matured events, then show each scorer's IC against abnormal returns."""

    async def _run() -> tuple[int, list[Any], dict[str, int]]:
        from sqlalchemy import text

        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.events.labels import label_events, report

        engine = await init_db(get_settings().database_url)
        try:
            written = await label_events(engine)
            async with engine.connect() as conn:
                counts = {
                    "events": (await conn.execute(text("SELECT count(*) FROM events"))).scalar()
                    or 0,
                    "scored": (
                        await conn.execute(text("SELECT count(*) FROM event_scores"))
                    ).scalar()
                    or 0,
                }
            return written, await report(engine, threshold=threshold), counts
        finally:
            await engine.dispose()

    written, rows, counts = asyncio.run(_run())
    console.print(f"{counts['events']} events, {counts['scored']} scores; {written} new label(s)")
    if not rows:
        console.print("[yellow]no labelled scores yet: horizons need sessions to elapse[/yellow]")
        return
    for r in rows:
        above = f"{r.above_abn:+.2%}" if r.above_abn is not None else "  n/a"
        below = f"{r.below_abn:+.2%}" if r.below_abn is not None else "  n/a"
        console.print(
            f"  {r.scorer:40} {r.horizon:>2}d  n={r.events:<5} IC {r.ic:+.3f}  "
            f">= {threshold:g}: {r.above} events, mean abn {above}; below: {below}"
        )
