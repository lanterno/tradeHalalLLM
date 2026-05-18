"""ML artefact CLI (Wave K) — list versions stored in ``ml_artefacts``.

Replaces the previous "look at the modification time of
``models/*.pkl``" diagnostic for which model is currently live. The
table stores a strictly-versioned history per artefact name; the
loader picks the highest version, so the most-recent row in this
listing is the model the bot will load on next start.
"""

from __future__ import annotations

import asyncio

import click
from rich.table import Table

from halal_trader.logging import console


@click.group("ml")
def ml_group() -> None:
    """ML model artefact operations."""


@ml_group.command("versions")
@click.option(
    "--name",
    default=None,
    help="Filter to one artefact name (e.g. 'anomaly_detector'). Omit for all.",
)
@click.option("--limit", default=50, show_default=True)
def versions(name: str | None, limit: int) -> None:
    """List rows in ``ml_artefacts`` (newest first)."""
    asyncio.run(_run_versions(name=name, limit=limit))


async def _run_versions(*, name: str | None, limit: int) -> None:
    from halal_trader.config import get_settings
    from halal_trader.db import init_db
    from halal_trader.db.ml_artefacts import list_versions

    settings = get_settings()
    engine = await init_db(settings.database_url)
    rows = await list_versions(engine=engine, name=name)
    if not rows:
        console.print(
            "[yellow]No ml_artefacts rows yet — the bot will use the legacy "
            "models/*.pkl path or fall back to defaults.[/yellow]"
        )
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim", justify="right")
    table.add_column("Name")
    table.add_column("Version", justify="right")
    table.add_column("Format", style="dim")
    table.add_column("sklearn")
    table.add_column("Feature hash", style="dim")
    table.add_column("Created at")

    # Newest-per-name first. ``list_versions`` already sorts by
    # ``created_at desc``; truncate to the requested limit.
    for r in rows[:limit]:
        table.add_row(
            str(r["id"]),
            r["name"],
            str(r["version"]),
            r["payload_format"],
            r.get("sklearn_version") or "—",
            (r.get("feature_hash") or "—")[:16],
            r.get("created_at", ""),
        )
    console.print(table)
    console.print(f"\n[green]{len(rows)} row(s).[/green]")
