"""Daily SQLite backup command."""

from __future__ import annotations

import asyncio

import click

from halal_trader.logging import console


@click.command("backup")
@click.option(
    "--retention-days",
    default=None,
    type=int,
    help="Override BACKUP_RETENTION_DAYS for this run.",
)
@click.option(
    "--weekly-count",
    default=None,
    type=int,
    help="Override BACKUP_WEEKLY_COUNT for this run.",
)
def backup(retention_days: int | None, weekly_count: int | None) -> None:
    """Create a gzipped SQLite backup and prune old ones."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.backup import prune_backups, run_backup

        settings = get_settings()
        retention = retention_days if retention_days is not None else settings.backup.retention_days
        weekly = weekly_count if weekly_count is not None else settings.backup.weekly_count

        result = run_backup(
            db_path=settings.resolve_db_path(),
            backup_dir=settings.backup.dir,
        )
        console.print(
            f"[green]Backup written[/green]: {result.path} "
            f"([dim]{result.size_bytes / 1024:.1f} KB[/dim])"
        )
        deleted = prune_backups(
            backup_dir=settings.backup.dir,
            retention_days=retention,
            weekly_count=weekly,
        )
        if deleted:
            console.print(f"[dim]Pruned {len(deleted)} old backup file(s)[/dim]")

    asyncio.run(_run())
