"""Daily backup command — currently a stub under the Postgres baseline.

The legacy SQLite gzipped-snapshot path is no longer applicable. Use
``pg_dump`` (or your managed-DB provider's snapshot feature) for now;
this command stays in the CLI as a placeholder so the scheduler can be
re-wired once a pg_dump-based module lands in ``db/backup.py``.
"""

from __future__ import annotations

import click

from halal_trader.logging import console


@click.command("backup")
def backup() -> None:
    """Stub — backups via pg_dump (not yet wired)."""
    console.print(
        "[yellow]`halal-trader backup` is disabled under the Postgres baseline. "
        "Use `pg_dump` against $DATABASE_URL or your managed-DB snapshot tool.[/]"
    )
