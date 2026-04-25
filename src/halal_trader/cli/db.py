"""Database / Alembic schema management commands."""

from __future__ import annotations

import click

from halal_trader.logging import console


@click.group("db")
def db_group() -> None:
    """Database / Alembic schema management."""


@db_group.command("migrate")
@click.option("--revision", default="head", help="Target revision (default: head)")
def db_migrate(revision: str) -> None:
    """Apply Alembic migrations forward to the target revision."""
    from halal_trader.db import admin

    console.print(f"[yellow]Applying migrations up to {revision}...[/yellow]")
    admin.upgrade(revision)
    console.print(f"[green]Database migrated to {admin.current() or 'unknown'}[/green]")


@db_group.command("stamp")
@click.argument("revision", default="head")
def db_stamp(revision: str) -> None:
    """Mark the DB as being at REVISION without running migrations.

    Use this once when adopting a database that was previously managed by
    SQLModel.metadata.create_all (no alembic_version table).
    """
    from halal_trader.db import admin

    console.print(f"[yellow]Stamping database at revision {revision}...[/yellow]")
    admin.stamp(revision)
    console.print(f"[green]Database stamped at {admin.current() or 'unknown'}[/green]")


@db_group.command("current")
def db_current() -> None:
    """Show the current and head Alembic revisions."""
    from halal_trader.db import admin

    cur = admin.current()
    head = admin.head()
    cur_str = cur or "[red]uninitialized[/red]"
    style = "green" if cur == head else "yellow"
    console.print(f"Current: [{style}]{cur_str}[/{style}]")
    console.print(f"Head:    [bold]{head}[/bold]")


@db_group.command("revision")
@click.option("-m", "--message", required=True, help="Short description of the migration")
@click.option("--autogenerate", is_flag=True, help="Diff models against DB to autogenerate")
def db_revision(message: str, autogenerate: bool) -> None:
    """Create a new Alembic revision file under alembic/versions/."""
    from halal_trader.db import admin

    admin.revision(message=message, autogenerate=autogenerate)
    console.print(f"[green]Created revision: {message}[/green]")
