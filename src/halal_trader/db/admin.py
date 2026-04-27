"""Thin wrapper around Alembic for the `halal-trader db` CLI subgroup."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _alembic_config():
    """Build an Alembic Config pointed at the project's alembic.ini."""
    from alembic.config import Config

    ini = Path(__file__).resolve().parent.parent.parent.parent / "alembic.ini"
    if not ini.exists():
        raise FileNotFoundError(f"alembic.ini not found at {ini}")
    return Config(str(ini))


def upgrade(revision: str = "head") -> None:
    """Apply migrations forward to the given revision (default: head).

    Test-suite shortcut: when ``DATABASE_URL`` points at SQLite (used
    only by the test fixtures), bypass Alembic entirely and create
    every SQLModel-declared table via ``metadata.create_all``. The
    production migrations are Postgres-shaped and won't run cleanly
    against SQLite anyway; tests don't need migration history fidelity.
    """
    from halal_trader.config import get_settings

    if "sqlite" in get_settings().database_url:
        _create_all_sqlite_for_tests()
        return

    from alembic import command

    command.upgrade(_alembic_config(), revision)


def _create_all_sqlite_for_tests() -> None:
    """Materialise every SQLModel table on a SQLite engine for tests.

    Also writes a synthetic ``alembic_version`` row at the production
    head revision, so ``init_db`` accepts the schema as up-to-date
    without us having to ship sqlite-shaped migrations.
    """
    from sqlalchemy import create_engine, text
    from sqlmodel import SQLModel

    # Touch every model so SQLModel.metadata is fully populated.
    import halal_trader.db.models  # noqa: F401
    from halal_trader.config import get_settings
    from halal_trader.db.models import _alembic_head_revision

    sync_url = get_settings().database_url
    if sync_url.startswith("sqlite+aiosqlite://"):
        sync_url = sync_url.replace("sqlite+aiosqlite://", "sqlite://")
    engine = create_engine(sync_url)
    try:
        SQLModel.metadata.create_all(engine)
        head_rev = _alembic_head_revision()
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS alembic_version "
                    "(version_num VARCHAR(32) NOT NULL PRIMARY KEY)"
                )
            )
            conn.execute(text("DELETE FROM alembic_version"))
            conn.execute(
                text("INSERT INTO alembic_version (version_num) VALUES (:v)"),
                {"v": head_rev},
            )
    finally:
        engine.dispose()


def stamp(revision: str = "head") -> None:
    """Mark the DB as being at the given revision without running migrations."""
    from alembic import command

    command.stamp(_alembic_config(), revision)


def current() -> str | None:
    """Return the DB's current Alembic revision (None if uninitialized)."""
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import create_engine

    from halal_trader.config import get_settings

    settings = get_settings()
    sync_engine = create_engine(settings.database_url_sync())
    try:
        with sync_engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            return ctx.get_current_revision()
    finally:
        sync_engine.dispose()


def head() -> str:
    """Return the head revision id from the migration tree."""
    from halal_trader.db.models import _alembic_head_revision

    return _alembic_head_revision()


def revision(message: str, autogenerate: bool = False) -> None:
    """Create a new migration revision file."""
    from alembic import command

    command.revision(_alembic_config(), message=message, autogenerate=autogenerate)
