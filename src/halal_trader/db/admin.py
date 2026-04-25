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
    """Apply migrations forward to the given revision (default: head)."""
    from alembic import command

    command.upgrade(_alembic_config(), revision)


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
    sync_engine = create_engine(f"sqlite:///{settings.resolve_db_path()}")
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
