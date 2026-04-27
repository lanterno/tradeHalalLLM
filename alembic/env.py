"""Alembic environment — async-aware for Postgres (asyncpg)."""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlmodel import SQLModel

from alembic import context

# Import every table so SQLModel.metadata is fully populated for autogenerate.
from halal_trader.db.models import (  # noqa: F401
    CryptoDailyPnl,
    CryptoHalalCache,
    CryptoTrade,
    DailyPnl,
    HalalCache,
    IndicatorSnapshot,
    LlmDecision,
    StrategyAdjustment,
    Trade,
)

# ── Alembic config ──────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    # disable_existing_loggers=False so that pre-configured loggers
    # (e.g. pytest's caplog) are not wiped when alembic runs in-process
    # during tests.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = SQLModel.metadata


def _resolve_url() -> str:
    """Return the DB URL — Settings.database_url takes precedence over alembic.ini.

    This keeps `alembic upgrade` and `init_db` in sync regardless of CWD.
    Falls back to the alembic.ini value if Settings cannot be loaded (e.g.
    running offline migrations without an env).
    """
    try:
        from halal_trader.config import get_settings

        settings = get_settings()
        return settings.database_url
    except Exception:
        return config.get_main_option("sqlalchemy.url") or ""


# ── Offline mode (SQL script generation) ────────────────────────
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL to stdout."""
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # render_as_batch=False — Postgres supports proper ALTER TABLE.
    )

    with context.begin_transaction():
        context.run_migrations()


# ── Online mode (async engine) ──────────────────────────────────
def do_run_migrations(connection) -> None:  # noqa: ANN001
    """Configure context with a live connection and run migrations."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # render_as_batch=False — Postgres supports proper ALTER TABLE.
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations inside a connection."""
    section = config.get_section(config.config_ini_section, {}) or {}
    section["sqlalchemy.url"] = _resolve_url()
    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode via the async engine."""
    asyncio.run(run_async_migrations())


# ── Entrypoint ──────────────────────────────────────────────────
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
