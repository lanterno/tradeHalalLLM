"""Alembic environment — async-aware for SQLite + aiosqlite."""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlmodel import SQLModel

from alembic import context

# Import all models so SQLModel.metadata is populated for autogenerate.
from halal_trader.db.models import (  # noqa: F401
    CryptoDailyPnl,
    CryptoHalalCache,
    CryptoTrade,
    DailyPnl,
    HalalCache,
    LlmDecision,
    Trade,
)

# ── Alembic config ──────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


# ── Offline mode (SQL script generation) ────────────────────────
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL to stdout."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # required for SQLite ALTER TABLE support
    )

    with context.begin_transaction():
        context.run_migrations()


# ── Online mode (async engine) ──────────────────────────────────
def do_run_migrations(connection) -> None:  # noqa: ANN001
    """Configure context with a live connection and run migrations."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # required for SQLite ALTER TABLE support
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations inside a connection."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
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
