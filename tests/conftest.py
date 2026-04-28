"""Test fixtures — Postgres-backed, isolated per test.

Strategy
========
* One **session-scoped** fixture (`_pg_test_db_ready`) drops + recreates the
  ``halal_trader_test`` database and runs Alembic to head exactly once.
* Per-test fixtures (`database_url`, `engine`) take a fresh slate by
  TRUNCATEing every table (except ``alembic_version``) before each test.
  TRUNCATE on an empty schema is microseconds, and avoids the cost of
  per-test migrations.

Tests that previously did
``monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite://...")`` should
now request the ``database_url`` fixture instead — it sets the env var
to the test database and clears the ``get_settings()`` cache so any
code that re-reads settings sees the test URL.

Postgres connection details come from the ``TEST_PG_*`` env vars (see
defaults below). The pgvector container at ``localhost:5433`` from the
project's docker-compose is the expected target.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import psycopg
import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# Touch every model so SQLModel.metadata is fully populated for the
# session-scoped migration step.
import halal_trader.db.models  # noqa: F401

PG_HOST = os.environ.get("TEST_PG_HOST", "localhost")
PG_PORT = os.environ.get("TEST_PG_PORT", "5433")
PG_USER = os.environ.get("TEST_PG_USER", "trader")
PG_PASS = os.environ.get("TEST_PG_PASS", "trader-dev-only")
PG_DBNAME = os.environ.get("TEST_PG_DB", "halal_trader_test")

PG_TEST_URL_ASYNC = f"postgresql+asyncpg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"
PG_TEST_URL_SYNC = f"postgresql+psycopg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"
_ADMIN_DSN_SYNC = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/postgres"


def _terminate_and_drop(dbname: str) -> None:
    with psycopg.connect(_ADMIN_DSN_SYNC, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (dbname,),
            )
            cur.execute(f'DROP DATABASE IF EXISTS "{dbname}"')


@pytest.fixture(scope="session")
def _pg_test_db_ready() -> str:
    """Recreate the test DB once per session and run Alembic to head."""
    _terminate_and_drop(PG_DBNAME)
    with psycopg.connect(_ADMIN_DSN_SYNC, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{PG_DBNAME}"')

    os.environ["DATABASE_URL"] = PG_TEST_URL_ASYNC
    import halal_trader.config as _config

    _config._settings = None
    from halal_trader.db.admin import upgrade

    upgrade()
    return PG_TEST_URL_ASYNC


def _truncate_all_tables(sync_url: str) -> None:
    eng = create_engine(sync_url)
    try:
        with eng.begin() as conn:
            rows = conn.execute(
                sa.text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' AND tablename != 'alembic_version'"
                )
            ).fetchall()
            tables = [r[0] for r in rows]
            if tables:
                qnames = ", ".join(f'"{t}"' for t in tables)
                conn.execute(sa.text(f"TRUNCATE {qnames} RESTART IDENTITY CASCADE"))
    finally:
        eng.dispose()


@pytest.fixture
def database_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _pg_test_db_ready: str,
) -> Iterator[str]:
    """Set ``DATABASE_URL`` to a freshly-truncated test DB.

    Also points ``DATA_DIR`` at ``tmp_path`` so any test that touches
    ``settings.resolve_data_dir()`` (round-trip purification, replay,
    regime memory, exception queue, …) writes into the per-test tmp
    tree instead of leaking into the dev workspace's ``./data/``.

    Code that calls ``get_settings()`` will pick up both URLs because
    we also bust the singleton cache.
    """
    url = _pg_test_db_ready
    _truncate_all_tables(PG_TEST_URL_SYNC)
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    import halal_trader.config as _config

    monkeypatch.setattr(_config, "_settings", None, raising=False)
    yield url


@pytest.fixture
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    """Async engine bound to the test database."""
    eng = create_async_engine(database_url)
    try:
        yield eng
    finally:
        await eng.dispose()
