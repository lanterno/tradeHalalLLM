"""Test fixtures — Postgres-backed, isolated per test (and per xdist worker).

Strategy
========
* Each pytest-xdist worker gets its **own** test database, e.g.
  ``halal_trader_test_gw0``, ``halal_trader_test_gw1``, …
  (Single-process runs use ``halal_trader_test``.) The worker
  initializes its DB once per session under a per-DB Postgres
  advisory lock so concurrent invocations of the same worker id
  block instead of corrupting state.
* Per-test fixtures (`database_url`, `engine`) take a fresh slate by
  TRUNCATEing every table (except ``alembic_version``) before each
  test. TRUNCATE on an empty schema is microseconds.

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
import zlib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

# Set COLUMNS *before* importing anything that touches Rich's Console
# (logging.console captures the terminal width at import time). 240 is
# wide enough for every CLI table the suite renders. xdist subprocesses
# inherit COLUMNS=80 from the captured stdout otherwise, which truncates
# wide-table assertions.
os.environ.setdefault("COLUMNS", "240")

import psycopg  # noqa: E402
import pytest  # noqa: E402
import sqlalchemy as sa  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine  # noqa: E402

# Touch every model so SQLModel.metadata is fully populated for the
# session-scoped migration step.
import halal_trader.db.models  # noqa: F401, E402

PG_HOST = os.environ.get("TEST_PG_HOST", "localhost")
PG_PORT = os.environ.get("TEST_PG_PORT", "5433")
PG_USER = os.environ.get("TEST_PG_USER", "trader")
PG_PASS = os.environ.get("TEST_PG_PASS", "trader-dev-only")
_PG_TEST_DB_BASE = os.environ.get("TEST_PG_DB", "halal_trader_test")

_ADMIN_DSN_SYNC = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/postgres"


def _worker_id() -> str:
    """Pytest-xdist worker id — ``"master"`` outside xdist."""
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


def _worker_dbname() -> str:
    """Per-worker DB name. Master collapses to the canonical name."""
    wid = _worker_id()
    if wid == "master":
        return _PG_TEST_DB_BASE
    return f"{_PG_TEST_DB_BASE}_{wid}"


def _worker_lock_key(dbname: str) -> int:
    """64-bit advisory-lock key derived from the DB name.

    Two pytest sessions that target the same DB name share the lock;
    different worker DBs lock independently, so xdist's parallel
    workers don't serialise on the lock.
    """
    return zlib.crc32(dbname.encode("utf-8")) & 0xFFFFFFFF


def _async_url(dbname: str) -> str:
    return f"postgresql+asyncpg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{dbname}"


def _sync_url(dbname: str) -> str:
    return f"postgresql+psycopg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{dbname}"


# Backwards-compatible aliases for tests that imported these directly.
PG_DBNAME = _worker_dbname()
PG_TEST_URL_ASYNC = _async_url(PG_DBNAME)
PG_TEST_URL_SYNC = _sync_url(PG_DBNAME)


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
def _pg_test_db_ready() -> Iterator[str]:
    """Recreate the per-worker test DB once per session and run Alembic.

    Holds a per-DB Postgres advisory lock for the lifetime of the
    session so two concurrent pytest invocations targeting the same
    worker can't race on ``DROP DATABASE`` / ``CREATE DATABASE``.
    Different xdist workers use different DB names → different lock
    keys → they don't serialise on each other.
    """
    dbname = _worker_dbname()
    lock_key = _worker_lock_key(dbname)
    async_url = _async_url(dbname)

    lock_conn = psycopg.connect(_ADMIN_DSN_SYNC, autocommit=True)
    try:
        with lock_conn.cursor() as cur:
            cur.execute("SET statement_timeout = '60s'")
            try:
                cur.execute("SELECT pg_advisory_lock(%s)", (lock_key,))
            except psycopg.errors.QueryCanceled:
                pytest.exit(
                    f"Another pytest session is holding the {dbname!r} "
                    "advisory lock. Wait for it to finish or kill it.",
                    returncode=1,
                )

        _terminate_and_drop(dbname)
        with psycopg.connect(_ADMIN_DSN_SYNC, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'CREATE DATABASE "{dbname}"')

        os.environ["DATABASE_URL"] = async_url
        import halal_trader.config as _config

        _config._settings = None
        from halal_trader.db.admin import upgrade

        upgrade()
        try:
            yield async_url
        finally:
            with lock_conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
    finally:
        lock_conn.close()


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
    _truncate_all_tables(_sync_url(_worker_dbname()))
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
