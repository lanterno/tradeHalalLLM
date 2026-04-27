"""Tests for init_db's Alembic schema-authority behavior.

All tests use SQLite via the test fixtures — production runs Postgres,
but the schema-authority logic is dialect-agnostic and tests don't
require a running Postgres.
"""

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from halal_trader.db import admin
from halal_trader.db.models import SchemaError, init_db


def _url(tmp_path, name="x.db") -> str:
    return f"sqlite+aiosqlite:///{tmp_path / name}"


async def test_init_db_rejects_empty_uninitialized(tmp_path):
    with pytest.raises(SchemaError, match="empty"):
        await init_db(_url(tmp_path, "empty.db"))


async def test_init_db_rejects_create_all_db_with_adoption_hint(tmp_path):
    """A DB populated by SQLModel.create_all (no alembic_version) must be rejected."""
    url = _url(tmp_path, "legacy.db")
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await engine.dispose()

    with pytest.raises(SchemaError, match="db stamp head"):
        await init_db(url)


async def test_init_db_rejects_wrong_revision(tmp_path):
    """A DB at a non-head revision must be rejected with migrate hint."""
    url = _url(tmp_path, "old.db")
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await conn.execute(
            sa.text("INSERT INTO alembic_version (version_num) VALUES ('b9da4efd8872')")
        )
    await engine.dispose()

    with pytest.raises(SchemaError, match="db migrate"):
        await init_db(url)


async def test_init_db_succeeds_at_head(tmp_path):
    """A DB at head must open cleanly."""
    url = _url(tmp_path, "head.db")
    engine = create_async_engine(url)
    head = admin.head()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await conn.execute(sa.text(f"INSERT INTO alembic_version (version_num) VALUES ('{head}')"))
    await engine.dispose()

    opened = await init_db(url)
    try:
        async with opened.connect() as conn:
            row = await conn.execute(sa.text("SELECT version_num FROM alembic_version"))
            assert row.first()[0] == head
    finally:
        await opened.dispose()


def test_alembic_head_resolves_to_a_revision_id():
    """Sanity check — head() returns *some* revision id (squashed migration)."""
    head = admin.head()
    assert isinstance(head, str)
    assert len(head) >= 12
