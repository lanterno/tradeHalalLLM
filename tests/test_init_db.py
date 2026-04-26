"""Tests for init_db's Alembic schema-authority behavior."""

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from halal_trader.db import admin
from halal_trader.db.models import SchemaError, init_db


async def test_init_db_rejects_empty_uninitialized(tmp_path):
    db_path = tmp_path / "empty.db"
    with pytest.raises(SchemaError, match="empty"):
        await init_db(str(db_path))


async def test_init_db_rejects_create_all_db_with_adoption_hint(tmp_path):
    """A DB populated by SQLModel.create_all (no alembic_version) must be rejected."""
    db_path = tmp_path / "legacy.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await engine.dispose()

    with pytest.raises(SchemaError, match="db stamp head"):
        await init_db(str(db_path))


async def test_init_db_rejects_wrong_revision(tmp_path):
    """A DB at a non-head revision must be rejected with migrate hint."""
    db_path = tmp_path / "old.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
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
        await init_db(str(db_path))


async def test_init_db_succeeds_at_head(tmp_path):
    """A DB at head must open cleanly."""
    db_path = tmp_path / "head.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    head = admin.head()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await conn.execute(sa.text(f"INSERT INTO alembic_version (version_num) VALUES ('{head}')"))
    await engine.dispose()

    opened = await init_db(str(db_path))
    try:
        async with opened.connect() as conn:
            row = await conn.execute(sa.text("SELECT version_num FROM alembic_version"))
            assert row.first()[0] == head
    finally:
        await opened.dispose()


def test_alembic_head_matches_known_revision():
    """If somebody adds a revision without updating tests, this fails loudly."""
    assert admin.head() == "e6f7a8b9c0d1"


async def test_init_db_applies_wal_and_busy_timeout(tmp_path):
    """Every connection from the engine must open in WAL mode with busy_timeout set."""
    db_path = tmp_path / "wal.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    head = admin.head()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await conn.execute(sa.text(f"INSERT INTO alembic_version (version_num) VALUES ('{head}')"))
    await engine.dispose()

    opened = await init_db(str(db_path))
    try:
        async with opened.connect() as conn:
            mode = (await conn.execute(sa.text("PRAGMA journal_mode"))).scalar_one()
            timeout = (await conn.execute(sa.text("PRAGMA busy_timeout"))).scalar_one()
            fk = (await conn.execute(sa.text("PRAGMA foreign_keys"))).scalar_one()
            assert str(mode).lower() == "wal"
            assert int(timeout) == 30000
            assert int(fk) == 1
    finally:
        await opened.dispose()
