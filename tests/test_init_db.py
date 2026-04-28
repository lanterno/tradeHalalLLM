"""Tests for init_db's Alembic schema-authority behavior."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import psycopg
import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

import halal_trader.db.models  # noqa: F401  — populate metadata
from halal_trader.db import admin
from halal_trader.db.models import SchemaError, init_db
from tests.conftest import (
    _ADMIN_DSN_SYNC,
    PG_HOST,
    PG_PASS,
    PG_PORT,
    PG_USER,
    _terminate_and_drop,
)


def _scratch_db_url() -> tuple[str, str]:
    """Create a one-off Postgres database for a single test; return its URL + name."""
    name = f"halal_trader_init_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(_ADMIN_DSN_SYNC, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{name}"')
    return (
        f"postgresql+asyncpg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{name}",
        name,
    )


@pytest.fixture
async def scratch_db() -> AsyncIterator[str]:
    url, name = _scratch_db_url()
    try:
        yield url
    finally:
        _terminate_and_drop(name)


async def test_init_db_rejects_empty_uninitialized(scratch_db):
    with pytest.raises(SchemaError, match="empty"):
        await init_db(scratch_db)


async def test_init_db_rejects_create_all_db_with_adoption_hint(scratch_db):
    """A DB populated by SQLModel.create_all (no alembic_version) must be rejected."""
    engine = create_async_engine(scratch_db)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await engine.dispose()

    with pytest.raises(SchemaError, match="db stamp head"):
        await init_db(scratch_db)


async def test_init_db_rejects_wrong_revision(scratch_db):
    """A DB at a non-head revision must be rejected with migrate hint."""
    engine = create_async_engine(scratch_db)
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
        await init_db(scratch_db)


async def test_init_db_succeeds_at_head(scratch_db):
    """A DB at head must open cleanly."""
    engine = create_async_engine(scratch_db)
    head = admin.head()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await conn.execute(sa.text(f"INSERT INTO alembic_version (version_num) VALUES ('{head}')"))
    await engine.dispose()

    opened = await init_db(scratch_db)
    try:
        async with opened.connect() as conn:
            row = await conn.execute(sa.text("SELECT version_num FROM alembic_version"))
            assert row.first()[0] == head
    finally:
        await opened.dispose()


def test_alembic_head_resolves_to_a_revision_id():
    head = admin.head()
    assert isinstance(head, str)
    assert len(head) >= 12
