"""Postgres fixtures for the engine's own tables.

Reuses the parent conftest's ``database_url`` fixture (which provisions the
session test DB), then bootstraps the engine's ``hb_*`` tables and truncates
them per test for isolation. Requires the same Postgres on :5433 the rest of
the suite uses.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from halabot.platform.db import bootstrap_schema, metadata


@pytest.fixture
async def halabot_engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(database_url)
    await bootstrap_schema(eng)  # additive + idempotent
    async with eng.begin() as conn:
        for table in reversed(metadata.sorted_tables):
            await conn.execute(sa.text(f"TRUNCATE {table.name} RESTART IDENTITY CASCADE"))
    try:
        yield eng
    finally:
        await eng.dispose()
