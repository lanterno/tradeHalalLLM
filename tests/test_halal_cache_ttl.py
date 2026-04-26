"""Halal cache TTL + mid-cycle refresh tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from halal_trader.config import HalalSettings
from halal_trader.db import admin
from halal_trader.db.repository import Repository
from halal_trader.halal.cache import HalalScreener


async def _engine_repo(tmp_path):
    db_path = tmp_path / "halal_ttl.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    head = admin.head()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(
            sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await conn.execute(sa.text(f"INSERT INTO alembic_version (version_num) VALUES ('{head}')"))
    return engine, Repository(engine)


def _settings(*, ttl=6, midcycle=4) -> HalalSettings:
    return HalalSettings(cache_max_age_hours=ttl, midcycle_refresh_hours=midcycle)


async def test_default_settings_use_six_hour_ttl():
    """Pin the new default so a regression to 24h fails loudly."""
    s = HalalSettings()
    assert s.cache_max_age_hours == 6
    assert s.midcycle_refresh_hours == 4


async def test_ensure_cache_skips_when_fresh(tmp_path):
    """A row updated 1h ago should not trigger a refresh under a 6h TTL."""
    engine, repo = await _engine_repo(tmp_path)
    try:
        await repo.cache_halal_status(symbol="AAPL", compliance="halal")
        zoya = MagicMock()
        zoya.api_key = "x"
        zoya.screen_bulk = AsyncMock(return_value=[])
        screener = HalalScreener(repo, zoya=zoya, halal_settings=_settings(ttl=6))
        await screener.ensure_cache()
        zoya.screen_bulk.assert_not_called()
    finally:
        await engine.dispose()


async def test_ensure_cache_refreshes_when_stale(tmp_path):
    """A 7h-old cache row must trigger a refresh under a 6h TTL."""
    engine, repo = await _engine_repo(tmp_path)
    try:
        # Seed a stale row by writing then back-dating updated_at.
        await repo.cache_halal_status(symbol="AAPL", compliance="halal")
        async with engine.begin() as conn:
            await conn.execute(
                sa.text("UPDATE halal_cache SET updated_at = :ts WHERE symbol = 'AAPL'"),
                {"ts": datetime.now(UTC) - timedelta(hours=7)},
            )

        zoya = MagicMock()
        zoya.api_key = "x"
        zoya.screen_bulk = AsyncMock(
            return_value=[{"symbol": "AAPL", "compliance": "halal", "detail": "fresh"}]
        )
        screener = HalalScreener(repo, zoya=zoya, halal_settings=_settings(ttl=6))
        await screener.ensure_cache(symbols=["AAPL"])
        zoya.screen_bulk.assert_awaited_once()
    finally:
        await engine.dispose()


async def test_refresh_if_stale_no_op_when_within_midcycle_window(tmp_path):
    """Cache 1h old; midcycle window 4h → refresh_if_stale returns False."""
    engine, repo = await _engine_repo(tmp_path)
    try:
        await repo.cache_halal_status(symbol="AAPL", compliance="halal")
        zoya = MagicMock()
        zoya.api_key = "x"
        zoya.screen_bulk = AsyncMock(return_value=[])
        screener = HalalScreener(repo, zoya=zoya, halal_settings=_settings(midcycle=4))
        ran = await screener.refresh_if_stale()
        assert ran is False
        zoya.screen_bulk.assert_not_called()
    finally:
        await engine.dispose()


async def test_refresh_if_stale_fires_when_past_midcycle_window(tmp_path):
    """Cache 5h old; midcycle window 4h → refresh fires (regardless of hard TTL)."""
    engine, repo = await _engine_repo(tmp_path)
    try:
        await repo.cache_halal_status(symbol="AAPL", compliance="halal")
        async with engine.begin() as conn:
            await conn.execute(
                sa.text("UPDATE halal_cache SET updated_at = :ts WHERE symbol = 'AAPL'"),
                {"ts": datetime.now(UTC) - timedelta(hours=5)},
            )

        zoya = MagicMock()
        zoya.api_key = "x"
        zoya.screen_bulk = AsyncMock(
            return_value=[{"symbol": "AAPL", "compliance": "halal", "detail": "fresh"}]
        )
        # Hard TTL is the default 6h; midcycle 4h is the stricter window.
        screener = HalalScreener(repo, zoya=zoya, halal_settings=_settings(ttl=6, midcycle=4))
        ran = await screener.refresh_if_stale(symbols=["AAPL"])
        assert ran is True
        zoya.screen_bulk.assert_awaited_once()
    finally:
        await engine.dispose()
