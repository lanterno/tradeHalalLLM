"""Halal cache TTL + mid-cycle refresh tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import sqlalchemy as sa

from halal_trader.config import HalalSettings
from halal_trader.db.repository import Repository
from halal_trader.halal.cache import HalalScreener


def _settings(*, ttl=6, midcycle=4) -> HalalSettings:
    return HalalSettings(cache_max_age_hours=ttl, midcycle_refresh_hours=midcycle)


async def test_default_settings_use_six_hour_ttl():
    s = HalalSettings()
    assert s.cache_max_age_hours == 6
    assert s.midcycle_refresh_hours == 4


async def test_ensure_cache_skips_when_fresh(engine):
    repo = Repository(engine)
    await repo.cache_halal_status(symbol="AAPL", compliance="halal")
    zoya = MagicMock()
    zoya.api_key = "x"
    zoya.screen_bulk = AsyncMock(return_value=[])
    screener = HalalScreener(repo, zoya=zoya, halal_settings=_settings(ttl=6))
    await screener.ensure_cache()
    zoya.screen_bulk.assert_not_called()


async def test_ensure_cache_refreshes_when_stale(engine):
    repo = Repository(engine)
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


async def test_refresh_if_stale_no_op_when_within_midcycle_window(engine):
    repo = Repository(engine)
    await repo.cache_halal_status(symbol="AAPL", compliance="halal")
    zoya = MagicMock()
    zoya.api_key = "x"
    zoya.screen_bulk = AsyncMock(return_value=[])
    screener = HalalScreener(repo, zoya=zoya, halal_settings=_settings(midcycle=4))
    ran = await screener.refresh_if_stale()
    assert ran is False
    zoya.screen_bulk.assert_not_called()


async def test_refresh_if_stale_fires_when_past_midcycle_window(engine):
    repo = Repository(engine)
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
    screener = HalalScreener(repo, zoya=zoya, halal_settings=_settings(ttl=6, midcycle=4))
    ran = await screener.refresh_if_stale(symbols=["AAPL"])
    assert ran is True
    zoya.screen_bulk.assert_awaited_once()
