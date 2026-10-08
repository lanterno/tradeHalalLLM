"""Halal cache TTL + mid-cycle refresh tests.

A refresh without a database engine reloads the default list, so a refresh
shows up as a write to the cache.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import sqlalchemy as sa

from halal_trader.config import HalalSettings
from halal_trader.db.repository import Repository
from halal_trader.halal.cache import DEFAULT_HALAL_SYMBOLS, HalalScreener


def _settings(*, ttl=6, midcycle=4) -> HalalSettings:
    return HalalSettings(cache_max_age_hours=ttl, midcycle_refresh_hours=midcycle)


async def _age_cache(engine, hours: float) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            sa.text("UPDATE halal_cache SET updated_at = :ts"),
            {"ts": datetime.now(UTC) - timedelta(hours=hours)},
        )


async def _cached(repo: Repository) -> set[str]:
    return set(await repo.get_halal_symbols())


async def test_default_settings_use_six_hour_ttl():
    s = HalalSettings()
    assert s.cache_max_age_hours == 6
    assert s.midcycle_refresh_hours == 4


async def test_ensure_cache_skips_when_fresh(engine):
    repo = Repository(engine)
    await repo.cache_halal_status(symbol="ZZZZ", compliance="halal")
    screener = HalalScreener(repo, halal_settings=_settings(ttl=6))
    await screener.ensure_cache()
    assert await _cached(repo) == {"ZZZZ"}


async def test_ensure_cache_refreshes_when_stale(engine):
    repo = Repository(engine)
    await repo.cache_halal_status(symbol="ZZZZ", compliance="halal")
    await _age_cache(engine, 7)
    screener = HalalScreener(repo, halal_settings=_settings(ttl=6))
    await screener.ensure_cache()
    assert set(DEFAULT_HALAL_SYMBOLS) <= await _cached(repo)


async def test_refresh_if_stale_no_op_when_within_midcycle_window(engine):
    repo = Repository(engine)
    await repo.cache_halal_status(symbol="ZZZZ", compliance="halal")
    screener = HalalScreener(repo, halal_settings=_settings(midcycle=4))
    assert await screener.refresh_if_stale() is False
    assert await _cached(repo) == {"ZZZZ"}


async def test_refresh_if_stale_fires_when_past_midcycle_window(engine):
    repo = Repository(engine)
    await repo.cache_halal_status(symbol="ZZZZ", compliance="halal")
    await _age_cache(engine, 5)
    screener = HalalScreener(repo, halal_settings=_settings(ttl=6, midcycle=4))
    assert await screener.refresh_if_stale() is True
    assert set(DEFAULT_HALAL_SYMBOLS) <= await _cached(repo)
