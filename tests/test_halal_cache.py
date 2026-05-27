"""Tests for the halal cache and screener."""

import pytest

from halal_trader.db.repository import Repository
from halal_trader.halal.cache import DEFAULT_HALAL_SYMBOLS, HalalScreener


@pytest.fixture
async def repo(engine):
    return Repository(engine)


@pytest.fixture
async def screener(repo):
    """Create a screener without Zoya API (uses defaults)."""
    return HalalScreener(repo, zoya=None)


class TestHalalCache:
    async def test_cache_halal_status(self, repo):
        await repo.cache_halal_status("AAPL", "halal", "Test")
        status = await repo.get_halal_status("AAPL")
        assert status == "halal"

    async def test_get_halal_symbols(self, repo):
        await repo.cache_halal_status("AAPL", "halal")
        await repo.cache_halal_status("BAD", "not_halal")
        await repo.cache_halal_status("MEH", "doubtful")

        symbols = await repo.get_halal_symbols()
        assert "AAPL" in symbols
        assert "BAD" not in symbols
        assert "MEH" not in symbols

    async def test_is_cache_fresh(self, repo):
        # Empty cache is not fresh
        assert not await repo.is_cache_fresh()

        # After adding data, it should be fresh
        await repo.cache_halal_status("AAPL", "halal")
        assert await repo.is_cache_fresh(max_age_hours=24)


class TestHalalScreener:
    async def test_ensure_cache_defaults(self, screener, repo):
        """Without Zoya API, should load default symbols."""
        await screener.ensure_cache()

        symbols = await screener.get_halal_symbols()
        assert len(symbols) == len(DEFAULT_HALAL_SYMBOLS)
        assert "AAPL" in symbols
        assert "NVDA" in symbols

    async def test_is_halal(self, screener):
        await screener.ensure_cache()

        assert await screener.is_halal("AAPL")
        assert await screener.is_halal("MSFT")
        assert not await screener.is_halal("UNKNOWN_TICKER")

    async def test_filter_halal(self, screener):
        await screener.ensure_cache()

        filtered = await screener.filter_halal(["AAPL", "UNKNOWN", "NVDA", "FAKE"])
        assert filtered == ["AAPL", "NVDA"]

    async def test_cache_not_refreshed_when_fresh(self, screener, repo):
        """Second call should skip refresh if cache is fresh."""
        await screener.ensure_cache()
        count_before = len(await screener.get_halal_symbols())

        # Second call — should skip
        await screener.ensure_cache()
        count_after = len(await screener.get_halal_symbols())

        assert count_before == count_after

    async def test_transient_zoya_errors_are_not_cached(self, repo):
        """A transient Zoya failure must NOT overwrite a good cached verdict
        — otherwise a momentary outage poisons the cache for the full TTL
        and starves the universe all day (observed 2026-05-27)."""
        from unittest.mock import AsyncMock, MagicMock

        # Seed a known-good prior verdict.
        await repo.cache_halal_status("AAPL", "halal", "prior good verdict")

        zoya = MagicMock()
        # Zoya blips: AAPL errors (transient), MSFT returns a real verdict.
        zoya.api_key = "k"
        zoya.screen_bulk = AsyncMock(
            return_value=[
                {
                    "symbol": "AAPL",
                    "compliance": "doubtful",
                    "detail": "ConnectError",
                    "error": True,
                },
                {"symbol": "MSFT", "compliance": "halal", "detail": "ok"},
            ]
        )
        screener = HalalScreener(repo, zoya=zoya)
        await screener.ensure_cache(symbols=["AAPL", "MSFT"], force=True)

        # AAPL's prior 'halal' verdict survives (error not cached);
        # MSFT's fresh verdict is written.
        symbols = await screener.get_halal_symbols()
        assert "AAPL" in symbols
        assert "MSFT" in symbols
