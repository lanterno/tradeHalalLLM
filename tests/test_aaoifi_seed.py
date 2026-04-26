"""AAOIFI seed-list screener tests."""

from __future__ import annotations

from halal_trader.halal.aaoifi_seed import (
    AAOIFI_SEED_HALAL_STOCKS,
    AAOIFISeedScreener,
)
from halal_trader.halal.corroborate import CorroboratingScreener


async def test_known_seed_symbols_are_halal():
    s = AAOIFISeedScreener()
    assert await s.is_halal("AAPL") is True
    assert await s.is_halal("aapl") is True  # case-insensitive
    assert await s.is_halal("MSFT") is True


async def test_unknown_symbol_is_not_halal():
    s = AAOIFISeedScreener()
    assert await s.is_halal("UNKNOWN") is False


async def test_filter_halal_returns_only_seeded():
    s = AAOIFISeedScreener()
    out = await s.filter_halal(["AAPL", "ZZZZ", "MSFT"])
    assert out == ["AAPL", "MSFT"]


async def test_get_halal_symbols_is_full_seed_list_sorted():
    s = AAOIFISeedScreener()
    out = await s.get_halal_symbols()
    assert sorted(AAOIFI_SEED_HALAL_STOCKS) == out


async def test_ensure_cache_is_noop():
    """No-op so the corroboration wrapper can call it without special-casing."""
    s = AAOIFISeedScreener()
    await s.ensure_cache()  # must not raise


async def test_seed_satisfies_compliance_screener_protocol():
    """Type-level smoke: pass the seed as a CorroboratingScreener secondary."""
    primary = AAOIFISeedScreener()
    secondary = AAOIFISeedScreener()
    wrapper = CorroboratingScreener(primary, secondary)
    # Both halal → unanimous halal.
    assert await wrapper.is_halal("AAPL") is True
    # Unknown symbol → unanimous not halal.
    assert await wrapper.is_halal("ZZZZ") is False
