"""Tests for the catalyst-source adapters in :mod:`trading`.

Adapters wrap a domain signal (Fed-speak rolling drift) into the
``CatalystSource`` protocol the cycle's
``StockCatalystFeed`` consumes. The wrapped sources have their own
unit tests; this file pins the adapter shape (correct kind label,
empty-symbols short-circuit, snapshot → Catalyst conversion).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.trading.fed_speak_adapter import FedSpeakCatalystSource

# ── FedSpeakCatalystSource ────────────────────────────────────


@pytest.mark.asyncio
async def test_fed_speak_source_empty_symbols_returns_empty():
    """Universe-wide signal still skips the fetch when no symbols
    are requested — saves an HTTP roundtrip on a quiet cycle."""
    fetcher = MagicMock()
    fetcher.fetch = AsyncMock()
    src = FedSpeakCatalystSource(fetcher=fetcher)
    out = await src.fetch([])
    assert out == []
    fetcher.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_fed_speak_source_aclose_delegates():
    fetcher = MagicMock()
    fetcher.aclose = AsyncMock()
    src = FedSpeakCatalystSource(fetcher=fetcher)
    await src.aclose()
    fetcher.aclose.assert_awaited_once()
