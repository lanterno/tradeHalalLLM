"""Tests for the catalyst-source adapters in :mod:`trading`.

Adapters wrap a domain signal (Fed-speak rolling drift, options IV
surface) into the ``CatalystSource`` protocol the cycle's
``StockCatalystFeed`` consumes. The wrapped sources have their own
unit tests; this file pins the adapter shape (correct kind label,
empty-symbols short-circuit, snapshot → Catalyst conversion).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.trading.fed_speak_adapter import FedSpeakCatalystSource
from halal_trader.trading.options_catalyst_adapter import (
    OptionsIVCatalystSource,
    _kind_for,
)
from halal_trader.trading.options_iv import OptionsIVSnapshot

# ── _kind_for label mapping ───────────────────────────────────


def _snap(*, atm_iv: float, put_call_skew: float = 0.0, **kwargs) -> OptionsIVSnapshot:
    base = dict(
        symbol="AAPL",
        spot=180.0,
        atm_iv=atm_iv,
        put_call_skew=put_call_skew,
        call_volume=100,
        put_volume=100,
        call_open_interest=1_000,
        put_open_interest=1_000,
    )
    base.update(kwargs)
    return OptionsIVSnapshot(**base)


def test_kind_for_elevated_iv():
    """``atm_iv >= 0.6`` → label "elevated_iv" → kind "options_iv_elevated"."""
    snap = _snap(atm_iv=0.65)
    assert snap.label == "elevated_iv"
    assert _kind_for(snap) == "options_iv_elevated"


def test_kind_for_downside_premium_returns_skew():
    """``put_call_skew >= 0.05`` (puts richer than calls) → "downside_premium"
    label → kind "options_iv_skew"."""
    snap = _snap(atm_iv=0.30, put_call_skew=0.06)
    assert snap.label == "downside_premium"
    assert _kind_for(snap) == "options_iv_skew"


def test_kind_for_normal_returns_default():
    """Calm IV + neutral skew → generic ``options_iv`` tag."""
    snap = _snap(atm_iv=0.25, put_call_skew=0.0)
    assert _kind_for(snap) == "options_iv"


# ── OptionsIVCatalystSource ───────────────────────────────────


@pytest.mark.asyncio
async def test_options_iv_source_empty_symbols_returns_empty():
    """Avoid the network entirely when no symbols are requested."""
    fetcher = MagicMock()
    fetcher.fetch = AsyncMock()
    src = OptionsIVCatalystSource(fetcher=fetcher)
    out = await src.fetch([])
    assert out == []
    fetcher.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_options_iv_source_no_snapshots_returns_empty():
    fetcher = MagicMock()
    fetcher.fetch = AsyncMock(return_value={})
    src = OptionsIVCatalystSource(fetcher=fetcher)
    out = await src.fetch(["AAPL"])
    assert out == []


@pytest.mark.asyncio
async def test_options_iv_source_renders_one_catalyst_per_snapshot():
    snap = _snap(atm_iv=0.65, put_call_skew=-0.04, put_volume=160, call_volume=100)
    fetcher = MagicMock()
    fetcher.fetch = AsyncMock(return_value={"AAPL": snap})
    src = OptionsIVCatalystSource(fetcher=fetcher)
    out = await src.fetch(["AAPL"])
    assert len(out) == 1
    cat = out[0]
    assert cat.symbol == "AAPL"
    assert cat.kind == "options_iv_elevated"
    assert "ATM IV 65%" in cat.title
    assert cat.source == "yahoo-options"
    assert cat.extra["atm_iv"] == 0.65


@pytest.mark.asyncio
async def test_options_iv_source_aclose_delegates():
    fetcher = MagicMock()
    fetcher.aclose = AsyncMock()
    src = OptionsIVCatalystSource(fetcher=fetcher)
    await src.aclose()
    fetcher.aclose.assert_awaited_once()


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
