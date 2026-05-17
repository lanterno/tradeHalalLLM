"""Tests for :class:`SentimentManager`'s init / enabled / update orchestration.

The DB-backed sentiment-score persistence and the live HTTP collectors
have their own tests; this file focuses on the manager's role: deciding
whether sources are configured, fanning out to whichever collectors
are wired, and composing per-pair signals.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.sentiment.manager import SentimentManager


def test_disabled_when_no_sources_configured():
    """Both source-keys empty → no collectors spun up, manager disabled."""
    m = SentimentManager(trading_pairs=["BTCUSDT"])
    assert m.enabled is False
    assert m._reddit is None
    assert m._cryptopanic is None


def test_enabled_when_only_reddit_configured():
    m = SentimentManager(
        trading_pairs=["BTCUSDT"],
        reddit_client_id="x",
        reddit_client_secret="y",
    )
    assert m.enabled is True
    assert m._reddit is not None
    assert m._cryptopanic is None


def test_enabled_when_only_cryptopanic_configured():
    m = SentimentManager(
        trading_pairs=["BTCUSDT"],
        cryptopanic_api_key="key",
    )
    assert m.enabled is True
    assert m._reddit is None
    assert m._cryptopanic is not None


def test_partial_reddit_creds_treated_as_unconfigured():
    """A bare client_id without a secret should NOT spin up a collector
    (Reddit needs both); avoids surprises from copy-paste env errors."""
    m = SentimentManager(
        trading_pairs=["BTCUSDT"],
        reddit_client_id="x",  # no secret
    )
    assert m._reddit is None
    assert m.enabled is False


@pytest.mark.asyncio
async def test_start_no_op_when_disabled():
    """`start` must not spin up the background task when no sources
    are wired — otherwise we'd burn an idle task forever."""
    m = SentimentManager(trading_pairs=["BTCUSDT"])
    await m.start()
    assert m._task is None
    assert m._running is False


@pytest.mark.asyncio
async def test_update_with_no_collectors_returns_empty_signals():
    m = SentimentManager(trading_pairs=["BTCUSDT"])
    out = await m.update()
    assert out == {}
    assert m.latest_signals == {}


@pytest.mark.asyncio
async def test_update_swallows_collector_failure():
    """A failing collector mustn't crash the cycle — the cycle keeps
    going with sparse data rather than throwing."""
    m = SentimentManager(trading_pairs=["BTCUSDT"])
    crashing = MagicMock()
    crashing.collect = AsyncMock(side_effect=RuntimeError("rate limit"))
    m._reddit = crashing  # inject a bad collector
    out = await m.update()
    # No signal generated, but no exception either.
    assert out == {}


@pytest.mark.asyncio
async def test_latest_signals_property_reflects_last_update():
    """``latest_signals`` is the cached snapshot the cycle reads each tick."""
    m = SentimentManager(trading_pairs=["BTCUSDT"])
    # Start empty.
    assert m.latest_signals == {}
    # Simulate a previous update populating the cache.
    sig = MagicMock()
    m._latest_signals = {"BTCUSDT": sig}
    assert m.latest_signals == {"BTCUSDT": sig}
