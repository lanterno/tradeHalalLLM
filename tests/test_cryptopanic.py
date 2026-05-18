"""Tests for the pure helpers + cache/disabled paths in CryptoPanicCollector.

The actual HTTP fetch is exercised by the live integration; this file
locks the bits that don't need network: pair → currency derivation,
empty-key short-circuit, cache-TTL hit, and the backoff window.
"""

from __future__ import annotations

import time

import pytest

from halal_trader.sentiment.cryptopanic import (
    CryptoPanicCollector,
    CryptoPanicData,
    _pair_to_currency,
)

# ── _pair_to_currency ──────────────────────────────────────────


def test_pair_to_currency_strips_usdt_suffix():
    assert _pair_to_currency("BTCUSDT") == "BTC"


def test_pair_to_currency_strips_busd_suffix():
    assert _pair_to_currency("ETHBUSD") == "ETH"


def test_pair_to_currency_lowercase_input_returns_uppercase():
    """Output is always uppercase (matches CryptoPanic's API param)."""
    assert _pair_to_currency("btcusdt") == "BTC"


def test_pair_to_currency_returns_none_for_unknown_quote():
    """A pair that doesn't end in a known stablecoin → None (no fetch)."""
    assert _pair_to_currency("BTCEUR") is None
    assert _pair_to_currency("AAPL") is None


# ── CryptoPanicCollector.collect() — cache / disabled paths ───


@pytest.mark.asyncio
async def test_collect_returns_empty_when_no_api_key():
    """Without an API key the collector returns immediately."""
    c = CryptoPanicCollector(api_key="", trading_pairs=["BTCUSDT"])
    out = await c.collect()
    assert out == {}


@pytest.mark.asyncio
async def test_collect_returns_cached_within_ttl():
    """A second collect within the TTL window must reuse the cached
    result without hitting the network."""
    c = CryptoPanicCollector(api_key="k", trading_pairs=["BTCUSDT"], cache_ttl_seconds=300)
    cached = {"BTCUSDT": CryptoPanicData(pair="BTCUSDT")}
    c._cache = cached
    c._cache_time = time.monotonic()  # fresh
    out = await c.collect()
    assert out is cached  # exact same object


@pytest.mark.asyncio
async def test_collect_returns_cache_when_in_backoff_window():
    """When the API is in backoff, return whatever's cached rather
    than empty (avoid losing the last good signal during a drop-out)."""
    c = CryptoPanicCollector(api_key="k", trading_pairs=["BTCUSDT"], cache_ttl_seconds=1)
    cached = {"BTCUSDT": CryptoPanicData(pair="BTCUSDT", sentiment_score=0.4)}
    c._cache = cached
    c._cache_time = 0.0  # stale
    c._disabled_until = time.monotonic() + 60  # in backoff
    out = await c.collect()
    assert out == cached


@pytest.mark.asyncio
async def test_collect_returns_empty_when_no_pairs_resolve_to_currencies():
    """If every configured pair has an unrecognised quote (no USDT/BUSD),
    the collector returns empty without trying the API."""
    c = CryptoPanicCollector(api_key="k", trading_pairs=["BTCEUR", "AAPL"])
    out = await c.collect()
    assert out == {}


# ── healthy property ──────────────────────────────────────────


def test_healthy_true_when_disabled_until_zero():
    c = CryptoPanicCollector(api_key="k", trading_pairs=["BTCUSDT"])
    assert c.healthy is True


def test_healthy_false_when_in_backoff():
    c = CryptoPanicCollector(api_key="k", trading_pairs=["BTCUSDT"])
    c._disabled_until = time.monotonic() + 60
    assert c.healthy is False
