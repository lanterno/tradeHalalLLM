"""Tests for the no-OAuth Reddit public-JSON fetcher."""

from __future__ import annotations

import time

import httpx
import pytest

from halal_trader.sentiment.reddit_public import (
    DEFAULT_CRYPTO_SUBS,
    DEFAULT_STOCK_SUBS,
    RedditPublicFetcher,
)


def _post(symbol: str, age_seconds: float, score: int = 10) -> dict:
    """One reddit search-result child wrapper."""
    return {
        "kind": "t3",
        "data": {
            "title": f"Some thread about {symbol}",
            "selftext": "lorem ipsum",
            "created_utc": time.time() - age_seconds,
            "score": score,
            "subreddit": "CryptoCurrency",
            "permalink": "/r/x/comments/y/z",
        },
    }


def _payload(posts: list[dict]) -> dict:
    return {"data": {"children": posts}}


def _client_with(payload: dict) -> httpx.AsyncClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _client_per_sub(payload_by_sub: dict[str, dict]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        for sub, payload in payload_by_sub.items():
            if f"/r/{sub}/" in str(request.url):
                return httpx.Response(200, json=payload)
        return httpx.Response(200, json={"data": {"children": []}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── Disabled paths ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_user_agent_returns_empty() -> None:
    f = RedditPublicFetcher(user_agent="")
    out = await f.fetch_for_symbols(["BTC"])
    assert out == []


@pytest.mark.asyncio
async def test_no_symbols_returns_empty() -> None:
    f = RedditPublicFetcher(subreddits=("CryptoCurrency",))
    f._client = _client_with(_payload([_post("BTC", 60)]))
    assert await f.fetch_for_symbols([]) == []
    await f.aclose()


# ── Happy path ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetches_mentions_per_symbol() -> None:
    f = RedditPublicFetcher(subreddits=("CryptoCurrency",))
    f._client = _client_with(_payload([_post("BTC", 60), _post("BTC", 120)]))
    out = await f.fetch_for_symbols(["BTC"])
    assert len(out) == 2
    assert all(m.symbol == "BTC" for m in out)
    assert all(m.source == "reddit:CryptoCurrency" for m in out)
    await f.aclose()


@pytest.mark.asyncio
async def test_normalises_symbol_to_upper() -> None:
    f = RedditPublicFetcher(subreddits=("CryptoCurrency",))
    f._client = _client_with(_payload([_post("btc", 60)]))
    out = await f.fetch_for_symbols(["btc"])
    assert len(out) == 1
    assert out[0].symbol == "BTC"
    await f.aclose()


@pytest.mark.asyncio
async def test_fans_out_across_subreddits() -> None:
    f = RedditPublicFetcher(
        subreddits=("CryptoCurrency", "Bitcoin", "ethfinance"),
    )
    f._client = _client_per_sub(
        {
            "CryptoCurrency": _payload([_post("BTC", 30)]),
            "Bitcoin": _payload([_post("BTC", 60), _post("BTC", 90)]),
            "ethfinance": _payload([]),
        }
    )
    out = await f.fetch_for_symbols(["BTC"])
    assert len(out) == 3
    sources = {m.source for m in out}
    assert sources == {"reddit:CryptoCurrency", "reddit:Bitcoin"}
    await f.aclose()


@pytest.mark.asyncio
async def test_passes_score_through() -> None:
    f = RedditPublicFetcher(subreddits=("CryptoCurrency",))
    f._client = _client_with(_payload([_post("BTC", 30, score=42)]))
    out = await f.fetch_for_symbols(["BTC"])
    assert out[0].score == 42
    await f.aclose()


@pytest.mark.asyncio
async def test_skips_posts_without_timestamp() -> None:
    bad = {
        "kind": "t3",
        "data": {"title": "x", "score": 1},  # no created_utc
    }
    f = RedditPublicFetcher(subreddits=("CryptoCurrency",))
    f._client = _client_with(_payload([bad, _post("BTC", 30)]))
    out = await f.fetch_for_symbols(["BTC"])
    assert len(out) == 1
    await f.aclose()


# ── Error handling ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_http_error_per_sub_isolated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "Bitcoin" in str(request.url):
            return httpx.Response(429)  # rate-limited
        return httpx.Response(200, json=_payload([_post("BTC", 30)]))

    f = RedditPublicFetcher(subreddits=("CryptoCurrency", "Bitcoin"))
    f._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = await f.fetch_for_symbols(["BTC"])
    assert len(out) == 1
    assert out[0].source == "reddit:CryptoCurrency"
    await f.aclose()


@pytest.mark.asyncio
async def test_malformed_response_returns_empty() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    f = RedditPublicFetcher(subreddits=("CryptoCurrency",))
    f._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = await f.fetch_for_symbols(["BTC"])
    assert out == []
    await f.aclose()


# ── Caching ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_caches_per_pair() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_payload([_post("BTC", 30)]))

    f = RedditPublicFetcher(subreddits=("CryptoCurrency",))
    f._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await f.fetch_for_symbols(["BTC"])
    await f.fetch_for_symbols(["BTC"])  # second call — cached
    await f.fetch_for_symbols(["BTC"])
    assert calls == 1
    await f.aclose()


@pytest.mark.asyncio
async def test_cache_per_subreddit_independent() -> None:
    calls_by_sub: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        for sub in ("CryptoCurrency", "Bitcoin"):
            if f"/r/{sub}/" in str(request.url):
                calls_by_sub[sub] = calls_by_sub.get(sub, 0) + 1
                return httpx.Response(200, json=_payload([_post("BTC", 30)]))
        return httpx.Response(200, json=_payload([]))

    f = RedditPublicFetcher(subreddits=("CryptoCurrency", "Bitcoin"))
    f._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await f.fetch_for_symbols(["BTC"])
    await f.fetch_for_symbols(["BTC"])
    # Each sub hit exactly once across two outer calls.
    assert calls_by_sub == {"CryptoCurrency": 1, "Bitcoin": 1}
    await f.aclose()


# ── Smoke ────────────────────────────────────────────────────────


def test_default_sub_lists_present() -> None:
    assert "CryptoCurrency" in DEFAULT_CRYPTO_SUBS
    assert "wallstreetbets" in DEFAULT_STOCK_SUBS
