"""Tests for sentiment/events.py — news event reactor."""

from __future__ import annotations

from typing import Any

import pytest

from halal_trader.sentiment.events import NewsEvent, NewsEventReactor


def _reactor(api_key: str = "secret") -> NewsEventReactor:
    return NewsEventReactor(
        api_key=api_key,
        trading_pairs=["BTCUSDT", "ETHUSDT"],
        poll_interval_seconds=1,
        importance_filter="hot",
    )


def _api_response(items: list[dict[str, Any]]) -> dict:
    return {"results": items}


def _item(
    url: str,
    title: str,
    *,
    currencies: list[str] | None = None,
    pos: int = 0,
    neg: int = 0,
    kind: str = "news",
) -> dict:
    return {
        "url": url,
        "title": title,
        "published_at": "2026-04-25T12:00:00Z",
        "source": {"title": "test-feed"},
        "kind": kind,
        "currencies": [{"code": c} for c in (currencies or [])],
        "votes": {"positive": pos, "negative": neg},
    }


def test_enabled_property():
    assert _reactor("k").enabled is True
    assert _reactor("").enabled is False


def test_pair_to_currency_mapping():
    r = _reactor()
    assert "BTC" in r._currency_to_pairs
    assert r._currency_to_pairs["BTC"] == ["BTCUSDT"]


def test_on_event_appends_callback():
    r = _reactor()

    async def cb(event: NewsEvent) -> None:
        pass

    r.on_event(cb)
    assert cb in r._callbacks


@pytest.mark.asyncio
async def test_check_for_events_yields_unseen_items(monkeypatch):
    r = _reactor()

    items = [
        _item("https://x.test/1", "Bitcoin halving!", currencies=["BTC"], pos=5),
        _item("https://x.test/2", "Ethereum merge", currencies=["ETH"], neg=3),
    ]

    class _StubClient:
        is_closed = False

        async def get(self, url, params):
            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    pass

                @staticmethod
                def json():
                    return _api_response(items)

            return _Resp()

        async def aclose(self):
            pass

    r._client = _StubClient()
    events = await r._check_for_events()
    assert len(events) == 2
    assert events[0].sentiment == "positive"
    assert events[1].sentiment == "negative"
    assert events[0].affected_pairs == ["BTCUSDT"]


@pytest.mark.asyncio
async def test_check_for_events_dedupes_by_url(monkeypatch):
    r = _reactor()

    items = [_item("https://x.test/1", "first", currencies=["BTC"])]

    class _StubClient:
        is_closed = False

        async def get(self, url, params):
            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    pass

                @staticmethod
                def json():
                    return _api_response(items)

            return _Resp()

        async def aclose(self):
            pass

    r._client = _StubClient()
    first = await r._check_for_events()
    second = await r._check_for_events()
    assert len(first) == 1
    assert second == []


@pytest.mark.asyncio
async def test_check_for_events_handles_no_currencies():
    r = NewsEventReactor(api_key="k", trading_pairs=[])
    events = await r._check_for_events()
    assert events == []


@pytest.mark.asyncio
async def test_check_for_events_skips_404_to_next_url():
    r = _reactor()

    class _StubClient:
        is_closed = False
        calls: list[str] = []

        async def get(self, url, params):
            self.calls.append(url)

            class _Resp404:
                status_code = 404

                def raise_for_status(self):
                    raise RuntimeError("404")

                @staticmethod
                def json():
                    return {}

            class _Resp200:
                status_code = 200

                def raise_for_status(self):
                    pass

                @staticmethod
                def json():
                    return _api_response([_item("https://x/y", "ok", currencies=["BTC"])])

            if "developer" in url:
                return _Resp404()
            return _Resp200()

        async def aclose(self):
            pass

    r._client = _StubClient()
    events = await r._check_for_events()
    assert len(events) == 1
    assert any("growth" in c or "enterprise" in c for c in r._client.calls)


@pytest.mark.asyncio
async def test_seen_urls_pruned_at_1000():
    r = _reactor()

    items = [_item(f"https://x/{i}", "t", currencies=["BTC"]) for i in range(1001)]

    class _StubClient:
        is_closed = False

        async def get(self, url, params):
            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    pass

                @staticmethod
                def json():
                    return _api_response(items)

            return _Resp()

        async def aclose(self):
            pass

    r._client = _StubClient()
    await r._check_for_events()
    # Set is pruned to ~500 once it grows past 1000.
    assert len(r._seen_urls) <= 1000
