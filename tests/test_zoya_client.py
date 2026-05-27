"""Tests for `halal.zoya.ZoyaClient`.

The Zoya client is the live-endpoint Shariah-compliance screener.
A regression here would either misclassify halal vs non-halal stocks
(the gate that decides every stock trade) or break the operator's
sandbox-vs-live toggle. None of the client's surface had direct tests.

We use `httpx.MockTransport` to test the HTTP layer without a real
network call.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from halal_trader.halal.zoya import ZoyaClient

# ── _normalize_status (static, pure) ───────────────────────


def test_normalize_status_compliant_to_halal():
    """`COMPLIANT` is the explicit halal signal."""
    assert ZoyaClient._normalize_status("COMPLIANT") == "halal"


def test_normalize_status_non_compliant_to_not_halal():
    assert ZoyaClient._normalize_status("NON_COMPLIANT") == "not_halal"


def test_normalize_status_doubtful_passes_through():
    assert ZoyaClient._normalize_status("DOUBTFUL") == "doubtful"


def test_normalize_status_unknown_status_defaults_to_doubtful():
    """Defensive: an unrecognised Zoya status string → "doubtful"
    rather than crashing or silently passing through. Halal compliance
    is non-negotiable; an unknown status must NOT be treated as halal."""
    assert ZoyaClient._normalize_status("PENDING_REVIEW") == "doubtful"
    assert ZoyaClient._normalize_status("") == "doubtful"
    assert ZoyaClient._normalize_status("WEIRD") == "doubtful"


# ── Constructor URL selection ──────────────────────────────


def test_constructor_uses_live_url_by_default():
    """Default toggle is production — pin so a refactor that changes
    the default to sandbox is intentional."""
    client = ZoyaClient("test-key")
    assert "api.zoya.finance" in client._url
    assert "sandbox" not in client._url


def test_constructor_uses_sandbox_when_enabled():
    """`use_sandbox=True` → swaps to the sandbox endpoint. The
    operator wires this in dev/CI."""
    client = ZoyaClient("test-key", use_sandbox=True)
    assert "sandbox-api.zoya.finance" in client._url


def test_constructor_stores_api_key():
    client = ZoyaClient("my-secret-key")
    assert client.api_key == "my-secret-key"


# ── HTTP layer (mocked) ────────────────────────────────────


def _client_with_handler(handler):
    """Build a ZoyaClient whose underlying httpx client uses the
    given mock transport handler."""
    client = ZoyaClient("test-key")
    transport = httpx.MockTransport(handler)
    client._http = httpx.AsyncClient(transport=transport, timeout=30.0)
    return client


@pytest.mark.asyncio
async def test_screen_stock_compliant_returns_halal():
    """A `COMPLIANT` GraphQL response → `compliance="halal"` shape."""

    def handler(req):
        return httpx.Response(
            200,
            json={
                "data": {
                    "basicCompliance": {
                        "report": {
                            "symbol": "AAPL",
                            "name": "Apple Inc.",
                            "status": "COMPLIANT",
                            "reportDate": "2026-04-01",
                        }
                    }
                }
            },
        )

    client = _client_with_handler(handler)
    out = await client.screen_stock("AAPL")
    assert out["symbol"] == "AAPL"
    assert out["compliance"] == "halal"
    assert "Apple Inc." in out["detail"]
    assert "raw" in out


@pytest.mark.asyncio
async def test_screen_stock_non_compliant_returns_not_halal():
    def handler(req):
        return httpx.Response(
            200,
            json={
                "data": {
                    "basicCompliance": {
                        "report": {
                            "symbol": "TSLA",
                            "status": "NON_COMPLIANT",
                        }
                    }
                }
            },
        )

    client = _client_with_handler(handler)
    out = await client.screen_stock("TSLA")
    assert out["compliance"] == "not_halal"


@pytest.mark.asyncio
async def test_screen_stock_no_report_returns_doubtful():
    """The Zoya API can return `null` for report when the symbol isn't
    in their universe — must NOT default to halal. Pin so a refactor
    doesn't accidentally allow trading on un-screened symbols."""

    def handler(req):
        return httpx.Response(
            200,
            json={"data": {"basicCompliance": {"report": None}}},
        )

    client = _client_with_handler(handler)
    out = await client.screen_stock("UNKNOWN")
    assert out["compliance"] == "doubtful"
    assert "No report found" in out["detail"]


@pytest.mark.asyncio
async def test_screen_stock_graphql_error_returns_doubtful():
    """GraphQL errors come back HTTP 200 with `errors` array — `_query`
    raises, `screen_stock` swallows + returns doubtful. Critical
    safety: a Zoya API hiccup must NOT silently pass non-halal stocks."""

    def handler(req):
        return httpx.Response(
            200,
            json={"errors": [{"message": "rate limit hit"}]},
        )

    client = _client_with_handler(handler)
    out = await client.screen_stock("AAPL")
    assert out["compliance"] == "doubtful"
    assert "rate limit hit" in out["detail"]


@pytest.mark.asyncio
async def test_screen_stock_http_error_returns_doubtful():
    """HTTP 5xx → swallowed, returns doubtful with status code in
    the detail message (operator can grep)."""

    def handler(req):
        return httpx.Response(503, json={"error": "service unavailable"})

    client = _client_with_handler(handler)
    out = await client.screen_stock("AAPL")
    assert out["compliance"] == "doubtful"
    assert "503" in out["detail"]


@pytest.mark.asyncio
async def test_screen_stock_transport_error_detail_includes_type():
    """A transport error (e.g. ConnectError) stringifies to '' — the old
    log/detail was a useless blank ('Zoya API error for X: '). The detail
    must now carry the exception TYPE so the operator can diagnose the
    outage that starved the halal universe on 2026-05-27."""

    def handler(req):
        raise httpx.ConnectError("")  # empty message, like the real outage

    client = _client_with_handler(handler)
    out = await client.screen_stock("AAPL")
    assert out["compliance"] == "doubtful"
    assert "ConnectError" in out["detail"]


@pytest.mark.asyncio
async def test_screen_stock_http_error_detail_includes_body():
    """HTTP error detail now includes the response body, not just the code."""

    def handler(req):
        return httpx.Response(401, text="invalid api key")

    client = _client_with_handler(handler)
    out = await client.screen_stock("AAPL")
    assert out["compliance"] == "doubtful"
    assert "401" in out["detail"]
    assert "invalid api key" in out["detail"]


@pytest.mark.asyncio
async def test_screen_stock_doubtful_status_passes_through():
    """A native `DOUBTFUL` response (rare but valid) preserves the
    doubtful flag."""

    def handler(req):
        return httpx.Response(
            200,
            json={"data": {"basicCompliance": {"report": {"symbol": "X", "status": "DOUBTFUL"}}}},
        )

    client = _client_with_handler(handler)
    out = await client.screen_stock("X")
    assert out["compliance"] == "doubtful"


@pytest.mark.asyncio
async def test_screen_stock_unknown_status_falls_back_to_doubtful():
    """A Zoya response with an unrecognised status → doubtful.
    Symmetric with `_normalize_status` defensive fallback."""

    def handler(req):
        return httpx.Response(
            200,
            json={"data": {"basicCompliance": {"report": {"symbol": "X", "status": "PENDING"}}}},
        )

    client = _client_with_handler(handler)
    out = await client.screen_stock("X")
    assert out["compliance"] == "doubtful"


@pytest.mark.asyncio
async def test_screen_stock_sends_authorization_header():
    """The api_key threads through to the `Authorization` header.
    Pin so a refactor that drops auth doesn't silently call the
    public endpoint (which would return errors / wrong data)."""
    captured: dict[str, Any] = {}

    def handler(req):
        captured["headers"] = dict(req.headers)
        return httpx.Response(
            200,
            json={"data": {"basicCompliance": {"report": None}}},
        )

    client = _client_with_handler(handler)
    client.api_key = "secret-key-xyz"
    await client.screen_stock("AAPL")
    assert captured["headers"].get("authorization") == "secret-key-xyz"


@pytest.mark.asyncio
async def test_screen_stock_sends_symbol_in_variables():
    """The symbol parameter flows into GraphQL `variables` — pin so
    a refactor doesn't accidentally hardcode a query string."""
    captured: dict[str, Any] = {}

    def handler(req):
        import json

        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"data": {"basicCompliance": {"report": None}}})

    client = _client_with_handler(handler)
    await client.screen_stock("MSFT")
    assert captured["body"]["variables"] == {"symbol": "MSFT"}


# ── Bulk screening ────────────────────────────────────────


@pytest.mark.asyncio
async def test_screen_bulk_calls_screen_stock_per_symbol():
    """Bulk is just iterated single-stock screens — pin so a
    refactor that batches doesn't silently change the request shape
    Zoya rate-limits on."""
    call_count = {"n": 0}

    def handler(req):
        call_count["n"] += 1
        return httpx.Response(
            200,
            json={"data": {"basicCompliance": {"report": {"symbol": "X", "status": "COMPLIANT"}}}},
        )

    client = _client_with_handler(handler)
    out = await client.screen_bulk(["AAPL", "MSFT", "GOOG"])
    assert len(out) == 3
    assert call_count["n"] == 3
    assert all(r["compliance"] == "halal" for r in out)


@pytest.mark.asyncio
async def test_screen_bulk_empty_list_returns_empty():
    """No symbols → empty list, no HTTP calls."""
    call_count = {"n": 0}

    def handler(req):
        call_count["n"] += 1
        return httpx.Response(200, json={})

    client = _client_with_handler(handler)
    out = await client.screen_bulk([])
    assert out == []
    assert call_count["n"] == 0


@pytest.mark.asyncio
async def test_screen_bulk_isolates_per_symbol_failure():
    """A failure on one symbol doesn't drop the others — bulk screen
    is best-effort. Each result is returned; failed ones are
    `doubtful`."""

    def handler(req):
        import json

        body = json.loads(req.content)
        sym = body["variables"]["symbol"]
        if sym == "AAPL":
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={"data": {"basicCompliance": {"report": {"symbol": sym, "status": "COMPLIANT"}}}},
        )

    client = _client_with_handler(handler)
    out = await client.screen_bulk(["AAPL", "MSFT"])
    assert len(out) == 2
    assert out[0]["symbol"] == "AAPL"
    assert out[0]["compliance"] == "doubtful"  # failed
    assert out[1]["symbol"] == "MSFT"
    assert out[1]["compliance"] == "halal"
