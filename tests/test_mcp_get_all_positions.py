"""Pin the response-shape handling of ``AlpacaMCPClient.get_all_positions``.

Upstream Alpaca MCP wraps the array as ``{"result": [...]}`` (same
shape change ``get_calendar`` saw). Before the unwrap fix, this
parser silently returned ``[]`` even when the broker really held
positions — producing 100% "phantom-position" drift on every
reconcile pass. These tests pin the three shapes we need to keep
working: wrapped dict, bare list (legacy), empty.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from halal_trader.mcp.client import AlpacaMCPClient


def _client_with_response(response: object) -> AlpacaMCPClient:
    c = AlpacaMCPClient()
    c.call_tool = AsyncMock(return_value=response)  # type: ignore[method-assign]
    return c


@pytest.mark.asyncio
async def test_wrapped_result_shape_parsed():
    """The current upstream shape: ``{"result": [...]}``."""
    resp = {
        "result": [
            {
                "symbol": "AMZN",
                "qty": "75",
                "avg_entry_price": "261.88",
                "current_price": "263.53",
                "unrealized_pl": "123.78",
                "unrealized_plpc": "0.0063",
            },
            {
                "symbol": "NVDA",
                "qty": "193",
                "avg_entry_price": "221.76",
                "current_price": "219.45",
            },
        ]
    }
    positions = await _client_with_response(resp).get_all_positions()
    assert len(positions) == 2
    assert positions[0].symbol == "AMZN"
    assert positions[0].qty == 75.0
    assert positions[0].avg_entry_price == 261.88
    assert positions[1].symbol == "NVDA"
    assert positions[1].qty == 193.0


@pytest.mark.asyncio
async def test_bare_list_shape_still_parsed():
    """Legacy: some older Alpaca MCP versions / fixtures return a bare list."""
    resp = [
        {"symbol": "AAPL", "qty": "10", "avg_entry_price": "200", "current_price": "201"},
    ]
    positions = await _client_with_response(resp).get_all_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_empty_wrapped_result_returns_empty():
    """Empty ``{"result": []}`` means no positions, not a parse failure."""
    positions = await _client_with_response({"result": []}).get_all_positions()
    assert positions == []


@pytest.mark.asyncio
async def test_empty_dict_returns_empty():
    """Some MCP versions return ``{}`` when there are no positions."""
    positions = await _client_with_response({}).get_all_positions()
    assert positions == []


@pytest.mark.asyncio
async def test_string_response_returns_empty():
    """Defensive: a text response (e.g. \"no positions\") → empty list."""
    positions = await _client_with_response("no positions").get_all_positions()
    assert positions == []


@pytest.mark.asyncio
async def test_non_dict_items_in_result_skipped():
    """Result list contains a non-dict entry → skip it, parse the rest."""
    resp = {
        "result": [
            "junk-line",
            {"symbol": "AAPL", "qty": "5", "avg_entry_price": "180", "current_price": "181"},
        ]
    }
    positions = await _client_with_response(resp).get_all_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
