"""A refused account or positions read raises; it is never an empty account or a flat book."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.mcp.client import AlpacaMCPClient, MCPToolError


def _client(*, is_error: bool, text: str) -> AlpacaMCPClient:
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(isError=is_error, content=[SimpleNamespace(text=text)])
    )
    client = AlpacaMCPClient()
    client.session = session
    client._tools = {"get_account_info": object(), "get_all_positions": object()}
    return client


@pytest.mark.asyncio
async def test_a_refused_account_read_raises() -> None:
    client = _client(is_error=True, text="Error: request to the broker timed out")
    with pytest.raises(MCPToolError):
        await client.get_account_info()


@pytest.mark.asyncio
async def test_a_refused_positions_read_raises() -> None:
    client = _client(is_error=True, text="Error: request to the broker timed out")
    with pytest.raises(MCPToolError):
        await client.get_all_positions()


@pytest.mark.asyncio
async def test_reads_the_server_answers_still_parse() -> None:
    account = await _client(is_error=False, text='{"equity": "103684.71"}').get_account_info()
    assert account.equity == 103684.71
    flat = await _client(is_error=False, text='{"result": []}').get_all_positions()
    assert flat == []
