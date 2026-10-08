"""Invariant: the end-of-day flatten records a close only if the broker accepted it.

From 2026-07 to 2026-10-01 every EOD close was refused by the broker (the
MCP ``close_position`` tool was called with an argument it doesn't take),
the refusal came back as an ordinary payload, and ``close_all`` stamped the
buys closed and wrote synthetic "filled" sells anyway. Positions stayed open
at Alpaca for months while the books said flat. A close the broker did not
accept must leave the trades open.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.mcp.client import AlpacaMCPClient, MCPToolError
from halal_trader.trading.executor import TradeExecutor, _accepted_closes


def _executor(broker: MagicMock, opens: list[Any]) -> tuple[Any, MagicMock]:
    repo = MagicMock()
    repo.get_open_trades = AsyncMock(return_value=opens)
    repo.close_open_trades_for_symbol = AsyncMock(return_value=1)
    repo.record_trade = AsyncMock(return_value=1)
    executor = TradeExecutor(
        broker,
        repo,
        max_position_pct=1.0,
        max_simultaneous_positions=10,
        max_sector_pct=0,
    )
    return executor, repo


def _buy(symbol: str, qty: float, entry_type: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol, side="buy", filled_quantity=qty, filled_price=100.0, entry_type=entry_type
    )


def _held(*symbols: str) -> AsyncMock:
    return AsyncMock(
        return_value=[SimpleNamespace(symbol=s, qty=1, current_price=101.0) for s in symbols]
    )


@pytest.mark.asyncio
async def test_refused_selective_close_leaves_trades_open() -> None:
    """The 2026-10-01 case: a reactor hold, and every other close refused."""
    broker = MagicMock()
    broker.get_all_positions = _held("MSFT", "AMD", "ORCL")
    broker.close_position = AsyncMock(side_effect=MCPToolError("close_position: refused"))
    executor, repo = _executor(
        broker, [_buy("MSFT", 24, "reactor_momentum"), _buy("AMD", 8), _buy("ORCL", 38)]
    )

    await executor.close_all()

    repo.close_open_trades_for_symbol.assert_not_awaited()
    repo.record_trade.assert_not_awaited()


@pytest.mark.asyncio
async def test_partly_refused_selective_close_records_only_the_accepted() -> None:
    broker = MagicMock()
    broker.get_all_positions = _held("MSFT", "AMD", "ORCL")

    async def close(symbol: str) -> dict[str, str]:
        if symbol == "ORCL":
            raise MCPToolError("close_position: refused")
        return {"id": f"o-{symbol}"}

    broker.close_position = AsyncMock(side_effect=close)
    executor, repo = _executor(
        broker, [_buy("MSFT", 24, "reactor_momentum"), _buy("AMD", 8), _buy("ORCL", 38)]
    )

    await executor.close_all()

    assert {c.args[0] for c in repo.close_open_trades_for_symbol.await_args_list} == {"AMD"}
    assert {c.kwargs["symbol"] for c in repo.record_trade.await_args_list} == {"AMD"}


@pytest.mark.asyncio
async def test_batch_close_records_only_symbols_with_a_2xx() -> None:
    broker = MagicMock()
    broker.get_all_positions = _held("AMD", "ORCL")
    broker.close_all_positions = AsyncMock(
        return_value={
            "result": [
                {"symbol": "AMD", "status": 200, "body": {"id": "o-1"}},
                {"symbol": "ORCL", "status": 403, "body": {"message": "insufficient qty"}},
            ]
        }
    )
    executor, repo = _executor(broker, [_buy("AMD", 8), _buy("ORCL", 38)])

    await executor.close_all()

    assert {c.args[0] for c in repo.close_open_trades_for_symbol.await_args_list} == {"AMD"}


@pytest.mark.asyncio
async def test_unrecognised_batch_reply_records_nothing() -> None:
    broker = MagicMock()
    broker.get_all_positions = _held("AMD")
    broker.close_all_positions = AsyncMock(return_value="Error: validation failed")
    executor, repo = _executor(broker, [_buy("AMD", 8)])

    await executor.close_all()

    repo.close_open_trades_for_symbol.assert_not_awaited()
    repo.record_trade.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_snapshot_requires_an_accepted_close() -> None:
    """Without the pre-close snapshot nothing is known to be a phantom."""
    broker = MagicMock()
    broker.get_all_positions = AsyncMock(side_effect=RuntimeError("broker down"))
    broker.close_all_positions = AsyncMock(return_value={"result": []})
    executor, repo = _executor(broker, [_buy("AMD", 8)])

    await executor.close_all()

    repo.close_open_trades_for_symbol.assert_not_awaited()


def test_accepted_closes_shapes() -> None:
    assert _accepted_closes({"closed": ["amd"]}) == {"AMD"}
    assert _accepted_closes([{"symbol": "AMD", "status": 200}]) == {"AMD"}
    assert _accepted_closes([{"symbol": "AMD", "status": "200"}]) == set()
    assert _accepted_closes({"result": "closed"}) == set()
    assert _accepted_closes(None) == set()


# ── the MCP adapter: right argument, refusals raise ─────────────────


def _session_returning(*, is_error: bool, text: str) -> MagicMock:
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(isError=is_error, content=[SimpleNamespace(text=text)])
    )
    return session


def _client(session: MagicMock) -> AlpacaMCPClient:
    client = AlpacaMCPClient()
    client.session = session
    client._tools = {"close_position": object(), "close_all_positions": object()}
    return client


@pytest.mark.asyncio
async def test_close_position_sends_the_servers_argument_name() -> None:
    session = _session_returning(is_error=False, text='{"id": "o-1"}')
    await _client(session).close_position("AMD")
    session.call_tool.assert_awaited_once_with("close_position", {"symbol_or_asset_id": "AMD"})


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["close_position", "close_all_positions"])
async def test_refused_close_raises(method: str) -> None:
    session = _session_returning(is_error=True, text="validation error: missing argument")
    client = _client(session)
    with pytest.raises(MCPToolError):
        if method == "close_position":
            await client.close_position("AMD")
        else:
            await client.close_all_positions()
