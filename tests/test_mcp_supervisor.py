"""MCP subprocess supervisor tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.mcp.supervisor import MCPSupervisor, SupervisorPolicy


def _client(connect_side_effects=None, disconnect_side_effect=None):
    c = MagicMock()
    c.connect = AsyncMock(side_effect=connect_side_effects)
    c.disconnect = AsyncMock(side_effect=disconnect_side_effect)
    return c


def _fast_policy() -> SupervisorPolicy:
    return SupervisorPolicy(
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.005,
        factor=2,
        max_consecutive_failures=3,
    )


async def test_start_marks_connected_on_success():
    c = _client(connect_side_effects=[None])
    sup = MCPSupervisor(c, policy=_fast_policy())
    await sup.start()
    assert sup.connected is True


async def test_start_propagates_first_connect_failure():
    """If we can't connect on startup, surface the error to the operator."""
    c = _client(connect_side_effects=[RuntimeError("no socket")])
    sup = MCPSupervisor(c, policy=_fast_policy())
    with pytest.raises(RuntimeError):
        await sup.start()
    assert sup.connected is False


async def test_reconnect_succeeds_after_failures():
    # Two failures, then success.
    side = [RuntimeError("fail 1"), RuntimeError("fail 2"), None]
    c = _client(connect_side_effects=side)
    sup = MCPSupervisor(c, policy=_fast_policy())
    ok = await sup.reconnect()
    assert ok is True
    assert sup.connected is True


async def test_reconnect_gives_up_after_max_failures_and_alerts():
    side = [RuntimeError("dead")] * 10
    c = _client(connect_side_effects=side)
    alerts = MagicMock()
    alerts.notify = AsyncMock()
    sup = MCPSupervisor(c, policy=_fast_policy(), alerts=alerts)
    ok = await sup.reconnect()
    assert ok is False
    alerts.notify.assert_awaited_once()
    args, _ = alerts.notify.await_args
    assert args[0] == "mcp.crash"


async def test_call_with_recovery_succeeds_first_try():
    sup = MCPSupervisor(_client(connect_side_effects=[None]), policy=_fast_policy())
    fn = AsyncMock(return_value="ok")
    result = await sup.call_with_recovery(fn)
    assert result == "ok"
    assert fn.await_count == 1


async def test_call_with_recovery_retries_after_reconnect():
    """First call fails → reconnect succeeds → second call returns."""
    c = _client(connect_side_effects=[None])  # initial connect ok
    sup = MCPSupervisor(c, policy=_fast_policy())
    await sup.start()

    # After reconnect, get_open_orders succeeds — script the connect side too.
    c.connect.side_effect = [None]
    fn = AsyncMock(side_effect=[RuntimeError("broken pipe"), "recovered"])
    result = await sup.call_with_recovery(fn)
    assert result == "recovered"
    assert fn.await_count == 2


async def test_call_with_recovery_raises_when_reconnect_gives_up():
    c = _client(connect_side_effects=[None])
    sup = MCPSupervisor(
        c,
        policy=SupervisorPolicy(
            initial_backoff_seconds=0.001,
            max_backoff_seconds=0.005,
            factor=2,
            max_consecutive_failures=2,
        ),
    )
    await sup.start()
    # Future reconnect attempts always fail.
    c.connect.side_effect = [RuntimeError("nope")] * 5
    fn = AsyncMock(side_effect=RuntimeError("call failed"))
    with pytest.raises(RuntimeError):
        await sup.call_with_recovery(fn)


async def test_stop_clears_connected():
    sup = MCPSupervisor(_client(connect_side_effects=[None]), policy=_fast_policy())
    await sup.start()
    await sup.stop()
    assert sup.connected is False
