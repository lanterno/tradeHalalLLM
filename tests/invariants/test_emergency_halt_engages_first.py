"""INVARIANT: `halal-trader halt` engages the kill-switch before anything else.

The emergency stop used to liquidate first and halt second, so a broker that
could not be reached aborted the command and the kill-switch was never set:
the bot kept trading while the operator believed it was stopped.
"""

from __future__ import annotations

import asyncio

import pytest
from click.testing import CliRunner

from halal_trader.cli.halt import halt


def _is_halted(database_url: str) -> bool:
    from sqlalchemy.ext.asyncio import create_async_engine

    from halal_trader.core.halt import is_halted

    async def go() -> bool:
        eng = create_async_engine(database_url)
        try:
            return await is_halted(eng)
        finally:
            await eng.dispose()

    return asyncio.run(go())


class _Broker:
    fail_connect = False

    async def connect(self) -> None:
        if self.fail_connect:
            raise ConnectionError("alpaca-mcp-server did not start")

    async def disconnect(self) -> None: ...


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> type[_Broker]:
    import halal_trader.mcp.client as client

    _Broker.fail_connect = False
    monkeypatch.setattr(client, "AlpacaMCPClient", _Broker)
    return _Broker


def test_plain_halt_engages(database_url: str) -> None:
    result = CliRunner().invoke(halt, ["--reason", "drill"])

    assert result.exit_code == 0, result.output
    assert _is_halted(database_url)


def test_unreachable_broker_still_leaves_the_bot_halted(
    database_url: str, broker: type[_Broker]
) -> None:
    broker.fail_connect = True

    result = CliRunner().invoke(halt, ["--reason", "broker down", "--close-all", "stocks"])

    assert result.exit_code == 1  # the liquidation failure is reported...
    assert "IS engaged" in result.output
    assert _is_halted(database_url)  # ...but the halt holds


def test_liquidation_runs_only_after_the_halt_is_set(
    database_url: str, broker: type[_Broker], monkeypatch: pytest.MonkeyPatch
) -> None:
    import halal_trader.core.halt as halt_module
    import halal_trader.core.liquidate as liquidate

    real_set_halt = halt_module.set_halt
    order: list[str] = []

    async def spy_set_halt(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        order.append("halt")
        return await real_set_halt(*args, **kwargs)

    async def spy_liquidate(_mcp: object) -> list[object]:
        order.append("liquidate")
        return []  # no open positions

    monkeypatch.setattr(halt_module, "set_halt", spy_set_halt)
    monkeypatch.setattr(liquidate, "liquidate_stocks", spy_liquidate)

    result = CliRunner().invoke(halt, ["--reason", "flatten", "--close-all", "stocks"])

    assert result.exit_code == 0, result.output
    assert order == ["halt", "liquidate"]
