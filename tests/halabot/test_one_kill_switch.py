"""The shadow obeys the operator's one kill-switch (`halal-trader halt`)."""

from __future__ import annotations

import pytest

from halabot.api import queries
from halabot.app import _make_halt_check
from halal_trader.core.halt import clear_halt, set_halt


@pytest.mark.asyncio
async def test_the_shadow_and_its_health_read_the_bots_kill_switch(halabot_engine) -> None:  # type: ignore[no-untyped-def]
    halted = _make_halt_check(halabot_engine)
    assert await halted() is False

    await set_halt(halabot_engine, reason="drill", set_by="test")
    assert await halted() is True
    assert (await queries.system_health(halabot_engine))["halted"] is True

    await clear_halt(halabot_engine)
    assert await halted() is False
