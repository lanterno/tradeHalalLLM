"""`halabot shadow` start-up order: the LLM spend meter is live before the engine."""

from __future__ import annotations

import pytest

import halabot.app
import halal_trader.db.models
from halabot.cli import _run_shadow
from halal_trader.core.llm import spend


class _Stop(Exception):
    pass


@pytest.mark.asyncio
async def test_spend_meter_is_installed_before_the_engine_bootstraps(monkeypatch):
    # Bootstrap replay runs inside build_engine; it used to run before the
    # meter was installed, so anything it spent was never counted.
    order: list[str] = []

    async def fake_init_db(url):
        return object()

    async def fake_build_engine(**kwargs):
        order.append("build_engine")
        raise _Stop

    monkeypatch.setattr(halal_trader.db.models, "init_db", fake_init_db)
    monkeypatch.setattr(spend, "install", lambda meter: order.append("install"))
    monkeypatch.setattr(halabot.app, "build_engine", fake_build_engine)
    with pytest.raises(_Stop):
        await _run_shadow(once=True, interval=1.0, timeframe="1Hour", days=1)
    assert order == ["install", "build_engine"]
