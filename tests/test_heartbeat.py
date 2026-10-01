"""Cross-process liveness: the bot's heartbeat rows, and what the web makes of them."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from halal_trader.core.heartbeat import (
    STOCK_CYCLE,
    STOCK_MONITOR,
    STOCK_PROCESS,
    Beat,
    beat,
    read_beats,
)


async def test_beat_upserts_one_row_per_component(engine: AsyncEngine) -> None:
    t0 = datetime(2026, 10, 1, 14, 0, tzinfo=UTC)
    await beat(engine, STOCK_PROCESS, {"n": 1}, now=t0)
    await beat(engine, STOCK_PROCESS, {"n": 2}, now=t0 + timedelta(minutes=1))
    await beat(engine, STOCK_CYCLE, now=t0)

    beats = await read_beats(engine)

    assert set(beats) == {STOCK_PROCESS, STOCK_CYCLE}
    assert beats[STOCK_PROCESS].beat_at == t0 + timedelta(minutes=1)
    assert beats[STOCK_PROCESS].detail == {"n": 2}
    assert beats[STOCK_CYCLE].detail is None


async def test_beat_never_raises() -> None:
    dead = create_async_engine("postgresql+asyncpg://nobody:x@127.0.0.1:1/none")
    try:
        await beat(dead, STOCK_PROCESS)  # unreachable DB: logged, not raised
    finally:
        await dead.dispose()
    await beat(None, STOCK_PROCESS)  # no engine: a no-op


def test_staleness_follows_each_components_cadence() -> None:
    now = datetime(2026, 10, 1, 14, 0, tzinfo=UTC)

    def at(component: str, minutes_ago: float) -> Beat:
        return Beat(component, now - timedelta(minutes=minutes_ago), None)

    assert not at(STOCK_PROCESS, 2).is_stale(now)
    assert at(STOCK_PROCESS, 4).is_stale(now)
    assert not at(STOCK_MONITOR, 4).is_stale(now)
    assert at(STOCK_MONITOR, 6).is_stale(now)
    assert not at(STOCK_CYCLE, 30).is_stale(now)
    assert not at("unknown.component", 10_000).is_stale(now)  # no rule, never "stale"


@pytest.fixture
def client(database_url: str, tmp_path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from halal_trader.web import app as web_app

    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    with TestClient(web_app.create_app()) as c:
        yield c


def _beat_now(database_url: str, component: str, minutes_ago: float = 0) -> None:
    async def go() -> None:
        eng = create_async_engine(database_url)
        try:
            await beat(eng, component, now=datetime.now(UTC) - timedelta(minutes=minutes_ago))
        finally:
            await eng.dispose()

    asyncio.run(go())


def test_health_reports_no_bot_until_it_beats(client: TestClient, database_url: str) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "running"  # the web itself is up
    assert body["bot_alive"] is False
    assert client.get("/api/health/bot").status_code == 503
    assert client.get("/api/system/status").json()["bot_running"] is False

    _beat_now(database_url, STOCK_PROCESS)

    assert client.get("/api/health").json()["bot_alive"] is True
    assert client.get("/api/health/bot").status_code == 200
    assert client.get("/api/system/status").json()["bot_running"] is True


def test_a_stale_process_beat_is_a_dead_bot(client: TestClient, database_url: str) -> None:
    _beat_now(database_url, STOCK_PROCESS, minutes_ago=10)

    r = client.get("/api/health/bot")

    assert r.status_code == 503
    assert r.json()["bot"][STOCK_PROCESS]["stale"] is True


async def test_monitor_reports_each_completed_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    from halal_trader.trading import monitor as monitor_mod
    from halal_trader.trading.monitor import StockPositionMonitor

    monkeypatch.setattr(monitor_mod, "is_market_open_local", lambda: True)
    repo = MagicMock()
    repo.get_open_trades = AsyncMock(return_value=[])
    ticks: list[dict] = []
    mon: StockPositionMonitor

    async def on_tick(detail: dict) -> None:
        ticks.append(detail)
        mon._running = False  # one tick is enough

    mon = StockPositionMonitor(MagicMock(), repo, check_interval=0, on_tick=on_tick)
    mon._running = True
    await mon._run_loop()

    assert ticks == [{"market_open": True, "open_trades": 0}]
