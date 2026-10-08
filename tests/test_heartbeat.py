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
def market_open(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """Pin whether cycles are due (default: no) so liveness doesn't depend on when tests run."""
    import halal_trader.core.heartbeat as hb

    state = {"open": False}
    monkeypatch.setattr(hb, "cycles_due_at", lambda now: state["open"])
    return state


@pytest.fixture
def client(database_url: str, tmp_path, monkeypatch: pytest.MonkeyPatch, market_open):  # type: ignore[no-untyped-def]
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
    assert client.get("/api/operations").json()["fleet"]["alive"] is False

    _beat_now(database_url, STOCK_PROCESS)

    assert client.get("/api/health").json()["bot_alive"] is True
    assert client.get("/api/health/bot").status_code == 200
    assert client.get("/api/operations").json()["fleet"]["alive"] is True


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


# ── the liveness verdict ──────────────────────────────────────────────


NOW = datetime(2026, 10, 1, 15, 0, tzinfo=UTC)


def _beats(**ages_min: float) -> dict[str, Beat]:
    names = {"process": STOCK_PROCESS, "cycle": STOCK_CYCLE}
    return {names[k]: Beat(names[k], NOW - timedelta(minutes=v), None) for k, v in ages_min.items()}


@pytest.mark.parametrize(
    ("beats", "cycles_due", "alive"),
    [
        ({}, False, False),  # never beat
        ({"process": 10}, False, False),  # process dead
        ({"process": 1}, False, True),  # off-hours: process is enough
        ({"process": 1, "cycle": 600}, False, True),  # yesterday's cycle is fine off-hours
        ({"process": 1, "cycle": 600}, True, False),  # market open, no recent cycle: hung
        ({"process": 1}, True, False),  # market open, never cycled
        ({"process": 1, "cycle": 10}, True, True),  # market open, cycling
    ],
)
def test_bot_liveness_verdict(beats: dict, cycles_due: bool, alive: bool) -> None:
    from halal_trader.core.heartbeat import bot_liveness

    verdict, reason = bot_liveness(_beats(**beats), now=NOW, cycles_due=cycles_due)

    assert verdict is alive
    assert (reason is None) is alive  # a dead verdict always says why


def test_a_hung_cycle_during_market_hours_is_a_dead_bot(
    client: TestClient,
    database_url: str,
    market_open: dict[str, bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import halal_trader.config as config

    monkeypatch.setattr(config, "_settings", None)
    market_open["open"] = True
    _beat_now(database_url, STOCK_PROCESS)  # the process is turning...
    _beat_now(database_url, STOCK_CYCLE, minutes_ago=60)  # ...but no cycle for an hour

    r = client.get("/api/health/bot")

    assert r.status_code == 503
    assert "trading cycle" in r.json()["bot"]["_verdict"]["reason"]

    _beat_now(database_url, STOCK_CYCLE)
    assert client.get("/api/health/bot").status_code == 200


# ── the core's 15:40 run and Friday's digest are watched like the other jobs ──


def test_the_core_trade_is_stale_after_a_missed_1540_run() -> None:
    from halal_trader.core.heartbeat import CORE_TRADE, assess
    from halal_trader.market_hours import MARKET_TZ

    # Wed 7 Oct 2026 17:00 ET; the last beat was Tue's run.
    now = datetime(2026, 10, 7, 17, 0, tzinfo=MARKET_TZ)
    beats = {CORE_TRADE: Beat(CORE_TRADE, datetime(2026, 10, 6, 15, 41, tzinfo=MARKET_TZ), None)}
    st = assess(beats, now=now, cycles_due=False)
    assert st[CORE_TRADE].status == "stale"
    beats[CORE_TRADE] = Beat(CORE_TRADE, datetime(2026, 10, 7, 15, 42, tzinfo=MARKET_TZ), None)
    assert assess(beats, now=now, cycles_due=False)[CORE_TRADE].status == "ok"


def test_the_core_trade_reports_disabled_when_the_bot_says_the_core_is_off() -> None:
    from halal_trader.core.heartbeat import CORE_TRADE, assess

    now = datetime(2026, 10, 7, 21, 0, tzinfo=UTC)
    off = {STOCK_PROCESS: Beat(STOCK_PROCESS, now, {"core": False})}
    st = assess(off, now=now, cycles_due=False)
    assert st[CORE_TRADE].status == "disabled" and not st[CORE_TRADE].failing


def test_the_weekly_digest_is_only_owed_on_fridays() -> None:
    from halal_trader.core.heartbeat import DAILY_JOBS, WEEKLY_DIGEST, last_due
    from halal_trader.market_hours import MARKET_TZ

    job = DAILY_JOBS[WEEKLY_DIGEST]
    # Tue 6 Oct: the newest owed run is Fri 2 Oct 17:15 ET.
    due = last_due(job, datetime(2026, 10, 6, 21, 0, tzinfo=MARKET_TZ))
    assert due == datetime(2026, 10, 2, 17, 15, tzinfo=MARKET_TZ)


def test_describe_lists_a_watched_job_that_has_never_run() -> None:
    from halal_trader.core.heartbeat import CORE_TRADE, assess, describe

    now = datetime(2026, 10, 7, 21, 0, tzinfo=UTC)
    st = assess({}, now=now, cycles_due=False)
    out = describe({}, st, now=now)
    assert out[CORE_TRADE]["beat_at"] is None
    assert out[CORE_TRADE]["status"] == "unknown"


def test_the_core_counts_as_on_until_the_bot_says_otherwise() -> None:
    from halal_trader.core.heartbeat import core_on

    now = datetime(2026, 10, 7, 21, 0, tzinfo=UTC)
    assert core_on({}) is True  # no bot yet: a missed core run is not excused
    assert core_on({STOCK_PROCESS: Beat(STOCK_PROCESS, now, None)}) is True
    assert core_on({STOCK_PROCESS: Beat(STOCK_PROCESS, now, {"core": True})}) is True
    assert core_on({STOCK_PROCESS: Beat(STOCK_PROCESS, now, {"core": False})}) is False


def test_the_web_reads_the_cores_state_from_the_bots_beat(
    client: TestClient, database_url: str
) -> None:
    """The web never sees the core's keys (compose blanks them), so the bot's
    process beat is what tells it whether the core runs."""

    async def say(core: bool) -> None:
        engine = create_async_engine(database_url)
        try:
            await beat(engine, STOCK_PROCESS, detail={"core": core})
        finally:
            await engine.dispose()

    def core_trading() -> str:
        rows = client.get("/api/operations").json()["config"]["core"]
        return dict(rows)["Core trading"]

    asyncio.run(say(False))
    assert core_trading().startswith("off")
    asyncio.run(say(True))
    assert core_trading() == "on"
