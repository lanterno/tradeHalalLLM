"""The fleet watchdog, the calendar-aware heartbeat verdicts it uses, and the healthcheck."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.heartbeat import (
    ALERT_MARKS,
    MARKET_SNAPSHOT,
    RESEARCH,
    SHADOW_PROCESS,
    STOCK_CYCLE,
    STOCK_EOD,
    STOCK_LEDGER,
    STOCK_MONITOR,
    STOCK_PROCESS,
    WATCHDOG,
    Beat,
    assess,
    beat,
    claim_once,
    describe,
    read_beats,
)
from halal_trader.web import watchdog

ET = ZoneInfo("America/New_York")


def et(y: int, mo: int, d: int, h: int, mi: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=ET)


def _fresh_processes(now: datetime) -> dict[str, Beat]:
    return {
        c: Beat(c, now - timedelta(seconds=30), None)
        for c in (STOCK_PROCESS, SHADOW_PROCESS, STOCK_MONITOR)
    }


# ── assess ────────────────────────────────────────────────────────────


def test_a_retired_day_traders_cycle_is_disabled_never_stale() -> None:
    now = et(2026, 10, 6, 11)  # a Tuesday, mid-session
    beats = _fresh_processes(now) | {STOCK_CYCLE: Beat(STOCK_CYCLE, now - timedelta(days=4), None)}

    retired = assess(beats, now=now, cycles_due=False, day_trader_enabled=False)
    running = assess(beats, now=now, cycles_due=True, day_trader_enabled=True)

    assert retired[STOCK_CYCLE].status == "disabled"
    assert not retired[STOCK_CYCLE].failing
    assert running[STOCK_CYCLE].status == "stale"


def test_research_is_stale_only_once_a_trading_days_run_is_missed() -> None:
    last_run = et(2026, 10, 2, 20, 40)  # Friday evening's run
    beats = {RESEARCH: Beat(RESEARCH, last_run, None)}

    def status(now: datetime) -> str:
        return assess(beats, now=now, cycles_due=False, day_trader_enabled=False)[RESEARCH].status

    assert status(et(2026, 10, 5, 12)) == "ok"  # the weekend owed nothing
    assert status(et(2026, 10, 5, 22)) == "ok"  # Monday's run is still within its grace
    assert status(et(2026, 10, 6, 0)) == "stale"  # Monday 20:30 + 3 h passed, no beat


def test_a_daily_job_that_never_ran_is_unknown_not_an_alarm() -> None:
    st = assess({}, now=et(2026, 10, 6, 23), cycles_due=False, day_trader_enabled=False)

    assert st[STOCK_LEDGER].status == "unknown"
    assert not st[STOCK_LEDGER].failing
    assert st[STOCK_PROCESS].status == "missing"  # a continuous loop with no row is


def test_the_end_of_day_job_is_judged_at_its_early_close_time() -> None:
    # 2026-11-27, the day after Thanksgiving, closes at 13:00; EOD runs at 12:50.
    beats = {STOCK_EOD: Beat(STOCK_EOD, et(2026, 11, 25, 15, 51), None)}  # Wednesday's

    st = assess(beats, now=et(2026, 11, 27, 13, 30), cycles_due=False, day_trader_enabled=False)

    assert st[STOCK_EOD].status == "stale"
    assert "12:50" in (st[STOCK_EOD].reason or "")


def test_the_market_snapshot_is_judged_during_the_session_only() -> None:
    now_in, now_out = et(2026, 10, 6, 11), et(2026, 10, 6, 19)
    old = {MARKET_SNAPSHOT: Beat(MARKET_SNAPSHOT, now_in - timedelta(minutes=20), None)}

    assert assess(old, now=now_in, cycles_due=False, day_trader_enabled=False)[
        MARKET_SNAPSHOT
    ].failing
    assert not assess(old, now=now_out, cycles_due=False, day_trader_enabled=False)[
        MARKET_SNAPSHOT
    ].failing


def test_describe_hides_the_alert_ledger_and_says_why() -> None:
    now = et(2026, 10, 6, 11)
    beats = {
        STOCK_CYCLE: Beat(STOCK_CYCLE, now - timedelta(days=4), None),
        ALERT_MARKS: Beat(ALERT_MARKS, now, {"x": "y"}),
    }
    st = assess(beats, now=now, cycles_due=False, day_trader_enabled=False)

    out = describe(beats, st, now=now)

    assert ALERT_MARKS not in out
    assert out[STOCK_CYCLE]["status"] == "disabled"
    assert out[STOCK_CYCLE]["stale"] is False
    assert "retired" in out[STOCK_CYCLE]["reason"]


# ── claim_once ────────────────────────────────────────────────────────


async def test_an_alert_is_claimed_once_across_processes(engine: AsyncEngine) -> None:
    results = await asyncio.gather(*(claim_once(engine, "llm.budget:2026-10-06") for _ in range(5)))

    assert sorted(results) == [False, False, False, False, True]
    assert await claim_once(engine, "llm.budget:2026-10-07") is True


async def test_old_alert_claims_are_pruned(engine: AsyncEngine) -> None:
    from sqlalchemy import text

    await claim_once(engine, "old")
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE heartbeats SET detail = jsonb_build_object('old', "
                "to_jsonb(now() - interval '90 days')) WHERE component = :c"
            ),
            {"c": ALERT_MARKS},
        )
    await claim_once(engine, "new")

    marks = (await read_beats(engine))[ALERT_MARKS].detail
    assert set(marks or {}) == {"new"}


# ── the watchdog ──────────────────────────────────────────────────────


class FakeSender:
    def __init__(self, ok: bool = True) -> None:
        self.messages: list[str] = []
        self.ok = ok

    @property
    def enabled(self) -> bool:
        return True

    async def send(self, message: str, *, parse_mode: str = "HTML") -> bool:
        self.messages.append(message)
        return self.ok


async def _beat_all(engine: AsyncEngine, now: datetime, *, skip: str = "") -> None:
    for c in (STOCK_PROCESS, SHADOW_PROCESS, STOCK_MONITOR, MARKET_SNAPSHOT):
        if c != skip:
            await beat(engine, c, now=now)


async def test_watchdog_alerts_after_two_failing_passes_then_recovers(engine: AsyncEngine) -> None:
    sender = FakeSender()
    t0 = et(2026, 10, 6, 11).astimezone(UTC)
    await _beat_all(engine, t0)
    await beat(engine, SHADOW_PROCESS, now=t0 - timedelta(minutes=10))  # the shadow died

    first = await watchdog.check_once(engine, sender, day_trader_enabled=False, now=t0)
    assert first["alerted"] == []  # one failing pass could be a deploy

    t1 = t0 + timedelta(minutes=5)
    await _beat_all(engine, t1, skip=SHADOW_PROCESS)
    second = await watchdog.check_once(engine, sender, day_trader_enabled=False, now=t1)
    assert second["alerted"] == [SHADOW_PROCESS]
    assert "shadow engine" in sender.messages[-1]

    t2 = t1 + timedelta(minutes=5)
    await _beat_all(engine, t2, skip=SHADOW_PROCESS)
    third = await watchdog.check_once(engine, sender, day_trader_enabled=False, now=t2)
    assert third["alerted"] == []  # already alerted: not repeated
    assert len(sender.messages) == 1

    t3 = t2 + timedelta(minutes=5)
    await _beat_all(engine, t3)
    fourth = await watchdog.check_once(engine, sender, day_trader_enabled=False, now=t3)
    assert fourth["recovered"] == [SHADOW_PROCESS]
    assert "is back" in sender.messages[-1]


async def test_watchdog_state_survives_a_web_restart(engine: AsyncEngine) -> None:
    t0 = et(2026, 10, 6, 11).astimezone(UTC)
    first_web = FakeSender()
    for i in range(2):
        now = t0 + timedelta(minutes=5 * i)
        await _beat_all(engine, now, skip=STOCK_PROCESS)
        await watchdog.check_once(engine, first_web, day_trader_enabled=False, now=now)
    assert len(first_web.messages) == 1

    second_web = FakeSender()  # a new process: nothing in memory
    now = t0 + timedelta(minutes=10)
    await _beat_all(engine, now, skip=STOCK_PROCESS)
    await watchdog.check_once(engine, second_web, day_trader_enabled=False, now=now)

    assert second_web.messages == []
    assert (await read_beats(engine))[WATCHDOG].detail == {
        "alerting": [STOCK_PROCESS],
        "suspect": [STOCK_PROCESS],
    }


async def test_an_alert_that_failed_to_send_is_retried(engine: AsyncEngine) -> None:
    t0 = et(2026, 10, 6, 11).astimezone(UTC)
    down = FakeSender(ok=False)
    for i in range(2):
        now = t0 + timedelta(minutes=5 * i)
        await _beat_all(engine, now, skip=STOCK_PROCESS)
        await watchdog.check_once(engine, down, day_trader_enabled=False, now=now)

    up = FakeSender()
    now = t0 + timedelta(minutes=10)
    await _beat_all(engine, now, skip=STOCK_PROCESS)
    result = await watchdog.check_once(engine, up, day_trader_enabled=False, now=now)

    assert result["alerted"] == [STOCK_PROCESS]


# ── the container healthcheck ─────────────────────────────────────────


async def test_healthcheck_exit_codes(
    engine: AsyncEngine, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from halal_trader.core import healthcheck

    monkeypatch.setenv("DATABASE_URL", database_url)
    run = asyncio.to_thread

    assert await run(healthcheck.main, [STOCK_PROCESS, "180"]) == 1  # never beat
    await beat(engine, STOCK_PROCESS)
    assert await run(healthcheck.main, [STOCK_PROCESS, "180"]) == 0
    await beat(engine, STOCK_PROCESS, now=datetime.now(UTC) - timedelta(minutes=10))
    assert await run(healthcheck.main, [STOCK_PROCESS, "180"]) == 1

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://nobody:x@127.0.0.1:1/none")
    assert await run(healthcheck.main, [STOCK_PROCESS, "180"]) == 2
