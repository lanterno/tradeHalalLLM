"""Every scheduled job runs one way (TradingBot._job) and on one schedule (DAILY_JOBS)."""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.heartbeat import (
    CORE_TRADE,
    DAILY_JOBS,
    RECOMMENDATION,
    STOCK_EOD,
    STOCK_LEDGER,
    WEEKLY_DIGEST,
    read_beats,
)
from halal_trader.core.observability import job_context, job_id_var
from halal_trader.trading import scheduler as sched
from halal_trader.trading.scheduler import Skipped, TradingBot, plan_catch_up

ET = ZoneInfo("America/New_York")
EARLY = date(2026, 11, 27)  # the day after Thanksgiving: a 13:00 close
NORMAL = date(2026, 11, 25)


def _bot(engine: AsyncEngine | None = None) -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    bot._engine = engine
    bot._alerts = MagicMock(notify=AsyncMock())
    bot._notifier = MagicMock(send=AsyncMock(return_value=True), notify_daily_summary=AsyncMock())
    bot.scheduler = MagicMock(running=True)
    return bot


async def test_a_finished_job_beats_with_its_detail(engine: AsyncEngine) -> None:
    bot = _bot(engine)
    assert await bot._job(
        "x", AsyncMock(return_value={"n": 3}), alert="x.failed", beat_as=STOCK_LEDGER
    )
    assert (await read_beats(engine))[STOCK_LEDGER].detail == {"n": 3}
    bot._alerts.notify.assert_not_awaited()


async def test_a_failed_job_alerts_and_leaves_no_beat(engine: AsyncEngine) -> None:
    bot = _bot(engine)
    body = AsyncMock(side_effect=RuntimeError("broker down"))
    assert not await bot._job("x", body, alert="x.failed", beat_as=STOCK_LEDGER)
    assert STOCK_LEDGER not in await read_beats(engine)
    key, details = bot._alerts.notify.await_args.args
    assert key == "x.failed" and "broker down" in details


async def test_a_skipped_job_leaves_no_beat(engine: AsyncEngine) -> None:
    bot = _bot(engine)
    assert await bot._job(
        "x", AsyncMock(return_value=Skipped("holiday")), alert="x", beat_as=STOCK_LEDGER
    )
    assert STOCK_LEDGER not in await read_beats(engine)


@pytest.mark.parametrize(
    ("day", "fired", "runs"),
    [
        (NORMAL, time(15, 50), True),
        (NORMAL, time(12, 50), False),
        (EARLY, time(12, 50), True),
        (EARLY, time(15, 50), False),  # it ran at 12:50: never twice
    ],
)
def test_end_of_day_runs_once_a_day_at_that_days_time(
    monkeypatch: pytest.MonkeyPatch, day: date, fired: time, runs: bool
) -> None:
    monkeypatch.setattr(sched, "today_eastern", lambda: day)
    assert (sched._off_schedule(STOCK_EOD, fired) is None) is runs
    assert sched._off_schedule(STOCK_EOD, None) is None  # a catch-up always runs


def test_a_core_run_missed_on_an_early_close_is_caught_up() -> None:
    now = datetime.combine(EARLY, time(12, 45), ET)
    assert ("core_trade", EARLY) in plan_catch_up(now, {}, core_due=True)


def test_every_daily_job_is_scheduled_from_daily_jobs() -> None:
    bot = TradingBot.__new__(TradingBot)
    bot.scheduler = AsyncIOScheduler()
    for component, job_id in (
        (RECOMMENDATION, "daily_recommendation"),
        (STOCK_EOD, "end_of_day"),
        (CORE_TRADE, "core_trade"),
        (WEEKLY_DIGEST, "weekly_digest"),
    ):
        bot._add_daily(AsyncMock(), component, job_id, grace_s=60)
    fields = {
        job.id: ({f.name: str(f) for f in job.trigger.fields}, job.kwargs)
        for job in bot.scheduler.get_jobs()
    }

    def at(job_id: str) -> tuple[str, str, str]:
        f = fields[job_id][0]
        return f["day_of_week"], f["hour"], f["minute"]

    assert at("daily_recommendation") == ("mon-fri", "9", "5")
    assert fields["daily_recommendation"][1] == {}
    assert at("end_of_day") == ("mon-fri", "15", "50")
    assert at("end_of_day_early_close") == ("mon-fri", "12", "50")
    assert fields["end_of_day_early_close"][1] == {
        "scheduled": DAILY_JOBS[STOCK_EOD].early_close_at
    }
    assert at("core_trade_early_close") == ("mon-fri", "12", "40")
    assert at("weekly_digest") == ("fri", "17", "15")


async def test_a_digest_telegram_refused_is_not_done(engine: AsyncEngine, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import halal_trader.notifications.digest as digest

    monkeypatch.setattr(digest, "build", AsyncMock(return_value="week"))
    bot = _bot(engine)
    bot.settings = MagicMock()
    bot._notifier.send = AsyncMock(return_value=False)

    await bot.weekly_digest()

    assert WEEKLY_DIGEST not in await read_beats(engine)
    assert bot._alerts.notify.await_args.args[0] == "digest.failed"


def test_a_jobs_log_records_carry_its_job_id(caplog: pytest.LogCaptureFixture) -> None:
    from halal_trader.core.observability import ObservabilityFilter

    logger = logging.getLogger("halal_trader.test.jobs")
    logger.addFilter(ObservabilityFilter())
    with (
        caplog.at_level(logging.INFO, logger="halal_trader.test.jobs"),
        job_context("end_of_day") as jid,
    ):
        logger.info("inside")
    assert jid.startswith("end_of_day-") and caplog.records[-1].job_id == jid
    assert job_id_var.get() == ""
