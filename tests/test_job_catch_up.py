"""A restart near a job's time must not silently skip it (plan_catch_up), and
the jobs that report their runs do so."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.heartbeat import (
    RECOMMENDATION,
    RESEARCH,
    STOCK_EOD,
    STOCK_LEDGER,
    Beat,
    read_beats,
)
from halal_trader.trading.scheduler import TradingBot, plan_catch_up

ET = ZoneInfo("America/New_York")


def et(mo: int, d: int, h: int, mi: int = 0) -> datetime:
    return datetime(2026, mo, d, h, mi, tzinfo=ET)


def beats(**at: datetime) -> dict[str, Beat]:
    names = {
        "rec": RECOMMENDATION,
        "eod": STOCK_EOD,
        "ledger": STOCK_LEDGER,
        "research": RESEARCH,
    }
    return {names[k]: Beat(names[k], v, None) for k, v in at.items()}


UP_TO_DATE_MON = beats(
    rec=et(10, 5, 9, 6), eod=et(10, 5, 15, 51), ledger=et(10, 5, 16, 31), research=et(10, 5, 20, 40)
)


def test_nothing_to_catch_up_when_every_job_ran() -> None:
    assert plan_catch_up(et(10, 6, 8), UP_TO_DATE_MON, core_due=False) == []


def test_a_restart_over_the_evening_runs_the_missed_jobs_for_the_day_they_missed() -> None:
    # Monday's ledger sync and research never ran; the bot restarts Tuesday 08:00.
    stale = beats(rec=et(10, 5, 9, 6), eod=et(10, 5, 15, 51))

    plan = plan_catch_up(et(10, 6, 8), stale, core_due=True)

    assert plan == [
        ("sync_broker_ledger", date(2026, 10, 5)),
        ("research_daily", date(2026, 10, 5)),
    ]


def test_a_restart_across_the_end_of_day_flattens_only_while_the_market_is_open() -> None:
    before_eod = beats(rec=et(10, 6, 9, 6), eod=et(10, 5, 15, 51))

    assert ("end_of_day", date(2026, 10, 6)) in plan_catch_up(
        et(10, 6, 15, 55), before_eod, core_due=False
    )
    # After the close a flatten would queue market orders for tomorrow's open.
    assert ("end_of_day", date(2026, 10, 6)) not in plan_catch_up(
        et(10, 6, 16, 5), before_eod, core_due=False
    )


def test_the_core_is_caught_up_only_before_its_cutoff() -> None:
    fresh = beats(rec=et(10, 6, 9, 6))

    assert ("core_trade", date(2026, 10, 6)) in plan_catch_up(
        et(10, 6, 15, 45), fresh, core_due=True
    )
    assert ("core_trade", date(2026, 10, 6)) not in plan_catch_up(
        et(10, 6, 15, 56), fresh, core_due=True
    )
    assert ("core_trade", date(2026, 10, 6)) not in plan_catch_up(
        et(10, 6, 15, 45),
        fresh,
        core_due=False,  # already ran, or not enabled
    )
    assert ("core_trade", date(2026, 10, 6)) not in plan_catch_up(
        et(10, 6, 15, 30),
        fresh,
        core_due=True,  # its time has not come yet
    )


def test_the_recommendation_is_caught_up_during_the_session_only() -> None:
    old = beats(rec=et(10, 5, 9, 6))

    assert ("recommend", date(2026, 10, 6)) in plan_catch_up(et(10, 6, 11), old, core_due=False)
    assert ("recommend", date(2026, 10, 6)) not in plan_catch_up(et(10, 6, 17), old, core_due=False)
    assert ("recommend", date(2026, 10, 6)) not in plan_catch_up(
        et(10, 6, 9),
        old,
        core_due=False,  # before 09:05: the cron will run it
    )


def test_weekends_owe_nothing_and_old_misses_are_let_go() -> None:
    friday_done = beats(ledger=et(10, 2, 16, 31), research=et(10, 2, 20, 40))
    assert plan_catch_up(et(10, 4, 12), friday_done, core_due=False) == []

    long_gone = beats(ledger=et(9, 1, 16, 31), research=et(9, 1, 20, 40))
    plan = plan_catch_up(et(10, 6, 8), long_gone, core_due=False)
    assert plan == [
        ("sync_broker_ledger", date(2026, 10, 5)),
        ("research_daily", date(2026, 10, 5)),
    ]


async def test_catch_up_queues_one_off_jobs(engine: AsyncEngine) -> None:
    bot = TradingBot.__new__(TradingBot)
    bot._engine = engine
    bot.settings = SimpleNamespace(core=SimpleNamespace(enabled=False))  # type: ignore[assignment]
    bot.scheduler = MagicMock()

    plan = await bot._catch_up_missed_jobs(now=et(10, 6, 8).astimezone(UTC))

    assert [job for job, _ in plan] == ["sync_broker_ledger", "research_daily"]
    kwargs = [c.kwargs["kwargs"] for c in bot.scheduler.add_job.call_args_list]
    assert kwargs == [{"day": date(2026, 10, 5)}, {"day": date(2026, 10, 5)}]


# ── the jobs' own reporting ───────────────────────────────────────────


def _bot(engine: AsyncEngine | None) -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    bot._engine = engine
    bot._alerts = MagicMock(notify=AsyncMock())
    bot.scheduler = MagicMock(running=True)
    return bot


@pytest.fixture
def trading_day(monkeypatch: pytest.MonkeyPatch) -> None:
    import halal_trader.trading.scheduler as sched

    monkeypatch.setattr(sched, "now_eastern", lambda: et(10, 6, 9, 5))


async def test_a_failed_recommendation_is_retried_once_before_it_alerts(
    engine: AsyncEngine, trading_day: None
) -> None:
    bot = _bot(engine)
    bot._recommendation = MagicMock(generate=AsyncMock(side_effect=TimeoutError()))

    await bot.recommend()

    bot._alerts.notify.assert_not_awaited()
    retry = bot.scheduler.add_job.call_args
    assert retry.kwargs["kwargs"] == {"is_retry": True}

    await bot.recommend(is_retry=True)

    bot._alerts.notify.assert_awaited_once()
    assert "TimeoutError" in bot._alerts.notify.await_args.args[1]  # not an empty message


async def test_a_recommendation_records_its_heartbeat(
    engine: AsyncEngine, trading_day: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import halal_trader.quant.conformal as conformal
    import halal_trader.recommendation.scorecard as scorecard

    monkeypatch.setattr(scorecard, "backfill_outcomes", AsyncMock(return_value={}))
    monkeypatch.setattr(conformal, "update_band_conformal", AsyncMock(return_value={}))
    bot = _bot(engine)
    bot.broker = MagicMock()
    bot._repo = MagicMock()
    bot._recommendation = MagicMock(
        generate=AsyncMock(return_value={"symbol": "AAPL", "conviction": 0.7})
    )

    await bot.recommend()

    assert (await read_beats(engine))[RECOMMENDATION].detail == {"symbol": "AAPL"}


async def test_end_of_day_beats_and_prunes_the_audit_log(engine: AsyncEngine) -> None:
    bot = _bot(engine)
    bot._news_reactor = None
    bot._self_review = None
    bot._notifier = None
    executor = MagicMock(close_all=AsyncMock(return_value={}))
    portfolio = MagicMock(record_day_end=AsyncMock(return_value={}))
    bot._require_initialized = MagicMock(return_value=(None, executor, portfolio, None))  # type: ignore[method-assign]
    bot._prune_audit_log = AsyncMock()  # type: ignore[method-assign]

    import halal_trader.trading.scheduler as sched

    real_sleep = sched.asyncio.sleep
    sched.asyncio.sleep = AsyncMock()  # type: ignore[assignment]
    try:
        await bot.end_of_day()
    finally:
        sched.asyncio.sleep = real_sleep  # type: ignore[assignment]

    assert STOCK_EOD in await read_beats(engine)
    bot._prune_audit_log.assert_awaited_once()

    # A failed end-of-day leaves no beat (so it is caught up), but still prunes.
    executor.close_all = AsyncMock(side_effect=RuntimeError("broker down"))
    other = _bot(engine)
    other._require_initialized = bot._require_initialized  # type: ignore[method-assign]
    other._prune_audit_log = AsyncMock()  # type: ignore[method-assign]
    await other.end_of_day()
    other._prune_audit_log.assert_awaited_once()
    other._alerts.notify.assert_awaited_once()


async def test_the_day_trader_account_watch_uses_its_paper_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import halal_trader.execution.alpaca_broker as ab

    seen: list[bool] = []

    class FakeBroker:
        def __init__(self, key: str, secret: str, *, paper: bool) -> None:
            seen.append(paper)

        async def get_account_info(self) -> SimpleNamespace:
            return SimpleNamespace(status="ACTIVE")

        async def disconnect(self) -> None:
            return None

    monkeypatch.setattr(ab, "AlpacaRestBroker", FakeBroker)
    bot = _bot(None)
    bot.settings = SimpleNamespace(  # type: ignore[assignment]
        alpaca=SimpleNamespace(api_key="k", secret_key="s", paper_trade=False),
        core=SimpleNamespace(alpaca_api_key="", alpaca_secret_key="", enabled=False),
    )

    assert await bot.account_watch() == []
    assert seen == [False]
