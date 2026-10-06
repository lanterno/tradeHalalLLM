"""The evening research run: top up bars, re-screen weekly, advance forward books.

Run by the bot after the extended session (so the day's bars are final) and
by `halal-trader books run`. Each step is independent: a failed screen still
lets the books advance on the last screen, and the books never trade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.config import Settings
from halal_trader.data.alpaca_market import AlpacaMarketData
from halal_trader.data.store import BENCHMARKS

logger = logging.getLogger(__name__)

SCREEN_EVERY = timedelta(days=7)
_BACKFILL_FROM = date(2016, 1, 1)


@dataclass
class ResearchRun:
    bars_stored: dict[str, int] = field(default_factory=dict)
    monthly_rows: int | None = None  # None: monthly bars current, or never synced
    screened: int | None = None  # None: the screen was fresh enough to skip
    rescreen_for: list[str] = field(default_factory=list)  # core holdings that just reported
    books: dict[str, int] = field(default_factory=dict)  # name -> sessions appended
    event_labels: int | None = None  # labels written for matured events
    event_refresh: dict[str, int] = field(default_factory=dict)  # step -> rows added
    purification: dict[str, int] = field(default_factory=dict)  # account -> accruals added
    zakat: dict[str, float] = field(default_factory=dict)  # account -> amount due, on a hawl
    core_ready_now: bool = False  # the core passed its live-money gate for the first time
    errors: list[str] = field(default_factory=list)


async def _stored_symbols(engine: AsyncEngine) -> list[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(text("SELECT DISTINCT symbol FROM daily_bars"))
        return sorted(r.symbol for r in rows)


async def _screen_age(engine: AsyncEngine, today: date) -> timedelta | None:
    async with engine.connect() as conn:
        newest = (await conn.execute(text("SELECT max(as_of) FROM halal_screen_results"))).scalar()
    return None if newest is None else today - newest


UNIVERSE = 1500  # screened and kept current: wider than any book's universe


async def _members(engine: AsyncEngine, today: date) -> list[str] | None:
    """Today's point-in-time universe plus benchmarks; None before monthly bars exist."""
    from halal_trader.data.universe import universe_at

    names = await universe_at(engine, today, top_n=UNIVERSE)
    return sorted(set(names) | set(BENCHMARKS)) if names else None


async def _refresh_monthly(
    engine: AsyncEngine, market: AlpacaMarketData, today: date
) -> int | None:
    """Fetch the months since the newest stored one once a month has closed.

    None when monthly bars were never synced (`data pit-universe` does that)
    or are already current.
    """
    from halal_trader.data.universe import sync_monthly_bars

    first = today.replace(day=1)
    last_month = date(first.year - (first.month == 1), (first.month - 2) % 12 + 1, 1)
    async with engine.connect() as conn:
        newest = (await conn.execute(text("SELECT max(month) FROM monthly_bars"))).scalar()
    if newest is None or newest >= last_month:
        return None
    _, rows = await sync_monthly_bars(engine, market, since=newest)
    return rows


async def run_research(engine: AsyncEngine, settings: Settings, *, today: date) -> ResearchRun:
    from halal_trader.compliance.runner import run_screen
    from halal_trader.compliance.sec import SecClient
    from halal_trader.data.store import update_bars
    from halal_trader.research.forward_book import advance_book, book_names

    run = ResearchRun()
    if not await _stored_symbols(engine):
        run.errors.append("no stored bars: run `halal-trader data backfill` first")
        return run

    try:
        market = AlpacaMarketData(settings.alpaca.api_key, settings.alpaca.secret_key)
        try:
            run.monthly_rows = await _refresh_monthly(engine, market, today)
            # The point-in-time universe once monthly bars exist (a new entrant
            # is backfilled from 2016 by update_bars); every stored name before.
            symbols = await _members(engine, today) or await _stored_symbols(engine)
            run.bars_stored = await update_bars(engine, market, symbols, since=_BACKFILL_FROM)
        finally:
            await market.aclose()
    except Exception as exc:  # noqa: BLE001 -- each step reports and the next still runs
        logger.error("research: bar update failed: %r", exc)
        run.errors.append(f"bars: {exc!r}"[:300])

    age = await _screen_age(engine, today)
    reported: list[str] = []
    if age is not None and timedelta(days=1) <= age < SCREEN_EVERY:
        # A core holding's new 10-Q/10-K can change its ratios: screen again
        # now rather than at the end of the week, so a failing holding is
        # sold days sooner.
        sec = SecClient(settings.edgar.user_agent)
        try:
            reported = await _holdings_reported(engine, sec, today - age)
        except Exception as exc:  # noqa: BLE001
            logger.error("research: holdings filings check failed: %r", exc)
            run.errors.append(f"holdings filings: {exc!r}"[:300])
        finally:
            await sec.aclose()
        if reported:
            logger.info("research: re-screening early, new reports from %s", reported)
            run.rescreen_for = reported
    if age is None or age >= SCREEN_EVERY or reported:
        sec = SecClient(settings.edgar.user_agent)
        try:
            from halal_trader.compliance.etf_holdings import sync_holdings

            # The index veto reads the halal ETFs' newest filed holdings.
            await sync_holdings(sec, engine)
            universe = await _members(engine, today) or await _stored_symbols(engine)
            stocks = [s for s in universe if s not in BENCHMARKS]
            run.screened = len(await run_screen(sec, engine, stocks, today))
            run.errors += await _validate_screen(engine, today)
        except Exception as exc:  # noqa: BLE001
            logger.error("research: halal screen failed: %r", exc)
            run.errors.append(f"screen: {exc!r}"[:300])
        finally:
            await sec.aclose()

    try:
        from halal_trader.events.daily import refresh_events

        refreshed = await refresh_events(engine, settings, today=today)
        run.event_refresh = refreshed.counts
        run.errors += refreshed.errors
    except Exception as exc:  # noqa: BLE001
        logger.error("research: event refresh failed: %r", exc)
        run.errors.append(f"event refresh: {exc!r}"[:300])

    try:
        from halal_trader.events.labels import label_events

        run.event_labels = await label_events(engine)
    except Exception as exc:  # noqa: BLE001
        logger.error("research: event labels failed: %r", exc)
        run.errors.append(f"event labels: {exc!r}"[:300])

    for name in await book_names(engine):
        try:
            run.books[name] = len(await advance_book(engine, name, through=today))
        except Exception as exc:  # noqa: BLE001
            logger.error("research: forward book %s failed: %r", name, exc)
            run.errors.append(f"book {name}: {exc!r}"[:300])
    try:
        run.purification = await _purify(engine, settings, today)
    except Exception as exc:  # noqa: BLE001
        logger.error("research: purification failed: %r", exc)
        run.errors.append(f"purification: {exc!r}"[:300])

    try:
        run.errors += await _backup_health(engine)
    except Exception as exc:  # noqa: BLE001
        logger.error("research: backup check failed: %r", exc)

    try:
        run.core_ready_now = await _core_readiness(engine, today)
    except Exception as exc:  # noqa: BLE001
        logger.error("research: core readiness failed: %r", exc)
        run.errors.append(f"core readiness: {exc!r}"[:300])

    try:
        run.errors += await _execution_quality(engine, today)
    except Exception as exc:  # noqa: BLE001
        logger.error("research: execution quality failed: %r", exc)
        run.errors.append(f"execution quality: {exc!r}"[:300])

    try:
        run.zakat = await _zakat(engine, settings, today)
    except Exception as exc:  # noqa: BLE001
        logger.error("research: zakat failed: %r", exc)
        run.errors.append(f"zakat: {exc!r}"[:300])
    return run


async def _zakat(engine: AsyncEngine, settings: Settings, today: date) -> dict[str, float]:
    """On (or after) the hawl, record each account's zakat by both methods, once.

    The run is weekday-only and 1 Ramadan can fall on a weekend, so a hawl
    that has passed without an assessment is recorded on the next run, still
    valued at the hawl day's close. Returns account -> amount due, empty on
    every other day.
    """
    from halal_trader.compliance import zakat as z
    from halal_trader.research.forward_book import book_names

    if not settings.zakat.hawl_hijri:
        return {}
    start, hawl = z.hawl_period(z.parse_hawl(settings.zakat.hawl_hijri), today)
    if (today - hawl).days > 14:  # only the hawl just passed; older ones are history
        return {}
    async with engine.connect() as conn:
        done = {
            r.account
            for r in await conn.execute(
                text("SELECT account FROM zakat_assessments WHERE hawl_date = :h"), {"h": hawl}
            )
        }
    out: dict[str, float] = {}
    for account in ["paper", "core", *(f"book:{b}" for b in await book_names(engine))]:
        if account in done:
            continue
        assessment = await z.assess(engine, account, period_start=start, hawl_date=hawl)
        await z.record(engine, assessment)
        out[account] = assessment.amount
    return out


async def _purify(engine: AsyncEngine, settings: Settings, today: date) -> dict[str, int]:
    """New dividends of anything held in the last 400 days, then accrue every account."""
    from halal_trader.compliance.purification import (
        accrue_account,
        accrue_book,
        accrue_paper,
        held_symbols,
        sync_dividends,
    )
    from halal_trader.research.forward_book import book_names

    since = today - timedelta(days=400)
    market = AlpacaMarketData(settings.alpaca.api_key, settings.alpaca.secret_key)
    try:
        await sync_dividends(
            engine, market, await held_symbols(engine, since), start=since, end=today
        )
    finally:
        await market.aclose()
    out = {"paper": len(await accrue_paper(engine, through=today))}
    out["core"] = len(await accrue_account(engine, "core", through=today))
    for book in await book_names(engine):
        out[f"book:{book}"] = len(await accrue_book(engine, book, through=today))
    return out


async def _core_readiness(engine: AsyncEngine, today: date) -> bool:
    """Check the core's live-money gate; True only the first evening it passes.

    The verdict is kept in the ``core.readiness`` heartbeat, so the alert
    fires on the transition, not every evening after.
    """
    from halal_trader.core.heartbeat import beat, read_beats
    from halal_trader.portfolio.readiness import check

    result = await check(engine, today=today)
    previous = (await read_beats(engine)).get("core.readiness")
    was_ready = bool(previous and (previous.detail or {}).get("ready"))
    await beat(
        engine,
        "core.readiness",
        {
            "ready": result.ready,
            "days": result.days,
            "tracking_error": result.tracking_error,
            "gap": result.gap,
            "failures": result.failures,
        },
    )
    return result.ready and not was_ready


BACKUP_MAX_AGE_H = 36
RESTORE_DRILL_MAX_AGE_D = 40


async def _backup_health(engine: AsyncEngine) -> list[str]:
    """Problems with the nightly backup or the monthly restore drill (empty when fine).

    Both write a heartbeat (trader justfile, home-backup); a missing or stale one
    is reported through the evening run's alert, so a backup that stopped is
    noticed within a day rather than when it is needed.
    """
    from datetime import UTC, datetime

    from halal_trader.core.heartbeat import read_beats

    beats = await read_beats(engine)
    now = datetime.now(UTC)
    problems = []
    nightly = beats.get("backup.nightly")
    if nightly is None or (now - nightly.beat_at).total_seconds() > BACKUP_MAX_AGE_H * 3600:
        problems.append(
            "backup: no nightly dump in the last 36 h"
            if nightly
            else "backup: no nightly dump recorded yet"
        )
    drill = beats.get("backup.restore_drill")
    if drill is not None and (now - drill.beat_at).days > RESTORE_DRILL_MAX_AGE_D:
        problems.append(f"backup: last restore drill {drill.beat_at:%Y-%m-%d}, over 40 days ago")
    return problems


async def _validate_screen(engine: AsyncEngine, today: date) -> list[str]:
    """After a weekly screen: agreement with SPUS/HLAL, kept for the digest; a large
    pass neither ETF holds is reported (under the strict option it should not happen)."""
    from halal_trader.compliance.validate import weekly_check
    from halal_trader.core.heartbeat import beat

    v = await weekly_check(engine, today)
    suspects = [x.symbol for x in v.large_halal_not_in_etfs]
    await beat(
        engine,
        "screen.validation",
        {
            "agreement": round(v.agreement, 3),
            "etf_names": v.screened_etf_names,
            "rejected_etf_names": len(v.etf_held_we_reject),
            "large_passes_no_etf_holds": suspects,
            "missing_data": [x.symbol for x in v.missing_data],
        },
    )
    if suspects:
        return [f"screen: passes large names no halal ETF holds: {', '.join(suspects[:8])}"]
    return []


async def _core_holdings(engine: AsyncEngine) -> set[str]:
    """What the core holds or is about to: the core book's newest weights and the
    core account's orders of the last two months."""
    async with engine.connect() as conn:
        weights = (
            await conn.execute(
                text(
                    "SELECT weights FROM forward_book_days WHERE book = 'core' "
                    "ORDER BY day DESC LIMIT 1"
                )
            )
        ).scalar()
        ordered = {
            r.symbol
            for r in await conn.execute(
                text("SELECT DISTINCT symbol FROM core_orders WHERE submitted_at >= :d"),
                {"d": datetime.now(UTC) - timedelta(days=62)},
            )
        }
    return set(dict(weights or {})) | ordered


async def _holdings_reported(engine: AsyncEngine, sec: Any, screened: date) -> list[str]:
    """Core holdings with a 10-Q or 10-K filed on or after the newest screen's date.

    Their submissions are fetched fresh (about a hundred requests): the
    weekly filings refresh is too slow to act on.
    """
    from halal_trader.events.daily import refresh_filings
    from halal_trader.events.history import covered_companies

    held = await _core_holdings(engine)
    if not held:
        return []
    companies = {c: s for c, s in (await covered_companies(engine)).items() if s in held}
    await refresh_filings(engine, sec, companies)
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT DISTINCT symbol FROM events WHERE kind IN ('10-q', '10-k') "
                "AND symbol = ANY(:s) AND published_at >= :d ORDER BY symbol"
            ),
            {"s": sorted(held), "d": datetime.combine(screened, time(), UTC)},
        )
        return [r.symbol for r in rows]


async def _execution_quality(engine: AsyncEngine, today: date) -> list[str]:
    """The core's fills of the last 30 days against arrival and the close, kept in
    the ``core.execution`` heartbeat for the Core page and the digest. An order
    from today still unfilled after the close is reported: a market order that
    did not fill is a position the account does not hold and its book does."""
    from halal_trader.core.heartbeat import beat
    from halal_trader.portfolio.execution_quality import report

    recent = await report(engine, today - timedelta(days=30), today)
    if not recent.orders:
        return []
    await beat(engine, "core.execution", recent.summary())
    todays = await report(engine, today)
    unfilled = [o.symbol for o in todays.orders if o.status == "unfilled"]
    if unfilled:
        return [
            f"core: {len(unfilled)} order(s) unfilled after the close: {', '.join(unfilled[:8])}"
        ]
    return []
