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

from halal_trader.compliance.sec import filed_at
from halal_trader.config import Settings
from halal_trader.data.alpaca_market import AlpacaMarketData
from halal_trader.data.store import BENCHMARKS, stored_symbols
from halal_trader.halal import strict
from halal_trader.portfolio.core_account import CORE_PAPER, DAY_TRADER, core_account

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


async def _screen_age(engine: AsyncEngine, today: date) -> timedelta | None:
    newest = await strict.newest_screen(engine)
    return None if newest is None else today - newest


async def _screened_at(engine: AsyncEngine) -> datetime | None:
    """When the newest screen ran: a filing accepted after it is not in it."""
    async with engine.connect() as conn:
        value: datetime | None = (
            await conn.execute(
                text(
                    "SELECT max(screened_at) FROM halal_screen_results "
                    "WHERE as_of = (SELECT max(as_of) FROM halal_screen_results)"
                )
            )
        ).scalar()
    return value


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
    if not await stored_symbols(engine):
        run.errors.append("no stored bars: run `halal-trader data backfill` first")
        return run

    try:
        market = AlpacaMarketData.from_settings(settings)
        try:
            run.monthly_rows = await _refresh_monthly(engine, market, today)
            # The point-in-time universe once monthly bars exist (a new entrant
            # is backfilled from 2016 by update_bars); every stored name before.
            symbols = await _members(engine, today) or await stored_symbols(engine)
            run.bars_stored = await update_bars(engine, market, symbols, since=_BACKFILL_FROM)
        finally:
            await market.aclose()
    except Exception as exc:  # noqa: BLE001 -- each step reports and the next still runs
        logger.error("research: bar update failed: %r", exc)
        run.errors.append(f"bars: {exc!r}"[:300])

    age = await _screen_age(engine, today)
    reported: list[str] = []
    if age is not None and timedelta(days=1) <= age < SCREEN_EVERY:
        # A core holding's new report (or an amended one) can change its
        # ratios: screen again now rather than at the end of the week, so a
        # failing holding is sold days sooner.
        sec = SecClient(settings.edgar.user_agent)
        try:
            since = await _screened_at(engine) or datetime.combine(today - age, time(), UTC)
            reported = await _holdings_reported(engine, sec, since)
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
            universe = await _members(engine, today) or await stored_symbols(engine)
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
        from halal_trader.core.llm import credits

        run.errors += await credits.check(engine, settings, today)
    except Exception as exc:  # noqa: BLE001
        logger.error("research: LLM credit check failed: %r", exc)

    try:
        run.errors += await _calendar_check(settings, today)
    except Exception as exc:  # noqa: BLE001
        logger.error("research: calendar check failed: %r", exc)

    try:
        run.core_ready_now = await _core_readiness(engine, today)
    except Exception as exc:  # noqa: BLE001
        logger.error("research: core readiness failed: %r", exc)
        run.errors.append(f"core readiness: {exc!r}"[:300])

    try:
        run.errors += await _execution_quality(engine, today, core_account(settings.core.paper))
    except Exception as exc:  # noqa: BLE001
        logger.error("research: execution quality failed: %r", exc)
        run.errors.append(f"execution quality: {exc!r}"[:300])

    try:
        run.zakat = await _zakat(engine, settings, today)
    except Exception as exc:  # noqa: BLE001
        logger.error("research: zakat failed: %r", exc)
        run.errors.append(f"zakat: {exc!r}"[:300])
    return run


def _core_ledgers(settings: Settings) -> list[str]:
    """The core's accounts to purify and assess: paper always (its history
    stays), live as well once the keys are live's (portfolio/core_account.py)."""
    return list(dict.fromkeys([CORE_PAPER, core_account(settings.core.paper)]))


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
    for account in [
        DAY_TRADER,
        *_core_ledgers(settings),
        *(f"book:{b}" for b in await book_names(engine)),
    ]:
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
    market = AlpacaMarketData.from_settings(settings)
    try:
        await sync_dividends(
            engine, market, await held_symbols(engine, since), start=since, end=today
        )
    finally:
        await market.aclose()
    out = {DAY_TRADER: len(await accrue_paper(engine, through=today))}
    for account in _core_ledgers(settings):
        out[account] = len(await accrue_account(engine, account, through=today))
    for book in await book_names(engine):
        out[f"book:{book}"] = len(await accrue_book(engine, book, through=today))
    return out


async def _core_readiness(engine: AsyncEngine, today: date) -> bool:
    """Check the core's live-money gate; True only the first evening it passes.

    The verdict is kept in the ``core.readiness`` heartbeat, so the alert
    fires on the transition, not every evening after.
    """
    from halal_trader.core.heartbeat import CORE_READINESS, beat, read_beats
    from halal_trader.portfolio.readiness import check

    result = await check(engine, today=today)
    previous = (await read_beats(engine)).get(CORE_READINESS)
    was_ready = bool(previous and (previous.detail or {}).get("ready"))
    await beat(
        engine,
        CORE_READINESS,
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

    Each writes a heartbeat: the dump and the drill in `just backup`, the
    off-site copy in infra/server/backup.sh once restic has stored it. A
    missing or stale one is reported through the evening run's alert, so a
    backup that stopped is noticed within a day rather than when it is needed.
    A dump that never leaves the server counts as a problem: the last machine
    took its backups with it.
    """
    from datetime import UTC, datetime

    from halal_trader.core.heartbeat import (
        BACKUP_NIGHTLY,
        BACKUP_OFFSITE,
        BACKUP_RESTORE_DRILL,
        read_beats,
    )

    beats = await read_beats(engine)
    now = datetime.now(UTC)
    problems = []
    for component, what in (
        (BACKUP_NIGHTLY, "nightly dump"),
        (BACKUP_OFFSITE, "off-site copy"),
    ):
        last = beats.get(component)
        if last is None:
            problems.append(f"backup: no {what} recorded yet")
        elif (now - last.beat_at).total_seconds() > BACKUP_MAX_AGE_H * 3600:
            problems.append(f"backup: no {what} in the last 36 h")
    drill = beats.get(BACKUP_RESTORE_DRILL)
    if drill is not None and (now - drill.beat_at).days > RESTORE_DRILL_MAX_AGE_D:
        problems.append(f"backup: last restore drill {drill.beat_at:%Y-%m-%d}, over 40 days ago")
    return problems


CALENDAR_HORIZON = timedelta(days=90)


async def _calendar_check(settings: Settings, today: date) -> list[str]:
    """Days in the next 90 where market_hours' static calendar disagrees with
    the broker's (empty when they agree). Every schedule, gate and catch-up
    trusts market_hours; a missing holiday or early close there would run
    jobs on a closed market or miss a session, and the table ends in 2027."""
    from halal_trader.execution.alpaca_broker import AlpacaRestBroker
    from halal_trader.market_hours import calendar_mismatches

    a = settings.alpaca
    if not a.api_key:
        return []
    end = today + CALENDAR_HORIZON
    broker = AlpacaRestBroker(a.api_key, a.secret_key, paper=a.paper_trade)
    try:
        days = await broker.get_calendar(today.isoformat(), end.isoformat())
    finally:
        await broker.disconnect()
    found = calendar_mismatches(days, today, end)
    if found:
        logger.error("market_hours disagrees with the broker's calendar: %s", "; ".join(found))
    return [f"calendar: {m} (fix market_hours.py)" for m in found[:5]]


async def _validate_screen(engine: AsyncEngine, today: date) -> list[str]:
    """After a weekly screen: agreement with SPUS/HLAL, kept for the digest. Reported:
    a large pass neither ETF holds (under the strict option it should not happen),
    and any pass whose interest expense implies debt over the limit."""
    from halal_trader.compliance.validate import weekly_check
    from halal_trader.core.heartbeat import beat

    v = await weekly_check(engine, today)
    suspects = [x.symbol for x in v.large_halal_not_in_etfs]
    implied = [x.symbol for x in v.implied_debt_suspects]
    await beat(
        engine,
        "screen.validation",
        {
            "agreement": round(v.agreement, 3),
            "etf_names": v.screened_etf_names,
            "rejected_etf_names": len(v.etf_held_we_reject),
            "rejected_by": {kind: len(items) for kind, items in v.rejected_by_kind.items()},
            "large_passes_no_etf_holds": suspects,
            "implied_debt_suspects": implied,
            "missing_data": [x.symbol for x in v.missing_data],
        },
    )
    errors = []
    if suspects:
        errors.append(f"screen: passes large names no halal ETF holds: {', '.join(suspects[:8])}")
    if implied:
        errors.append(
            "screen: passes whose interest expense implies debt over 30% of market cap: "
            f"{', '.join(implied[:8])}"
        )
    return errors


async def _core_holdings(engine: AsyncEngine) -> set[str]:
    """What the core holds or is about to: the core book's newest weights and the
    core account's orders of the last two months."""
    from halal_trader.research.forward_book import latest_weights

    weights = await latest_weights(engine, "core")
    async with engine.connect() as conn:
        ordered = {
            r.symbol
            for r in await conn.execute(
                text("SELECT DISTINCT symbol FROM core_orders WHERE submitted_at >= :d"),
                {"d": datetime.now(UTC) - timedelta(days=62)},
            )
        }
    return set(weights) | ordered


# Filings that can change a screen's inputs: annual and quarterly reports,
# their amendments and transition reports, and a foreign issuer's (whose
# interim reports come on 6-K).
SCREEN_FORMS = frozenset(
    {
        "10-K", "10-K/A", "10-KT", "10-KT/A",
        "10-Q", "10-Q/A", "10-QT", "10-QT/A",
        "20-F", "20-F/A", "40-F", "40-F/A", "6-K", "6-K/A",
    }
)  # fmt: skip


def _accepted(recent: dict[str, Any], i: int) -> datetime | None:
    """When filing ``i`` of a submissions page was accepted (UTC); its date at 17:00 ET
    if the time is missing, as the filings history records it."""
    stamps = recent.get("acceptanceDateTime") or []
    dates = recent.get("filingDate") or []
    filed = str(dates[i] or "") if i < len(dates) else ""
    if not filed:
        return None
    return filed_at(date.fromisoformat(filed), str(stamps[i] or "") if i < len(stamps) else "")


async def _holdings_reported(engine: AsyncEngine, sec: Any, screened: datetime) -> list[str]:
    """Core holdings with a report (``SCREEN_FORMS``) accepted after the newest screen ran.

    ``screened`` is that screen's own timestamp, not its date: a report
    accepted that afternoon, before the evening's screen, is already in it
    and must not bring the next one forward. Submissions are fetched fresh
    (about a hundred requests; the weekly filings refresh is too slow to act
    on), and their 10-Q/10-K/8-K rows are recorded as events on the way.
    """
    from halal_trader.events.history import covered_companies, filing_records
    from halal_trader.events.store import EventRecorder

    held = await _core_holdings(engine)
    if not held:
        return []
    companies = {c: s for c, s in (await covered_companies(engine)).items() if s in held}
    recorder = EventRecorder(engine, raise_errors=True)
    out: set[str] = set()
    for cik, symbol in sorted(companies.items()):
        subs = await sec.submissions(cik)
        if not subs:
            continue
        recent = subs["filings"]["recent"]
        await recorder.record(filing_records(recent, symbol))
        for i, form in enumerate(recent.get("form") or []):
            accepted = _accepted(recent, i) if form in SCREEN_FORMS else None
            if accepted is not None and accepted > screened:
                out.add(symbol)
                break
    return sorted(out)


async def _execution_quality(
    engine: AsyncEngine, today: date, account: str = CORE_PAPER
) -> list[str]:
    """The core's fills of the last 30 days against arrival and the close, kept in
    the ``core.execution`` heartbeat for the Core page and the digest. An order
    from today still unfilled after the close is reported: a market order that
    did not fill is a position the account does not hold and its book does."""
    from halal_trader.core.heartbeat import beat
    from halal_trader.portfolio.execution_quality import report

    recent = await report(engine, today - timedelta(days=30), today, account=account)
    if not recent.orders:
        return []
    await beat(engine, "core.execution", recent.summary())
    todays = await report(engine, today, account=account)
    unfilled = [o.symbol for o in todays.orders if o.status == "unfilled"]
    if unfilled:
        return [
            f"core: {len(unfilled)} order(s) unfilled after the close: {', '.join(unfilled[:8])}"
        ]
    return []
