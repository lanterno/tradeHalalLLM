"""The evening research run: top up bars, re-screen weekly, advance forward books.

Run by the bot after the extended session (so the day's bars are final) and
by `halal-trader books run`. Each step is independent: a failed screen still
lets the books advance on the last screen, and the books never trade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

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
    books: dict[str, int] = field(default_factory=dict)  # name -> sessions appended
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
    if age is None or age >= SCREEN_EVERY:
        sec = SecClient(settings.edgar.user_agent)
        try:
            universe = await _members(engine, today) or await _stored_symbols(engine)
            stocks = [s for s in universe if s not in BENCHMARKS]
            run.screened = len(await run_screen(sec, engine, stocks, today))
        except Exception as exc:  # noqa: BLE001
            logger.error("research: halal screen failed: %r", exc)
            run.errors.append(f"screen: {exc!r}"[:300])
        finally:
            await sec.aclose()

    for name in await book_names(engine):
        try:
            run.books[name] = len(await advance_book(engine, name, through=today))
        except Exception as exc:  # noqa: BLE001
            logger.error("research: forward book %s failed: %r", name, exc)
            run.errors.append(f"book {name}: {exc!r}"[:300])
    return run
