"""Keeping the event store current, each evening (called by research/daily.py).

The backfills fill history once; this tops it up so the forward record --
the only evidence a strategy cannot have been fitted to -- has no gaps:

* news of the last week (finished days are skipped, so normally one day);
* every covered company's **recent** SEC filings, weekly (older pages
  never change), each new one stamped from its EDGAR header;
* EPS facts again for companies that filed a 10-Q or 10-K in the last week;
* any insider-transaction quarter SEC has published since the last run;
* earnings facts from the new headlines, and LLM scores for new
  company headlines (research budget pool; stops at its cap).

Each step reports its own error; none blocks the others.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

FILINGS_EVERY = timedelta(days=7)
# EDGAR header requests per filings refresh (about 0.4 s each: 3 minutes at
# most): a typical week's new filings, 200 to 500. An earnings week's
# overflow keeps the JSON time until a later run's catch-up.
HEADER_BUDGET = 500
CATCH_UP = timedelta(days=30)


@dataclass
class EventRefresh:
    counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


async def _last_unit(engine: AsyncEngine, task: str) -> date | None:
    async with engine.connect() as conn:
        value = (
            await conn.execute(
                text("SELECT max(done_at) FROM backfill_progress WHERE task = :t"), {"t": task}
            )
        ).scalar()
    return value.date() if value is not None else None


async def _mark(engine: AsyncEngine, task: str, unit: str, items: int) -> None:
    from halal_trader.events.history import _mark as mark

    await mark(engine, task, unit, items)


async def refresh_filings(
    engine: AsyncEngine,
    sec: Any,
    companies: dict[int, str],
    *,
    header_budget: int = HEADER_BUDGET,
    today: date | None = None,
) -> int:
    """Store the covered companies' recent filings that are new to the store.

    Each new filing is stamped at its EDGAR header's acceptance time before
    it is stored (the submissions JSON's is hours late for a third of them)
    and its unit marked in ``filing-times``: one request per filing, at most
    ``header_budget`` per run, each company's structural and earnings 8-Ks
    first. Past the budget, or when the header is missing or EDGAR stops
    answering, a filing keeps the JSON's time, marked ``time_source: json``.
    With ``today``, what is left of the budget corrects the stored filings of
    the last ``CATCH_UP`` that no pass has checked: those, and what another
    path stored with the JSON's time.
    """
    import httpx

    from halal_trader.compliance.sec import SecUnavailable
    from halal_trader.events import history
    from halal_trader.events.store import EventRecord, EventRecorder

    # EDGAR down, or refusing (a 403 when over its rate): the filings are
    # still stored, with the JSON's time.
    edgar_failed = (SecUnavailable, httpx.HTTPError)
    recorder = EventRecorder(engine, raise_errors=True)
    budget = header_budget
    written = 0
    for cik, symbol in sorted(companies.items()):
        subs = await sec.submissions(cik)
        if not subs:
            continue
        fresh = await history.unstored_filings(
            engine, history.filing_records(subs["filings"]["recent"], symbol)
        )
        fresh.sort(key=lambda r: history.filing_priority(r.kind, r.payload.get("items") or []))
        stamped: list[EventRecord] = []
        deltas: dict[str, int] = {}
        for record in fresh:
            accepted = None
            if budget > 0:
                budget -= 1
                try:
                    accepted = await sec.acceptance(cik, record.source_id)
                except edgar_failed as exc:
                    logger.warning("filing headers unavailable, using the JSON times: %r", exc)
                    budget = 0
            if accepted is not None:
                deltas[record.source_id] = history.time_delta(record.published_at, accepted)
            stamped.append(history.header_stamped(record, accepted))
        written += await recorder.record(stamped)
        await history.mark_units(engine, history.TIMES_TASK, deltas)
    if today is not None and budget > 0:
        try:
            await history.correct_filing_times(
                engine,
                sec,
                start=today - CATCH_UP,
                end=today,
                forms=sorted(history.FILING_FORMS),
                limit=budget,
            )
        except edgar_failed as exc:
            logger.warning("filing-time catch-up stopped: %r", exc)
    return written


async def recent_reporters(engine: AsyncEngine, since: date) -> dict[int, str]:
    from halal_trader.events.history import covered_companies

    companies = await covered_companies(engine)
    by_symbol = {s: c for c, s in companies.items()}
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT DISTINCT symbol FROM events WHERE kind IN ('10-q', '10-k') "
                "AND published_at >= :d"
            ),
            {"d": since},
        )
        return {by_symbol[r.symbol]: r.symbol for r in rows if r.symbol in by_symbol}


async def refresh_events(engine: AsyncEngine, settings: Any, *, today: date) -> EventRefresh:
    from halal_trader.compliance.sec import SecClient
    from halal_trader.data.alpaca_market import AlpacaMarketData
    from halal_trader.events import history

    run = EventRefresh()
    sec = SecClient(settings.edgar.user_agent)
    try:
        companies = await history.covered_companies(engine)
        steps: list[tuple[str, Any]] = []

        async def news() -> int:
            market = AlpacaMarketData.from_settings(settings)
            try:
                return await history.backfill_news(
                    engine,
                    market,
                    symbols=await history.covered_symbols(engine),
                    start=today - timedelta(days=7),
                )
            finally:
                await market.aclose()

        async def filings() -> int:
            last = await _last_unit(engine, "filings-refresh")
            if last is not None and today - last < FILINGS_EVERY:
                return 0
            n = await refresh_filings(engine, sec, companies, today=today)
            await _mark(engine, "filings-refresh", today.isoformat(), n)
            return n

        async def eps() -> int:
            reporters = await recent_reporters(engine, today - timedelta(days=7))
            async with engine.begin() as conn:  # let backfill_eps re-read them
                await conn.execute(
                    text("DELETE FROM backfill_progress WHERE task = 'eps' AND unit = ANY(:u)"),
                    {"u": [str(c) for c in reporters]},
                )
            return await history.backfill_eps(engine, sec, reporters)

        async def insiders() -> int:
            return await history.backfill_insiders(
                engine, sec, companies, start=today - timedelta(days=400)
            )

        async def facts() -> int:
            from halal_trader.events.earnings_parse import extract_all

            return await extract_all(engine)

        async def scores() -> int:
            from halal_trader.core.llm import create_classifier_llm, spend
            from halal_trader.events.llm_score import score_all

            # Context-local: the bot's own calls stay on the bot's meter.
            with spend.metered(spend.research_meter(engine, settings)):
                return await score_all(
                    create_classifier_llm(settings),
                    engine,
                    model=settings.llm.model,
                    max_pairs=5000,
                )

        steps = [
            ("news", news),
            ("filings", filings),
            ("eps", eps),
            ("insiders", insiders),
            ("facts", facts),
            ("llm_scores", scores),
        ]
        for name, step in steps:
            try:
                run.counts[name] = await step()
            except Exception as exc:  # noqa: BLE001 -- each step reports, the next still runs
                logger.error("event refresh %s failed: %r", name, exc)
                run.errors.append(f"events {name}: {exc!r}"[:300])
    finally:
        await sec.aclose()
    return run
