"""Test helpers for the story builder: stored events, aliases, a week of news, a fake context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events import earnings_parse, stories
from halal_trader.events.aliases import BUILDER_VERSION
from halal_trader.events.earnings_parse import parse_headline
from halal_trader.events.history import mark_units, write_times
from halal_trader.events.store import EventRecord, EventRecorder
from halal_trader.market_hours import MARKET_TZ
from tests._renames import mark_renamed_news_done

MON, TUE, WED, THU, FRI = (date(2024, 5, d) for d in (6, 7, 8, 9, 10))
DOWNGRADE = "Morgan Stanley Downgrades Apple to Equal-Weight"
MISS = "Apple Q2 Adj. EPS $1.40 Misses $1.50 Estimate, Sales $90.00B Miss $91.00B Estimate"


def ny(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), MARKET_TZ)


def headline_row(
    n: int, symbol: str, at: datetime, headline: str, tagged: list[str] | None = None
) -> EventRecord:
    """A news row; ``tagged`` None is a live row (no ``payload.symbols``)."""
    payload: dict[str, Any] = {"headline": headline}
    if tagged is not None:
        payload["symbols"] = tagged
    return EventRecord("alpaca", f"alpaca:{n}", "news", symbol, at, at, payload)


def news_row(n: int, symbol: str, at: datetime, headline: str) -> EventRecord:
    return headline_row(n, symbol, at, headline, [symbol])


def filing_row(
    acc: str, symbol: str, at: datetime, items: list[str], kind: str = "8-k"
) -> EventRecord:
    return EventRecord("sec", acc, kind, symbol, at, at, {"form": kind.upper(), "items": items})


async def store(
    engine: AsyncEngine, rows: list[EventRecord], *, facts: bool = True
) -> dict[str, int]:
    """Store the rows and, as ``extract_all`` would, the current extractor's facts
    of each news headline (a ``none`` row for one without); source_id -> event id."""
    await EventRecorder(engine, raise_errors=True).record(rows)
    async with engine.begin() as conn:
        ids = {
            r.source_id: int(r.id)
            for r in await conn.execute(text("SELECT source_id, id FROM events"))
        }
        if facts:
            x = earnings_parse.EXTRACTOR
            for r in rows:
                if r.kind != "news":
                    continue
                e = ids[r.source_id]
                parsed = parse_headline(r.payload.get("headline") or "")
                values = [
                    {"e": e, "x": x, "k": f.kind, "f": json.dumps(f.fields)} for f in parsed
                ] or [{"e": e, "x": x, "k": "none", "f": "{}"}]
                await conn.execute(
                    text(
                        "INSERT INTO event_facts (event_id, extractor, kind, fields) "
                        "VALUES (:e, :x, :k, CAST(:f AS JSONB))"
                    ),
                    values,
                )
    return ids


async def time_filings(engine: AsyncEngine) -> None:
    """Every stored 8-K and 8-K/A read from its EDGAR header, which says the time
    it is stored at (as ``events filings fix-times`` leaves a filing that was right)."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT source_id, min(published_at) AS at FROM events "
                "WHERE kind IN ('8-k', '8-k/a') GROUP BY source_id"
            )
        )
        fixes = {str(r.source_id): (r.at, 0) for r in rows}
    await write_times(engine, fixes)


async def mark_built(engine: AsyncEngine, start: date, end: date, items: int = 0) -> None:
    """Record [start, end] as completely built from the current inputs (stories
    persisted outside build_range)."""
    unit = stories.build_unit(start, end, await stories.inputs_sha(engine))
    await mark_units(engine, stories.TASK, {unit: items})


async def add_aliases(engine: AsyncEngine, rows: list[tuple[str, str, str]]) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO story_aliases (builder_version, symbol, alias, source) "
                "VALUES (:v, :s, :a, :src)"
            ),
            [{"v": BUILDER_VERSION, "s": s, "a": a, "src": src} for s, a, src in rows],
        )


async def stored_rows(engine: AsyncEngine) -> list[dict[str, Any]]:
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT * FROM news_stories ORDER BY story_id"))
        return [dict(r._mapping) for r in result]


WEEK = [
    # AAPL: Monday's probe is structural, so Tuesday to Thursday follow it (rule a);
    # Friday's three sessions back hold only followers, so it has no parent.
    news_row(1, "AAPL", ny(MON, 8), "Apple Faces SEC Probe Into App Store Disclosures"),
    news_row(2, "AAPL", ny(TUE, 8), "Jefferies Downgrades Apple to Hold"),
    news_row(3, "AAPL", ny(WED, 8), "Apple Unveils New Mac"),
    news_row(4, "AAPL", ny(THU, 8), DOWNGRADE),
    news_row(
        5, "AAPL", ny(THU, 9), "Goldman Sachs Maintains Neutral on Apple, Lowers Price Target"
    ),
    news_row(6, "AAPL", ny(FRI, 7), MISS),
    filing_row("acc-1", "AAPL", ny(THU, 18), ["2.02", "9.01"]),  # public Fri 06:00
    news_row(7, "MSFT", ny(TUE, 9), "Microsoft Unveils New Surface"),
    news_row(8, "MSFT", ny(WED, 9), "Samsung Unveils New Phone"),  # fails the entity check
    news_row(9, "MSFT", ny(THU, 10), "Barclays Downgrades Microsoft to Equal-Weight"),
    filing_row("acc-2", "MSFT", ny(WED, 8), [], kind="10-q"),
]
ALIAS_ROWS = [
    ("AAPL", "Apple", "name"),
    ("AAPL", "AAPL", "ticker"),
    ("MSFT", "Microsoft", "name"),
    ("MSFT", "MSFT", "ticker"),
]


async def seed_week(engine: AsyncEngine) -> None:
    """The week's events, aliases stored, every renamed-news month fetched (needs
    small_map), the 8-K read from its header."""
    await store(engine, WEEK)
    await add_aliases(engine, ALIAS_ROWS)
    await mark_renamed_news_done(engine)
    await time_filings(engine)


@dataclass
class Verdict:
    eligible: bool
    tech: bool
    reason: str


class FakeContext:
    """PRIMARY: every symbol but MSFT; Technology: AAPL. Records each question."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, date, datetime]] = []

    def eligibility(self, symbol: str, session: date, *, at_news: datetime) -> Verdict:
        self.asked.append((symbol, session, at_news))
        if symbol == "MSFT":
            return Verdict(False, True, "rank")
        return Verdict(True, symbol == "AAPL", "ok")


def fake_context(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[list[str], date, date]], list[FakeContext]]:
    """Every context the counts load is a FakeContext; returns (loads, contexts)."""
    loads: list[tuple[list[str], date, date]] = []
    made: list[FakeContext] = []

    async def load(engine: Any, symbols: Any, lo: date, hi: date) -> FakeContext:
        loads.append((sorted(symbols), lo, hi))
        made.append(FakeContext())
        return made[-1]

    monkeypatch.setattr(stories, "_context", load)
    return loads, made
