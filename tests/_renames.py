"""Test helpers for renamed tickers: a fake news client and stored rows."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data.alpaca_market import Asset, NewsArticle
from halal_trader.events.store import EventRecord, EventRecorder
from halal_trader.market_hours import MARKET_TZ


def article(n: int, symbols: tuple[str, ...], when: datetime, headline: str = "") -> NewsArticle:
    return NewsArticle(
        n, headline or f"headline {n}", "s", f"https://x/{n}", "benzinga", symbols, when
    )


def ny(day: date, hour: int = 12) -> datetime:
    return datetime(day.year, day.month, day.day, hour, tzinfo=MARKET_TZ)


class NewsMarket:
    """Answers every news request with one article tagging the asked ticker and
    AAPL; records each request."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], datetime, datetime, int]] = []

    async def news(
        self, symbols: list[str], *, start: datetime, end: datetime, max_pages: int
    ) -> list[NewsArticle]:
        self.calls.append((list(symbols), start, end, max_pages))
        return [article(len(self.calls), (symbols[0], "AAPL"), start + timedelta(hours=12))]

    async def inactive_assets(self) -> list[Asset]:
        return []

    async def aclose(self) -> None:
        return None


async def screen(engine: AsyncEngine, rows: list[tuple[date, str, str]]) -> None:
    """Halal-screen rows: (as_of, symbol, verdict)."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) "
                "VALUES (:a, :s, 1, '', :v, '[]', '{}', 't', now())"
            ),
            [{"a": a, "s": s, "v": v} for a, s, v in rows],
        )


async def news(engine: AsyncEngine, rows: list[tuple[int, str, datetime]]) -> None:
    """News rows (id, symbol, published) tagging only their own symbol."""
    await EventRecorder(engine, raise_errors=True).record(
        [
            EventRecord("alpaca", f"alpaca:{n}", "news", symbol, at, at, {"symbols": [symbol]})
            for n, symbol, at in rows
        ]
    )


async def seed_candidates_data(engine: AsyncEngine) -> None:
    """Screens and news that make LATE, META and SILENT seeding candidates."""
    q = date(2016, 9, 30)
    await screen(
        engine,
        [
            (q, "LATE", "halal"),
            (q, "EARLY", "halal"),
            (q, "EDGE", "halal"),
            (q, "SILENT", "halal"),
            (date(2026, 9, 30), "FRESH", "halal"),
            (q, "NEVER", "not_halal"),
            (q, "META", "not_halal"),
            (date(2017, 3, 31), "META", "halal"),
        ],
    )
    await news(
        engine,
        [
            (1, "LATE", ny(date(2018, 1, 2))),
            (2, "LATE", ny(date(2019, 1, 2))),
            (3, "EARLY", ny(date(2016, 10, 3))),
            (4, "EDGE", ny(q + timedelta(days=365), 23)),  # 23:00 New York = next day UTC
            (5, "NEVER", ny(date(2020, 1, 2))),
            (6, "META", ny(date(2021, 6, 30))),
        ],
    )
