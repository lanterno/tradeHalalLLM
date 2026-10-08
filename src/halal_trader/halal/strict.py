"""The strict in-house screen's verdicts: how every consumer reads them.

``halal_screen_results`` holds every run of the in-house screen
(compliance/runner.py): the stricter of AAOIFI and S&P Shariah on every
axis, an index board's exclusion a veto (operator decision 2026-10-02).
Both strategies trade on it (the day-trader through halal/cache.py, the
core through portfolio/core_executor.py), and the web, the forward books
and the research tools read it too. They all read it here.

Rules, all failing CLOSED:

* only the **newest** screen on or before the day counts, and only while it
  is **fresh** (``MAX_SCREEN_AGE``): a stale or missing screen makes nothing
  halal;
* a symbol absent from that screen is not halal;
* the verdict is the newest method's (``halal_screen_current``): one row per
  symbol and day.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

# The screen runs weekly, so 10 days allows one missed run.
MAX_SCREEN_AGE = timedelta(days=10)


@dataclass(frozen=True, slots=True)
class Verdict:
    halal: bool
    reason: str
    screen_as_of: date | None


@dataclass(frozen=True, slots=True)
class ScreenRow:
    """One symbol's verdict on one screen, with the inputs consumers use."""

    symbol: str
    verdict: str  # halal | doubtful | not_halal
    cik: int | None
    sic_description: str | None
    price: float | None
    shares: float | None
    market_cap: float | None


async def newest_screen(engine: AsyncEngine, *, on_or_before: date | None = None) -> date | None:
    """The newest screen's date (on or before ``on_or_before`` when given)."""
    sql = "SELECT max(as_of) FROM halal_screen_results"
    params: dict[str, date] = {}
    if on_or_before is not None:
        sql += " WHERE as_of <= :d"
        params["d"] = on_or_before
    async with engine.connect() as conn:
        value = (await conn.execute(text(sql), params)).scalar()
    return value if isinstance(value, date) else None


async def screen_rows(
    engine: AsyncEngine,
    as_of: date,
    *,
    halal_only: bool = False,
    symbols: Sequence[str] | None = None,
) -> list[ScreenRow]:
    """The rows of the screen of ``as_of``: every one, the halal ones, or those
    of ``symbols``."""
    where = ["as_of = :a"]
    params: dict[str, object] = {"a": as_of}
    if halal_only:
        where.append("verdict = 'halal'")
    if symbols is not None:
        where.append("symbol = ANY(:s)")
        params["s"] = list(symbols)
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT symbol, verdict, cik, sic_description, "
                "(metrics->>'price')::float AS price, "
                "(metrics->>'shares_outstanding')::float AS shares, "
                "(metrics->>'market_cap')::float AS market_cap "
                f"FROM halal_screen_current WHERE {' AND '.join(where)} ORDER BY symbol"
            ),
            params,
        )
        return [
            ScreenRow(
                r.symbol, r.verdict, r.cik, r.sic_description, r.price, r.shares, r.market_cap
            )
            for r in rows
        ]


async def verdict(engine: AsyncEngine, symbol: str, *, today: date) -> Verdict:
    """The order-boundary check: does the newest fresh screen hold ``symbol`` halal?"""
    as_of = await newest_screen(engine, on_or_before=today)
    if as_of is None:
        return Verdict(False, "no strict screen on record", None)
    if today - as_of > MAX_SCREEN_AGE:
        return Verdict(False, f"the strict screen of {as_of} is stale", as_of)
    rows = await screen_rows(engine, as_of, symbols=[symbol])
    if not rows:
        return Verdict(False, f"{symbol} is not in the strict screen of {as_of}", as_of)
    if rows[0].verdict != "halal":
        return Verdict(False, f"the strict screen of {as_of} does not pass {symbol}", as_of)
    return Verdict(True, f"halal in the strict screen of {as_of}", as_of)


async def halal_universe(
    engine: AsyncEngine, *, today: date, limit: int
) -> tuple[date | None, list[str]]:
    """The ``limit`` largest names (by market cap) the newest fresh screen passes.

    Market cap is the liquidity proxy: the largest names trade the most.
    Returns ``(screen date, symbols)``; no symbols when the screen is stale
    or missing.
    """
    as_of = await newest_screen(engine, on_or_before=today)
    if as_of is None or today - as_of > MAX_SCREEN_AGE:
        return as_of, []
    rows = await screen_rows(engine, as_of, halal_only=True)
    rows.sort(key=lambda r: (r.market_cap is None, -(r.market_cap or 0.0), r.symbol))
    return as_of, [r.symbol for r in rows[:limit]]
