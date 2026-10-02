"""A point-in-time liquidity universe, delisted companies included.

The first S1 backtest picked its universe by *today's* dollar volume: the
names that grew into the most traded are, by construction, the winners,
and every one of them was still listed. Here the universe at any date is
ranked from what was known then -- the trailing twelve months of monthly
dollar volume -- over every NYSE/NASDAQ/AMEX stock Alpaca has ever listed,
including the ~2,300 it no longer does.

Monthly bars keep that cheap: ~10k symbols x ~130 months. Daily bars are
then fetched only for names that were ever in the universe.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data.alpaca_market import AlpacaMarketData, Asset, DailyBar

logger = logging.getLogger(__name__)

EXCHANGES = ("NYSE", "NASDAQ", "AMEX")
# Plain tickers only: Alpaca writes share classes as "BRK.B", units with "/",
# and renames a delisted holder of a reused ticker to "XYZ_DELISTED".
_PLAIN = re.compile(r"^[A-Z]{1,5}$")
_MIN_MONTHS = 6


def stock_symbols(assets: Sequence[Asset]) -> list[str]:
    return sorted({a.symbol for a in assets if a.exchange in EXCHANGES and _PLAIN.match(a.symbol)})


async def store_monthly(engine: AsyncEngine, bars: Sequence[DailyBar]) -> int:
    async with engine.begin() as conn:
        for start in range(0, len(bars), 1000):
            chunk = bars[start : start + 1000]
            await conn.execute(
                text(
                    "INSERT INTO monthly_bars (symbol, month, close, volume, vwap) "
                    "VALUES (:s, :m, :c, :v, :vw) ON CONFLICT (symbol, month) DO UPDATE SET "
                    "close = EXCLUDED.close, volume = EXCLUDED.volume, vwap = EXCLUDED.vwap"
                ),
                [
                    {
                        "s": b.symbol,
                        "m": b.day.replace(day=1),
                        "c": b.close,
                        "v": b.volume,
                        "vw": b.vwap,
                    }
                    for b in chunk
                ],
            )
    return len(bars)


async def sync_monthly_bars(
    engine: AsyncEngine, client: AlpacaMarketData, *, since: date
) -> tuple[int, int]:
    """Monthly raw bars for every listed and delisted stock; (symbols, rows)."""
    assets = [*(await client.assets()), *(await client.inactive_assets())]
    symbols = stock_symbols(assets)
    rows = 0
    for start in range(0, len(symbols), 500):
        batch = symbols[start : start + 500]
        bars = await client.daily_bars(batch, start=since, timeframe="1Month")
        rows += await store_monthly(engine, bars)
    logger.info("monthly bars: %d rows for %d symbols", rows, len(symbols))
    return len(symbols), rows


async def universe_at(
    engine: AsyncEngine, as_of: date, *, top_n: int, min_price: float = 5.0
) -> list[str]:
    """The ``top_n`` most traded names by mean monthly dollar volume over the
    twelve months before ``as_of``'s month (known at ``as_of``).

    A name needs six of those months, and a mean close of ``min_price``.
    """
    first_of_month = as_of.replace(day=1)
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                """
                SELECT symbol FROM monthly_bars
                WHERE month < :m AND month >= :since
                GROUP BY symbol
                HAVING count(*) >= :min_months AND avg(close) >= :min_price
                ORDER BY avg(coalesce(vwap, close) * volume) DESC
                LIMIT :n
                """
            ),
            {
                "m": first_of_month,
                "since": date(first_of_month.year - 1, first_of_month.month, 1),
                "min_months": _MIN_MONTHS,
                "min_price": min_price,
                "n": top_n,
            },
        )
        return [r.symbol for r in rows]


def month_starts(start: date, end: date) -> list[date]:
    out, d = [], start.replace(day=1)
    while d <= end:
        out.append(d)
        d = date(d.year + d.month // 12, d.month % 12 + 1, 1)
    return out


async def universe_history(
    engine: AsyncEngine, *, start: date, end: date, top_n: int
) -> dict[date, list[str]]:
    """The universe at the first of every month from ``start`` to ``end``."""
    return {m: await universe_at(engine, m, top_n=top_n) for m in month_starts(start, end)}
