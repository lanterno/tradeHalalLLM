"""Persist research market data, and choose what to persist.

* :func:`sync_assets` -- Alpaca's active asset list into ``market_assets``.
* :func:`liquid_universe` -- the most liquid listed common stocks by median
  daily dollar volume, plus benchmark ETFs. Backfilling all ~8.5k listed
  names would be mostly illiquid noise; a strategy can only trade what it
  can fill.
* :func:`store_bars` / :func:`update_bars` -- daily bars, idempotent upserts,
  incremental from each symbol's last stored session.

Known limitation, stated so nobody forgets it: the universe is built from
names listed *today*. Backtests over it are survivorship-biased (delisted
names are missing). Fixing that needs a point-in-time constituent history,
a later Phase 3 item.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data.alpaca_market import Adjustment, AlpacaMarketData, DailyBar

logger = logging.getLogger(__name__)

# Always stored, whatever the liquidity screen says: what results are
# measured against (S&P 500, Nasdaq-100, and the halal ETFs SPUS / HLAL).
BENCHMARKS: tuple[str, ...] = ("SPY", "QQQ", "SPUS", "HLAL")
_STOCK_EXCHANGES = ("NYSE", "NASDAQ")
_CHUNK = 2000


@dataclass(frozen=True, slots=True)
class UniverseMember:
    symbol: str
    median_dollar_volume: float


async def sync_assets(engine: AsyncEngine, client: AlpacaMarketData) -> int:
    assets = await client.assets()
    async with engine.begin() as conn:
        for a in assets:
            await conn.execute(
                text(
                    """
                    INSERT INTO market_assets (symbol, name, exchange, tradable, fractionable,
                        status, synced_at)
                    VALUES (:s, :n, :e, :t, :f, :st, now())
                    ON CONFLICT (symbol) DO UPDATE SET name = EXCLUDED.name,
                        exchange = EXCLUDED.exchange, tradable = EXCLUDED.tradable,
                        fractionable = EXCLUDED.fractionable, status = EXCLUDED.status,
                        synced_at = EXCLUDED.synced_at
                    """
                ),
                {
                    "s": a.symbol,
                    "n": a.name,
                    "e": a.exchange,
                    "t": a.tradable,
                    "f": a.fractionable,
                    "st": a.status,
                },
            )
    return len(assets)


async def liquid_universe(
    engine: AsyncEngine,
    client: AlpacaMarketData,
    *,
    top_n: int,
    as_of: date,
    lookback_days: int = 90,
    min_price: float = 5.0,
) -> list[UniverseMember]:
    """Top ``top_n`` tradable NYSE/NASDAQ names by median daily dollar volume.

    Penny stocks (median close below ``min_price``) are excluded. Symbols with
    a '.' or '/' (share classes and units in Alpaca's notation) are skipped.
    """
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT symbol FROM market_assets WHERE tradable AND status = 'active' "
                "AND exchange = ANY(:ex)"
            ),
            {"ex": list(_STOCK_EXCHANGES)},
        )
        candidates = [r.symbol for r in rows if "." not in r.symbol and "/" not in r.symbol]
    bars = await client.daily_bars(
        candidates, start=as_of - timedelta(days=lookback_days), end=as_of
    )
    by_symbol: dict[str, list[DailyBar]] = defaultdict(list)
    for b in bars:
        by_symbol[b.symbol].append(b)
    ranked = []
    for sym, rows_ in by_symbol.items():
        if len(rows_) < 20:  # too new or too thinly traded to judge
            continue
        if statistics.median(b.close for b in rows_) < min_price:
            continue
        ranked.append(UniverseMember(sym, statistics.median(b.close * b.volume for b in rows_)))
    ranked.sort(key=lambda m: m.median_dollar_volume, reverse=True)
    logger.info("liquidity screen: %d candidates, %d ranked", len(candidates), len(ranked))
    return ranked[:top_n]


async def stored_symbols(engine: AsyncEngine, adjustment: Adjustment | None = None) -> list[str]:
    """Every symbol with stored daily bars (of ``adjustment`` when given), sorted."""
    sql = "SELECT DISTINCT symbol FROM daily_bars"
    params: dict[str, str] = {}
    if adjustment is not None:
        sql += " WHERE adjustment = :adj"
        params["adj"] = adjustment
    async with engine.connect() as conn:
        return sorted(r.symbol for r in await conn.execute(text(sql), params))


async def last_closes(
    engine: AsyncEngine, symbols: Sequence[str], *, on_or_before: date | None = None
) -> dict[str, tuple[date, float]]:
    """Each symbol's newest stored as-traded (raw) close, and its day; a symbol
    without one is absent."""
    if not symbols:
        return {}
    sql = "SELECT DISTINCT ON (symbol) symbol, day, close FROM daily_bars "
    sql += "WHERE adjustment = 'raw' AND symbol = ANY(:s) AND close > 0 "
    params: dict[str, object] = {"s": list(symbols)}
    if on_or_before is not None:
        sql += "AND day <= :d "
        params["d"] = on_or_before
    async with engine.connect() as conn:
        rows = await conn.execute(text(sql + "ORDER BY symbol, day DESC"), params)
        return {r.symbol: (r.day, float(r.close)) for r in rows}


async def store_bars(engine: AsyncEngine, bars: Sequence[DailyBar], adjustment: Adjustment) -> int:
    """Upsert bars (a re-fetched adjusted bar replaces the older vintage)."""
    sql = text(
        """
        INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume,
            vwap, trades, fetched_at)
        VALUES (:symbol, :day, :adj, :open, :high, :low, :close, :volume, :vwap, :trades, now())
        ON CONFLICT (symbol, day, adjustment) DO UPDATE SET open = EXCLUDED.open,
            high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close,
            volume = EXCLUDED.volume, vwap = EXCLUDED.vwap, trades = EXCLUDED.trades,
            fetched_at = EXCLUDED.fetched_at
        """
    )
    for i in range(0, len(bars), _CHUNK):
        chunk = bars[i : i + _CHUNK]
        async with engine.begin() as conn:
            await conn.execute(
                sql,
                [
                    {
                        "symbol": b.symbol,
                        "day": b.day,
                        "adj": adjustment,
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                        "vwap": b.vwap,
                        "trades": b.trades,
                    }
                    for b in chunk
                ],
            )
    return len(bars)


async def update_bars(
    engine: AsyncEngine,
    client: AlpacaMarketData,
    symbols: Sequence[str],
    *,
    since: date,
    adjustments: Sequence[Adjustment] = ("raw", "all"),
) -> dict[str, int]:
    """Bring every symbol up to date from its last stored session (or ``since``).

    Grouped by start date so a fresh backfill and a daily top-up both batch
    100 symbols per request. Adjusted bars are re-fetched for the last 5
    sessions on each top-up, so a dividend's retroactive adjustment of the
    most recent bars is picked up.
    """
    stored: dict[str, int] = {}
    for adjustment in adjustments:
        async with engine.connect() as conn:
            rows = await conn.execute(
                text(
                    "SELECT symbol, max(day) AS last FROM daily_bars "
                    "WHERE adjustment = :a AND symbol = ANY(:s) GROUP BY symbol"
                ),
                {"a": adjustment, "s": list(symbols)},
            )
            last = {r.symbol: r.last for r in rows}
        groups: dict[date, list[str]] = defaultdict(list)
        overlap = timedelta(days=7) if adjustment == "all" else timedelta(days=1)
        for sym in symbols:
            groups[last[sym] - overlap if sym in last else since].append(sym)
        total = 0
        for start, syms in sorted(groups.items()):
            bars = await client.daily_bars(syms, start=start, adjustment=adjustment)
            total += await store_bars(engine, bars, adjustment)
        stored[adjustment] = total
        logger.info(
            "daily bars (%s): %d rows stored for %d symbols", adjustment, total, len(symbols)
        )
    return stored
