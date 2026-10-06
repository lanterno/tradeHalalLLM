"""A screening history: what the halal screen would have said at each quarter end.

The screen is point in time by construction -- SEC frames are keyed by
period, prices and the 36-month average are read up to ``as_of`` -- so it
can be run for past dates. Each quarter screens that date's liquidity
universe (data/universe.py), which includes companies since delisted.

Known limits, each counted rather than hidden:

* ticker -> CIK comes from SEC's *current* ticker file. A delisted
  company is not in it, so it screens ``doubtful`` (never halal): the
  residual survivorship bias, measured by the backtest as the share of
  each universe that could not be mapped.
* a reused ticker maps to today's company. That company has no XBRL facts
  for periods before it filed, so it also screens ``doubtful`` there.
* frames carry the latest filed value for a period; a later restatement
  can differ from what was first reported.
"""

from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.runner import METHOD, method_rank, run_screen
from halal_trader.compliance.sec import SecClient
from halal_trader.data.store import BENCHMARKS
from halal_trader.data.universe import universe_at

logger = logging.getLogger(__name__)


def quarter_ends(start: date, end: date) -> list[date]:
    out = []
    for year in range(start.year, end.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            d = date(year, month, day)
            if start <= d <= end:
                out.append(d)
    return out


async def _screened(engine: AsyncEngine) -> set[date]:
    async with engine.connect() as conn:
        rows = await conn.execute(text("SELECT DISTINCT as_of FROM halal_screen_results"))
        return {r.as_of for r in rows}


async def screen_history(
    sec: SecClient, engine: AsyncEngine, *, start: date, end: date, top_n: int
) -> dict[date, int]:
    """Screen each quarter end's universe; quarters already stored are skipped."""
    done = await _screened(engine)
    out: dict[date, int] = {}
    for as_of in quarter_ends(start, end):
        if as_of in done:
            continue
        symbols = [s for s in await universe_at(engine, as_of, top_n=top_n) if s not in BENCHMARKS]
        if not symbols:
            logger.warning("screen history: no universe at %s (run `data pit-universe`)", as_of)
            continue
        results = await run_screen(sec, engine, symbols, as_of)
        out[as_of] = sum(1 for r in results if r.verdict == "halal")
        logger.info("screen history %s: %d of %d halal", as_of, out[as_of], len(results))
    return out


async def rescreen_stale(sec: SecClient, engine: AsyncEngine) -> dict[date, int]:
    """Re-screen every stored verdict whose newest method is older than this one.

    Returns as_of -> names re-screened. The older verdicts stay stored
    beside the new ones (the key includes the method). Resumable: a
    re-screened name has a current-method row and is skipped next time; a
    process running an older method than one already stored re-screens nothing.
    """
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT as_of, array_agg(symbol) AS symbols FROM halal_screen_current "
                "WHERE coalesce(halal_screen_method_rank(method), -1) < :rank "
                "GROUP BY as_of ORDER BY as_of"
            ),
            {"rank": method_rank(METHOD)},
        )
        todo = [(r.as_of, list(r.symbols)) for r in rows]
    out: dict[date, int] = {}
    for as_of, symbols in todo:
        results = await run_screen(sec, engine, symbols, as_of)
        out[as_of] = len(symbols)
        halal = sum(1 for r in results if r.verdict == "halal")
        logger.info("rescreen %s: %d re-screened, %d halal", as_of, len(symbols), halal)
    return out
