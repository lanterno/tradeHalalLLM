"""Annual fundamentals from SEC XBRL frames: the quality factor's input.

Quality is gross profitability (Novy-Marx): gross profit over total
assets. Gross profit is ``GrossProfit`` where a filer reports it, else
revenue minus cost of revenue; assets are the year-end instant.

Point in time by a publication lag, not by filing dates (frames do not
carry them): a calendar year's figures are used only from May 1 of the
next year, four months after a December year end, by which every 10-K is
due. Fiscal years ending earlier in the year are therefore used later
than they could be, never earlier.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.sec import Fact, SecClient

logger = logging.getLogger(__name__)

_GROSS_PROFIT = "GrossProfit"
_REVENUE = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
_COST = ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold")
_PUBLISHED_FROM = (5, 1)  # month, day of the following year


def usable_year(as_of: date) -> int:
    """The newest calendar year whose annual figures are certainly published by ``as_of``."""
    return as_of.year - 1 if (as_of.month, as_of.day) >= _PUBLISHED_FROM else as_of.year - 2


def _first(frames: list[dict[int, Fact]], cik: int) -> float | None:
    for frame in frames:
        if cik in frame:
            return frame[cik].val
    return None


async def sync_annual(sec: SecClient, engine: AsyncEngine, years: Iterable[int]) -> int:
    """Store gross profit and assets for every filer in each calendar year; returns rows."""
    stored = 0
    for year in years:
        period = f"CY{year}"
        gp = await sec.frame("us-gaap", _GROSS_PROFIT, "USD", period)
        revenue = [await sec.frame("us-gaap", c, "USD", period) for c in _REVENUE]
        cost = [await sec.frame("us-gaap", c, "USD", period) for c in _COST]
        assets = await sec.frame("us-gaap", "Assets", "USD", f"{period}Q4I")
        rows = []
        for cik in set(gp) | set(assets) | set().union(*revenue):
            gross = gp[cik].val if cik in gp else None
            if gross is None:
                rev, cst = _first(revenue, cik), _first(cost, cik)
                gross = rev - cst if rev is not None and cst is not None else None
            total = assets[cik].val if cik in assets else None
            if gross is not None or total is not None:
                rows.append({"c": cik, "y": year, "g": gross, "a": total})
        async with engine.begin() as conn:
            for start in range(0, len(rows), 1000):
                await conn.execute(
                    text(
                        "INSERT INTO annual_fundamentals (cik, year, gross_profit, assets) "
                        "VALUES (:c, :y, :g, :a) ON CONFLICT (cik, year) DO UPDATE SET "
                        "gross_profit = EXCLUDED.gross_profit, assets = EXCLUDED.assets"
                    ),
                    rows[start : start + 1000],
                )
        logger.info("fundamentals %d: %d filers", year, len(rows))
        stored += len(rows)
    return stored


async def quality_by_year(engine: AsyncEngine) -> dict[int, dict[int, float]]:
    """year -> CIK -> gross profit / assets (positive assets only)."""
    out: dict[int, dict[int, float]] = {}
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT cik, year, gross_profit / assets AS q FROM annual_fundamentals "
                "WHERE gross_profit IS NOT NULL AND assets > 0"
            )
        )
        for r in rows:
            out.setdefault(r.year, {})[r.cik] = float(r.q)
    return out
