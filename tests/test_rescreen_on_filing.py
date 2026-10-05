"""A core holding's new 10-Q/10-K brings the weekly screen forward."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.research.daily import _holdings_reported

SCREENED = date(2026, 10, 2)


class FakeSec:
    def __init__(self, filings: dict[int, list[tuple[str, str]]]) -> None:
        self.filings = filings
        self.asked: list[int] = []

    async def submissions(self, cik: int) -> dict[str, Any]:
        self.asked.append(cik)
        rows = self.filings.get(cik, [])
        return {
            "filings": {
                "recent": {
                    "form": [f for f, _ in rows],
                    "filingDate": [d for _, d in rows],
                    "acceptanceDateTime": [f"{d}T16:05:00.000Z" for _, d in rows],
                    "accessionNumber": [f"{cik}-{i}" for i in range(len(rows))],
                }
            }
        }


async def _seed(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, :s, :c, '', 'halal', '[]', "
                "'{}', 'test', now())"
            ),
            [
                {"a": SCREENED, "s": s, "c": c}
                for s, c in (("HELD", 1), ("OLDQ", 2), ("NOTHELD", 3))
            ],
        )
        await conn.execute(
            text(
                "INSERT INTO forward_books (name, strategy, params, created_at) "
                "VALUES ('core', 'core-strict-cap', '{}', now())"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO forward_book_days (book, day, nav, day_return, turnover, weights, "
                "rebalance_next, recorded_at) VALUES ('core', :d, 1.0, 0, 0, CAST(:w AS JSONB), "
                "false, now())"
            ),
            {"d": SCREENED, "w": json.dumps({"HELD": 0.5, "OLDQ": 0.5})},
        )


async def test_only_holdings_reporting_since_the_screen_count(engine: AsyncEngine) -> None:
    await _seed(engine)
    sec = FakeSec(
        {
            1: [("10-Q", "2026-10-04")],  # held, filed after the screen
            2: [("10-Q", "2026-08-01"), ("8-K", "2026-10-04")],  # held, nothing new that matters
            3: [("10-K", "2026-10-04")],  # filed, but not held
        }
    )

    assert await _holdings_reported(engine, sec, SCREENED) == ["HELD"]
    assert sorted(sec.asked) == [1, 2]  # only the holdings are fetched


async def test_nothing_held_asks_sec_nothing(engine: AsyncEngine) -> None:
    sec = FakeSec({})
    assert await _holdings_reported(engine, sec, SCREENED) == []
    assert sec.asked == []
