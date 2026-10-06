"""Screen history is kept per method; every reader takes the newest method's verdict."""

from __future__ import annotations

import json
from datetime import date

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.runner import method_rank

AS_OF = date(2026, 10, 1)


async def _row(
    engine: AsyncEngine, symbol: str, verdict: str, method: str, *, as_of: date = AS_OF
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, :s, 1, 'Software', :v, '[]', "
                "CAST(:m AS JSONB), :method, now())"
            ),
            {
                "a": as_of,
                "s": symbol,
                "v": verdict,
                "method": method,
                "m": json.dumps(
                    {
                        "price": 100.0,
                        "shares_outstanding": 1_000.0,
                        "market_cap": 1e5,
                        "impure_income_ratio": 0.01 if verdict == "halal" else 0.2,
                    }
                ),
            },
        )


def test_method_rank_reads_the_version_number() -> None:
    assert method_rank("aaoifi-sec-v11") == 11
    assert method_rank("aaoifi-sec-v9") == 9
    assert method_rank("test") is None


async def test_both_methods_are_kept_and_the_view_reads_the_newest(engine: AsyncEngine) -> None:
    # v10 sorts before v9 as a string; the view must compare the numbers.
    await _row(engine, "AAA", "halal", "aaoifi-sec-v9")
    await _row(engine, "AAA", "not_halal", "aaoifi-sec-v10")
    await _row(engine, "BBB", "halal", "aaoifi-sec-v10")

    async with engine.connect() as conn:
        stored = (await conn.execute(text("SELECT count(*) FROM halal_screen_results"))).scalar()
        current = {
            r.symbol: (r.verdict, r.method)
            for r in await conn.execute(
                text("SELECT symbol, verdict, method FROM halal_screen_current")
            )
        }
        sql_rank = (
            await conn.execute(text("SELECT halal_screen_method_rank('aaoifi-sec-v11')"))
        ).scalar()
    assert stored == 3
    assert current == {
        "AAA": ("not_halal", "aaoifi-sec-v10"),
        "BBB": ("halal", "aaoifi-sec-v10"),
    }
    assert sql_rank == 11


async def test_a_new_method_adds_rows_and_the_old_verdicts_are_untouched(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from halal_trader.compliance import runner

    from .test_compliance_screen import FakeSec

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                "volume, fetched_at) VALUES ('SOFT', '2026-09-30', 'raw', 100, 100, 100, 100, "
                "1, now())"
            )
        )
    monkeypatch.setattr(runner, "METHOD", "aaoifi-sec-v10")
    await runner.run_screen(FakeSec(), engine, ["SOFT"], AS_OF)  # type: ignore[arg-type]
    async with engine.connect() as conn:
        first = (
            await conn.execute(text("SELECT screened_at, metrics FROM halal_screen_results"))
        ).one()

    monkeypatch.setattr(runner, "METHOD", "aaoifi-sec-v11")
    await runner.run_screen(FakeSec(), engine, ["SOFT"], AS_OF)  # type: ignore[arg-type]

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT method, screened_at, metrics FROM halal_screen_results ORDER BY method"
                )
            )
        ).all()
    assert [r.method for r in rows] == ["aaoifi-sec-v10", "aaoifi-sec-v11"]
    assert (rows[0].screened_at, rows[0].metrics) == (first.screened_at, first.metrics)


async def test_the_core_gate_reads_the_newest_method(engine: AsyncEngine) -> None:
    """A name a newer method fails must not stay buyable on an older pass."""
    from halal_trader.portfolio.core_executor import is_halal_now

    await _row(engine, "AAA", "halal", "aaoifi-sec-v10")
    await _row(engine, "AAA", "not_halal", "aaoifi-sec-v11")
    await _row(engine, "BBB", "not_halal", "aaoifi-sec-v10")
    await _row(engine, "BBB", "halal", "aaoifi-sec-v11")

    assert await is_halal_now(engine, "AAA", date(2026, 10, 2)) == (False, AS_OF)
    assert await is_halal_now(engine, "BBB", date(2026, 10, 2)) == (True, AS_OF)


async def test_books_purification_and_the_home_page_read_the_newest_method(
    engine: AsyncEngine,
) -> None:
    from halal_trader.compliance.purification import impure_ratio
    from halal_trader.research.forward_book import _halal_as_of, _screen_caps

    await _row(engine, "AAA", "halal", "aaoifi-sec-v10")
    await _row(engine, "AAA", "not_halal", "aaoifi-sec-v11")
    await _row(engine, "BBB", "halal", "aaoifi-sec-v11")

    assert await _halal_as_of(engine, date(2026, 10, 2)) == {"BBB"}
    caps = await _screen_caps(engine, date(2026, 10, 2))
    assert caps is not None and set(caps[1]) == {"BBB"}
    ratio, as_of, known = await impure_ratio(engine, "AAA", date(2026, 10, 2))
    assert (ratio, as_of, known) == (pytest.approx(0.2), AS_OF, True)
