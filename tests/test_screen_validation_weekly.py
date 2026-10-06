"""The weekly screen-vs-ETF check: stored for the digest, a large unheld pass reported."""

from __future__ import annotations

import json
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.research.daily import _validate_screen

TODAY = date(2026, 10, 9)


async def _seed(engine: AsyncEngine, verdicts: dict[str, tuple[str, float]]) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, :s, '', :v, '[]', "
                "CAST(:m AS JSONB), 'test', now())"
            ),
            [
                {"a": TODAY, "s": s, "v": v, "m": json.dumps({"market_cap": cap})}
                for s, (v, cap) in verdicts.items()
            ],
        )
        await conn.execute(
            text(
                "INSERT INTO etf_holdings (etf, filed, period_end, ticker, name, cusip, "
                "weight_pct) VALUES ('SPUS', :f, :p, :t, :t, :t, 1.0)"
            ),
            [{"f": date(2026, 9, 1), "p": date(2026, 7, 31), "t": t} for t in ("AAA", "BBB")],
        )


async def _beat(engine: AsyncEngine) -> dict:  # type: ignore[type-arg]
    async with engine.connect() as conn:
        return dict(
            (
                await conn.execute(
                    text("SELECT detail FROM heartbeats WHERE component = 'screen.validation'")
                )
            ).scalar_one()
        )


async def test_agreement_is_stored_and_a_clean_screen_raises_nothing(engine: AsyncEngine) -> None:
    await _seed(engine, {"AAA": ("halal", 2e11), "BBB": ("not_halal", 2e11), "SML": ("halal", 1e9)})

    assert await _validate_screen(engine, TODAY) == []
    beat = await _beat(engine)
    assert beat["agreement"] == 0.5
    assert beat["large_passes_no_etf_holds"] == []
    assert beat["missing_data"] == []


async def test_a_large_pass_no_etf_holds_is_reported(engine: AsyncEngine) -> None:
    await _seed(engine, {"AAA": ("halal", 2e11), "BIG": ("halal", 9e11)})

    errors = await _validate_screen(engine, TODAY)
    assert errors and "BIG" in errors[0]
    assert (await _beat(engine))["large_passes_no_etf_holds"] == ["BIG"]


async def test_a_pass_whose_interest_expense_implies_heavy_debt_is_reported(
    engine: AsyncEngine,
) -> None:
    await _seed(engine, {"AAA": ("halal", 2e11), "BBB": ("not_halal", 2e11)})
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, 'SM', '', 'halal', '[]', "
                "CAST(:m AS JSONB), 'test', now())"
            ),
            {"a": TODAY, "m": json.dumps({"market_cap": 7.7e9, "implied_debt_ratio": 0.37})},
        )

    errors = await _validate_screen(engine, TODAY)

    assert any("interest expense" in e and "SM" in e for e in errors)
    beat = await _beat(engine)
    assert beat["implied_debt_suspects"] == ["SM"]
    assert beat["rejected_by"] == {"ratio": 1}  # BBB, held by the ETF and rejected
