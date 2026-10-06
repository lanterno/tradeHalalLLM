"""Share counts are put on the basis of the price after a split (screen v11)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.runner import rescale, run_screen, split_ratio
from halal_trader.compliance.sec import Company, Fact

AS_OF = date(2026, 10, 5)


def test_split_ratios_are_recognised_and_anything_else_is_not() -> None:
    # The one-session jumps the stored bars show for 2025-26 splits.
    assert split_ratio(1.9998) == 2.0  # APH, MNST
    assert split_ratio(10.0000) == 10.0  # KLAC
    assert split_ratio(15.0006) == 15.0  # ORLY
    assert split_ratio(0.1000) == pytest.approx(0.1)  # LCID's 1:10
    assert split_ratio(1.5001) == 1.5
    assert split_ratio(1.4933) is None  # Fortive's spin-off of Ralliant, 0.45% off 3:2
    assert split_ratio(2.3909) is None  # DuPont's spin-off of Qnity
    assert split_ratio(1.25) is None  # 5:4 is left out on purpose


def test_a_count_is_rescaled_by_the_splits_after_it_was_known() -> None:
    cover = Fact(1_233_000_000.0, date(2026, 7, 28), "a")
    split = [(date(2026, 9, 3), 2.0)]
    assert rescale(cover, split, known=cover.end, through=AS_OF) == (2_466_000_000.0, None)
    # Already on the new basis: known on or after the split.
    assert rescale(cover, split, known=date(2026, 9, 3), through=AS_OF)[0] == cover.val
    # A split after the price day is not this screen's business.
    assert rescale(cover, split, known=cover.end, through=date(2026, 9, 2))[0] == cover.val


def test_a_filing_issued_after_the_split_is_not_rescaled_again() -> None:
    # KLAC's FY2026 10-K (filed 2026-08-06) restated its 2025 balance-sheet count
    # ten-fold for the May split; the fact's period end is before the split.
    restated = Fact(1_320_227_000.0, date(2025, 6, 30), "10-k")
    split = [(date(2026, 5, 15), 10.0)]
    assert rescale(restated, split, known=date(2026, 8, 6), through=AS_OF)[0] == restated.val


def test_with_no_filing_date_only_reverse_splits_apply() -> None:
    fact = Fact(1_000.0, date(2026, 3, 31), "unknown")
    assert rescale(fact, [(date(2026, 5, 1), 2.0)], known=None, through=AS_OF)[0] == 1_000.0
    assert rescale(fact, [(date(2026, 5, 1), 0.1)], known=None, through=AS_OF)[0] == 100.0


def test_an_adjustment_that_is_no_split_leaves_the_count_unknown() -> None:
    fact = Fact(1_000.0, date(2026, 7, 28), "a")
    value, issue = rescale(fact, [(date(2026, 9, 1), None)], known=fact.end, through=AS_OF)
    assert value is None and issue is not None and "spin-off" in issue


# ── through the stored bars ────────────────────────────────────────────────

COVER = date(2026, 7, 28)


class SplitSec:
    """CIK -> (cover count, diluted count, diluted accession); filing dates by accession."""

    FILED = {"q1-10q": date(2026, 4, 30), "q2-10q": date(2026, 7, 31)}

    def __init__(self, filers: dict[str, tuple[int, float, float, str]]) -> None:
        self.filers = filers

    async def companies(self) -> dict[str, Company]:
        return {s: Company(cik, s, s) for s, (cik, *_) in self.filers.items()}

    async def sic(self, cik: int) -> tuple[int | None, str]:
        return 3678, "Electronic connectors"

    async def foreign_filer(self, cik: int) -> bool:
        return False

    def filed(self, accn: str) -> date | None:
        return self.FILED.get(accn)

    async def frame(self, taxonomy: str, concept: str, unit: str, period: str) -> dict[int, Fact]:
        out: dict[int, Fact] = {}
        for cik, cover, diluted, accn in self.filers.values():
            if concept == "EntityCommonStockSharesOutstanding" and period == "CY2026Q2I":
                out[cik] = Fact(cover, COVER, "q2-10q")
            elif concept == "WeightedAverageNumberOfDilutedSharesOutstanding" and (
                period == "CY2026Q1"
            ):
                out[cik] = Fact(diluted, date(2026, 3, 31), accn)
            elif concept == "LongTermDebt" and period.endswith("I"):
                out[cik] = Fact(30_000.0, date(2026, 6, 30), "q2-10q")
            elif concept == "CashAndCashEquivalentsAtCarryingValue":
                out[cik] = Fact(1_000.0, date(2026, 6, 30), "q2-10q")
            elif concept == "Revenues" and period == "CY2025":
                out[cik] = Fact(50_000.0, date(2025, 12, 31), "k")
        return out


async def _bars(engine: AsyncEngine, symbol: str, rows: list[tuple[date, float, float]]) -> None:
    """(day, raw close, adjusted close) sessions."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                "volume, fetched_at) VALUES (:s, :d, :a, :c, :c, :c, :c, 1, now())"
            ),
            [
                {"s": symbol, "d": d, "a": a, "c": c}
                for d, raw, adj in rows
                for a, c in (("raw", raw), ("all", adj))
            ],
        )


async def test_a_split_after_the_cover_date_doubles_the_count(engine: AsyncEngine) -> None:
    # APH: 2:1 on 2026-09-03, after its 2026-07-28 cover page. Adjusted bars
    # fetched after the split carry it; raw ones halve on the day.
    await _bars(
        engine,
        "APH",
        [
            (date(2026, 9, 2), 174.0, 87.0),
            (date(2026, 9, 3), 87.2, 87.2),
            (date(2026, 10, 2), 87.0, 87.0),
        ],
    )
    sec = SplitSec({"APH": (1, 1_000.0, 1_050.0, "q2-10q")})

    (r,) = await run_screen(sec, engine, ["APH"], AS_OF)  # type: ignore[arg-type]

    # 2,000 shares x 87 = 174,000: debt 30,000 is 17%, not the 34% that failed it.
    assert r.verdict == "halal"
    assert r.metrics["debt_ratio"] == pytest.approx(30_000 / 174_000)


async def test_a_pre_split_diluted_count_no_longer_wins_the_mis_scale_guard(
    engine: AsyncEngine,
) -> None:
    # KLAC: 10:1 in May. Cover page (2026-07-28) post-split; the newest
    # quarterly diluted count is Q1's, filed 2026-04-30, pre-split.
    await _bars(
        engine,
        "KLAC",
        [
            (date(2026, 5, 14), 1_500.0, 150.0),
            (date(2026, 5, 15), 151.0, 151.0),
            (date(2026, 10, 2), 200.0, 200.0),
        ],
    )
    sec = SplitSec({"KLAC": (2, 10_000.0, 1_005.0, "q1-10q")})

    (r,) = await run_screen(sec, engine, ["KLAC"], AS_OF)  # type: ignore[arg-type]

    async with engine.connect() as conn:
        shares = (
            await conn.execute(
                text("SELECT (metrics->>'shares_outstanding')::float FROM halal_screen_results")
            )
        ).scalar_one()
    assert shares == 10_000.0  # v10 took the 1,005 pre-split count: a 10x smaller cap
    assert r.verdict == "halal"


async def test_a_reverse_split_divides_the_count(engine: AsyncEngine) -> None:
    await _bars(
        engine,
        "REV",
        [
            (date(2026, 8, 9), 5.0, 50.0),
            (date(2026, 8, 10), 50.0, 50.0),
            (date(2026, 10, 2), 40.0, 40.0),
        ],
    )
    sec = SplitSec({"REV": (3, 10_000.0, 10_000.0, "q1-10q")})

    (r,) = await run_screen(sec, engine, ["REV"], AS_OF)  # type: ignore[arg-type]

    # 1,000 shares x 40 = 40,000: debt 30,000 is 75%. Unscaled it read 7.5%.
    assert r.verdict == "not_halal"
    assert r.metrics["debt_ratio"] == pytest.approx(0.75)


async def test_an_adjustment_that_is_no_split_makes_the_verdict_doubtful(
    engine: AsyncEngine,
) -> None:
    await _bars(
        engine,
        "SPIN",
        [
            (date(2026, 9, 1), 100.0, 77.0),  # raw / adjusted 1.3: a spin-off
            (date(2026, 9, 2), 77.0, 77.0),
            (date(2026, 10, 2), 80.0, 80.0),
        ],
    )
    sec = SplitSec({"SPIN": (4, 1_000.0, 1_000.0, "q2-10q")})

    (r,) = await run_screen(sec, engine, ["SPIN"], AS_OF)  # type: ignore[arg-type]

    assert r.verdict == "doubtful"
    assert any("no split" in reason for reason in r.reasons)


async def test_adjusted_bars_never_re_adjusted_for_a_split_fall_back_to_the_spot_price(
    engine: AsyncEngine,
) -> None:
    # Adjusted bars fetched before a 2:1 split and only topped up since: they
    # halve with the raw ones, and their 36-month average mixes the two bases.
    rows = [
        (d, 100.0, 100.0)
        for d in (date(2025, 9, 30) + timedelta(days=30 * m) for m in range(11))
        if d < date(2026, 9, 1)
    ]
    rows += [(date(2026, 9, 2), 100.0, 100.0), (date(2026, 9, 3), 50.0, 50.0)]
    rows += [(date(2026, 9, 30), 50.0, 50.0), (date(2026, 10, 2), 50.0, 50.0)]
    await _bars(engine, "STALE", rows)
    sec = SplitSec({"STALE": (5, 2_000.0, 2_000.0, "q2-10q")})

    (r,) = await run_screen(sec, engine, ["STALE"], AS_OF)  # type: ignore[arg-type]

    assert r.metrics["market_cap_basis"] == 0.0  # spot, not the mixed-basis average
    assert r.metrics["market_cap"] == pytest.approx(2_000 * 50.0)
