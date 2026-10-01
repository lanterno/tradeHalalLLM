"""In-house AAOIFI screen: the rules, and a run end to end with a fake SEC."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.aaoifi import Fundamentals, prohibited_activity, screen
from halal_trader.compliance.runner import recent_quarter_instants, run_screen
from halal_trader.compliance.sec import Company, Fact


def _f(**kw: object) -> Fundamentals:
    base: dict[str, object] = dict(
        symbol="CO",
        sic=3571,  # electronic computers
        shares_outstanding=1_000.0,
        price=100.0,  # market cap 100,000
        interest_bearing_debt=10_000.0,  # 10%
        cash_and_securities=20_000.0,  # 20%
        interest_income=100.0,
        revenue=50_000.0,  # 0.2%
    )
    base.update(kw)
    return Fundamentals(**base)  # type: ignore[arg-type]


def test_a_clean_operating_company_passes() -> None:
    r = screen(_f())
    assert r.verdict == "halal" and r.reasons == []
    assert r.metrics["debt_ratio"] == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("sic", "reason"),
    [
        (6021, "banking"),
        (6311, "insurance"),
        (2085, "alcohol"),
        (2111, "tobacco"),
        (7011, "casinos"),
    ],
)
def test_prohibited_activities_fail_whatever_the_ratios(sic: int, reason: str) -> None:
    r = screen(_f(sic=sic))
    assert r.verdict == "not_halal" and reason in r.reasons[0]


def test_real_estate_is_not_excluded_by_activity() -> None:
    assert prohibited_activity(6512) is None  # operators of buildings


@pytest.mark.parametrize(
    ("kw", "fragment"),
    [
        ({"interest_bearing_debt": 30_000.0}, "interest-bearing debt"),  # exactly at 30%: fails
        ({"cash_and_securities": 45_000.0}, "cash and interest-bearing"),
        ({"interest_income": 2_600.0}, "interest income"),  # 5.2% of revenue
    ],
)
def test_each_ratio_limit_bites(kw: dict, fragment: str) -> None:
    r = screen(_f(**kw))
    assert r.verdict == "not_halal" and fragment in r.reasons[0]


@pytest.mark.parametrize(
    "kw", [{"price": None}, {"shares_outstanding": None}, {"revenue": None}, {"sic": None}]
)
def test_anything_not_computable_is_doubtful_not_halal(kw: dict) -> None:
    assert screen(_f(**kw)).verdict == "doubtful"  # fail closed


def test_unreported_interest_income_counts_as_zero_when_revenue_exists() -> None:
    assert screen(_f(interest_income=None)).verdict == "halal"


def test_recent_quarter_instants_are_completed_quarters_newest_first() -> None:
    assert recent_quarter_instants(date(2026, 10, 1), n=3) == [
        "CY2026Q3I",
        "CY2026Q2I",
        "CY2026Q1I",
    ]
    assert recent_quarter_instants(date(2026, 2, 15), n=2) == ["CY2025Q4I", "CY2025Q3I"]


class FakeSec:
    """CIK 1 = a clean company; CIK 2 = a bank; CIK 3 = tags no debt at all."""

    async def companies(self) -> dict[str, Company]:
        return {s: Company(c, s, s) for s, c in (("SOFT", 1), ("BANK", 2), ("NODEBT", 3))}

    async def sic(self, cik: int) -> tuple[int | None, str]:
        return {
            1: (7372, "Prepackaged software"),
            2: (6021, "National banks"),
            3: (3571, "Computers"),
        }[cik]

    async def frame(self, taxonomy: str, concept: str, unit: str, period: str) -> dict[int, Fact]:
        end = date(2026, 6, 30)
        if concept == "EntityCommonStockSharesOutstanding":
            return {c: Fact(1_000.0, end, "a") for c in (1, 2, 3)}
        if concept == "LongTermDebt" and period.endswith("I"):
            return {1: Fact(5_000.0, end, "a"), 2: Fact(90_000.0, end, "a")}
        if concept == "CashAndCashEquivalentsAtCarryingValue":
            return {c: Fact(1_000.0, end, "a") for c in (1, 2, 3)}
        if concept == "Revenues" and not period.endswith("I"):
            return {c: Fact(10_000.0, date(2025, 12, 31), "a") for c in (1, 2, 3)}
        return {}


async def test_a_run_screens_and_stores_point_in_time_verdicts(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        for sym in ("SOFT", "BANK", "NODEBT"):
            await conn.execute(
                text(
                    "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                    "volume, fetched_at) "
                    "VALUES (:s, '2026-09-30', 'raw', 100, 100, 100, 100, 1, now())"
                ),
                {"s": sym},
            )

    results = await run_screen(
        FakeSec(), engine, ["SOFT", "BANK", "NODEBT", "ZZZZ"], date(2026, 10, 1)
    )  # type: ignore[arg-type]

    verdicts = {r.symbol: r.verdict for r in results}
    assert verdicts == {"SOFT": "halal", "BANK": "not_halal", "NODEBT": "halal", "ZZZZ": "doubtful"}
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT symbol, verdict, metrics->>'debt_ratio' "
                    "FROM halal_screen_results ORDER BY symbol"
                )
            )
        ).all()
    assert [r[0] for r in rows] == ["BANK", "NODEBT", "SOFT", "ZZZZ"]
    assert float(dict((r[0], r[2]) for r in rows)["SOFT"]) == pytest.approx(0.05)
