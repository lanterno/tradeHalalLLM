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


@pytest.mark.parametrize("sic", [6792, 6794, 6795, 6798])
def test_reits_and_royalty_owners_face_the_ratios_not_an_activity_ban(sic: int) -> None:
    assert prohibited_activity(sic) is None
    assert screen(_f(sic=sic)).verdict == "halal"


@pytest.mark.parametrize("sic", [6700, 6719, 6726, 6770, 6793, 6799])
def test_holding_companies_funds_and_shells_stay_excluded(sic: int) -> None:
    assert screen(_f(sic=sic)).verdict == "not_halal"


def test_a_mortgage_reit_fails_on_interest_income() -> None:
    """SIC 6798 also files mortgage REITs: their revenue is interest."""
    r = screen(_f(sic=6798, interest_income=45_000.0, revenue=50_000.0))
    assert r.verdict == "not_halal" and "interest income" in r.reasons[0]


def test_the_average_price_sets_market_cap_when_known() -> None:
    """A spot rally must not wash a 36%-of-average-cap debt load down below 30%."""
    spot = screen(_f(interest_bearing_debt=36_000.0, price=150.0))  # 24% of spot cap
    averaged = screen(_f(interest_bearing_debt=36_000.0, price=150.0, average_price=100.0))
    assert spot.verdict == "halal"
    assert averaged.verdict == "not_halal" and averaged.metrics["market_cap"] == 100_000.0
    assert averaged.metrics["market_cap_basis"] == 36.0


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


def test_a_foreign_issuer_never_passes_on_ratios_but_still_fails_on_them() -> None:
    clean = _f(foreign_filer=True)
    result = screen(clean)
    assert result.verdict == "doubtful"
    assert any("foreign issuer" in r for r in result.reasons)
    # The overstated cap only shrinks the ratios, so a failure is still a failure.
    assert screen(_f(foreign_filer=True, interest_bearing_debt=40_000.0)).verdict == "not_halal"


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

    async def foreign_filer(self, cik: int) -> bool:
        return False

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


async def test_a_run_uses_the_36_month_average_when_there_is_a_year_of_history(
    engine: AsyncEngine,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                "volume, fetched_at) VALUES ('SOFT', '2026-09-30', 'raw', 100, 100, 100, 100, "
                "1, now())"
            )
        )
        # Month-end adjusted closes of 50 for 14 months (plus a mid-month bar
        # that is not a month end and must not count).
        for m in range(14):
            y, mo = divmod(8 - m, 12)
            await conn.execute(
                text(
                    "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                    "volume, fetched_at) VALUES ('SOFT', :d, 'all', 50, 50, 50, 50, 1, now())"
                ),
                {"d": date(2026 + y, mo + 1, 28)},
            )
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                "volume, fetched_at) VALUES ('SOFT', '2026-09-10', 'all', 999, 999, 999, 999, "
                "1, now())"
            )
        )

    (r,) = await run_screen(FakeSec(), engine, ["SOFT"], date(2026, 10, 1))  # type: ignore[arg-type]

    assert r.metrics["market_cap"] == pytest.approx(50_000.0)  # 1,000 shares x 50
    assert r.metrics["debt_ratio"] == pytest.approx(0.10)


class FilerShapes(FakeSec):
    """Real filing shapes that v2 misread.

    CIK 10 (like CVX): debt only as LongTermDebtAndCapitalLeaseObligations-
    IncludingCurrentMaturities, cash only including restricted cash.
    CIK 11 (like XOM): noncurrent debt with leases + DebtCurrent.
    CIK 12 (like GOOG): no dei shares (multi-class), diluted shares only;
    revenue under the Including-assessed-tax concept.
    """

    async def companies(self) -> dict[str, Company]:
        return {s: Company(c, s, s) for s, c in (("OIL", 10), ("OIL2", 11), ("MULTI", 12))}

    async def sic(self, cik: int) -> tuple[int | None, str]:
        return (2911, "Petroleum refining") if cik in (10, 11) else (7370, "Services")

    async def frame(self, taxonomy: str, concept: str, unit: str, period: str) -> dict[int, Fact]:
        end = date(2026, 6, 30)
        instant = period.endswith("I")
        table: dict[str, dict[int, float]] = {
            "EntityCommonStockSharesOutstanding": {10: 1_000.0, 11: 1_000.0} if instant else {},
            "WeightedAverageNumberOfDilutedSharesOutstanding": ({} if instant else {12: 2_000.0}),
            "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities": (
                {10: 37_000.0} if instant else {}
            ),
            "LongTermDebtAndCapitalLeaseObligations": {11: 33_000.0} if instant else {},
            "DebtCurrent": {11: 14_000.0} if instant else {},
            "ShortTermBorrowings": {10: 400.0} if instant else {},
            "CashAndCashEquivalentsAtCarryingValue": {11: 1_000.0, 12: 1_000.0} if instant else {},
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": (
                {10: 2_000.0} if instant else {}
            ),
            "Revenues": {10: 50_000.0, 11: 50_000.0} if not instant else {},
            "RevenueFromContractWithCustomerIncludingAssessedTax": (
                {12: 50_000.0} if not instant else {}
            ),
        }
        return {c: Fact(v, end, "a") for c, v in table.get(concept, {}).items()}


async def test_v3_reads_every_debt_family_and_falls_back_for_shares_cash_revenue(
    engine: AsyncEngine,
) -> None:
    async with engine.begin() as conn:
        for sym in ("OIL", "OIL2", "MULTI"):
            await conn.execute(
                text(
                    "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                    "volume, fetched_at) VALUES (:s, '2026-09-30', 'raw', 100, 100, 100, 100, "
                    "1, now())"
                ),
                {"s": sym},
            )

    results = await run_screen(FilerShapes(), engine, ["OIL", "OIL2", "MULTI"], date(2026, 10, 1))  # type: ignore[arg-type]
    by = {r.symbol: r for r in results}

    # 37,000 + 400 of debt on a 100,000 market cap: 37.4%, not v2's 0.4%.
    assert by["OIL"].metrics["debt_ratio"] == pytest.approx(0.374)
    assert by["OIL"].verdict == "not_halal"
    assert by["OIL"].metrics["cash_ratio"] == pytest.approx(0.02)  # restricted-inclusive cash
    assert by["OIL2"].metrics["debt_ratio"] == pytest.approx(0.47)  # 33,000 + 14,000
    # 2,000 diluted shares x 100 = 200,000 market cap; revenue found.
    assert by["MULTI"].metrics["market_cap"] == pytest.approx(200_000.0)
    assert by["MULTI"].verdict == "halal"


async def test_the_sec_client_memoises_frames_and_sic_codes() -> None:
    import httpx

    from halal_trader.compliance.sec import SecClient

    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        if "frames" in str(request.url):
            if "Missing" in str(request.url):
                return httpx.Response(404)
            row = {"cik": 1, "val": 5.0, "end": "2026-06-30", "accn": "a"}
            return httpx.Response(200, json={"data": [row]})
        forms = ["20-F", "6-K"] if "CIK0000000002" in str(request.url) else ["10-K", "20-F"]
        return httpx.Response(
            200,
            json={
                "sic": "3571",
                "sicDescription": "Computers",
                "filings": {"recent": {"form": forms}},
            },
        )

    sec = SecClient(
        "test test@example.invalid",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        min_interval_s=0.0,
    )
    for _ in range(3):
        assert (await sec.frame("us-gaap", "LongTermDebt", "USD", "CY2026Q2I"))[1].val == 5.0
        assert await sec.frame("us-gaap", "Missing", "USD", "CY2026Q2I") == {}
        assert await sec.sic(1) == (3571, "Computers")

    assert len(urls) == 3  # one per distinct request, however often asked
    # The submissions record already read for the SIC code says who files 20-F only.
    assert await sec.foreign_filer(1) is False  # a 10-K filer, whatever else it filed
    assert await sec.foreign_filer(2) is True
    assert len(urls) == 4


def test_quarter_ends_within_the_window() -> None:
    from halal_trader.compliance.history import quarter_ends

    assert quarter_ends(date(2025, 5, 1), date(2026, 3, 31)) == [
        date(2025, 6, 30),
        date(2025, 9, 30),
        date(2025, 12, 31),
        date(2026, 3, 31),
    ]


async def test_screen_history_screens_each_quarters_universe_once(engine: AsyncEngine) -> None:
    from halal_trader.compliance.history import screen_history

    async with engine.begin() as conn:
        for sym in ("SOFT", "BANK"):
            await conn.execute(
                text(
                    "INSERT INTO monthly_bars (symbol, month, close, volume, vwap) "
                    "SELECT :s, m::date, 100, 1e6, 100 FROM generate_series("
                    "'2025-01-01'::date, '2026-06-01'::date, '1 month') AS m"
                ),
                {"s": sym},
            )
            await conn.execute(
                text(
                    "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                    "volume, fetched_at) VALUES (:s, '2025-12-30', 'raw', 100, 100, 100, 100, "
                    "1, now())"
                ),
                {"s": sym},
            )

    first = await screen_history(
        FakeSec(), engine, start=date(2025, 12, 1), end=date(2026, 3, 31), top_n=10
    )  # type: ignore[arg-type]
    again = await screen_history(
        FakeSec(), engine, start=date(2025, 12, 1), end=date(2026, 3, 31), top_n=10
    )  # type: ignore[arg-type]

    assert first == {date(2025, 12, 31): 1, date(2026, 3, 31): 1}  # SOFT halal, BANK not
    assert again == {}


async def test_rescreen_foreign_turns_a_foreign_issuers_past_pass_doubtful(
    engine: AsyncEngine,
) -> None:
    from halal_trader.compliance.history import rescreen_foreign

    class Foreign(FakeSec):
        async def foreign_filer(self, cik: int) -> bool:
            return cik == 3

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                "volume, fetched_at) VALUES (:s, '2026-06-30', 'raw', 100, 100, 100, 100, 1, now())"
            ),
            [{"s": s} for s in ("SOFT", "NODEBT")],
        )
    as_of = date(2026, 7, 1)
    await run_screen(FakeSec(), engine, ["SOFT", "NODEBT"], as_of)  # type: ignore[arg-type]

    assert await rescreen_foreign(Foreign(), engine) == {as_of: 1}  # type: ignore[arg-type]
    async with engine.connect() as conn:
        rows = await conn.execute(text("SELECT symbol, verdict FROM halal_screen_results"))
        verdicts = {r.symbol: r.verdict for r in rows}
    assert verdicts == {"SOFT": "halal", "NODEBT": "doubtful"}
