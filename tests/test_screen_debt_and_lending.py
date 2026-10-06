"""v11: debt the way REITs and homebuilders file it, interest expense as a check,
lending income as impure income, and mortgage REITs as lenders.

The fixtures are the companies' own XBRL facts as SEC's frames API served
them for 2026 (values in USD; market caps as the 2026-10-05 screen read them).
Each was a pass under v10.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.aaoifi import Fundamentals, screen
from halal_trader.compliance.runner import debt_of, run_screen
from halal_trader.compliance.sec import Company, Fact

B = 1e9


def _debt(facts: dict[str, float]) -> float | None:
    return debt_of(facts.get)[0]


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        # KRC: only SecuredDebt + UnsecuredDebt. v10 read 0.
        ({"SecuredDebt": 0.591 * B, "UnsecuredDebt": 3.998 * B}, 4.589 * B),
        # SLG: secured mortgages, senior notes and its revolver, plus finance leases.
        (
            {
                "SecuredDebt": 2.492 * B,
                "SeniorNotes": 1.143 * B,
                "LineOfCredit": 0.814 * B,
                "FinanceLeaseLiability": 0.109 * B,
            },
            4.558 * B,
        ),
        # NNN: NotesPayable + LoansPayable (a term loan).
        ({"NotesPayable": 4.476 * B, "LoansPayable": 0.497 * B}, 4.973 * B),
        # SM: SeniorNotes is the total of SeniorLongTermNotes + SeniorNotesCurrent:
        # synonyms within a kind are not added.
        (
            {
                "SeniorNotes": 7.036 * B,
                "SeniorLongTermNotes": 6.620 * B,
                "SeniorNotesCurrent": 0.416 * B,
            },
            7.036 * B,
        ),
        # KBH: one total, NotesAndLoansPayable.
        ({"NotesAndLoansPayable": 1.969 * B}, 1.969 * B),
        # OPEN: asset-backed credit lines and a convertible.
        ({"LongTermLineOfCredit": 1.071 * B, "ConvertibleDebtCurrent": 0.194 * B}, 1.265 * B),
        # FR: the filed total (DebtInstrumentCarryingAmount) beats the parts it is made of;
        # the two readings are never added.
        (
            {
                "DebtInstrumentCarryingAmount": 2.581 * B,
                "UnsecuredDebt": 1.440 * B,
                "OtherLongTermDebt": 0.993 * B,
                "LineOfCredit": 0.123 * B,
                "SecuredDebt": 0.009 * B,
            },
            2.581 * B,
        ),
        # MAA: NotesPayable already includes the secured debt; the kinds overstate by it
        # (+6%), which errs strict.
        (
            {"NotesPayable": 5.657 * B, "UnsecuredDebt": 5.296 * B, "SecuredDebt": 0.360 * B},
            6.017 * B,
        ),
        # CNH: DebtAndCapitalLeaseObligations, a total v10 did not read.
        ({"DebtAndCapitalLeaseObligations": 25.966 * B, "OtherLongTermDebt": 1.3 * B}, 25.966 * B),
        # AKAM: convertible notes, noncurrent + current.
        (
            {
                "ConvertibleLongTermNotesPayable": 5.857 * B,
                "ConvertibleNotesPayableCurrent": 1.706 * B,
            },
            7.563 * B,
        ),
    ],
)
def test_debt_reads_every_kind_without_adding_synonyms(
    facts: dict[str, float], expected: float
) -> None:
    assert _debt(facts) == pytest.approx(expected)


def test_debt_is_none_only_when_no_debt_concept_is_filed() -> None:
    assert _debt({}) is None
    assert _debt({"FinanceLeaseLiability": 1.0}) == 1.0
    debt, readings = debt_of({"LongTermDebt": 10.0, "SecuredDebt": 4.0, "SeniorNotes": 9.0}.get)
    assert debt == 13.0  # the kinds (4 + 9) beat the filed total (10)
    assert readings == {"debt_from_totals": 10.0, "debt_from_kinds": 13.0, "debt_extras": None}


def _f(**kw: object) -> Fundamentals:
    base: dict[str, object] = dict(
        symbol="CO",
        sic=4991,
        shares_outstanding=1e9,
        price=9.91,  # AES: market cap 9.9B
        interest_bearing_debt=0.735 * B,  # its finance leases: all the standard tags hold
        cash_and_securities=1.0 * B,
        interest_income=0.05 * B,
        revenue=12.0 * B,
    )
    base.update(kw)
    return Fundamentals(**base)  # type: ignore[arg-type]


def test_interest_expense_implying_far_more_debt_than_the_tags_is_doubtful() -> None:
    aes = screen(_f(interest_expense=1.407 * B))  # ~$23B of debt at 6%: 237% of the cap
    assert aes.verdict == "doubtful"
    assert "interest expense implies" in aes.reasons[0]
    assert aes.metrics["implied_debt_ratio"] == pytest.approx(1.407 / 0.06 / 9.91)
    # A company whose tags explain its interest bill passes as before.
    assert screen(_f(interest_expense=0.02 * B)).verdict == "halal"
    assert (
        screen(_f(interest_expense=0.2 * B, interest_bearing_debt=2.0 * B)).verdict == "halal"
    )  # implies 3.3B (34%); 2.0B read explains over half; the 20% ratio stands


def test_lending_income_counts_as_interest_income() -> None:
    # STWD without its balance sheet: 82% of revenue is commercial-loan interest.
    r = screen(_f(sic=6798, interest_income=None, lender_income=1.519 * B, revenue=1.844 * B))
    assert r.verdict == "not_halal" and "interest income / revenue" in r.reasons[0]
    assert r.metrics["lender_income_ratio"] == pytest.approx(1.519 / 1.844)


def test_a_reit_holding_mostly_loans_is_a_lender() -> None:
    stwd = screen(_f(sic=6798, loans_receivable=22.03 * B, total_assets=61.05 * B))
    assert stwd.verdict == "not_halal"
    assert "mortgage REIT" in stwd.reasons[0] and stwd.reasons[0].startswith("business activity")
    # An equity REIT's mezzanine loans are a few percent of assets: no activity failure.
    slg = screen(_f(sic=6798, loans_receivable=0.12 * B, total_assets=11.76 * B))
    assert slg.verdict == "halal"
    # Only REITs: a manufacturer's financing receivables face the ratios instead.
    assert screen(_f(sic=3523, loans_receivable=30 * B, total_assets=40 * B)).verdict == "halal"


def test_an_implausible_share_count_is_doubtful_not_a_ratio_failure() -> None:
    # MCD: 711 diluted shares (millions, unscaled) x $290 = a $0.2M market cap.
    mcd = screen(_f(sic=5812, shares_outstanding=711.0, price=290.0, interest_bearing_debt=40 * B))
    assert mcd.verdict == "doubtful"
    assert "mis-scaled" in mcd.reasons[-1]
    # A heavily indebted but real company still fails on the ratio.
    assert screen(_f(interest_bearing_debt=30 * B)).verdict == "not_halal"


# ── end to end, through the frames ─────────────────────────────────────────

END = date(2026, 6, 30)
# symbol -> (cik, sic, price, shares, instant facts, annual facts)
FILERS: dict[str, tuple[int, int, float, float, dict[str, float], dict[str, float]]] = {
    "KRC": (
        1025996,
        6798,
        3.8,
        1e9,
        {"SecuredDebt": 0.591 * B, "UnsecuredDebt": 3.998 * B, "Assets": 10.77 * B},
        {"Revenues": 1.1 * B, "InterestExpenseNonoperating": 0.126 * B},
    ),
    "STWD": (
        1465128,
        6798,
        5.9,
        1e9,
        {
            "SecuredDebt": 14.0 * B,
            "UnsecuredDebt": 4.883 * B,
            "MortgageLoansOnRealEstate": 22.03 * B,
            "NotesReceivableNet": 19.82 * B,
            "Assets": 61.05 * B,
        },
        {"Revenues": 1.844 * B, "InterestAndFeeIncomeLoansCommercial": 1.519 * B},
    ),
    "AES": (
        874761,
        4991,
        9.91,
        1e9,
        {"FinanceLeaseLiability": 0.735 * B},
        {"Revenues": 12.0 * B, "InterestExpense": 1.407 * B, "InterestPaidNet": 1.21 * B},
    ),
    "CAH": (
        721371,
        5122,
        35.5,
        1e9,
        {"DebtAndCapitalLeaseObligations": 8.886 * B, "ReceivablesNetCurrent": 18.0 * B},
        {"Revenues": 220.0 * B},
    ),
    "CLEAN": (  # debt-free, with a small interest bill (finance leases)
        865752,
        2086,
        32.0,
        1e9,
        {},
        {"Revenues": 8.0 * B, "InterestPaidNet": 0.005 * B},
    ),
}


class FrameSec:
    async def companies(self) -> dict[str, Company]:
        return {s: Company(cik, s, s) for s, (cik, *_) in FILERS.items()}

    async def sic(self, cik: int) -> tuple[int | None, str]:
        sic = next(s for c, s, *_ in FILERS.values() if c == cik)
        return sic, f"SIC {sic}"

    async def foreign_filer(self, cik: int) -> bool:
        return False

    def filed(self, accn: str) -> date | None:
        return None

    async def frame(self, taxonomy: str, concept: str, unit: str, period: str) -> dict[int, Fact]:
        out: dict[int, Fact] = {}
        for cik, _, _, shares, instant, annual in FILERS.values():
            if period.endswith("I"):
                if concept == "EntityCommonStockSharesOutstanding":
                    out[cik] = Fact(shares, END, "a")
                elif concept == "CashAndCashEquivalentsAtCarryingValue":
                    out[cik] = Fact(0.1 * B, END, "a")
                elif concept in instant:
                    out[cik] = Fact(instant[concept], END, "a")
            elif period == "CY2025" and concept in annual:
                out[cik] = Fact(annual[concept], date(2025, 12, 31), "a")
        return out


async def test_v10_passes_that_the_new_reads_catch(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                "volume, fetched_at) VALUES (:s, '2026-10-02', 'raw', :p, :p, :p, :p, 1, now())"
            ),
            [{"s": s, "p": p} for s, (_, _, p, *_) in FILERS.items()],
        )

    results = await run_screen(FrameSec(), engine, list(FILERS), date(2026, 10, 5))  # type: ignore[arg-type]
    by = {r.symbol: r for r in results}

    assert by["KRC"].verdict == "not_halal"
    assert by["KRC"].metrics["debt_ratio"] == pytest.approx(4.589 / 3.8)  # 121%, not 0%
    assert by["STWD"].verdict == "not_halal" and "mortgage REIT" in by["STWD"].reasons[0]
    assert by["AES"].verdict == "doubtful" and "interest expense" in by["AES"].reasons[0]
    assert by["CAH"].metrics["receivables_ratio"] == pytest.approx(18.0 / 35.5)
    assert by["CAH"].verdict == "not_halal"  # 51% receivables; v10 read none
    assert by["CLEAN"].verdict == "halal"

    async with engine.connect() as conn:
        metrics = (
            await conn.execute(
                text("SELECT metrics FROM halal_screen_results WHERE symbol = 'KRC'")
            )
        ).scalar_one()
    assert metrics["debt_from_kinds"] == pytest.approx(4.589 * B)
    assert metrics["interest_expense"] == pytest.approx(0.126 * B)
