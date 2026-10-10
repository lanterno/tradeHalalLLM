"""`compliance rescreen-renamed`: the stored rows the veto's old-ticker match changes."""

from __future__ import annotations

import asyncio
import json
from datetime import date

import pytest
from click.testing import CliRunner
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from halal_trader.cli import cli
from halal_trader.compliance.index_veto import IndexView
from halal_trader.compliance.renamed import (
    Stored,
    changes,
    describe,
    outcome,
    pre_veto,
    rescreen_renamed,
)
from halal_trader.compliance.runner import METHOD
from halal_trader.compliance.sec import Company, Fact

AS_OF = date(2020, 9, 30)
FILED = date(2020, 7, 28)  # SPUS's N-PORT for 2020-05-31, which lists Facebook as FB
PERIOD = date(2020, 5, 31)


def _veto(etf: str, floor: str = "12.2B") -> str:
    return (
        f"excluded by {etf}'s Shariah index (holdings filed {FILED}) "
        f"although within its size range (market cap >= {floor})"
    )


BOARD = (
    "business activity unverified: restaurants (alcohol); impermissible revenue is not "
    "in SEC data and no Shariah index (SPUS, HLAL) holds the company"
)


def _row(symbol: str, verdict: str, *reasons: str, cap: float = 20e9, sic: int = 7372) -> Stored:
    return Stored(symbol, verdict, list(reasons), cap, sic)  # type: ignore[arg-type]


# ── what a stored row was before the veto ─────────────────────


@pytest.mark.parametrize(
    ("row", "before"),
    [
        (_row("A", "halal"), "halal"),
        (_row("A", "not_halal", _veto("SPUS")), "halal"),
        (_row("A", "doubtful", BOARD, sic=5812), "halal"),
        (_row("A", "not_halal", "debt ratio 45% >= 30%"), "not_halal"),
        (_row("A", "doubtful", "no market cap"), "doubtful"),
    ],
)
def test_a_row_that_failed_on_the_veto_or_the_board_alone_was_a_pass(
    row: Stored, before: str
) -> None:
    result = pre_veto(row)
    assert result.verdict == before
    assert result.metrics["market_cap"] == row.market_cap
    assert result.reasons == ([] if before == "halal" else row.reasons)


def test_an_outcome_names_the_excluding_index_but_not_the_floor() -> None:
    assert outcome("not_halal", [_veto("SPUS", "12.2B")]) == outcome(
        "not_halal", [_veto("SPUS", "11.9B")]
    )
    assert outcome("not_halal", [_veto("SPUS")]) != outcome("not_halal", [_veto("HLAL")])
    assert describe(outcome("not_halal", [_veto("HLAL")])) == "not_halal (HLAL)"
    assert describe(outcome("doubtful", [BOARD])) == "doubtful (board)"
    assert describe(outcome("halal", [])) == "halal"


# ── which rows of a date change ───────────────────────────────

HELD = {f"H{i}": 10e9 + i * 1e9 for i in range(40)}  # SPUS's range starts near 17.8B


def _held_rows() -> list[Stored]:
    return [_row(s, "halal", cap=c) for s, c in HELD.items()]


def _spus(*tickers: str, period_end: date = PERIOD) -> IndexView:
    return IndexView("SPUS", FILED, frozenset({*HELD, *tickers}), frozenset(), period_end)


def test_a_company_held_under_its_old_ticker_passes() -> None:
    rows = [*_held_rows(), _row("META", "not_halal", _veto("SPUS"), cap=543e9)]
    (a,) = changes(AS_OF, rows, {"META": "Meta Platforms, Inc."}, [_spus("FB")])
    assert (a.as_of, a.symbol, a.stored, a.replayed) == (
        AS_OF,
        "META",
        "not_halal (SPUS)",
        "halal",
    )
    assert a.why == "SPUS holds it as FB (holdings of 2020-05-31)"


def test_the_other_indexs_exclusion_replaces_the_one_that_held_it() -> None:
    # HLAL held Corpay as FLT; SPUS held neither ticker. The stored row says HLAL.
    hlal = IndexView("HLAL", date(2020, 7, 7), frozenset({*HELD, "FLT"}), frozenset(), PERIOD)
    rows = [*_held_rows(), _row("CPAY", "not_halal", _veto("HLAL"), cap=19.8e9)]
    (a,) = changes(AS_OF, rows, {}, [hlal, _spus()])
    assert (a.symbol, a.stored, a.replayed) == ("CPAY", "not_halal (HLAL)", "not_halal (SPUS)")


def test_a_name_the_moved_size_range_crosses_is_affected() -> None:
    # 30 held names from 10B to 39B: the floor is 15.8B. Fortune Brands, held as
    # FBHS at 1B, joins the range and lowers it to 15.0B, under NEAR's 15.5B.
    held = {f"H{i}": (10 + i) * 1e9 for i in range(30)}
    rows = [
        *(_row(s, "halal", cap=c) for s, c in held.items()),
        _row("FBIN", "halal", cap=1e9),
        _row("NEAR", "halal", cap=15.5e9),
    ]
    view = IndexView(
        "SPUS", date(2021, 7, 29), frozenset({*held, "FBHS"}), frozenset(), date(2021, 5, 31)
    )
    (a,) = changes(date(2021, 9, 30), rows, {}, [view])
    assert (a.symbol, a.stored, a.replayed) == ("NEAR", "halal", "not_halal (SPUS)")
    assert a.why == "a renamed company moved an index's size range"


def test_a_row_already_at_its_new_outcome_and_a_date_without_old_tickers_are_left() -> None:
    rows = [*_held_rows(), _row("META", "halal", cap=543e9)]
    assert changes(AS_OF, rows, {}, [_spus("FB")]) == []
    vetoed = [*_held_rows(), _row("META", "not_halal", _veto("SPUS"), cap=543e9)]
    assert changes(AS_OF, vetoed, {}, [_spus()]) == []
    late = _spus("FB", period_end=date(2022, 8, 31))  # FB's days ended 2022-06-15
    assert changes(AS_OF, vetoed, {}, [late]) == []


def test_a_mixed_activity_pass_held_under_an_old_ticker_loses_its_doubt() -> None:
    rows = [*_held_rows(), _row("META", "doubtful", BOARD, cap=5e9, sic=5812)]
    (a,) = changes(AS_OF, rows, {}, [_spus("FB")])
    assert (a.stored, a.replayed) == ("doubtful (board)", "halal")


# ── the re-screen ─────────────────────────────────────────────


class RenamedSec:
    """SEC today: CIK 1 is Meta Platforms (META), a clean company in 2020."""

    def __init__(self, user_agent: str = "") -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def companies(self) -> dict[str, Company]:
        return {"META": Company(1, "META", "Meta Platforms, Inc.")}

    async def sic(self, cik: int) -> tuple[int | None, str]:
        return 7370, "Services-computer programming, data processing, etc."

    async def foreign_filer(self, cik: int) -> bool:
        return False

    async def frame(self, taxonomy: str, concept: str, unit: str, period: str) -> dict[int, Fact]:
        instant = period.endswith("I")
        table = {
            "EntityCommonStockSharesOutstanding": 1_000.0 if instant else None,
            "CashAndCashEquivalentsAtCarryingValue": 1_000.0 if instant else None,
            "Revenues": 50_000.0 if not instant else None,
        }
        value = table.get(concept)
        return {} if value is None else {1: Fact(value, date(2020, 6, 30), "a")}


async def seed(engine: AsyncEngine) -> None:
    """SPUS's 2020-05-31 holdings: 40 names and FB. The stored 2020-09-30 screen:
    the 40 pass, META is excluded by SPUS (screened before the old-ticker match)."""
    peers = {f"H{i}": 50_000.0 + i * 1_000.0 for i in range(40)}  # META's cap is 100k
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO etf_holdings (etf, filed, period_end, ticker, name, cusip, "
                "weight_pct) VALUES ('SPUS', :f, :p, :t, :n, '', 1.0)"
            ),
            [{"f": FILED, "p": PERIOD, "t": t, "n": f"{t} Inc"} for t in [*peers, "FB"]],
        )
        rows = [(s, "halal", [], c) for s, c in peers.items()]
        rows.append(("META", "not_halal", [_veto("SPUS", "0.0B")], 100_000.0))
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, "
                "verdict, reasons, metrics, method, screened_at) VALUES (:d, :s, NULL, '', "
                ":v, CAST(:r AS JSONB), CAST(:m AS JSONB), :method, now())"
            ),
            [
                {
                    "d": AS_OF,
                    "s": s,
                    "v": v,
                    "r": json.dumps(r),
                    "m": json.dumps({"market_cap": c, "sic": 7370.0}),
                    "method": METHOD,
                }
                for s, v, r, c in rows
            ],
        )
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                "volume, fetched_at) VALUES ('META', '2020-09-29', 'raw', 100, 100, 100, 100, "
                "1, now())"
            )
        )


async def _verdicts(engine: AsyncEngine) -> dict[str, str]:
    async with engine.connect() as conn:
        rows = await conn.execute(text("SELECT symbol, verdict FROM halal_screen_current"))
        return {r.symbol: r.verdict for r in rows}


async def test_exactly_the_affected_rows_are_rescreened_once(engine: AsyncEngine) -> None:
    await seed(engine)
    sec = RenamedSec()

    (planned,) = await rescreen_renamed(sec, engine, dry_run=True)  # type: ignore[arg-type]
    assert (planned[0].symbol, planned[0].stored, planned[0].replayed, planned[1]) == (
        "META",
        "not_halal (SPUS)",
        "halal",
        None,
    )
    assert (await _verdicts(engine))["META"] == "not_halal"  # a dry run writes nothing

    (done,) = await rescreen_renamed(sec, engine)  # type: ignore[arg-type]
    assert (done[0].symbol, done[1]) == ("META", "halal")
    after = await _verdicts(engine)
    assert after["META"] == "halal"
    assert len(after) == 41 and set(after.values()) == {"halal"}
    assert await rescreen_renamed(sec, engine) == []  # type: ignore[arg-type]


def test_the_command_lists_then_rescreens(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def go() -> None:
        engine = create_async_engine(database_url)
        try:
            await seed(engine)
        finally:
            await engine.dispose()

    asyncio.run(go())
    monkeypatch.setattr("halal_trader.compliance.sec.SecClient", RenamedSec)

    dry = CliRunner().invoke(cli, ["compliance", "rescreen-renamed", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "2020-09-30 META   not_halal (SPUS) -> halal" in dry.output
    assert "SPUS holds it as FB (holdings of 2020-05-31)" in dry.output
    assert "1 row(s) across 1 screen date(s) would be re-screened" in dry.output

    run = CliRunner().invoke(cli, ["compliance", "rescreen-renamed"])
    assert run.exit_code == 0, run.output
    assert "re-screened halal" in run.output
    assert "1 row(s) across 1 screen date(s) re-screened" in run.output
