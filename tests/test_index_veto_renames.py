"""The index veto recognises a renamed company held under its old ticker, on its own days only."""

from __future__ import annotations

from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.aaoifi import ScreenResult
from halal_trader.compliance.index_veto import (
    IndexView,
    apply_veto,
    held,
    held_as,
    name_key,
    require_board,
    views_at,
)

# 40 held names from 10B to 49B: the size range starts near 17.8B.
HELD = {f"H{i}": 10e9 + i * 1e9 for i in range(40)}
TITLES = {
    "META": "Meta Platforms, Inc.",
    "CPAY": "Corpay, Inc.",
    "TT": "Trane Technologies plc",
    "IR": "Ingersoll Rand Inc.",
}


def _view(etf: str, filed: date, period_end: date | None, *tickers: str) -> IndexView:
    return IndexView(etf, filed, frozenset({*HELD, *tickers}), frozenset(), period_end)


def _passes(**caps: float) -> list[ScreenResult]:
    return [ScreenResult(s, "halal", [], {"market_cap": c}) for s, c in {**HELD, **caps}.items()]


def _verdicts(view: IndexView, **caps: float) -> dict[str, str]:
    return {r.symbol: r.verdict for r in apply_veto(_passes(**caps), TITLES, [view])}


def test_spus_holding_fb_in_2020_holds_meta() -> None:
    spus = _view("SPUS", date(2020, 7, 28), date(2020, 5, 31), "FB")
    assert held_as(spus, "META") == "FB"
    assert _verdicts(spus, META=543e9)["META"] == "halal"


def test_hlal_holding_flt_and_zi_holds_corpay_and_zoominfo() -> None:
    hlal = _view("HLAL", date(2022, 7, 27), date(2022, 5, 31), "FLT", "ZI")
    assert set(_verdicts(hlal, CPAY=18.8e9, GTM=20.1e9).values()) == {"halal"}
    # A view without the old tickers excludes both.
    spus = _view("SPUS", date(2022, 7, 28), date(2022, 5, 31))
    by = _verdicts(spus, CPAY=18.8e9, GTM=20.1e9)
    assert (by["CPAY"], by["GTM"]) == ("not_halal", "not_halal")


def test_an_old_ticker_counts_through_its_grace_sessions_and_not_after() -> None:
    # FB's last session was 2022-06-08; Benzinga's grace runs to 2022-06-15.
    in_grace = _view("SPUS", date(2022, 7, 28), date(2022, 6, 15), "FB")
    after = _view("SPUS", date(2022, 10, 26), date(2022, 8, 31), "FB")
    assert held_as(in_grace, "META") == "FB"
    assert held_as(after, "META") is None
    assert _verdicts(after, META=400e9)["META"] == "not_halal"


def test_a_reused_ticker_never_counts_for_its_former_owner() -> None:
    # Ingersoll-Rand plc (now Trane, TT) traded as IR to 2020-02-28; Gardner
    # Denver took IR the next session. A 2020 holding of IR is Gardner Denver's.
    spus = _view("SPUS", date(2020, 7, 28), date(2020, 5, 31), "IR")
    assert held_as(spus, "TT") is None
    by = _verdicts(spus, TT=25e9, IR=20e9)
    assert by["TT"] == "not_halal"
    assert by["IR"] == "halal"
    before = _view("SPUS", date(2020, 3, 27), date(2020, 2, 28), "IR")
    assert held_as(before, "TT") == "IR"  # on its last session IR was still Trane's


def test_an_old_ticker_counts_only_from_the_day_it_named_the_company() -> None:
    # IAC named the old IAC (now Match Group) until 2020-06-30; People Inc.
    # (the spin-off, PPLI today) traded as IAC from 2020-07-01.
    old_owner = _view("HLAL", date(2020, 7, 7), date(2020, 5, 31), "IAC")
    new_owner = _view("HLAL", date(2022, 10, 24), date(2022, 8, 31), "IAC")
    assert held_as(old_owner, "PPLI") is None
    assert held_as(new_owner, "PPLI") == "IAC"


def test_an_undated_view_counts_no_old_ticker() -> None:
    undated = IndexView(
        "SPUS", date(2020, 7, 28), frozenset({"FB", "AAPL"}), frozenset({name_key("Facebook Inc")})
    )
    assert held_as(undated, "META") is None
    assert not held(undated, "META", TITLES["META"])
    assert held(undated, "AAPL", "")  # today's tickers and names still count
    assert held(undated, "META", "Facebook, Inc.")  # by name, had SEC kept the old one


def test_a_holding_under_an_old_ticker_joins_the_size_range() -> None:
    # 29 names held by today's ticker are too few to size the range (MIN_PRICED
    # is 30); FB makes the 30th, and the veto then applies to BIG.
    held_now = {f"H{i}": 10e9 + i * 1e9 for i in range(29)}
    results = [
        ScreenResult(s, "halal", [], {"market_cap": c})
        for s, c in {**held_now, "META": 543e9, "BIG": 300e9}.items()
    ]
    dated = IndexView(
        "SPUS", date(2020, 7, 28), frozenset({*held_now, "FB"}), frozenset(), date(2020, 5, 31)
    )
    undated = IndexView("SPUS", date(2020, 7, 28), dated.tickers, frozenset())
    by = {r.symbol: r.verdict for r in apply_veto(results, TITLES, [dated])}
    assert (by["META"], by["BIG"]) == ("halal", "not_halal")
    assert {r.verdict for r in apply_veto(results, TITLES, [undated])} == {"halal"}


def test_the_board_rule_accepts_a_holding_under_an_old_ticker() -> None:
    hlal = _view("HLAL", date(2022, 1, 20), date(2021, 11, 30), "FLT")
    result = ScreenResult("CPAY", "halal", [], {"market_cap": 21e9})
    (r,) = require_board([result], {"CPAY": 5812}, TITLES, [hlal])
    assert r.verdict == "halal"
    (r,) = require_board([result], {"CPAY": 5812}, TITLES, [_view("HLAL", hlal.filed, None, "FLT")])
    assert r.verdict == "doubtful"


async def test_views_carry_the_period_their_holdings_are_as_of(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO etf_holdings (etf, filed, period_end, ticker, name, cusip, "
                "weight_pct) VALUES (:e, :f, :p, :t, :n, :c, 1.0)"
            ),
            [
                {
                    "e": "SPUS",
                    "f": date(2020, 7, 28),
                    "p": date(2020, 5, 31),
                    "t": "FB",
                    "n": "Facebook Inc",
                    "c": "30303M102",
                },
                {
                    "e": "HLAL",
                    "f": date(2020, 7, 7),
                    "p": date(2020, 5, 31),
                    "t": "FLT",
                    "n": "FleetCor Technologies Inc",
                    "c": "339041105",
                },
            ],
        )
    views = {v.etf: v for v in await views_at(engine, date(2020, 9, 30))}
    assert views["SPUS"].period_end == date(2020, 5, 31)
    assert held(views["SPUS"], "META", TITLES["META"])
    assert held(views["HLAL"], "CPAY", TITLES["CPAY"])
    assert not held(views["HLAL"], "META", TITLES["META"])
