"""Screen v12: a pass in a mixed-activity sector needs a Shariah index's inclusion."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.compliance.aaoifi import ScreenResult, mixed_activity, prohibited_activity
from halal_trader.compliance.index_veto import IndexView, name_key, require_board
from halal_trader.compliance.validate import Verdict, rejection_kind

SPUS = IndexView("SPUS", date(2026, 9, 30), frozenset({"MCD", "COST"}), frozenset())
HLAL = IndexView("HLAL", date(2026, 9, 30), frozenset(), frozenset({name_key("Kroger Co")}))


def _halal(symbol: str) -> ScreenResult:
    return ScreenResult(symbol, "halal", [], {"market_cap": 5e9})


@pytest.mark.parametrize(("sic", "word"), [(5813, "bars"), (5993, "tobacco")])
def test_bars_and_tobacco_stores_are_excluded(sic: int, word: str) -> None:
    assert word in (prohibited_activity(sic) or "")


@pytest.mark.parametrize(
    ("sic", "mixed"),
    [
        (5812, True),  # restaurants
        (5411, True),  # grocery stores
        (5412, True),  # convenience stores
        (5331, True),  # variety stores
        (5399, True),  # warehouse clubs
        (5912, True),  # drug stores
        (5141, True),  # grocery wholesale
        (2015, True),  # poultry and processed food
        (2086, False),  # soft drinks carry no pork
        (7372, False),  # software
        (None, False),
    ],
)
def test_which_sectors_are_mixed(sic: int | None, mixed: bool) -> None:
    assert (mixed_activity(sic) is not None) is mixed


def test_a_restaurant_no_index_holds_is_doubtful() -> None:
    (r,) = require_board([_halal("TXRH")], {"TXRH": 5812}, {}, [SPUS, HLAL])
    assert r.verdict == "doubtful"
    assert r.reasons[-1].startswith("business activity unverified: restaurants")


def test_an_index_holding_by_ticker_or_by_name_lets_the_pass_stand() -> None:
    results = [_halal("MCD"), _halal("KR")]
    out = require_board(results, {"MCD": 5812, "KR": 5411}, {"KR": "KROGER CO"}, [SPUS, HLAL])
    assert [r.verdict for r in out] == ["halal", "halal"]


def test_with_no_index_holdings_on_file_every_mixed_pass_is_doubtful() -> None:
    (r,) = require_board([_halal("MCD")], {"MCD": 5812}, {}, [])
    assert r.verdict == "doubtful"


def test_other_sectors_and_other_verdicts_are_left_alone() -> None:
    software = _halal("ADBE")
    failed = ScreenResult("CASY", "not_halal", ["debt"], {})
    out = require_board([software, failed], {"ADBE": 7372, "CASY": 5412}, {}, [])
    assert out == [software, failed]


def test_validation_counts_it_as_an_activity_rejection() -> None:
    v = Verdict(
        "TXRH",
        "doubtful",
        ["business activity unverified: restaurants (alcohol); ..."],
        1e10,
    )
    assert rejection_kind(v) == "activity"


async def test_a_run_doubts_a_restaurant_until_an_index_holds_it(engine) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import text

    from halal_trader.compliance.runner import run_screen
    from tests.test_compliance_screen import FakeSec

    class RestaurantSec(FakeSec):
        async def sic(self, cik: int) -> tuple[int | None, str]:
            return (5812, "Retail-eating places") if cik == 1 else await super().sic(cik)

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                "volume, fetched_at) VALUES ('SOFT', '2026-09-30', 'raw', 100, 100, 100, 100, "
                "1, now())"
            )
        )
    as_of = date(2026, 10, 1)

    (alone,) = await run_screen(RestaurantSec(), engine, ["SOFT"], as_of)  # type: ignore[arg-type]
    assert alone.verdict == "doubtful"

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO etf_holdings (etf, filed, period_end, ticker, name, cusip, "
                "weight_pct) VALUES ('SPUS', '2026-09-01', '2026-07-31', 'SOFT', 'Soft Inc', "
                "'000000000', 1.0)"
            )
        )
    (held,) = await run_screen(RestaurantSec(), engine, ["SOFT"], as_of)  # type: ignore[arg-type]
    assert held.verdict == "halal"
