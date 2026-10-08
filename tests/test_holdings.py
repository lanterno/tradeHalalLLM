"""What an account holds and the screen's view of it (portfolio/holdings.py)."""

from __future__ import annotations

from halal_trader.portfolio.holdings import broker_position, failing_screen, sector_values

SCREEN = {
    "MSFT": ("halal", "Services-Prepackaged Software"),
    "ADBE": ("halal", "Services-Prepackaged Software"),
    "LLY": ("halal", "Pharmaceutical Preparations"),
    "NVDA": ("not_halal", "Semiconductors & Related Devices"),
}


def test_failing_is_anything_the_newest_screen_does_not_pass_or_hold() -> None:
    assert failing_screen(["MSFT", "NVDA", "GONE"], SCREEN) == ["NVDA", "GONE"]


def test_sector_values_sum_by_the_screens_sector_largest_first() -> None:
    values = {"MSFT": 300.0, "ADBE": 100.0, "LLY": 350.0}
    ranked = sector_values(values, SCREEN)
    assert [v for _, v in ranked] == [400.0, 350.0]
    assert ranked[0][1] == 400.0  # the two software names together


def test_a_snapshot_position_carries_its_cost() -> None:
    row = broker_position(
        {"symbol": "MSFT", "qty": 2, "price": 500.0, "market_value": 1000.0, "unrealized_pl": 100.0}
    )
    assert row["cost_basis"] == 900.0 and row["avg_entry"] == 450.0
    assert row["unrealized_pl_pct"] == round(100 / 900, 5)
