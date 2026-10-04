"""The core portfolio's rules: cap targets, share classes, bands, forced sales."""

from __future__ import annotations

import pytest

from halal_trader.portfolio.strict_core import (
    rebalance,
    split_share_classes,
    targets,
    turnover,
)


def test_targets_weight_the_largest_caps() -> None:
    t = targets({"A": 300.0, "B": 100.0, "C": 50.0, "D": 0.0}, top_n=2)
    assert t == pytest.approx({"A": 0.75, "B": 0.25})


def test_share_classes_split_one_company_cap() -> None:
    caps = split_share_classes(
        {"GOOG": 2.0, "GOOGL": 2.0, "X": 2.0}, {"GOOG": 7, "GOOGL": 7, "X": 8}
    )
    assert caps == {"GOOG": 1.0, "GOOGL": 1.0, "X": 2.0}


def test_inside_the_band_nothing_trades_outside_it_returns_to_target() -> None:
    target = {"A": 0.5, "B": 0.3, "C": 0.2}
    drifted = {"A": 0.55, "B": 0.27, "C": 0.18}  # all within 25% of target
    assert rebalance(drifted, target, {"A", "B", "C"}) == pytest.approx(drifted)
    far = {"A": 0.70, "B": 0.20, "C": 0.10}  # all three outside their bands
    assert rebalance(far, target, {"A", "B", "C"}) == pytest.approx(target)


def test_a_name_the_screen_stops_passing_is_sold_whatever_its_drift() -> None:
    current = {"A": 0.5, "B": 0.5}
    new = rebalance(current, {"A": 0.5, "B": 0.5}, eligible={"A"})
    assert new == {"A": 1.0}
    assert turnover(current, new) == pytest.approx(1.0)


def test_a_new_name_enters_and_holdings_inside_their_band_make_room_pro_rata() -> None:
    # A is out of its band: both go to target.
    assert rebalance({"A": 1.0}, {"A": 0.7, "B": 0.3}, {"A", "B"}) == pytest.approx(
        {"A": 0.7, "B": 0.3}
    )
    # A sits on its band's edge: held, and scaled down as B comes in.
    assert rebalance({"A": 1.0}, {"A": 0.8, "B": 0.2}, {"A", "B"}) == pytest.approx(
        {"A": 1 / 1.2, "B": 0.2 / 1.2}
    )
