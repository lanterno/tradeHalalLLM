"""A ticker that moved to a new SEC registrant reads its predecessor's facts."""

from __future__ import annotations

from datetime import date

from halal_trader.compliance.runner import _newest
from halal_trader.compliance.sec import Fact
from halal_trader.compliance.successors import PREDECESSOR, lineage


def test_lineage_follows_the_chain_newest_first() -> None:
    assert lineage(2115436) == (2115436, 34088)
    assert lineage(320193) == (320193,)


def test_a_successor_without_a_fact_falls_back_to_its_predecessor() -> None:
    successor, predecessor = next(iter(PREDECESSOR.items()))
    annual = [{predecessor: Fact(332e9, date(2025, 12, 31), "a")}]
    assert _newest(annual, successor) == 332e9
    # The successor's own figure wins once it has one.
    annual.insert(0, {successor: Fact(300e9, date(2026, 12, 31), "b")})
    assert _newest(annual, successor) == 300e9
