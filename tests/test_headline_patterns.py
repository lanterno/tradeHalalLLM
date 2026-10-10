"""The shared Benzinga headline templates (events/headline_patterns.py)."""

from __future__ import annotations

import pytest

from halal_trader.events.aliases import slot_of
from halal_trader.events.headline_patterns import ANALYST_ACTION, ANALYST_SLOT, sources


def _slot(headline: str) -> str | None:
    m = ANALYST_SLOT.search(headline)
    return None if m is None else next((g for g in m.groups() if g), None)


@pytest.mark.parametrize(
    ("headline", "slot"),
    [
        ("Morgan Stanley Downgrades Apple to Equal-Weight", "Apple"),
        ("Goldman Sachs Initiates Coverage On Snowflake with Buy Rating", "Snowflake"),
        ("Needham Maintains Buy on Apple", "Apple"),
        # "On", the 2016-2020 UPDATE wires.
        ("UPDATE: CFRA Maintains Strong Buy On Coca-Cola", "Coca-Cola"),
        ("Despite Transport Selloff, Stifel Reiterates 'Buy' On C.H. Robinson", "C.H. Robinson"),
        # "Initiates <Co> With <rating>", without "Coverage On".
        ("UPDATE: Canaccord Genuity Initiates Peloton With Buy", "Peloton"),
        ("Jefferies Initiates WW International With A Buy", "WW International"),
        ("Needham Initiates Lyft With Buy Rating And $48 Price Target", "Lyft"),
        ("Evercore Initiates Lyft With Outperform", "Lyft"),
        ("MKM Initiates Snap With Neutral", "Snap"),
        ("Bernstein Initiates Visa With Market Perform", "Visa"),
        ("Morgan Stanley Initiates Dell With Equal-weight", "Dell"),
        # A bare "Initiates" before anything but a rating names no company.
        ("Raymond James Initiates Coverage With Underperform Rating", None),
        ("Lilly Initiates Phase 3 Trial Of LY-CoV555 In Partnership With The NIH", None),
        ("Corcept Initiates Phase 1b Trial In Patients With Adrenal Cancer", None),
        ("Rambus Initiates $100M Accelerated Buyback", None),
        ("Apple Maintains Lead In Smartphones", None),
    ],
)
def test_the_analyst_slot(headline: str, slot: str | None) -> None:
    assert ANALYST_ACTION.search(headline)
    assert _slot(headline) == slot


def test_the_alias_learner_reads_the_new_slots() -> None:
    assert slot_of("UPDATE: Wedbush Maintains Outperform On Lululemon") == "Lululemon"
    assert slot_of("UPDATE: MKM Initiates Constellation Brands With Buy") == "Constellation Brands"
    assert slot_of("Alnylam Initiates Rolling Submission Of NDA For Patients With PH1") is None


def test_sources_hold_every_pattern() -> None:
    assert sources()["ANALYST_SLOT"] == ANALYST_SLOT.pattern
    assert set(sources()) == {
        "ANALYST_ACTION",
        "ANALYST_SLOT",
        "EARN_CO",
        "GUIDE_CO",
        "REISSUE_PREFIX",
        "CORRECTION",
    }
