"""The story builder (events/stories.py), pure parts: admission, time, grouping, labels.

Items are synthetic Benzinga headlines and 8-K rows; facts come from the real
earnings parser, types from the real taxonomy. No database and no price.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta

import pytest

from halabot.playbooks.interfaces import StoryView
from halal_trader.data.minutes import session_bounds
from halal_trader.events import earnings_parse, renames, stories
from halal_trader.events.aliases import BUILDER_VERSION, AliasMatcher
from halal_trader.events.earnings_parse import PARSER_SHA, parse_headline
from halal_trader.events.stories import (
    NEWS_LAG,
    RawItem,
    Story,
    StoryCounts,
    analyst_clause,
    build,
    counts_table,
    edgar_business_day,
    eligibility_at,
    fact_keys,
    federal_holidays,
    filing_public_at,
    has_analyst_slot,
    jaccard,
    reaction_session,
    shingles,
    story_row,
    window_of,
)
from halal_trader.events.taxonomy import TAXONOMY_SHA, ItemLike, StoryCard, resolve
from halal_trader.market_hours import MARKET_TZ

ALIASES = {
    "AAPL": AliasMatcher("AAPL", ("Apple",), ("AAPL",)),
    "INTC": AliasMatcher("INTC", ("Intel",), ("INTC",)),
    "AMD": AliasMatcher("AMD", ("AMD", "Advanced Micro"), ("AMD",)),
    "JBHT": AliasMatcher("JBHT", ("JB Hunt",), ("JBHT",)),
    "BRKB": AliasMatcher("BRKB", ("Berkshire",), ("BRK.B",)),
}
_ids = itertools.count(1)

# A plain week: Mon 2024-05-06 .. Fri 2024-05-10, then Mon 2024-05-13.
MON, TUE, WED, THU, FRI = (date(2024, 5, d) for d in (6, 7, 8, 9, 10))
NEXT_MON = date(2024, 5, 13)


def ny(day: date, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm, ss), MARKET_TZ)


def news(
    symbol: str,
    at: datetime,
    headline: str,
    *,
    n: int | None = 1,
    eid: int | None = None,
) -> RawItem:
    return RawItem(
        event_id=eid if eid is not None else next(_ids),
        source_id=f"alpaca:{next(_ids)}",
        kind="news",
        symbol=symbol,
        published_at=at.astimezone(UTC),
        seen_at=at.astimezone(UTC),
        headline=headline,
        n_symbols=n,
        facts=tuple(parse_headline(headline)),
    )


def filing(symbol: str, accepted: datetime, items: tuple[str, ...], kind: str = "8-k") -> RawItem:
    return RawItem(
        event_id=next(_ids),
        source_id=f"0000-{next(_ids)}",
        kind=kind,
        symbol=symbol,
        published_at=accepted.astimezone(UTC),
        seen_at=accepted.astimezone(UTC),
        headline="",
        n_symbols=None,
        items_8k=items,
    )


def one(items: list[RawItem], **kwargs: object) -> Story:
    built = build(items, ALIASES, **kwargs)  # type: ignore[arg-type]
    assert len(built) == 1, [s.story_id for s in built]
    return built[0]


def by_id(built: list[Story]) -> dict[str, Story]:
    return {s.story_id: s for s in built}


DOWNGRADE = "Morgan Stanley Downgrades Apple to Equal-Weight"
MISS = "Apple Q2 Adj. EPS $1.40 Misses $1.50 Estimate, Sales $90.00B Miss $91.00B Estimate"


# ── admission (spec §A.2) ─────────────────────────────────────


def test_only_news_and_8k_rows_are_story_items() -> None:
    c: Counter[str] = Counter()
    rows = [
        news("AAPL", ny(TUE, 8), DOWNGRADE),
        filing("AAPL", ny(TUE, 7), ("2.02",)),
        filing("AAPL", ny(TUE, 7, 30), ("5.02",), kind="8-k/a"),
        filing("AAPL", ny(TUE, 8), (), kind="10-q"),
        filing("AAPL", ny(TUE, 8), (), kind="insider_buy"),
    ]
    story = one(rows, counters=c)
    assert sorted(i.raw.kind for i in story.items) == ["8-k", "8-k/a", "news"]
    assert c["kind"] == 2 and c["admitted"] == 3 and c["rows"] == 5


def test_a_roundup_of_more_than_three_symbols_is_dropped_and_a_live_row_is_flagged() -> None:
    c: Counter[str] = Counter()
    rows = [
        news("AAPL", ny(TUE, 8), "Apple Unveils New Mac", n=4),
        news("AAPL", ny(TUE, 8, 1), "Apple Unveils New iPad", n=3),
        news("AAPL", ny(TUE, 8, 2), "Apple Unveils New Watch", n=None),
    ]
    story = one(rows, counters=c)
    assert [i.raw.headline for i in story.items] == [
        "Apple Unveils New iPad",
        "Apple Unveils New Watch",
    ]
    assert c["roundup"] == 1 and c["nsym_unknown"] == 1
    assert story_row(story)["flags"]["nsym_unknown"] == 1


def test_the_headline_must_name_the_company() -> None:
    c: Counter[str] = Counter()
    rows = [
        news("AAPL", ny(TUE, 8), "Samsung Unveils New Phone"),  # tagged AAPL, about Samsung
        news("AAPL", ny(TUE, 8, 1), "Pineapple Prices Soar"),  # no letter before the alias
        news("AAPL", ny(TUE, 8, 2), "Applebee's Opens Restaurant"),  # nor a lowercase one after
        news("AAPL", ny(TUE, 8, 3), "aapl Trends On Social Media"),  # tickers are case-sensitive
        news("AAPL", ny(TUE, 8, 4), "$AAPL Unveils New Mac"),
        news("JBHT", ny(TUE, 8, 5), "J.B. Hunt Unveils New Trucks"),  # dots removed both sides
        news("BRKB", ny(TUE, 8, 6), "BRK.B Unveils Something"),
    ]
    built = by_id(build(rows, ALIASES, counters=c))
    assert [i.raw.headline for i in built["AAPL:2024-05-07"].items] == ["$AAPL Unveils New Mac"]
    assert len(built["JBHT:2024-05-07"].items) == 1
    assert len(built["BRKB:2024-05-07"].items) == 1
    assert c["entity"] == 4
    assert all(i.entity_ok is True for s in built.values() for i in s.items)


def test_a_symbol_without_stored_aliases_is_matched_on_its_ticker_alone() -> None:
    rows = [
        news("ZZZZ", ny(TUE, 8), "ZZZZ Unveils New Product"),
        news("ZZZZ", ny(TUE, 8, 1), "Zeta Unveils New Product"),
    ]
    story = one(rows)
    assert [i.raw.headline for i in story.items] == ["ZZZZ Unveils New Product"]


def test_a_filing_has_no_entity_check() -> None:
    story = one([filing("AAPL", ny(TUE, 7), ("8.01",))])
    assert story.items[0].entity_ok is None and story.items[0].itype == "filing_other"


def test_an_analyst_headline_is_read_for_the_clause_that_names_the_company() -> None:
    headline = "Barclays Downgrades Intel, Upgrades AMD"
    built = by_id(
        build([news("INTC", ny(TUE, 8), headline), news("AMD", ny(TUE, 8), headline)], ALIASES)
    )
    assert built["INTC:2024-05-07"].items[0].itype == "analyst_downgrade"
    assert built["AMD:2024-05-07"].items[0].itype == "analyst_upgrade"
    joined = "Jefferies Downgrades Intel and Upgrades AMD"
    assert analyst_clause(joined, ALIASES["INTC"]) == "Jefferies Downgrades Intel"
    assert analyst_clause(joined, ALIASES["AMD"]) == "Upgrades AMD"


def test_the_clause_group_keeps_the_slotless_clauses_after_it() -> None:
    headline = "Baird Downgrades Apple to Neutral, Lowers Target to $265.00; Sees Risks"
    assert analyst_clause(headline, ALIASES["AAPL"]) == headline
    maintains = "Morgan Stanley Maintains Overweight on Apple, Raises Price Target to $200"
    assert analyst_clause(maintains, ALIASES["AAPL"]) == maintains
    assert one([news("AAPL", ny(TUE, 8), maintains)]).items[0].itype == "analyst_pt_raise"
    # The next action with a slot of its own ends the group.
    two = "UBS Upgrades Apple to Buy, Downgrades Intel to Sell"
    assert analyst_clause(two, ALIASES["AAPL"]) == "UBS Upgrades Apple to Buy"


def test_an_analyst_headline_with_no_clause_about_the_company_is_dropped() -> None:
    c: Counter[str] = Counter()
    rows = [
        # Apple is named, but in no action's slot: the action is about Intel.
        news("AAPL", ny(TUE, 8), "Morgan Stanley Downgrades Intel, Says Apple Orders Weak"),
        news("AAPL", ny(TUE, 8, 2), DOWNGRADE),
    ]
    story = one(rows, counters=c)
    assert [i.raw.headline for i in story.items] == [DOWNGRADE]
    assert c["entity_analyst"] == 1 and c["analyst_no_slot"] == 0


def test_an_analyst_headline_with_no_slot_is_checked_and_classified_whole() -> None:
    c: Counter[str] = Counter()
    rows = [
        news("AAPL", ny(TUE, 8), "Vetr Issues Downgrade To Hold On Apple"),
        news("AAPL", ny(TUE, 8, 1), "Apple Cuts FY24 Revenue Guidance, Reiterates Adj EPS Outlook"),
        news("AAPL", ny(TUE, 8, 2), "Apple Initiates Bankruptcy Proceedings For Its Unit"),
        news("AAPL", ny(TUE, 8, 3), "Apple Maintains Lead In Smartphones"),
        # No slot and no name: dropped as any other news.
        news("AAPL", ny(TUE, 8, 4), "Regulators Reiterate Concerns, Analyst Downgrade Looms"),
    ]
    story = one(rows, counters=c)
    assert [i.itype for i in story.items] == [
        "analyst_downgrade",
        "guidance_cut",
        "insolvency",
        "analyst_other",
    ]
    assert c["analyst_no_slot"] == 5 and c["entity"] == 1 and c["entity_analyst"] == 0
    assert story.card_at(story.close).structural is True  # the veto is no longer lost


def test_the_slot_forms_the_clause_rule_reads() -> None:
    intel, amd = ALIASES["INTC"], ALIASES["AMD"]
    on = "UPDATE: Morgan Stanley Maintains Overweight On Intel, Downgrades AMD To Equal-Weight"
    assert analyst_clause(on, intel) == "UPDATE: Morgan Stanley Maintains Overweight On Intel"
    assert analyst_clause(on, amd) == "Downgrades AMD To Equal-Weight"
    both = "Jefferies Initiates Intel With Buy; Initiates AMD With A Hold"
    assert analyst_clause(both, intel) == "Jefferies Initiates Intel With Buy"
    assert analyst_clause(both, amd) == "Initiates AMD With A Hold"
    for no_slot in (
        "Raymond James Initiates Coverage With Underperform Rating On Intel",
        "Intel Initiates Phase 3 Trial Of A Chip In Patients With Severe Boredom",
        "Intel Initiates $10B Buyback",
        "Apple Maintains Lead In Smartphones",
    ):
        assert not has_analyst_slot(no_slot), no_slot
    built = by_id(build([news("INTC", ny(TUE, 8), on), news("AMD", ny(TUE, 8), on)], ALIASES))
    assert built["INTC:2024-05-07"].items[0].itype == "analyst_other"
    assert built["AMD:2024-05-07"].items[0].itype == "analyst_downgrade"


# ── time (spec §A.2 step 4, §A.3) ─────────────────────────────


def test_federal_holidays_follow_the_observed_rules() -> None:
    assert federal_holidays(2023) == {
        date(2023, 1, 2),  # New Year's Day, a Sunday
        date(2023, 1, 16),
        date(2023, 2, 20),
        date(2023, 5, 29),
        date(2023, 6, 19),
        date(2023, 7, 4),
        date(2023, 9, 4),
        date(2023, 10, 9),  # Columbus Day
        date(2023, 11, 10),  # Veterans Day, a Saturday
        date(2023, 11, 23),
        date(2023, 12, 25),
    }
    assert date(2020, 6, 19) not in federal_holidays(2020)  # Juneteenth from 2021
    assert date(2021, 6, 18) in federal_holidays(2021)  # its first, a Saturday
    assert date(2021, 12, 31) in federal_holidays(2022)  # New Year's 2022, a Saturday


@pytest.mark.parametrize(
    ("day", "open_"),
    [
        (date(2024, 5, 7), True),
        (date(2024, 5, 11), False),  # Saturday
        (date(2016, 10, 10), False),  # Columbus Day: markets open, EDGAR closed
        (date(2016, 11, 11), False),  # Veterans Day
        (date(2021, 6, 18), False),  # Juneteenth's first observance
        (date(2020, 6, 19), True),
        (date(2021, 12, 31), False),  # New Year's Day 2022 observed
        (date(2018, 12, 5), False),  # day of mourning
        (date(2019, 12, 24), False),  # executive order
        (date(2017, 1, 20), True),  # inauguration: EDGAR open
    ],
)
def test_edgar_business_days(day: date, open_: bool) -> None:
    assert edgar_business_day(day) is open_


@pytest.mark.parametrize(
    ("accepted", "public"),
    [
        (ny(TUE, 6), ny(TUE, 6)),
        (ny(TUE, 17, 29, 59), ny(TUE, 17, 29, 59)),
        (ny(TUE, 17, 30), ny(WED, 6)),
        (ny(TUE, 5, 59), ny(TUE, 6)),  # before 06:00: that day's 06:00
        (ny(FRI, 18), ny(NEXT_MON, 6)),
        (ny(date(2024, 5, 11), 10), ny(NEXT_MON, 6)),  # a Saturday
        # After 17:30 before Columbus Day: Tuesday, though the market is open Monday.
        (ny(date(2023, 10, 6), 18), ny(date(2023, 10, 10), 6)),
        (ny(date(2016, 10, 10), 10), ny(date(2016, 10, 11), 6)),  # accepted on Columbus Day
        (ny(date(2023, 11, 9), 18), ny(date(2023, 11, 13), 6)),  # Veterans Day, observed Fri
        (ny(date(2021, 6, 17), 18), ny(date(2021, 6, 21), 6)),  # Juneteenth 2021
        (ny(date(2020, 6, 18), 18), ny(date(2020, 6, 19), 6)),  # not yet a holiday
        (ny(date(2021, 12, 30), 18), ny(date(2022, 1, 3), 6)),  # New Year's observed 12-31
        (ny(date(2018, 12, 4), 18), ny(date(2018, 12, 6), 6)),  # day of mourning
        (ny(date(2019, 12, 23), 18), ny(date(2019, 12, 26), 6)),  # 24th closed, 25th holiday
    ],
)
def test_a_filing_is_public_when_edgar_disseminates_it(
    accepted: datetime, public: datetime
) -> None:
    assert filing_public_at(accepted) == public
    assert filing_public_at(accepted).tzinfo == UTC


def test_times_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError):
        filing_public_at(datetime(2024, 5, 7, 12))
    with pytest.raises(ValueError):
        reaction_session(datetime(2024, 5, 7, 12))


@pytest.mark.parametrize(
    ("available", "session"),
    [
        (ny(TUE, 7), TUE),  # pre-open: that session
        (ny(TUE, 14, 30), TUE),  # exactly 90 minutes before the close
        (ny(TUE, 14, 30, 1), WED),
        (ny(TUE, 20), WED),
        (ny(FRI, 15), NEXT_MON),
        (ny(date(2024, 5, 11), 9), NEXT_MON),
        (ny(date(2024, 7, 3), 12), date(2024, 7, 5)),  # early close 13:00: cutoff 11:30
        (ny(date(2024, 7, 4), 10), date(2024, 7, 5)),  # a holiday
        (ny(date(2023, 11, 24), 11, 30), date(2023, 11, 24)),
        (ny(date(2023, 11, 24), 11, 30, 1), date(2023, 11, 27)),
    ],
)
def test_the_reaction_session(available: datetime, session: date) -> None:
    assert reaction_session(available) == session


def utc(y: int, mo: int, d: int, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    return datetime(y, mo, d, hh, mm, ss, tzinfo=UTC)


# The clocks change on Sunday 2024-03-10 (EST -> EDT) and Sunday 2024-11-03 (back).
@pytest.mark.parametrize(
    ("available", "session"),
    [
        (utc(2024, 3, 8, 19, 30), date(2024, 3, 8)),  # Friday 14:30 EST: the cutoff
        (utc(2024, 3, 8, 19, 30, 1), date(2024, 3, 11)),
        (utc(2024, 3, 11, 18, 30), date(2024, 3, 11)),  # Monday 14:30 EDT: an hour earlier
        (utc(2024, 3, 11, 18, 30, 1), date(2024, 3, 12)),
        (utc(2024, 3, 11, 19, 30), date(2024, 3, 12)),  # the winter cutoff is too late now
        (utc(2024, 3, 10, 7, 30), date(2024, 3, 11)),  # 03:30 EDT, just after the change
        (utc(2024, 11, 1, 18, 30), date(2024, 11, 1)),  # Friday 14:30 EDT
        (utc(2024, 11, 1, 19, 30), date(2024, 11, 4)),
        (utc(2024, 11, 4, 19, 30), date(2024, 11, 4)),  # Monday 14:30 EST
        (utc(2024, 11, 4, 19, 30, 1), date(2024, 11, 5)),
        (utc(2024, 11, 3, 5, 30), date(2024, 11, 4)),  # 01:30 EDT, the first of two
        (utc(2024, 11, 3, 6, 30), date(2024, 11, 4)),  # 01:30 EST, the second
    ],
)
def test_the_reaction_session_across_a_clock_change(available: datetime, session: date) -> None:
    assert reaction_session(available) == session


@pytest.mark.parametrize(
    ("accepted", "public"),
    [
        # Friday after 17:30 EST: Monday 06:00 EDT, which is 10:00 UTC, not 11:00.
        (utc(2024, 3, 8, 23), utc(2024, 3, 11, 10)),
        (utc(2024, 3, 10, 7, 30), utc(2024, 3, 11, 10)),  # Sunday, after the change
        (utc(2024, 3, 11, 9, 59), utc(2024, 3, 11, 10)),  # Monday 05:59 EDT
        (utc(2024, 3, 11, 21, 29), utc(2024, 3, 11, 21, 29)),  # Monday 17:29 EDT
        (utc(2024, 3, 11, 21, 30), utc(2024, 3, 12, 10)),
        # Friday after 17:30 EDT: Monday 06:00 EST, 11:00 UTC.
        (utc(2024, 11, 1, 22), utc(2024, 11, 4, 11)),
        (utc(2024, 11, 3, 6, 30), utc(2024, 11, 4, 11)),  # the repeated 01:30
        (utc(2024, 11, 4, 22, 29), utc(2024, 11, 4, 22, 29)),  # Monday 17:29 EST
    ],
)
def test_filings_become_public_across_a_clock_change(accepted: datetime, public: datetime) -> None:
    assert filing_public_at(accepted) == public


def test_a_story_across_a_clock_change() -> None:
    rows = [
        news("AAPL", utc(2024, 3, 9, 17), "Apple Unveils New Mac"),  # Saturday, EST
        news("AAPL", utc(2024, 3, 10, 16), "Apple Unveils New iPad"),  # Sunday, EDT
        filing("AAPL", utc(2024, 3, 8, 23), ("8.01",)),  # Friday evening: Monday 06:00 EDT
        news("AAPL", utc(2024, 3, 11, 13, 25), DOWNGRADE),  # 09:25 EDT: before the open
        news("INTC", utc(2024, 3, 11, 13, 30), "Barclays Downgrades Intel"),  # 09:30 EDT
    ]
    built = by_id(build(rows, ALIASES))
    apple, intel = built["AAPL:2024-03-11"], built["INTC:2024-03-11"]
    assert len(apple.items) == 4 and set(built) == {"AAPL:2024-03-11", "INTC:2024-03-11"}
    assert [i.at for i in apple.items] == [
        utc(2024, 3, 9, 17),
        utc(2024, 3, 10, 16),
        utc(2024, 3, 11, 10),  # the filing: Monday 06:00 EDT
        utc(2024, 3, 11, 13, 25),
    ]
    assert session_bounds(date(2024, 3, 11)) == (utc(2024, 3, 11, 13, 30), utc(2024, 3, 11, 20))
    assert apple.start_case() == "out" and intel.start_case() == "in"
    assert apple.close == utc(2024, 3, 11, 20) and intel.nsn_at(intel.close) is not None


def test_items_are_usable_ten_minutes_after_they_are_public() -> None:
    built = by_id(
        build(
            [
                news("AAPL", ny(TUE, 14, 20), DOWNGRADE),  # usable 14:30: today
                news("INTC", ny(TUE, 14, 20, 1), "Barclays Downgrades Intel"),  # 14:30:01
                filing("AMD", ny(TUE, 14, 25), ("8.01",)),  # usable 14:35: tomorrow
            ],
            ALIASES,
        )
    )
    assert set(built) == {"AAPL:2024-05-07", "INTC:2024-05-08", "AMD:2024-05-08"}
    item = built["AMD:2024-05-08"].items[0]
    assert item.at == ny(TUE, 14, 25) and item.available_at == item.at + NEWS_LAG
    assert NEWS_LAG == timedelta(seconds=600)


def test_an_8k_after_the_filing_cutoff_reacts_after_edgar_reopens() -> None:
    story = one([filing("AAPL", ny(date(2023, 10, 6), 18), ("2.02",))])
    assert story.session == date(2023, 10, 10)  # not Monday 10-09: EDGAR was closed
    assert story.items[0].at == ny(date(2023, 10, 10), 6)


# ── duplicates and corrections (spec §A.4) ────────────────────


def test_shingles_normalise_the_headline() -> None:
    assert shingles("UPDATE: Apple Q2 EPS $1.40") == frozenset({"apple q2 eps", "q2 eps #"})
    assert shingles("The Apple") == frozenset({"apple"})  # fewer than k tokens: one shingle
    assert shingles("") == frozenset()
    assert jaccard(frozenset(), frozenset()) == 0.0
    assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == pytest.approx(1 / 3)


def test_a_repackaged_item_is_a_duplicate_of_its_first_copy() -> None:
    a = news("AAPL", ny(TUE, 8), "Apple Recalls One Million iPhones Over Battery Fires")
    b = news("AAPL", ny(TUE, 8, 5), "UPDATE: Apple Recalls One Million iPhones Over Battery Fires")
    c = news(
        "AAPL", ny(TUE, 8, 9), "Apple Recalls One Million iPhones Over Battery Fires, Shares Dip"
    )
    d = news("AAPL", ny(TUE, 9), "Apple Unveils New Mac")
    f1, f2 = filing("AAPL", ny(TUE, 7), ("8.01",)), filing("AAPL", ny(TUE, 7, 1), ("8.01",))
    story = one([d, c, b, a, f2, f1])
    dups = {i.event_id: i.dup_of for i in story.items}
    assert dups[b.event_id] == a.event_id
    assert dups[c.event_id] == a.event_id  # the first copy, not the copy it matched
    assert dups[d.event_id] is None
    assert dups[f1.event_id] is None and dups[f2.event_id] is None  # filings have no text
    row = story_row(story)
    assert row["n_items"] == 6 and row["n_distinct"] == 4


def test_a_correction_replaces_the_facts_of_the_wire_it_corrects_from_its_own_time() -> None:
    wrong = news("AAPL", ny(TUE, 8), MISS)
    fixed = news("AAPL", ny(TUE, 9), "CORRECTION: Apple Q2 Adj. EPS $1.60 Beats $1.50 Estimate")
    other = news("AAPL", ny(TUE, 8, 30), "Apple Unveils New Mac")
    story = one([wrong, fixed, other])
    corr = next(i for i in story.items if i.event_id == fixed.event_id)
    assert corr.supersedes == (wrong.event_id,)  # shares (result, Q2, adj, eps); not the Mac
    before = story.card_at(ny(TUE, 9, 9, 59))
    assert before.type == "earnings_miss" and before.family == "NSN_CORE"
    after = story.card_at(ny(TUE, 9, 10))
    assert after.type == "earnings_beat" and after.family is None
    assert story_row(story)["flags"]["superseded"] == 1


def test_a_correction_also_supersedes_by_text_and_only_corrections_do() -> None:
    first = news("AAPL", ny(TUE, 8), "Apple Says Supplier Halted Production In China")
    corr = news("AAPL", ny(TUE, 9), "CORRECTED: Apple Says Supplier Halted Production In India")
    plain = news("AAPL", ny(TUE, 10), "UPDATE: Apple Says Supplier Halted Production In China")
    story = one([first, corr, plain])
    supersedes = {i.event_id: i.supersedes for i in story.items}
    assert jaccard(shingles(first.headline), shingles(corr.headline)) >= 0.5
    assert supersedes[corr.event_id] == (first.event_id,)
    assert supersedes[plain.event_id] == ()


def test_only_a_substantive_correction_supersedes() -> None:
    miss = news("AAPL", ny(TUE, 8), MISS)
    noise = news("AAPL", ny(TUE, 9), "CORRECTION: " + MISS + ": What To Expect")
    mover = news("AAPL", ny(TUE, 9, 30), "CORRECTION: Apple Shares Are Trading Lower After Miss")
    story = one([miss, noise, mover])
    assert [i.itype for i in story.items] == ["earnings_fact", "noise", "mover"]
    assert jaccard(shingles(miss.headline), shingles(noise.headline)) >= 0.5
    assert all(i.supersedes == () for i in story.items)
    assert story.card_at(story.close).type == "earnings_miss"


def test_nsn_is_re_read_at_every_item() -> None:
    # A cut guided below consensus vetoes the miss. Were a noise item to take the
    # guidance away (the builder never lets one), the card would turn NSN at it.
    guide = news("AAPL", ny(TUE, 7), "Apple Sees Q3 Sales $80.000B-$82.000B vs $85.000B Est")
    miss = news("AAPL", ny(TUE, 8), MISS)
    noise = news("AAPL", ny(TUE, 9), "Why Apple Shares Are Trading Lower")
    built = one([guide, miss, noise])
    assert built.nsn_at(built.close) is None
    g, m, n = built.items
    forced = Story(built.story_id, "AAPL", TUE, [g, m, replace(n, supersedes=(guide.event_id,))])
    assert forced.card_at(ny(TUE, 9, 9)).type == "guidance_cut"
    assert forced.card_at(ny(TUE, 9, 10)).family == "NSN_CORE"
    assert forced.nsn_at(forced.close) == ny(TUE, 9, 10)
    assert forced.at_news() == ny(TUE, 9)
    # Of items arriving together, the substantive one made it.
    dg = news("AAPL", ny(TUE, 10), DOWNGRADE)
    noisy = news("AAPL", ny(TUE, 10), "Why Apple Shares Are Trading Lower Today")
    together = one([noisy, dg])
    assert together.nsn_at(together.close) == ny(TUE, 10, 10)
    trigger = together._trigger()
    assert trigger is not None and trigger.itype == "analyst_downgrade"


def test_fact_keys_name_each_statement() -> None:
    assert fact_keys(parse_headline(MISS)) == {
        ("result", "Q2", "adj", "eps"),
        ("result", "Q2", "adj", "sales"),
    }
    guide = parse_headline("Apple Sees Q3 Sales $80.000B-$82.000B vs $85.000B Est")
    assert {k[0] for k in fact_keys(guide)} == {"guidance"}


# ── followers (spec §A.5) ─────────────────────────────────────


def test_a_story_of_reactive_items_follows_until_a_new_item_is_not_reactive() -> None:
    rows = [
        news("AAPL", ny(MON, 8), DOWNGRADE),
        news(
            "AAPL",
            ny(TUE, 8),
            "Goldman Sachs Maintains Neutral on Apple, Lowers Price Target to $140",
        ),
        news("AAPL", ny(TUE, 10), "Jefferies Downgrades Apple to Hold"),
    ]
    built = by_id(build(rows, ALIASES))
    parent, story = built["AAPL:2024-05-06"], built["AAPL:2024-05-07"]
    assert story.parent == parent.story_id and story.parent_type_close == "analyst_downgrade"
    assert story.follower_at(ny(TUE, 8, 10)) is True  # only a price-target cut so far
    assert story.follower_at(ny(TUE, 10, 9)) is True
    assert story.follower_at(ny(TUE, 10, 10)) is False  # the downgrade is not reactive
    assert story.card_at(ny(TUE, 8, 10)).follower is True
    # NSN is re-read at each item: it turns NSN when the downgrade arrives.
    assert story.nsn_at(ny(TUE, 10, 9)) is None
    assert story.nsn_at(ny(TUE, 15)) == ny(TUE, 10, 10)
    assert story.at_news() == ny(TUE, 10)
    assert story.start_case() == "in"
    assert story.card_at(ny(TUE, 8, 10)).family is None
    assert story_row(story)["follower_close"] is False


def test_a_story_after_a_structural_story_follows_it() -> None:
    rows = [
        news("AAPL", ny(MON, 8), "Apple Faces SEC Probe Into App Store Disclosures"),
        news("AAPL", ny(TUE, 8), "Jefferies Downgrades Apple to Hold"),
        news("AAPL", ny(WED, 8), "UBS Downgrades Apple to Sell"),
    ]
    built = by_id(build(rows, ALIASES))
    assert built["AAPL:2024-05-06"].type_close() == "fraud_probe"
    tue, wed = built["AAPL:2024-05-07"], built["AAPL:2024-05-08"]
    assert tue.parent_type_close == "fraud_probe" and tue.follower_at(tue.close)
    assert tue.nsn_at(tue.close) is None
    # A follower is nobody's parent: Wednesday's is still Monday's structural story.
    assert wed.parent == "AAPL:2024-05-06" and wed.nsn_at(wed.close) is None


def test_a_story_whose_first_item_reruns_a_parent_item_follows() -> None:
    rows = [
        news("AAPL", ny(MON, 8), "Morgan Stanley Downgrades Apple to Equal-Weight on China Demand"),
        news(
            "AAPL",
            ny(TUE, 8),
            "Morgan Stanley Downgrades Apple to Equal-Weight on China Demand Worry",
        ),
    ]
    story = by_id(build(rows, ALIASES))["AAPL:2024-05-07"]
    assert story.follower_at(story.close) is True
    assert story.nsn_at(story.close) is None


def test_the_parent_is_the_latest_non_follower_story_in_the_three_sessions_before() -> None:
    rows = [
        news("AAPL", ny(MON, 8), DOWNGRADE),
        news(
            "AAPL",
            ny(TUE, 8),
            "Goldman Sachs Maintains Neutral on Apple, Lowers Price Target to $140",
        ),
        news("AAPL", ny(WED, 8), "Why Apple Shares Are Trading Lower"),  # noise only
        news("AAPL", ny(THU, 8), "Apple Unveils New Mac"),
        news("AAPL", ny(NEXT_MON, 8), "Apple Unveils New Watch"),
    ]
    built = by_id(build(rows, ALIASES))
    assert built["AAPL:2024-05-07"].parent == "AAPL:2024-05-06"
    assert (
        built["AAPL:2024-05-08"].noise_only
        and built["AAPL:2024-05-08"].type_close() == "noise_only"
    )
    # Tuesday follows, Wednesday is noise: Thursday's parent is Monday's.
    assert built["AAPL:2024-05-09"].parent == "AAPL:2024-05-06"
    # Next Monday: Fri, Thu, Wed are the three sessions before; Thursday's is the parent.
    assert built["AAPL:2024-05-13"].parent == "AAPL:2024-05-09"


def test_a_story_four_sessions_later_has_no_parent() -> None:
    rows = [
        news("AAPL", ny(MON, 8), DOWNGRADE),
        news("AAPL", ny(FRI, 8), "Jefferies Downgrades Apple"),
    ]
    story = by_id(build(rows, ALIASES))["AAPL:2024-05-10"]
    assert story.parent is None and story.follower_at(story.close) is False
    assert story.nsn_at(story.close) == ny(FRI, 8, 10)


def test_history_gives_the_same_parents_as_one_build() -> None:
    rows = [
        news("AAPL", ny(MON, 8), "Apple Faces SEC Probe Into App Store Disclosures"),
        news("AAPL", ny(TUE, 8), "Jefferies Downgrades Apple to Hold"),
    ]
    together = by_id(build(rows, ALIASES))
    first = build(rows[:1], ALIASES)
    later = build(rows[1:], ALIASES, history={"AAPL": first})
    assert story_row(later[0]) == story_row(together["AAPL:2024-05-07"])


# ── labels in time (spec §A.6, §B.7) ──────────────────────────


def test_no_item_after_t_changes_the_card_at_t() -> None:
    rows = [
        news("AAPL", ny(MON, 8), "Apple Unveils New Mac"),
        news("AAPL", ny(TUE, 6), "Why Apple Shares Are Trading Lower"),
        news(
            "AAPL",
            ny(TUE, 7),
            "Goldman Sachs Maintains Neutral on Apple, Lowers Price Target to $140",
        ),
        news("AAPL", ny(TUE, 8), DOWNGRADE),
        news("AAPL", ny(TUE, 8, 20), MISS),
        filing("AAPL", ny(TUE, 9), ("2.02", "9.01")),
        news("AAPL", ny(TUE, 11), "Apple Faces SEC Probe Into App Store Disclosures"),
        news("AAPL", ny(TUE, 12), "UBS Upgrades Apple to Buy"),
    ]
    full = by_id(build(rows, ALIASES))["AAPL:2024-05-07"]
    times = sorted({i.available_at for i in full.items})
    probes = [t + d for t in times for d in (-timedelta(seconds=1), timedelta(0))]
    for t in probes:
        known = [r for r in rows if _available(r) <= t]
        truncated = by_id(build(known, ALIASES)).get("AAPL:2024-05-07")
        expected = truncated.card_at(t) if truncated is not None else resolve([], t, follower=False)
        assert full.card_at(t) == expected, t
        assert full.follower_at(t) == (truncated.follower_at(t) if truncated else False)
    assert full.card_at(ny(TUE, 11, 9)).structural is False
    assert full.card_at(ny(TUE, 11, 10)).structural is True


_POOL = {
    "AAPL": [
        DOWNGRADE,
        "Jefferies Downgrades Apple to Hold",
        "Goldman Sachs Maintains Neutral on Apple, Lowers Price Target to $140",
        "UBS Upgrades Apple to Buy",
        MISS,
        "Apple Q2 Adj. EPS $1.60 Beats $1.50 Estimate",
        "CORRECTION: Apple Q2 Adj. EPS $1.60 Beats $1.50 Estimate",
        "Apple Sees Q3 Sales $80.000B-$82.000B vs $85.000B Est",
        "Apple Faces SEC Probe Into App Store Disclosures",
        "Apple Unveils New Mac",
        "Why Apple Shares Are Trading Lower",
        "Apple Earnings Preview: What To Expect",
        "Rosen Law Firm Investigates Apple",
        "Samsung Unveils New Phone",  # fails the entity check
    ],
    "INTC": [
        "Barclays Downgrades Intel",
        "Barclays Downgrades Intel, Upgrades AMD",
        "Citi Maintains Buy on Intel, Lowers Price Target to $40",
        "Intel Q1 EPS $0.10 Misses $0.20 Estimate",
        "Intel Shares Are Trading Lower",
        "Intel Unveils New Chip",
    ],
}


def _random_items(rng: random.Random) -> list[RawItem]:
    """News and 8-Ks of two symbols over two weeks, at any hour (weekends too)."""
    out: list[RawItem] = []
    first = date(2024, 5, 3)  # a Friday: S-3..S-1 reach back over a weekend
    for _ in range(rng.randint(8, 18)):
        symbol = rng.choice(["AAPL", "INTC"])
        at = datetime.combine(first + timedelta(days=rng.randrange(12)), time(), MARKET_TZ)
        at += timedelta(minutes=rng.randrange(24 * 60))
        if rng.random() < 0.15:
            items = rng.choice([("2.02", "9.01"), ("8.01",), ("4.02",), ("5.02",)])
            out.append(filing(symbol, at, items))
        else:
            out.append(news(symbol, at, rng.choice(_POOL[symbol])))
    return out


def test_no_item_after_t_changes_any_story_known_at_t() -> None:
    # Many sessions, parents and followers: delete every item after T and rebuild.
    rng = random.Random(20261010)
    seen: Counter[str] = Counter()
    for _ in range(40):
        rows = _random_items(rng)
        full = build(rows, ALIASES)
        times = sorted({i.available_at for s in full for i in s.items})
        probes = [t + d for t in times for d in (-timedelta(seconds=1), timedelta(0))]
        for t in rng.sample(probes, min(10, len(probes))):
            truncated = by_id(build([r for r in rows if _available(r) <= t], ALIASES))
            started = {s.story_id: s for s in full if s.items[0].available_at <= t}
            assert set(truncated) == set(started), t
            for sid, story in started.items():
                cut = truncated[sid]
                assert (cut.parent, cut.parent_type_close) == (
                    story.parent,
                    story.parent_type_close,
                ), (sid, t)
                assert cut.card_at(t) == story.card_at(t), (sid, t)
                assert cut.follower_at(t) == story.follower_at(t), (sid, t)
                assert cut.nsn_at(t) == story.nsn_at(t), (sid, t)
                if story.nsn_at(t) is not None:
                    assert cut.at_news() == story.at_news(), (sid, t)
                seen["parent"] += story.parent is not None
                seen["follower"] += story.follower_at(t)
                seen["nsn"] += story.nsn_at(t) is not None
                seen["structural"] += story.card_at(t).structural
    assert min(seen[k] for k in ("parent", "follower", "nsn", "structural")) >= 20, seen


def _available(raw: RawItem) -> datetime:
    at: datetime = raw.published_at if raw.kind == "news" else filing_public_at(raw.published_at)
    return at + NEWS_LAG


def test_the_card_is_taxonomy_resolve_over_the_known_items() -> None:
    story = one([news("AAPL", ny(TUE, 8), DOWNGRADE), news("AAPL", ny(TUE, 9), MISS)])
    t = ny(TUE, 8, 30)
    item: ItemLike = story.items[0]
    assert story.card_at(t) == resolve([item], t, follower=False)
    assert isinstance(story.card_at(t), StoryCard)


def test_a_story_led_by_a_mover_never_turns_nsn() -> None:
    story = one(
        [
            news("AAPL", ny(TUE, 8), "Why Apple Shares Are Trading Lower"),
            news("AAPL", ny(TUE, 9), DOWNGRADE),
        ]
    )
    assert story.card_at(story.close).type == "analyst_downgrade"
    assert story.nsn_at(story.close) is None and story.at_news() is None
    assert story.detect_at() == ny(TUE, 9, 10)  # a mover never triggers a story
    assert story.type_detect() == "analyst_downgrade"


def test_noise_neither_triggers_nor_types_a_story() -> None:
    story = one(
        [
            news("AAPL", ny(TUE, 8), "Apple Earnings Preview: What To Expect"),
            news("AAPL", ny(TUE, 8, 30), "Rosen Law Firm Investigates Apple"),
            news("AAPL", ny(TUE, 9), "Apple Shares Are Trading Lower"),
        ]
    )
    assert [i.itype for i in story.items] == ["noise", "law_firm", "mover"]
    row = story_row(story)
    assert row["type_close"] == row["type_detect"] == "noise_only"
    assert row["detect_at"] is None and row["nsn_at"] is None and row["start_case"] == "out"
    assert row["flags"]["noise_items"] == 3


def test_nsn_detection_stops_at_its_cutoff_and_records_the_news_time() -> None:
    story = one([news("AAPL", ny(TUE, 7, 50), DOWNGRADE)])
    assert story.nsn_at(ny(TUE, 7, 59, 59)) is None
    assert story.nsn_at(ny(TUE, 8)) == ny(TUE, 8)
    assert story.at_news() == ny(TUE, 7, 50)
    assert story.start_case() == "out"  # pre-open news
    in_session = one([news("AAPL", ny(TUE, 9, 30), DOWNGRADE)])
    assert in_session.start_case() == "in"
    evening = one([news("AAPL", ny(MON, 18), DOWNGRADE)])
    assert evening.session == TUE and evening.start_case() == "out"


def test_a_story_is_a_story_view_for_the_simulator() -> None:
    story = one(
        [news("AAPL", ny(TUE, 8), DOWNGRADE), news("AAPL", ny(TUE, 9), "Apple Unveils Mac")]
    )
    view: StoryView = story
    assert view.session == TUE and view.symbol == "AAPL" and view.story_id == "AAPL:2024-05-07"
    assert list(view.news_times()) == [ny(TUE, 8, 10), ny(TUE, 9, 10)]
    assert view.card_at(ny(TUE, 8, 10)).family == "NSN_CORE"


def test_items_out_of_order_are_refused() -> None:
    story = one(
        [news("AAPL", ny(TUE, 8), DOWNGRADE), news("AAPL", ny(TUE, 9), "Apple Unveils Mac")]
    )
    with pytest.raises(ValueError):
        Story("AAPL:2024-05-07", "AAPL", TUE, list(reversed(story.items)))


# ── rows and determinism (spec §A.7) ──────────────────────────


def test_the_row_holds_every_column() -> None:
    rows = [news("AAPL", ny(MON, 8), "Apple Unveils New Mac"), news("AAPL", ny(TUE, 8), DOWNGRADE)]
    story = by_id(build(rows, ALIASES))["AAPL:2024-05-07"]
    row = story_row(story)
    assert row == {
        "builder_version": BUILDER_VERSION,
        "story_id": "AAPL:2024-05-07",
        "symbol": "AAPL",
        "session": TUE,
        "start_case": "out",
        "detect_at": ny(TUE, 8, 10),
        "nsn_at": ny(TUE, 8, 10),
        "at_news": ny(TUE, 8),
        "type_detect": "analyst_downgrade",
        "type_close": "analyst_downgrade",
        "family_ever": "NSN_CORE",
        "follower_close": False,
        "parent": "AAPL:2024-05-06",
        "n_items": 1,
        "n_distinct": 1,
        "items": [
            {
                "event_id": rows[1].event_id,
                "at": "2024-05-07T12:00:00+00:00",
                "available_at": "2024-05-07T12:10:00+00:00",
                "itype": "analyst_downgrade",
                "dup_of": None,
                "supersedes": [],
                "entity_ok": True,
            }
        ],
        "flags": {
            "nsym_unknown": 0,
            "corrections": 0,
            "superseded": 0,
            "noise_items": 0,
            "parent_type_close": "product",
        },
    }


def test_the_build_does_not_depend_on_the_order_of_its_items() -> None:
    rows = [
        news("AAPL", ny(MON, 8), DOWNGRADE),
        news(
            "AAPL",
            ny(TUE, 8),
            "Goldman Sachs Maintains Neutral on Apple, Lowers Price Target to $140",
        ),
        news(
            "AAPL",
            ny(TUE, 8),
            "UPDATE: Goldman Sachs Maintains Neutral on Apple, Lowers Price Target",
        ),
        news("AAPL", ny(TUE, 10), MISS),
        news("INTC", ny(TUE, 9), "Barclays Downgrades Intel, Upgrades AMD"),
        news("AMD", ny(TUE, 9), "Barclays Downgrades Intel, Upgrades AMD"),
        filing("AAPL", ny(TUE, 18), ("2.02",)),
    ]
    expected = [story_row(s) for s in build(rows, ALIASES)]
    for seed in range(5):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        assert [story_row(s) for s in build(shuffled, ALIASES)] == expected


# ── counts and pins ───────────────────────────────────────────


def test_windows_and_the_eligibility_time() -> None:
    assert window_of(date(2016, 10, 3)) == "train" and window_of(date(2021, 12, 31)) == "train"
    assert window_of(date(2022, 1, 3)) == "validation"
    assert window_of(date(2024, 12, 31)) == "validation" and window_of(date(2025, 1, 2)) == "other"
    open_ = session_bounds(TUE)[0]
    assert eligibility_at(ny(TUE, 8, 10), ny(TUE, 8), ny(TUE, 7, 10), TUE) == ny(TUE, 8)
    assert eligibility_at(None, None, ny(TUE, 7, 10), TUE) == ny(TUE, 7)
    assert eligibility_at(None, None, None, TUE) == open_


def test_the_counts_table_sums_years_and_windows() -> None:
    counts = StoryCounts()
    counts.add(
        session=date(2021, 3, 1), type_close="earnings_miss", nsn=True, universes=["all", "primary"]
    )
    counts.add(session=date(2022, 3, 1), type_close="earnings_miss", nsn=False, universes=["all"])
    counts.add(session=date(2022, 3, 2), type_close="noise_only", nsn=False, universes=["all"])
    counts.reasons[("all", "ok")] += 1
    lines = counts_table(counts, start=date(2021, 1, 4), end=date(2022, 12, 30))
    assert lines[1] == "== all stories =="
    assert lines[2].split() == ["type", "2021", "2022", "train", "valid", "total"]
    assert lines[3].split() == ["earnings_miss", "1", "1", "1", "1", "2"]
    assert lines[4].split() == ["noise_only", "0", "1", "0", "1", "1"]
    assert lines[5].split()[-5:] == ["1", "0", "1", "0", "1"]  # NSN_CORE
    assert lines[6].split() == ["ALL", "1", "2", "1", "2", "3"]
    primary = lines.index(next(x for x in lines if x.startswith("== PRIMARY")))
    assert lines[primary + 2].split() == ["earnings_miss", "1", "0", "1", "0", "1"]
    assert lines[-2] == "eligibility (every story): ok 1"


def test_the_builder_pins_its_own_constants() -> None:
    assert len(stories.STORIES_SHA) == 12 and len(stories.HEADLINE_PATTERNS_SHA) == 12
    assert stories.STORIES_SHA == stories._sha(stories.sources())
    assert stories.STORIES_SHA != stories._sha(stories.sources() | {"DUP_JACCARD": "0.61"})
    assert {"EDGAR_CLOSURES", "STOPWORDS", "CLAUSE_SPLIT", "NEWS_LAG_S"} <= stories.sources().keys()


async def test_pins_gather_every_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    async def alias_sha(engine: object) -> str:
        return "aliases12345"

    monkeypatch.setattr(stories, "alias_sha", alias_sha)
    pinned = await stories.pins(object())  # type: ignore[arg-type]
    assert pinned == {
        "builder_version": "stories-v1",
        "stories_sha": stories.STORIES_SHA,
        "alias_sha": "aliases12345",
        "renames_sha": renames.renames_sha(),
        "taxonomy_sha": TAXONOMY_SHA,
        "parser_sha": PARSER_SHA,
        "headline_patterns_sha": stories.HEADLINE_PATTERNS_SHA,
    }
    # The parser's pin is read when asked, as the extractor is.
    monkeypatch.setattr(earnings_parse, "PARSER_SHA", "parser123456")
    assert (await stories.pins(object()))["parser_sha"] == "parser123456"  # type: ignore[arg-type]


async def test_a_builds_inputs_hash_every_pin_and_the_extractor_read_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def returning(value: str) -> object:
        async def sha(engine: object) -> str:
            return value

        return sha

    monkeypatch.setattr(stories, "alias_sha", returning("aliases12345"))
    engine: object = object()
    inputs = await stories.inputs_sha(engine)  # type: ignore[arg-type]
    pinned = await stories.pins(engine)  # type: ignore[arg-type]
    assert inputs == stories._sha(pinned | {"extractor": earnings_parse.EXTRACTOR})
    assert len(inputs) == 12
    seen = {inputs}
    for module, name, value in [
        (stories, "STORIES_SHA", "rules1234567"),
        (stories, "TAXONOMY_SHA", "taxonomy1234"),
        (stories, "HEADLINE_PATTERNS_SHA", "patterns1234"),
        (earnings_parse, "PARSER_SHA", "parser123456"),
        (earnings_parse, "EXTRACTOR", "benzinga-earnings-next"),
        (renames, "renames_sha", lambda: "renames12345"),
        (stories, "alias_sha", returning("aliases99999")),
    ]:
        monkeypatch.setattr(module, name, value)
        seen.add(await stories.inputs_sha(engine))  # type: ignore[arg-type]
    assert len(seen) == 8  # each change gives another hash
