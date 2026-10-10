"""The story builder against the database: loading items, building a range,
persisting, and counting.

The renamed tickers are the two of ``small_map`` (OLDA, then OLDB, of NEWA),
with every month of their news marked fetched where a build needs it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, date, datetime
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events import earnings_parse, stories
from halal_trader.events.aliases import AliasMatcher, load_aliases
from halal_trader.events.context import PitContext
from halal_trader.events.earnings_parse import (
    EXTRACTOR,
    EXTRACTOR_V3,
    EarningsFacts,
    parse_headline,
)
from halal_trader.events.history import write_times
from halal_trader.events.stories import (
    RawItem,
    StoriesNotReady,
    build,
    build_range,
    build_unit,
    built_ranges,
    count_stories,
    inputs_sha,
    load_items,
    missing_facts,
    persist,
    story_row,
    untimed_filings,
)
from tests._renames import mark_renamed_news_done
from tests._stories import (
    ALIAS_ROWS,
    DOWNGRADE,
    FRI,
    MISS,
    MON,
    THU,
    TUE,
    WED,
    WEEK,
    add_aliases,
    fake_context,
    filing_row,
    headline_row,
    mark_built,
    news_row,
    ny,
    seed_week,
    store,
    stored_rows,
    time_filings,
)
from tests.test_event_context import NEXT, PREV, S, _world, et

APPLE = {"AAPL": AliasMatcher("AAPL", ("Apple",), ("AAPL",))}


async def collect(engine: AsyncEngine, **kwargs: Any) -> list[RawItem]:
    return [item async for item in load_items(engine, **kwargs)]


def raw_news(n: int, at: datetime, headline: str) -> RawItem:
    facts = tuple(parse_headline(headline))
    return RawItem(n, f"alpaca:{n}", "news", "AAPL", at.astimezone(UTC), at, headline, 1, (), facts)


@pytest.fixture
async def ready(engine: AsyncEngine, small_map: None) -> AsyncEngine:
    await seed_week(engine)
    return engine


# ── loading ───────────────────────────────────────────────────


async def test_items_come_with_their_v4_facts_and_filing_items(engine: AsyncEngine) -> None:
    rows = [
        news_row(1, "AAPL", ny(TUE, 8), MISS),
        headline_row(2, "AAPL", ny(TUE, 9), "Apple Unveils Mac", ["AAPL", "MSFT", "GOOG", "AMZN"]),
        headline_row(3, "AAPL", ny(TUE, 10), "Apple Unveils Watch"),  # a live row: no symbols
        filing_row("acc-1", "AAPL", ny(TUE, 16, 5), ["2.02", "9.01"]),
        filing_row("acc-2", "AAPL", ny(TUE, 16, 6), ["5.02"], kind="8-k/a"),
        filing_row("acc-3", "AAPL", ny(TUE, 16, 7), [], kind="10-q"),
        news_row(4, "AAPL", ny(MON, 23, 59), "Apple Unveils iPad"),  # the day before
        news_row(5, "AAPL", ny(WED, 0, 0), "Apple Unveils Vision"),  # the day after
    ]
    ids = await store(engine, rows)  # 'none' rows for the headlines without facts
    async with engine.begin() as conn:  # other extractors' rows are not read
        await conn.execute(
            text(
                "INSERT INTO event_facts (event_id, extractor, kind, fields) VALUES "
                "(:e, :v3, 'result', '{\"eps\": 9}')"
            ),
            {"e": ids["alpaca:1"], "v3": EXTRACTOR_V3},
        )
    items = await collect(engine, start=TUE, end=TUE, symbols=["AAPL"])
    assert [i.source_id for i in items] == ["alpaca:1", "alpaca:2", "alpaca:3", "acc-1", "acc-2"]
    miss, roundup, live, f1, f2 = items
    assert miss.facts == tuple(parse_headline(MISS)) and miss.n_symbols == 1
    assert miss.published_at == ny(TUE, 8) and miss.headline == MISS
    assert roundup.n_symbols == 4 and roundup.facts == ()
    assert live.n_symbols is None
    assert (f1.kind, f1.items_8k, f1.headline, f1.facts) == ("8-k", ("2.02", "9.01"), "", ())
    assert (f2.kind, f2.items_8k) == ("8-k/a", ("5.02",))


async def test_items_come_with_the_facts_of_the_extractor_current_when_read(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = await store(engine, [news_row(1, "AAPL", ny(TUE, 8), MISS)])
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO event_facts (event_id, extractor, kind, fields) "
                "VALUES (:e, 'benzinga-earnings-next', 'result', '{\"eps\": 9}')"
            ),
            {"e": ids["alpaca:1"]},
        )
    monkeypatch.setattr(earnings_parse, "EXTRACTOR", "benzinga-earnings-next")
    (item,) = await collect(engine, start=TUE, end=TUE, symbols=["AAPL"])
    assert item.facts == (EarningsFacts("result", {"eps": 9}),)
    (pinned,) = await collect(engine, start=TUE, end=TUE, symbols=["AAPL"], extractor=EXTRACTOR)
    assert pinned.facts == tuple(parse_headline(MISS))


@pytest.mark.usefixtures("small_map")
async def test_news_rows_that_are_not_their_symbols_own_are_dropped(engine: AsyncEngine) -> None:
    # OLDA named NEWA to 2016-02-10 (news window to 02-18); NEWA's own ticker from 03-05.
    day = date(2016, 1, 15)
    rows = [
        headline_row(1, "OLDA", ny(day, 10), "Old Co Unveils A", ["OLDA"]),  # copied under NEWA
        headline_row(1, "NEWA", ny(day, 10), "Old Co Unveils A", ["OLDA"]),  # the copy: kept
        headline_row(2, "NEWA", ny(day, 11), "Someone Else Unveils B", ["NEWA"]),  # before NEWA's
        headline_row(3, "NEWA", ny(date(2016, 3, 10), 11), "New Co Unveils C", ["NEWA"]),
        headline_row(4, "NEWA", ny(date(2016, 3, 11), 11), "New Co Unveils D"),  # live: kept
        filing_row("acc-1", "OLDA", ny(day, 12), ["8.01"]),  # filings are kept as they are
    ]
    await store(engine, rows, facts=False)
    c: Counter[str] = Counter()
    items = await collect(
        engine, start=date(2016, 1, 1), end=date(2016, 3, 31), symbols=["OLDA", "NEWA"], counters=c
    )
    assert [(i.symbol, i.source_id) for i in items] == [
        ("NEWA", "alpaca:1"),
        ("NEWA", "alpaca:3"),
        ("NEWA", "alpaca:4"),
        ("OLDA", "acc-1"),
    ]
    assert c["owner"] == 2


# ── building a range ──────────────────────────────────────────


async def test_a_build_waits_for_the_renamed_news_the_aliases_and_the_facts(
    engine: AsyncEngine, small_map: None
) -> None:
    ids = await store(engine, WEEK, facts=False)
    with pytest.raises(StoriesNotReady, match="renamed-ticker news"):
        await build_range(engine, start=MON, end=FRI)
    await mark_renamed_news_done(engine)
    with pytest.raises(StoriesNotReady, match="no story aliases"):
        await build_range(engine, start=MON, end=FRI)
    await add_aliases(engine, ALIAS_ROWS)
    with pytest.raises(StoriesNotReady, match=f"9 news event.*no {re.escape(EXTRACTOR)} facts row"):
        await build_range(engine, start=MON, end=FRI)
    first = min(i for s, i in ids.items() if s.startswith("alpaca:"))
    assert await missing_facts(engine, start=date(2016, 1, 1), end=FRI) == (9, first)
    assert await build_range(engine, start=MON, end=FRI, force=True) > 0
    assert await built_ranges(engine) == []  # a forced build is never marked complete
    await earnings_parse.extract_all(engine)
    assert await missing_facts(engine, start=date(2016, 1, 1), end=FRI) == (0, None)
    with pytest.raises(StoriesNotReady, match="1 8-K event.*not at their EDGAR header's time"):
        await build_range(engine, start=MON, end=FRI)
    await time_filings(engine)
    assert await build_range(engine, start=MON, end=FRI) > 0
    assert await built_ranges(engine) == [(MON, FRI)]
    with pytest.raises(ValueError):
        await build_range(engine, start=FRI, end=MON)


async def test_the_facts_a_build_waits_for_are_the_current_extractors(
    ready: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert await build_range(ready, start=MON, end=FRI) == 7
    monkeypatch.setattr(earnings_parse, "EXTRACTOR", "benzinga-earnings-next")
    with pytest.raises(StoriesNotReady, match="no benzinga-earnings-next facts row"):
        await build_range(ready, start=MON, end=FRI)
    await earnings_parse.extract_all(ready)  # parses under the current name
    assert await build_range(ready, start=MON, end=FRI) == 7


async def test_only_the_news_a_build_reads_must_be_parsed(ready: AsyncEngine) -> None:
    # Unparsed news after the range's end, and an unparsed filing, hold nothing up.
    later = date(2024, 5, 20)
    await store(
        ready,
        [
            news_row(50, "AAPL", ny(later, 8), DOWNGRADE),
            filing_row("acc-9", "AAPL", ny(TUE, 8), []),
        ],
        facts=False,
    )
    await time_filings(ready)
    assert await build_range(ready, start=MON, end=FRI) == 7
    with pytest.raises(StoriesNotReady, match="1 news event"):
        await build_range(ready, start=MON, end=later)


async def _event_id(engine: AsyncEngine, source_id: str, symbol: str) -> int:
    async with engine.connect() as conn:
        n = await conn.scalar(
            text("SELECT id FROM events WHERE source_id = :i AND symbol = :s"),
            {"i": source_id, "s": symbol},
        )
    return int(n)


async def test_a_build_waits_for_its_8ks_header_times(ready: AsyncEngine) -> None:
    history_from = date(2016, 1, 1)
    # seed_week's 8-K is settled; MSFT's 10-Q is no story item, so it is never asked.
    assert await untimed_filings(ready, start=history_from, end=FRI) == (0, None)
    later = date(2024, 5, 20)
    await store(
        ready,
        [
            filing_row("acc-5", "AAPL", ny(TUE, 17, 45), ["8.01"]),  # the JSON's late time
            filing_row("acc-6", "AAPL", ny(WED, 9), ["5.02"], kind="8-k/a"),
            filing_row("acc-7", "AAPL", ny(later, 9), ["8.01"]),  # after the range: not read
        ],
    )
    first = await _event_id(ready, "acc-5", "AAPL")
    refused = (
        r"2 8-K event\(s\) published 2016-01-01\.\.2024-05-10 are not at their EDGAR header's "
        rf"time yet \(first: event {first}\); run `halal-trader events filings fix-times "
        r"--start 2016-01-01 --end 2024-05-10` first"
    )
    with pytest.raises(StoriesNotReady, match=refused):
        await build_range(ready, start=MON, end=FRI)
    assert await untimed_filings(ready, start=history_from, end=FRI) == (2, first)
    assert await built_ranges(ready) == []
    # The header says 11:00; EDGAR has no header for the amendment: both are settled.
    await write_times(ready, {"acc-5": (ny(TUE, 11), 6 * 3600 + 45 * 60)}, ["acc-6"])
    assert await untimed_filings(ready, start=history_from, end=FRI) == (0, None)
    assert await build_range(ready, start=MON, end=FRI) > 0
    assert await built_ranges(ready) == [(MON, FRI)]
    with pytest.raises(StoriesNotReady, match="1 8-K event"):
        await build_range(ready, start=MON, end=later)
    # A row stored under a second symbol after its filing was read keeps the
    # JSON's time: unsettled until a pass gives it the header's.
    await store(ready, [filing_row("acc-5", "MSFT", ny(TUE, 17, 45), ["8.01"])])
    copy = await _event_id(ready, "acc-5", "MSFT")
    assert await untimed_filings(ready, start=history_from, end=FRI) == (1, copy)
    with pytest.raises(StoriesNotReady, match=rf"1 8-K event.*first: event {copy}\)"):
        await build_range(ready, start=MON, end=FRI)
    assert await build_range(ready, start=MON, end=FRI, force=True) > 0  # forced: not asked
    await write_times(ready, {"acc-5": (ny(TUE, 11), 0)})
    assert await untimed_filings(ready, start=history_from, end=FRI) == (0, None)
    assert await build_range(ready, start=MON, end=FRI) > 0
    assert await built_ranges(ready) == [(MON, FRI)]


async def test_a_built_range_round_trips_through_the_table(ready: AsyncEngine) -> None:
    c: Counter[str] = Counter()
    written = await build_range(ready, start=MON, end=FRI, counters=c)
    items = await collect(ready, start=date(2016, 1, 1), end=FRI, symbols=["AAPL", "MSFT"])
    expected = [story_row(s) for s in build(items, await load_aliases(ready))]
    assert written == len(expected) == 7
    assert await stored_rows(ready) == sorted(expected, key=lambda r: r["story_id"])
    assert c["kind"] == 1 and c["entity"] == 1 and c["admitted"] == 9
    by_id = {r["story_id"]: r for r in expected}
    for day in ("07", "08", "09"):
        row = by_id[f"AAPL:2024-05-{day}"]
        assert row["parent"] == "AAPL:2024-05-06" and row["follower_close"] is True
    assert by_id["AAPL:2024-05-09"]["family_ever"] is None  # a downgrade, but a follower
    friday = by_id["AAPL:2024-05-10"]
    assert friday["parent"] is None and friday["family_ever"] == "NSN_CORE"
    assert [i["itype"] for i in friday["items"]] == ["earnings_8k", "earnings_fact"]
    msft = by_id["MSFT:2024-05-09"]
    assert msft["parent"] == "MSFT:2024-05-07" and msft["family_ever"] == "NSN_CORE"


async def test_the_same_stories_whether_the_range_is_built_whole_or_in_parts(
    ready: AsyncEngine,
) -> None:
    await build_range(ready, start=MON, end=FRI)
    whole = await stored_rows(ready)
    async with ready.begin() as conn:
        await conn.execute(text("DELETE FROM news_stories"))
    await build_range(ready, start=WED, end=FRI)  # parents on Mon and Tue come from history
    await build_range(ready, start=MON, end=TUE)
    assert await stored_rows(ready) == whole
    await build_range(ready, start=MON, end=FRI)  # and a rebuild changes nothing
    assert await stored_rows(ready) == whole


async def test_a_build_replaces_its_range_only(ready: AsyncEngine) -> None:
    await build_range(ready, start=MON, end=FRI)
    row = (await stored_rows(ready))[0]
    stale = row | {"story_id": "ZZZ:2024-05-08", "symbol": "ZZZ", "session": WED}
    outside = row | {"story_id": "ZZZ:2024-06-03", "symbol": "ZZZ", "session": date(2024, 6, 3)}
    async with ready.begin() as conn:
        for r in (stale, outside):
            await conn.execute(
                text(
                    "INSERT INTO news_stories VALUES (:builder_version, :story_id, :symbol, "
                    ":session, :start_case, :detect_at, :nsn_at, :at_news, :type_detect, "
                    ":type_close, :family_ever, :follower_close, :parent, :n_items, :n_distinct, "
                    "CAST(:items AS JSONB), CAST(:flags AS JSONB))"
                ),
                r | {"items": json.dumps(r["items"]), "flags": json.dumps(r["flags"])},
            )
    await build_range(ready, start=TUE, end=WED)
    ids = {r["story_id"] for r in await stored_rows(ready)}
    assert "ZZZ:2024-05-08" not in ids and "ZZZ:2024-06-03" in ids
    assert "AAPL:2024-05-06" in ids and "AAPL:2024-05-08" in ids


async def test_persist_upserts_a_story(engine: AsyncEngine) -> None:
    raw = raw_news(1, ny(TUE, 8), DOWNGRADE)
    story = build([raw], APPLE)[0]
    assert await persist(engine, [story]) == 1
    later = build([raw, raw_news(2, ny(TUE, 9), MISS)], APPLE)
    assert await persist(engine, later) == 1
    rows = await stored_rows(engine)
    assert len(rows) == 1 and rows[0]["n_items"] == 2
    assert rows[0]["detect_at"] == ny(TUE, 8, 10) and rows[0]["detect_at"].tzinfo is not None


async def test_a_noise_only_story_is_stored_with_no_detection(engine: AsyncEngine) -> None:
    raws = [
        raw_news(1, ny(TUE, 8), "Why Apple Shares Are Trading Lower"),
        raw_news(2, ny(TUE, 9), "Rosen Law Firm Investigates Apple"),
    ]
    (story,) = build(raws, APPLE)
    assert await persist(engine, [story]) == 1
    (row,) = await stored_rows(engine)
    assert row == story_row(story)
    assert (row["type_close"], row["type_detect"]) == ("noise_only", "noise_only")
    assert row["detect_at"] is None and row["nsn_at"] is None and row["at_news"] is None
    assert row["family_ever"] is None and row["start_case"] == "out"


# ── complete builds ───────────────────────────────────────────


async def _progress(engine: AsyncEngine) -> dict[str, int]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT unit, items FROM backfill_progress WHERE task = :t"), {"t": stories.TASK}
        )
        return {r.unit: r.items for r in rows}


async def _counted(engine: AsyncEngine, start: date, end: date) -> int:
    """Every story ``count_stories`` counts in [start, end]."""
    counts = await count_stories(engine, start=start, end=end)
    return sum(n for (u, *_), n in counts.types.items() if u == "all")


async def test_a_build_marks_its_range_complete_with_its_inputs(ready: AsyncEngine) -> None:
    assert await built_ranges(ready) == []
    assert await build_range(ready, start=MON, end=FRI) == 7
    assert await built_ranges(ready) == [(MON, FRI)]
    inputs = await inputs_sha(ready)
    assert await _progress(ready) == {f"stories-v1:2024-05-06:2024-05-10:{inputs}": 7}
    assert build_unit(MON, FRI, "abc") == "stories-v1:2024-05-06:2024-05-10:abc"


async def test_counts_refuse_a_range_built_from_other_inputs(
    ready: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_context(monkeypatch)
    await build_range(ready, start=MON, end=FRI)
    assert await _counted(ready, MON, FRI) == 7
    before = await _progress(ready)
    await add_aliases(ready, [("AAPL", "Apple Inc", "name")])  # the aliases changed
    assert await built_ranges(ready) == []
    assert await _progress(ready) == before  # the mark is still there, naming the old inputs
    stale = r"5 session\(s\) in .*first: 2024-05-06\); 5 of them built from other inputs"
    with pytest.raises(StoriesNotReady, match=stale):
        await count_stories(ready, start=MON, end=FRI)
    await build_range(ready, start=MON, end=FRI)
    assert await built_ranges(ready) == [(MON, FRI)]
    assert await _counted(ready, MON, FRI) == 7
    # A pin of the code, or the extractor read, changes the inputs too.
    taxonomy = stories.TAXONOMY_SHA
    monkeypatch.setattr(stories, "TAXONOMY_SHA", "taxonomy-next")
    assert await built_ranges(ready) == []
    monkeypatch.setattr(stories, "TAXONOMY_SHA", taxonomy)
    assert await built_ranges(ready) == [(MON, FRI)]
    monkeypatch.setattr(earnings_parse, "EXTRACTOR", "benzinga-earnings-next")
    assert await built_ranges(ready) == []


async def test_rebuilding_part_of_a_range_after_its_inputs_changed_leaves_the_rest_refused(
    ready: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_context(monkeypatch)
    await build_range(ready, start=MON, end=FRI)
    old = await inputs_sha(ready)
    monkeypatch.setattr(stories, "STORIES_SHA", "rules-next")  # the builder's constants changed
    new = await inputs_sha(ready)
    assert new != old
    await build_range(ready, start=TUE, end=WED)
    assert await _progress(ready) == {
        f"stories-v1:2024-05-06:2024-05-06:{old}": 1,  # each outside part keeps its inputs
        f"stories-v1:2024-05-07:2024-05-08:{new}": 3,
        f"stories-v1:2024-05-09:2024-05-10:{old}": 3,
    }
    assert await built_ranges(ready) == [(TUE, WED)]
    assert await _counted(ready, TUE, WED) == 3
    with pytest.raises(StoriesNotReady, match=r"3 session.*3 of them built from other inputs"):
        await count_stories(ready, start=MON, end=FRI)
    # Sessions with no mark at all are not called stale.
    with pytest.raises(StoriesNotReady, match=r"4 session.*first: 2024-05-06\); 3 of them"):
        await count_stories(ready, start=MON, end=date(2024, 5, 13))
    await build_range(ready, start=MON, end=FRI)
    assert await built_ranges(ready) == [(MON, FRI)]


async def test_a_mark_that_names_no_inputs_matches_none(ready: AsyncEngine) -> None:
    async with ready.begin() as conn:  # a mark written before marks named their inputs
        await conn.execute(
            text(
                "INSERT INTO backfill_progress (task, unit, items, done_at) "
                "VALUES (:t, 'stories-v1:2024-05-06:2024-05-10', 7, now())"
            ),
            {"t": stories.TASK},
        )
    assert await built_ranges(ready) == []
    with pytest.raises(StoriesNotReady, match="5 of them built from other inputs"):
        await count_stories(ready, start=MON, end=FRI)
    await build_range(ready, start=TUE, end=WED)  # withdrawn over the range, split around it
    inputs = await inputs_sha(ready)
    assert await _progress(ready) == {
        "stories-v1:2024-05-06:2024-05-06:": 0,
        f"stories-v1:2024-05-07:2024-05-08:{inputs}": 3,
        "stories-v1:2024-05-09:2024-05-10:": 0,
    }
    assert await built_ranges(ready) == [(TUE, WED)]


async def test_the_inputs_are_hashed_before_the_aliases_are_read(
    ready: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Aliases changed while a build starts: it read the new ones, but its mark
    # names the inputs hashed before, so the counts refuse it (never the reverse).
    real_load = stories.load_aliases

    async def load_after_a_change(engine: AsyncEngine) -> Any:
        await add_aliases(engine, [("AAPL", "Apple Inc", "name")])
        return await real_load(engine)

    monkeypatch.setattr(stories, "load_aliases", load_after_a_change)
    await build_range(ready, start=MON, end=FRI)
    assert await built_ranges(ready) == []
    monkeypatch.setattr(stories, "load_aliases", real_load)
    await build_range(ready, start=MON, end=FRI)
    assert await built_ranges(ready) == [(MON, FRI)]


async def test_a_build_that_stops_part_way_leaves_its_range_unmarked(
    ready: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads, _ = fake_context(monkeypatch)
    await build_range(ready, start=MON, end=FRI)
    whole = await stored_rows(ready)
    async with ready.begin() as conn:  # mark every row so a rewrite shows
        await conn.execute(text("UPDATE news_stories SET n_items = 99"))
    monkeypatch.setattr(stories, "BATCH_SYMBOLS", 1)  # AAPL, then MSFT
    real_build = stories.build

    def failing(items: Any, aliases: Any, **kwargs: Any) -> Any:
        items = list(items)
        if items and items[0].symbol == "MSFT":
            raise RuntimeError("killed")
        return real_build(items, aliases, **kwargs)

    monkeypatch.setattr(stories, "build", failing)
    with pytest.raises(RuntimeError, match="killed"):
        await build_range(ready, start=MON, end=FRI)
    rows = {r["story_id"]: r for r in await stored_rows(ready)}
    # AAPL's batch was replaced whole; MSFT's delete was rolled back with its batch.
    assert {r["n_items"] for s, r in rows.items() if s.startswith("AAPL")} == {1, 2}
    assert {r["n_items"] for s, r in rows.items() if s.startswith("MSFT")} == {99}
    assert await built_ranges(ready) == []
    with pytest.raises(StoriesNotReady, match="5 session.*first: 2024-05-06"):
        await count_stories(ready, start=MON, end=FRI)
    assert loads == []
    monkeypatch.setattr(stories, "build", real_build)
    await build_range(ready, start=MON, end=FRI)
    assert await stored_rows(ready) == whole
    assert await built_ranges(ready) == [(MON, FRI)]


async def test_rebuilding_part_of_a_range_keeps_the_rest_complete(
    ready: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_context(monkeypatch)
    await build_range(ready, start=MON, end=FRI)
    async with ready.begin() as conn:  # another version's mark is not this one's
        await conn.execute(
            text(
                "INSERT INTO backfill_progress (task, unit, items, done_at) "
                "VALUES (:t, 'stories-v0:2016-01-01:2026-12-31', 1, now())"
            ),
            {"t": stories.TASK},
        )
    await build_range(ready, start=TUE, end=WED)
    assert await built_ranges(ready) == [(MON, MON), (TUE, WED), (THU, FRI)]
    inputs = await inputs_sha(ready)
    assert await _progress(ready) == {
        "stories-v0:2016-01-01:2026-12-31": 1,
        f"stories-v1:2024-05-06:2024-05-06:{inputs}": 1,  # the stories left in each part
        f"stories-v1:2024-05-07:2024-05-08:{inputs}": 3,
        f"stories-v1:2024-05-09:2024-05-10:{inputs}": 3,
    }
    assert sum((await count_stories(ready, start=MON, end=FRI)).types.values()) > 0
    with pytest.raises(StoriesNotReady, match=r"1 session\(s\) in .*first: 2024-05-13"):
        await count_stories(ready, start=MON, end=date(2024, 5, 13))
    weekend = await count_stories(ready, start=date(2024, 5, 11), end=date(2024, 5, 12))
    assert not weekend.types  # no session, nothing to cover
    await build_range(ready, start=date(2024, 5, 1), end=date(2024, 5, 31))
    assert await built_ranges(ready) == [(date(2024, 5, 1), date(2024, 5, 31))]


# ── counts ────────────────────────────────────────────────────


async def test_counts_split_types_and_nsn_by_universe(
    ready: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads, made = fake_context(monkeypatch)
    await build_range(ready, start=MON, end=FRI)
    counts = await count_stories(ready, start=MON, end=FRI)
    assert loads == [(["AAPL", "MSFT"], MON, FRI)]
    assert counts.types[("all", "fraud_probe", 2024, "validation")] == 1
    assert counts.types[("all", "analyst_downgrade", 2024, "validation")] == 3
    assert counts.types[("primary", "analyst_downgrade", 2024, "validation")] == 2  # no MSFT
    assert counts.types[("tech", "earnings_miss", 2024, "validation")] == 1
    assert sum(n for (u, *_), n in counts.types.items() if u == "all") == 7
    assert sum(n for (u, *_), n in counts.types.items() if u == "primary") == 5
    v = "validation"
    assert counts.nsn == Counter(
        {("all", 2024, v): 2, ("primary", 2024, v): 1, ("tech", 2024, v): 1}
    )
    assert counts.reasons == Counter(
        {("all", "ok"): 5, ("all", "rank"): 2, ("nsn", "ok"): 1, ("nsn", "rank"): 1}
    )
    asked = {(s, d): at for s, d, at in made[0].asked}
    assert asked[("AAPL", FRI)] == ny(FRI, 7)  # the NSN item's own time, not the 8-K's
    assert asked[("MSFT", THU)] == ny(THU, 10)
    assert asked[("AAPL", THU)] == ny(THU, 8)  # no NSN: the first item's time


async def test_counts_load_one_year_at_a_time(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads, _ = fake_context(monkeypatch)
    raws = [
        raw_news(1, ny(date(2021, 12, 30), 8), DOWNGRADE),
        raw_news(2, ny(date(2022, 1, 3), 8), MISS),
    ]
    await persist(engine, build(raws, APPLE))
    await mark_built(engine, date(2021, 6, 1), date(2022, 6, 30), 2)
    counts = await count_stories(engine, start=date(2021, 6, 1), end=date(2022, 6, 30))
    assert loads == [
        (["AAPL"], date(2021, 6, 1), date(2021, 12, 31)),
        (["AAPL"], date(2022, 1, 1), date(2022, 6, 30)),
    ]
    assert counts.nsn[("primary", 2021, "train")] == 1
    assert counts.nsn[("primary", 2022, "validation")] == 1


def _news_at(n: int, symbol: str, at: datetime, headline: str) -> RawItem:
    return RawItem(n, f"a:{n}", "news", symbol, at.astimezone(UTC), at, headline, 1, (), ())


async def test_counts_judge_eligibility_through_the_real_context(engine: AsyncEngine) -> None:
    # The point-in-time world of test_event_context: S = 2024-03-19, screens dated
    # 2024-03-08 (seen at S) and S itself (seen from the session after).
    await _world(engine)
    named = ["AAA", "AAB", "CHEAP", "GAP", "HALT", "SPLIT", "THIN", "VET"]
    aliases = {s: AliasMatcher(s, (), (s,)) for s in [*named, "BETA3"]}
    raws = [
        _news_at(n, s, et(S, 8), f"Morgan Stanley Downgrades {s} to Equal-Weight")
        for n, s in enumerate(named, 1)
    ]
    raws += [
        # The evening before S: judged at its own time, sigma through S-1.
        _news_at(20, "BETA3", et(PREV, 18), "Morgan Stanley Downgrades BETA3 to Equal-Weight"),
        # The day after: the screen dated S now counts for AAA; a product story, no NSN.
        _news_at(21, "AAA", et(NEXT, 8), "AAA Unveils New Product"),
    ]
    built = build(raws, aliases)
    await persist(engine, built)
    await mark_built(engine, S, NEXT, len(built))

    counts = await count_stories(engine, start=S, end=NEXT)

    ctx = await PitContext.load(engine, symbols=[*named, "BETA3"], start=S, end=NEXT)
    want = Counter(
        ("all", ctx.eligibility(st.symbol, st.session, at_news=st.items[0].at).reason)
        for st in built
    )
    assert want == Counter(
        {
            ("all", "ok"): 3,  # AAA, SPLIT and BETA3 at S
            ("all", "share_class"): 1,
            ("all", "price"): 1,
            ("all", "no_daily"): 1,
            ("all", "no_sigma"): 1,
            ("all", "rank"): 1,
            ("all", "not_halal"): 2,  # VET at S, AAA at NEXT
        }
    )
    assert Counter({k: n for k, n in counts.reasons.items() if k[0] == "all"}) == want
    assert counts.reasons[("nsn", "ok")] == 3
    assert counts.nsn[("all", 2024, "validation")] == 9
    assert counts.nsn[("primary", 2024, "validation")] == 3
    assert counts.nsn[("tech", 2024, "validation")] == 3  # software and semis
    assert counts.types[("primary", "analyst_downgrade", 2024, "validation")] == 3
    assert counts.types[("all", "product", 2024, "validation")] == 1
