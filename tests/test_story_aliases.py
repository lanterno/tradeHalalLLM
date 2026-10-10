"""The story builder's entity check: alias sources, the matcher, and the stored set."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data.alpaca_market import Asset
from halal_trader.events import renames
from halal_trader.events.aliases import (
    BUILDER_VERSION,
    AliasMatcher,
    AliasRow,
    alias_rows,
    alias_sha,
    build_aliases,
    learn_slots,
    learned_aliases,
    load_aliases,
    matcher_for,
    name_aliases,
    slot_of,
    tickers_of,
)
from halal_trader.events.store import EventRecord, EventRecorder

# ── source (a): names ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "aliases"),
    [
        # suffixes and share classes go; the first word stays when distinctive
        ("Meta Platforms, Inc. Class A Common Stock", ["Meta"]),
        ("Advanced Micro Devices, Inc.", ["Advanced Micro Devices", "Advanced Micro"]),
        ("Zoom Video Communications, Inc.", ["Zoom Video", "Zoom"]),
        ("Ford Motor Company", ["Ford Motor", "Ford"]),
        # a generic first word is not an alias on its own; the first two words are
        ("United Parcel Service, Inc.", ["United Parcel Service", "United Parcel"]),
        # three capitals are distinctive, three lowercase letters are not
        ("IBM Watson Health Corp", ["IBM Watson Health", "IBM Watson", "IBM"]),
        ("Box Office Inc", ["Box Office"]),
        ("AMC Networks Inc.", ["AMC"]),
        # the first two words need six characters and no single letter
        ("X Financial Corp", ["X Financial"]),
        ("Abc De Fgh", ["Abc De Fgh", "Abc De"]),
        ("Ab Cd Ef", ["Ab Cd Ef"]),
        # dots, commas and brackets split words before the suffixes go
        ("J.B. Hunt Transport Services, Inc.", ["J B Hunt Transport Services"]),
        (
            "Commerce.com, Inc. Series 1 Common Stock",
            ["Commerce com Series 1", "Commerce com", "Commerce"],
        ),
        ("", []),
        ("Inc.", []),
    ],
)
def test_names_are_cleaned_as_the_prototype_did(name: str, aliases: list[str]) -> None:
    assert name_aliases(name) == aliases


# ── source (b): Benzinga's slot ───────────────────────────────


@pytest.mark.parametrize(
    ("headline", "slot"),
    [
        ("Morgan Stanley Downgrades Apple to Equal-Weight", "Apple"),
        ("Needham Maintains Buy on Apple, Raises Price Target to $250", "Apple"),
        ("Goldman Sachs Initiates Coverage On Snowflake with Buy Rating", "Snowflake"),
        ("Barclays Downgrades Kohl's to Underweight", "Kohl"),
        ("Apple Q3 Adj. EPS $1.40 Beats $1.35 Estimate, Sales $85.8B", "Apple"),
        ("Alphabet Class A Q2 2024 EPS $1.89 Beats $1.84 Estimate", "Alphabet Class A"),
        ("Nvidia Sees Q3 Sales $32.5B +/- 2%", "Nvidia"),
        ("Stocks Moving In Tuesday's Pre-Market Session", None),
        # a slot too long to be a name is no slot (the first template still decides)
        ("UBS Upgrades " + "A" * 41 + " to Buy", None),
        ("X Upgrades Y to Buy", None),  # one letter is too short
    ],
)
def test_the_company_slot_is_read_from_benzinga_templates(headline: str, slot: str | None) -> None:
    assert slot_of(headline) == slot


def test_learned_slots_need_three_sightings_and_a_tenth_of_the_symbol() -> None:
    learned = learned_aliases(
        {
            "AAPL": Counter({"Apple": 10, "Apple Inc": 2, "Aple": 3}),
            "BIG": Counter({"Big Lots": 40, "A Co": 3}),  # 3 < 10% of 43
            "JBHT": Counter({"JB Hunt Transport Servs": 5}),
            "AMD": Counter({"AMD Inc": 4}),
            "GM": Counter({"General Motors": 5}),
            "NONE": Counter({"Rare": 2}),
        }
    )
    assert learned == {
        "AAPL": {"Apple", "Aple"},
        "BIG": {"Big Lots"},  # "Big" is generic
        "JBHT": {"JB Hunt Transport Servs"},  # "JB" is too short to stand alone
        "AMD": {"AMD Inc", "AMD"},  # three capitals
        "GM": {"General Motors"},
    }


async def _news(
    engine: AsyncEngine, rows: list[tuple[int, str, datetime, str, list[str] | None]]
) -> None:
    await EventRecorder(engine, raise_errors=True).record(
        [
            EventRecord(
                source="alpaca",
                source_id=f"alpaca:{n}",
                kind="news",
                symbol=symbol,
                published_at=at,
                seen_at=at,
                payload={"headline": headline}
                | ({"symbols": symbols} if symbols is not None else {}),
            )
            for n, symbol, at, headline, symbols in rows
        ]
    )


async def test_slots_are_learned_from_single_symbol_articles_in_the_window(
    engine: AsyncEngine,
) -> None:
    t = datetime(2020, 3, 2, 15, tzinfo=UTC)
    await _news(
        engine,
        [
            (1, "AAPL", t, "Morgan Stanley Downgrades Apple to Equal-Weight", ["AAPL"]),
            (2, "AAPL", t, "Apple Q1 EPS $4.99 Beats $4.55 Estimate", ["AAPL"]),
            # tags two symbols: not Benzinga's name for either
            (3, "AAPL", t, "Barclays Upgrades Apple to Buy", ["AAPL", "MSFT"]),
            # a live row has no tag list
            (4, "AAPL", t, "Apple Sees Q2 Sales $90B", None),
            # outside the window
            (5, "AAPL", datetime(2015, 12, 31, 12, tzinfo=UTC), "Apple Sees Q1 Sales", ["AAPL"]),
            (6, "AAPL", datetime(2026, 10, 10, 4, 30, tzinfo=UTC), "Apple Sees Q4 Sales", ["AAPL"]),
            (7, "MSFT", t, "Microsoft Sees Q3 Sales $60B", ["MSFT"]),
            (8, "MSFT", t, "Stocks Moving In Tuesday's Pre-Market Session", ["MSFT"]),
        ],
    )
    counts = await learn_slots(engine)
    assert counts == {"AAPL": Counter({"Apple": 2}), "MSFT": Counter({"Microsoft": 1})}
    # the window ends at midnight New York after its last day
    assert await learn_slots(engine, end=date(2026, 10, 10)) == {
        "AAPL": Counter({"Apple": 3}),
        "MSFT": Counter({"Microsoft": 1}),
    }


async def test_slots_are_learned_only_from_rows_that_are_their_symbols_own(
    engine: AsyncEngine,
) -> None:
    def at(day: date) -> datetime:
        return datetime(day.year, day.month, day.day, 15, tzinfo=UTC)

    pandora, everpure = date(2017, 5, 1), date(2026, 5, 4)
    await _news(
        engine,
        [
            # Pandora held P: its articles teach P nothing ...
            *((n, "P", at(pandora), "Pandora Q1 EPS $(0.20) Misses", ["P"]) for n in (1, 2, 3)),
            # ... Pure Storage's, copied from PSTG or after the switch, do
            *((n, "P", at(pandora), "Pure Storage Q4 EPS $0.10 Beats", ["PSTG"]) for n in (4, 5)),
            (6, "P", at(everpure), "Everpure Q4 EPS $0.40 Beats", ["P"]),
            # the old IAC's own rows count once, under PPLI (the copy), not under IAC
            *(
                (n, s, at(date(2021, 5, 4)), "IAC Q1 EPS $1.00 Beats", ["IAC"])
                for n, s in ((7, "IAC"), (7, "PPLI"))
            ),
            # Ingersoll-Rand's IR is Trane's now; Gardner Denver's IR from 2020-03-02
            (8, "IR", at(date(2019, 5, 1)), "Ingersoll-Rand Q1 EPS $1.30 Beats", ["IR"]),
            (8, "TT", at(date(2019, 5, 1)), "Ingersoll-Rand Q1 EPS $1.30 Beats", ["IR"]),
            (9, "IR", at(date(2021, 5, 4)), "Ingersoll Rand Q1 EPS $0.40 Beats", ["IR"]),
            # a metaverse ETF held META before Facebook took it
            (10, "META", at(date(2021, 11, 1)), "Roundhill Upgrades Metaverse to Buy", ["META"]),
        ],
    )
    assert await learn_slots(engine) == {
        "P": Counter({"Pure Storage": 2, "Everpure": 1}),
        "PPLI": Counter({"IAC": 1}),
        "TT": Counter({"Ingersoll-Rand": 1}),
        "IR": Counter({"Ingersoll Rand": 1}),
    }


# ── the matcher ───────────────────────────────────────────────


def test_aliases_match_any_case_and_tickers_only_as_written() -> None:
    m = AliasMatcher("META", ("Facebook", "Meta"), ("META", "FB"))
    assert m.matches("Facebook Q2 EPS Beats")
    assert m.matches("FACEBOOK shares slide")
    assert m.matches("Analysts weigh Meta's AI spending")
    assert m.matches("$META rallies after hours")
    assert m.matches("FB, AAPL lead the Nasdaq")
    assert not m.matches("fb shares")  # a ticker is case-sensitive
    assert not m.matches("Metamaterials jump")  # no letter may follow
    assert not m.matches("METAL prices rise")  # ...in any case, for an alias
    assert not m.matches("AFB declares dividend")  # nor precede


def test_dots_are_removed_on_both_sides() -> None:
    hunt = AliasMatcher("JBHT", ("J.B. Hunt",), ("JBHT",))
    assert hunt.aliases == ("JB Hunt",)
    assert hunt.matches("J.B. Hunt Transport Q3 EPS")
    assert hunt.matches("JB Hunt Q3 EPS")
    brk = AliasMatcher("BRK.B", (), ("BRK.B",))
    assert brk.tickers == ("BRKB",)
    assert brk.matches("BRK.B hits a record")
    assert brk.matches("$BRKB hits a record")


def test_aliases_are_normalised_longest_first_and_tickers_need_two_letters() -> None:
    m = AliasMatcher("AMD", ("AMD", "Advanced Micro", "AMD", " ", "Advanced Micro Devices"), ("D",))
    assert m.aliases == ("Advanced Micro Devices", "Advanced Micro", "AMD")
    assert m.tickers == ()
    assert m.matches("Advanced Micro Devices Q4")
    assert not m.matches("D shares jump")


def test_a_matcher_with_nothing_to_match_passes_every_headline() -> None:
    m = AliasMatcher("F", (), ("F",))
    assert m.empty
    assert m.matches("anything at all")
    assert not AliasMatcher("AAPL", (), ("AAPL",)).empty


def test_an_unknown_symbol_gets_only_its_own_ticker() -> None:
    known = {"AAPL": AliasMatcher("AAPL", ("Apple",), ("AAPL",))}
    assert matcher_for("AAPL", known) is known["AAPL"]
    caly = matcher_for("CALY", known)
    assert caly.aliases == ()
    # old tickers come from the stored set, never from TICKER_RENAMES as it is now
    assert caly.tickers == ("CALY",)
    assert caly.matches("CALY shares fall")
    assert not caly.matches("MODG shares fall")


def test_tickers_are_the_symbol_and_its_old_ones() -> None:
    assert tickers_of("META") == ("META", "FB")
    assert tickers_of("NXH") == ("NXH", "OSTK", "BYON", "BBBY")
    assert tickers_of("AAPL") == ("AAPL",)


def test_rows_combine_the_four_sources() -> None:
    rows = alias_rows(
        ["META", "F", "ZZZZ"],
        {"META": ["Meta Platforms, Inc. Class A Common Stock"], "F": ["Ford Motor Company"]},
        {"META": {"Facebook"}, "ZZZZ": {"Zed Corp"}},
    )
    assert rows == sorted(
        [
            AliasRow("F", "Ford", "name"),
            AliasRow("F", "Ford Motor", "name"),
            AliasRow("META", "Meta", "name"),
            AliasRow("META", "Facebook", "learned"),
            *(
                AliasRow("META", a, "override")
                for a in ("Facebook", "Meta", "Instagram", "WhatsApp", "FB")
            ),
            AliasRow("META", "META", "ticker"),
            AliasRow("META", "FB", "ticker"),  # the old ticker
            AliasRow("ZZZZ", "Zed Corp", "learned"),  # learned without any name
            AliasRow("ZZZZ", "ZZZZ", "ticker"),
        ]
    )  # "F" is one letter: no ticker row


# ── build, load, pin ──────────────────────────────────────────


class _Market:
    def __init__(self, inactive: list[Asset]) -> None:
        self.inactive = inactive

    async def inactive_assets(self) -> list[Asset]:
        return self.inactive


def _asset(symbol: str, name: str) -> Asset:
    return Asset(symbol, name, "NASDAQ", False, False, "inactive")


async def _seed(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO market_assets (symbol, name, exchange, tradable, fractionable, "
                "status, synced_at) VALUES (:s, :n, 'NASDAQ', true, true, 'active', now())"
            ),
            [
                {"s": "META", "n": "Meta Platforms, Inc. Class A Common Stock"},
                {"s": "AAPL", "n": "Apple Inc. Common Stock"},
                {"s": "XOM", "n": "Exxon Mobil Corporation"},  # no news, not screened
            ],
        )
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES "
                "('2020-03-31', :s, 1, '', 'halal', '[]', '{}', 't', now())"
            ),
            [{"s": "AAPL"}, {"s": "OLDCO"}],
        )
    t = datetime(2020, 3, 2, 15, tzinfo=UTC)
    await _news(
        engine,
        [(n, "META", t, "Facebook Q4 EPS $2.56 Beats $2.53 Estimate", ["FB"]) for n in range(1, 4)]
        + [(9, "AAPL", t, "Apple to unveil new iPhone", ["AAPL"])],
    )


async def test_build_stores_every_source_and_load_reads_them_back(engine: AsyncEngine) -> None:
    await _seed(engine)
    market = _Market(
        [_asset("OLDCO", "Old Dominion Freight Line"), _asset("AAPL", "Apple Computer")]
    )
    n = await build_aliases(engine, market)
    async with engine.connect() as conn:
        rows = {
            (r.symbol, r.alias, r.source)
            for r in await conn.execute(
                text("SELECT symbol, alias, source FROM story_aliases WHERE builder_version = :v"),
                {"v": BUILDER_VERSION},
            )
        }
    assert n == len(rows)
    assert ("META", "Facebook", "learned") in rows  # from the old ticker's articles
    assert ("META", "Meta", "name") in rows
    assert ("META", "FB", "ticker") in rows
    assert ("AAPL", "Apple", "name") in rows
    assert ("AAPL", "Apple Computer", "name") in rows  # the inactive list's name too
    assert ("OLDCO", "Old Dominion", "name") in rows
    assert ("OLDCO", "Old", "name") not in rows  # "old" is generic
    assert not any(s == "XOM" for s, _, _ in rows)  # no story can be about it
    assert {s for s, _, _ in rows} >= {"CALY", "XYZ"}  # renamed tickers' current symbols

    matchers = await load_aliases(engine)
    assert matchers["META"].matches("Facebook to buy WhatsApp")
    assert matchers["META"].matches("FB slides")
    assert matchers["AAPL"].matches("AAPL, MSFT lead")
    assert matchers["XYZ"].tickers == ("XYZ", "SQ")
    assert not matchers["AAPL"].matches("Microsoft Sees Q3 Sales")


async def test_a_rebuild_replaces_the_rows_and_the_hash_follows_them(engine: AsyncEngine) -> None:
    await _seed(engine)
    market = _Market([])
    first = await build_aliases(engine, market)
    sha = await alias_sha(engine)
    assert len(sha) == 12
    assert await build_aliases(engine, market) == first  # replaced, not duplicated
    assert await alias_sha(engine) == sha
    await build_aliases(engine, _Market([_asset("NEWCO", "Newco Widgets Inc")]))
    assert await alias_sha(engine) == sha  # NEWCO carries no news: not stored
    await build_aliases(engine, _Market([_asset("AAPL", "Apple Computer")]))
    assert await alias_sha(engine) != sha


async def test_editing_the_renames_changes_nothing_until_the_next_build(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed(engine)
    await build_aliases(engine, _Market([]))
    sha = await alias_sha(engine)
    before = await load_aliases(engine)

    monkeypatch.setattr(
        renames,
        "TICKER_RENAMES",
        {
            **renames.TICKER_RENAMES,
            "APPL": ("AAPL", date(2019, 1, 2)),
            "NEWT": ("NEWCO", date(2019, 1, 2)),
        },
    )
    after = await load_aliases(engine)
    assert after == before
    assert not after["AAPL"].matches("APPL shares rise")
    assert matcher_for("NEWCO", after).tickers == ("NEWCO",)
    assert await alias_sha(engine) == sha

    await build_aliases(engine, _Market([]))
    rebuilt = await load_aliases(engine)
    assert rebuilt["AAPL"].tickers == ("AAPL", "APPL")
    assert rebuilt["AAPL"].matches("APPL shares rise")
    assert rebuilt["NEWCO"].matches("NEWT shares rise")
    assert await alias_sha(engine) != sha


async def test_the_hash_ignores_the_order_rows_come_back_in(engine: AsyncEngine) -> None:
    async def insert(rows: list[tuple[str, str, str]]) -> str:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM story_aliases"))
            await conn.execute(
                text(
                    "INSERT INTO story_aliases (builder_version, symbol, alias, source) "
                    "VALUES (:v, :s, :a, :src)"
                ),
                [{"v": BUILDER_VERSION, "s": s, "a": a, "src": src} for s, a, src in rows],
            )
        return await alias_sha(engine)

    rows = [("b", "Zed", "name"), ("B", "alpha", "name"), ("a", "_x", "ticker")]
    assert await insert(rows) == await insert(rows[::-1])
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO story_aliases (builder_version, symbol, alias, source) "
                "VALUES ('stories-v0', 'a', 'other', 'name')"
            )
        )
    assert await alias_sha(engine) == await insert(rows)  # another version is not this one's
