"""Renamed tickers: the map, each old ticker's window, who a news row belongs to,
seeding, and the old-news backfill."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events import renames
from halal_trader.events.renames import (
    GRACE_SESSIONS,
    HELD_SINCE,
    TASK,
    TICKER_RENAMES,
    backfill_renamed_news,
    held_since,
    months,
    news_window,
    old_tickers,
    owner,
    renamed_records,
    seed_candidates,
    ticker_history,
    unit,
    window,
)
from halal_trader.market_hours import is_trading_day, next_trading_day
from tests._renames import NewsMarket, article, ny, seed_candidates_data

# ── the map ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("old", "current", "last"),
    [
        ("FB", "META", date(2022, 6, 8)),  # META from 2022-06-09
        ("SQ", "XYZ", date(2025, 1, 17)),  # XYZ from 2025-01-21 (MLK day between)
        ("PCLN", "BKNG", date(2018, 2, 26)),  # BKNG from 2018-02-27
        ("UTX", "RTX", date(2020, 4, 2)),  # RTX from 2020-04-03
        ("JCOM", "ZD", date(2021, 10, 7)),  # ZD regular-way from 2021-10-08
        ("CREE", "WOLF", date(2021, 10, 1)),  # WOLF from 2021-10-04
        ("FLT", "CPAY", date(2024, 3, 22)),  # CPAY from 2024-03-25
        ("IR", "TT", date(2020, 2, 28)),  # Trane from 2020-03-02 ...
        ("GDI", "IR", date(2020, 2, 28)),  # ... when Gardner Denver took IR
    ],
)
def test_known_renames(old: str, current: str, last: date) -> None:
    assert TICKER_RENAMES[old] == (current, last)


def test_every_old_ticker_ends_on_a_session_and_names_another_symbol() -> None:
    for old, (current, last) in TICKER_RENAMES.items():
        assert is_trading_day(last), old
        assert old != current
        assert old == old.upper() and current == current.upper()
        first, end = window(old)
        assert first <= end == last


def test_held_since_entries_are_sessions_of_known_tickers() -> None:
    for (ticker, current), first in HELD_SINCE.items():
        assert is_trading_day(first), ticker
        assert ticker == current or TICKER_RENAMES[ticker][0] == current


def test_a_chain_of_tickers_hands_over_on_consecutive_sessions() -> None:
    by_current: dict[str, list[str]] = {}
    for old, (current, _) in TICKER_RENAMES.items():
        by_current.setdefault(current, []).append(old)
    for current, olds in by_current.items():
        chain = old_tickers(current)
        assert sorted(chain) == sorted(olds)
        for before, after in zip(chain, chain[1:], strict=False):
            if (after, current) in HELD_SINCE:
                continue
            assert window(after)[0] == next_trading_day(window(before)[1]), (current, after)
    assert old_tickers("CALY") == ("ELY", "MODG")
    assert window("MODG") == (date(2022, 9, 7), date(2026, 1, 15))
    assert window("BBBY") == (date(2025, 8, 29), date(2026, 8, 14))
    assert window("AAXN") == (date(2017, 4, 6), date(2021, 1, 25))
    assert old_tickers("AAPL") == ()


def test_a_ticker_another_company_held_first_starts_when_this_one_took_it() -> None:
    assert window("IAC") == (date(2020, 7, 1), date(2026, 6, 3))  # not the old IAC (Match)
    assert window("BTX") == (date(2021, 3, 26), date(2022, 10, 14))  # not BioTime
    assert window("DWDP") == (date(2017, 9, 1), date(2019, 5, 31))  # DowDuPont's first session
    assert window("FB") == (date(2016, 1, 1), date(2022, 6, 8))  # from the news history's start


def test_a_current_ticker_is_held_from_the_day_after_its_last_old_one() -> None:
    assert held_since("META") == date(2022, 6, 9)
    assert held_since("IR") == date(2020, 2, 29)  # GDI's last session was Friday 02-28
    assert held_since("P") == date(2026, 4, 17)
    assert held_since("AXON") == date(2021, 1, 26)
    assert held_since("GAP") == date(2024, 8, 22)
    assert held_since("DD") == date(2019, 6, 1)
    assert held_since("PPLI") == date(2026, 6, 4)
    assert held_since("Q") == date(2025, 11, 3)  # Qnity, after Quintiles' Q
    assert held_since("AAPL") is None
    assert held_since("FB") is None  # an old ticker nobody holds today


def test_an_old_ticker_keeps_naming_its_company_a_few_sessions_unless_reused() -> None:
    assert GRACE_SESSIONS == 5
    assert news_window("FB") == (date(2016, 1, 1), date(2022, 6, 15))
    assert news_window("DWDP") == (date(2017, 9, 1), date(2019, 6, 7))
    assert news_window("GDI") == (date(2016, 1, 1), date(2020, 3, 6))
    # Gardner Denver took IR at the next session: no grace for Ingersoll-Rand's IR
    assert news_window("IR") == (date(2016, 1, 1), date(2020, 2, 28))
    # Qnity took Q eight years later: the grace applies
    assert news_window("Q") == (date(2016, 1, 1), date(2017, 11, 21))


def test_months_cut_a_window_at_calendar_months() -> None:
    assert months(date(2016, 11, 15), date(2017, 1, 10)) == [
        (date(2016, 11, 15), date(2016, 11, 30)),
        (date(2016, 12, 1), date(2016, 12, 31)),
        (date(2017, 1, 1), date(2017, 1, 10)),
    ]
    assert months(date(2024, 2, 1), date(2024, 2, 29)) == [(date(2024, 2, 1), date(2024, 2, 29))]
    assert months(date(2024, 3, 2), date(2024, 3, 1)) == []
    assert unit("FB", date(2016, 1, 1)) == "FB:2016-01"


def test_old_ticker_articles_become_events_of_the_current_symbol() -> None:
    when = datetime(2019, 5, 1, 14, tzinfo=UTC)
    recs = renamed_records(
        [article(1, ("FB", "AAPL"), when), article(2, ("AAPL",), when)], "FB", "META"
    )
    assert [(r.symbol, r.source_id, r.kind, r.published_at) for r in recs] == [
        ("META", "alpaca:1", "news", when)
    ]
    assert recs[0].payload["symbols"] == ["FB", "AAPL"]  # as Benzinga sent it
    assert recs[0].payload["backfill"] is True


# ── who a news row belongs to ─────────────────────────────────


@pytest.mark.parametrize(
    ("symbol", "day", "tagged", "expected"),
    [
        # IR: Ingersoll-Rand's until 2020-02-28 (Trane's now), Gardner Denver's after
        ("IR", date(2019, 6, 3), ["IR"], "TT"),
        ("IR", date(2020, 1, 31), ["GDI", "IR"], "IR"),  # tags Gardner Denver too
        ("IR", date(2020, 3, 1), ["IR"], "IR"),  # the Sunday after the switch
        ("IR", date(2024, 5, 1), ["IR"], "IR"),
        ("TT", date(2019, 6, 3), ["IR"], "TT"),  # the backfilled copy
        # P: Pandora's until Pure Storage took it
        ("P", date(2017, 5, 1), ["P"], None),
        ("P", date(2017, 5, 1), ["PSTG"], "P"),
        ("P", date(2017, 5, 1), ["P", "PSTG"], "P"),
        ("P", date(2026, 4, 17), ["P"], "P"),
        # IAC: the old IAC (Match) to 2020-06-30, then the spin-off now trading as PPLI
        ("IAC", date(2019, 6, 3), ["IAC", "MTCH"], None),
        ("IAC", date(2021, 6, 3), ["IAC"], "PPLI"),
        ("IAC", date(2026, 6, 4), ["IAC"], "PPLI"),  # a grace session
        ("IAC", date(2026, 7, 6), ["IAC"], None),
        ("PPLI", date(2021, 6, 3), ["IAC"], "PPLI"),
        ("PPLI", date(2021, 6, 3), ["PPLI"], None),
        # DWDP and DD: E.I. du Pont's DD, then DowDuPont's DWDP, then DuPont's DD
        ("DD", date(2017, 8, 24), ["DD"], None),
        ("DWDP", date(2018, 6, 4), ["DWDP"], "DD"),
        ("DD", date(2018, 6, 4), ["DWDP"], "DD"),
        ("DWDP", date(2019, 6, 3), ["DWDP"], "DD"),  # Benzinga still tagging DWDP
        ("DWDP", date(2021, 12, 10), ["DWDP", "GE"], None),
        ("DD", date(2019, 6, 6), ["DD"], "DD"),
        # META: a metaverse ETF's ticker before Facebook took it
        ("META", date(2021, 10, 28), ["META"], None),
        ("META", date(2021, 10, 28), ["FB", "META"], "META"),
        ("META", date(2022, 6, 9), ["META"], "META"),
        ("FB", date(2020, 1, 2), ["FB"], "META"),
        # GAP: Grupo Aeroportuario's tag before Gap Inc. moved from GPS
        ("GAP", date(2016, 8, 8), ["GAP"], None),
        ("GAP", date(2017, 11, 27), ["GAP", "GPS"], "GAP"),
        ("GAP", date(2024, 8, 22), ["GAP"], "GAP"),
        # AXON: Axovant's until Axon moved from AAXN
        ("AXON", date(2019, 2, 7), ["AXON"], None),
        ("AXON", date(2018, 2, 7), ["AAXN"], "AXON"),
        ("AXON", date(2021, 1, 26), ["AXON"], "AXON"),
        # Q: Quintiles IMS (IQV now), nobody, then Qnity
        ("Q", date(2017, 11, 9), ["Q"], "IQV"),
        ("Q", date(2020, 1, 2), ["Q"], None),
        ("Q", date(2025, 11, 4), ["Q"], "Q"),
        # untouched symbols, and live rows without tags
        ("AAPL", date(2016, 1, 4), None, "AAPL"),
        ("META", date(2026, 1, 5), None, "META"),
        ("META", date(2021, 10, 28), None, None),
    ],
)
def test_a_row_belongs_to_whoever_held_its_ticker_that_day(
    symbol: str, day: date, tagged: list[str] | None, expected: str | None
) -> None:
    assert owner(symbol, day, tagged) == expected


@pytest.mark.parametrize(
    ("old", "current", "day"),
    [("IAC", "PPLI", date(2021, 6, 3)), ("DWDP", "DD", date(2018, 6, 4))],
)
def test_an_old_ticker_the_screen_still_covers_counts_once(
    old: str, current: str, day: date
) -> None:
    """The original backfill stored the article under the old ticker (the screen
    maps it to the company's CIK) and the renames backfill under the current
    symbol: exactly one of the two rows is its symbol's own."""
    stored = [(old, [old]), (current, [old])]
    assert [s for s, tagged in stored if owner(s, day, tagged) == s] == [current]


@pytest.mark.usefixtures("small_map")
def test_the_lookups_follow_the_tables_they_were_built_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = ticker_history()
    assert ticker_history() is history  # built once per table
    assert owner("OLDA", date(2016, 1, 5), ["OLDA"]) == "NEWA"
    assert owner("NEWA", date(2016, 1, 5), ["NEWA"]) is None  # held from 2016-03-05
    assert owner("FB", date(2020, 1, 2), ["FB"]) == "FB"  # not in this map
    monkeypatch.setattr(renames, "TICKER_RENAMES", {"FB": ("META", date(2022, 6, 8))})
    assert ticker_history() is not history
    assert owner("FB", date(2020, 1, 2), ["FB"]) == "META"


# ── seeding ───────────────────────────────────────────────────


async def test_seeding_flags_news_that_starts_over_a_year_after_the_first_halal_screen(
    engine: AsyncEngine,
) -> None:
    await seed_candidates_data(engine)
    assert await seed_candidates(engine, today=date(2026, 10, 10)) == [
        ("LATE", date(2016, 9, 30), date(2018, 1, 2)),
        ("META", date(2017, 3, 31), date(2021, 6, 30)),
        ("SILENT", date(2016, 9, 30), None),  # halal for years, never in the news
    ]  # EDGE's first news is exactly 365 days on (New York date); FRESH is not a year old


# ── the backfill ──────────────────────────────────────────────

_SECOND = timedelta(seconds=1)


async def _stored(engine: AsyncEngine) -> list[tuple[str, str, list[str]]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT symbol, source_id, payload->'symbols' AS s FROM events ORDER BY id")
        )
        return [(r.symbol, r.source_id, r.s) for r in rows]


async def _units(engine: AsyncEngine) -> dict[str, int]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT unit, items FROM backfill_progress WHERE task = :t"), {"t": TASK}
        )
        return {r.unit: r.items for r in rows}


@pytest.mark.usefixtures("small_map")
async def test_backfill_fetches_each_old_month_and_files_it_under_the_current_symbol(
    engine: AsyncEngine,
) -> None:
    market = NewsMarket()
    stored = await backfill_renamed_news(engine, market, rate_per_min=600_000)
    assert stored == 4
    assert market.calls == [
        (["OLDA"], ny(date(2016, 1, 1), 0), ny(date(2016, 2, 1), 0) - _SECOND, 500),
        # five sessions past OLDA's last (2016-02-10), Presidents' Day between
        (["OLDA"], ny(date(2016, 2, 1), 0), ny(date(2016, 2, 19), 0) - _SECOND, 500),
        # OLDB names the company from the session after OLDA's last
        (["OLDB"], ny(date(2016, 2, 11), 0), ny(date(2016, 3, 1), 0) - _SECOND, 500),
        (["OLDB"], ny(date(2016, 3, 1), 0), ny(date(2016, 3, 12), 0) - _SECOND, 500),
    ]
    assert await _stored(engine) == [
        ("NEWA", "alpaca:1", ["OLDA", "AAPL"]),  # payload.symbols as Benzinga sent it
        ("NEWA", "alpaca:2", ["OLDA", "AAPL"]),
        ("NEWA", "alpaca:3", ["OLDB", "AAPL"]),
        ("NEWA", "alpaca:4", ["OLDB", "AAPL"]),
    ]
    assert await _units(engine) == {
        "OLDA:2016-01": 1,
        "OLDA:2016-02": 1,
        "OLDB:2016-02": 1,
        "OLDB:2016-03": 1,
    }
    again = NewsMarket()
    assert await backfill_renamed_news(engine, again, rate_per_min=600_000) == 0
    assert again.calls == []  # every unit is done


@pytest.mark.usefixtures("small_map")
async def test_a_month_not_over_yet_is_fetched_but_not_marked_done(engine: AsyncEngine) -> None:
    market = NewsMarket()
    await backfill_renamed_news(engine, market, rate_per_min=600_000, now=ny(date(2016, 3, 11), 20))
    assert len(market.calls) == 4
    assert "OLDB:2016-03" not in await _units(engine)
    again = NewsMarket()
    await backfill_renamed_news(engine, again, rate_per_min=600_000, now=ny(date(2016, 3, 12), 1))
    assert [(c[0], c[1]) for c in again.calls] == [(["OLDB"], ny(date(2016, 3, 1), 0))]
    assert (await _units(engine))["OLDB:2016-03"] == 0  # the same article again: nothing new
