"""Plan H (events/units.py) on a database: what each part selects, the plan, the fetch.

Stories are written straight into ``news_stories``; liquidity comes from
``monthly_bars`` through the real ``universe_at``. Most tests swap the point-
in-time context for a fake one; one builds a small daily-bar world and asks
the real ``PitContext``. Every bar is synthetic and no return is computed.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.loader import GATE_RANGES, gate_pins, register_gate_units
from halal_trader.data import minutes
from halal_trader.data.minutes import session_bounds
from halal_trader.data.universe import month_starts
from halal_trader.events import context, units
from halal_trader.events.context import PitContext
from halal_trader.events.stories import BUILDER_VERSION, StoriesNotReady
from halal_trader.events.study import Observation
from halal_trader.events.units import (
    LiquidityRanks,
    UnitPlan,
    calib_pairs,
    fetch,
    g1_stories,
    h1_plan,
    h1_windows,
    path,
    prev_close_s,
    reactor_headlines,
    read_stories,
    sessions_between,
    sue_complement,
    sue_event,
    window_units,
)
from halal_trader.market_hours import MARKET_TZ
from tests._stories import filing_row, mark_built, news_row, store


def ny(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), MARKET_TZ)


async def liquidity(engine: AsyncEngine, ranked: Sequence[str], first: date, last: date) -> None:
    """Monthly bars making ``ranked`` the most traded names, in that order."""
    rows = [
        {"s": s, "m": m, "c": 50.0, "v": 1e9 / (i + 1)}
        for i, s in enumerate(ranked)
        for m in month_starts(first, last)
    ]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO monthly_bars (symbol, month, close, volume, vwap) "
                "VALUES (:s, :m, :c, :v, :c)"
            ),
            rows,
        )


async def add_story(
    engine: AsyncEngine,
    symbol: str,
    session: date,
    items: Sequence[tuple[int, str]],
    *,
    nsn_at: datetime | None = None,
    type_close: str = "analyst_downgrade",
    version: str = BUILDER_VERSION,
) -> None:
    """A ``news_stories`` row whose items are (event_id, itype); an id with no
    ``events`` row (the tests use 100 and up) is an item of no known kind."""
    at = session_bounds(session)[0] - timedelta(hours=2)
    payload = [
        {
            "event_id": e,
            "at": at.isoformat(),
            "available_at": (at + timedelta(minutes=10)).isoformat(),
            "itype": itype,
            "dup_of": None,
            "supersedes": [],
            "entity_ok": True,
        }
        for e, itype in items
    ]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO news_stories (builder_version, story_id, symbol, session, start_case, "
                "detect_at, nsn_at, at_news, type_detect, type_close, family_ever, follower_close, "
                "parent, n_items, n_distinct, items, flags) VALUES (:v, :id, :s, :d, 'out', NULL, "
                ":nsn, :nsn, :tc, :tc, NULL, false, NULL, :n, :n, CAST(:items AS JSONB), '{}')"
            ),
            {
                "v": version,
                "id": f"{symbol}:{session.isoformat()}",
                "s": symbol,
                "d": session,
                "nsn": nsn_at,
                "tc": type_close,
                "n": len(items),
                "items": json.dumps(payload),
            },
        )


@dataclass(frozen=True)
class Elig:
    eligible: bool


@dataclass(frozen=True)
class Daily:
    close: float
    adj: float


@dataclass
class FakeContext:
    """BROAD-eligible: ``broad``, PRIMARY-eligible: ``primary``, a name in ``on``
    only at the sessions it admits; previous closes (in S units) from ``closes``."""

    broad: set[str]
    closes: dict[str, float]
    primary: set[str] = field(default_factory=set)
    on: dict[str, Callable[[date], bool]] = field(default_factory=dict)
    asked: list[tuple[str, date, datetime, str]] = field(default_factory=list)

    def eligibility(self, symbol: str, session: date, *, at_news: datetime, universe: str) -> Elig:
        self.asked.append((symbol, session, at_news, universe))
        members = self.broad if universe == "broad" else self.primary
        when = self.on.get(symbol)
        return Elig(symbol in members and (when is None or when(session)))

    def daily(self, symbol: str, day: date) -> Daily | None:
        close = self.closes.get(symbol)
        return Daily(close, 1.0) if close is not None else None

    def adj(self, symbol: str, day: date) -> float | None:
        return 1.0 if symbol in self.closes else None


def fake_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    broad: Sequence[str] = (),
    closes: dict[str, float] | None = None,
    primary: Sequence[str] = (),
    on: dict[str, Callable[[date], bool]] | None = None,
) -> tuple[list[tuple[list[str], date, date]], list[FakeContext]]:
    """Every context the plan loads is a FakeContext; returns (loads, contexts)."""
    loads: list[tuple[list[str], date, date]] = []
    made: list[FakeContext] = []

    async def load(engine: Any, symbols: Any, lo: date, hi: date) -> FakeContext:
        loads.append((sorted(symbols), lo, hi))
        made.append(FakeContext(set(broad), dict(closes or {}), set(primary), dict(on or {})))
        return made[-1]

    monkeypatch.setattr(units, "_context", load)
    return loads, made


# ── stories ───────────────────────────────────────────────────

S1 = date(2016, 10, 4)  # a Tuesday


async def test_stories_are_read_with_their_item_types_kinds_and_nsn_by_the_cutoff(
    engine: AsyncEngine,
) -> None:
    ids = await store(
        engine,
        [
            news_row(1, "AAA", ny(S1, 8), "Jefferies Downgrades AAA to Hold"),
            filing_row("acc-1", "CCC", ny(S1, 7), ["8.01"]),
        ],
        facts=False,
    )
    await add_story(engine, "AAA", S1, [(ids["alpaca:1"], "analyst_downgrade")], nsn_at=ny(S1, 10))
    await add_story(
        engine, "BBB", S1, [(990, "noise"), (991, "mover")], nsn_at=ny(S1, 15, 30)
    )  # after the 15:00 cutoff
    await add_story(engine, "CCC", S1, [(ids["acc-1"], "filing_other")], type_close="other")
    await add_story(engine, "DDD", S1, [(992, "analyst_downgrade")], version="stories-v0")
    await add_story(engine, "EEE", date(2016, 10, 10), [(993, "analyst_downgrade")])
    # NSN 5 minutes after the cutoff: by it when items are usable 60 s after their time.
    await add_story(engine, "FFF", S1, [(994, "analyst_downgrade")], nsn_at=ny(S1, 15, 5))

    got = {r.symbol: r for r in await read_stories(engine, S1, date(2016, 10, 7))}

    assert sorted(got) == ["AAA", "BBB", "CCC", "FFF"]
    assert (got["AAA"].nsn, got["AAA"].substantive, got["AAA"].has_8k) == (True, True, False)
    assert (got["BBB"].nsn, got["BBB"].substantive, got["BBB"].has_8k) == (False, False, False)
    assert (got["CCC"].nsn, got["CCC"].substantive, got["CCC"].has_8k) == (False, False, True)
    assert got["CCC"].type_close == "other" and got["AAA"].story_id == "AAA:2016-10-04"
    assert {s: (r.nsn, r.nsn_fast) for s, r in got.items()} == {
        "AAA": (True, True),
        "BBB": (False, False),  # 15:30 less 540 s is still after the cutoff
        "CCC": (False, False),
        "FFF": (False, True),
    }


async def test_train_takes_substantive_broad_liquid_stories_one_year_at_a_time(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = await store(engine, [filing_row("acc-1", "CCC", ny(S1, 7), ["2.02"])], facts=False)
    s2 = date(2017, 2, 7)
    await liquidity(engine, ["AAA", "BBB", "CCC", "EEE"], date(2015, 9, 1), date(2017, 1, 1))
    await add_story(engine, "AAA", S1, [(101, "analyst_pt_cut")])
    await add_story(engine, "BBB", S1, [(102, "noise"), (103, "other")])  # nothing substantive
    await add_story(engine, "CCC", S1, [(ids["acc-1"], "earnings_8k")])
    await add_story(engine, "DDD", S1, [(104, "analyst_downgrade")])  # not in the universe
    await add_story(engine, "EEE", S1, [(105, "analyst_downgrade")])  # not BROAD
    await add_story(engine, "AAA", s2, [(106, "product")])
    await add_story(engine, "AAA", date(2016, 9, 30), [(107, "product")])  # before train
    loads, _ = fake_context(monkeypatch, broad=["AAA", "BBB", "CCC", "DDD"])
    counts: Counter[str] = Counter()

    got = await window_units(engine, "train", counts=counts)

    expected = (
        {("AAA", d) for d in path(S1, 4, units.TRAIN[1])}
        | {("AAA", d) for d in path(s2, 4, units.TRAIN[1])}
        | {("CCC", d) for d in path(S1, 4, units.TRAIN[1])}
        | {("CCC", date(2016, 10, 3))}  # an 8-K story: S-1 too
    )
    assert got == expected
    assert loads == [
        (["AAA", "CCC", "EEE"], date(2016, 10, 3), date(2016, 12, 31)),
        (["AAA"], date(2017, 1, 1), date(2017, 12, 31)),
    ]
    assert counts["selected"] == 3 and counts["with_8k"] == 1


async def test_train_takes_primary_names_and_8k_stories_in_only_the_session_before(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    s2 = date(2017, 1, 3)  # the year's first session: S-1 is 2016-12-30
    ids = await store(
        engine,
        [
            filing_row("acc-1", "EKS", ny(s2, 7), ["2.02"]),
            filing_row("acc-2", "NOW", ny(S1, 7), ["2.02"]),
        ],
        facts=False,
    )
    await liquidity(engine, ["PRI", "EKS", "NOW"], date(2015, 9, 1), date(2017, 1, 1))
    await add_story(engine, "PRI", S1, [(101, "analyst_downgrade")])  # PRIMARY, not BROAD
    await add_story(engine, "EKS", s2, [(ids["acc-1"], "earnings_8k")])
    await add_story(engine, "NOW", S1, [(ids["acc-2"], "earnings_8k")])  # in at neither
    on = {"EKS": lambda d: d == date(2016, 12, 30), "NOW": lambda d: False}
    loads, _ = fake_context(monkeypatch, broad=["EKS", "NOW"], primary=["PRI"], on=on)
    counts: Counter[str] = Counter()

    got = await window_units(engine, "train", counts=counts)

    assert got == {("PRI", d) for d in path(S1, 4, units.TRAIN[1])} | {
        ("EKS", d) for d in path(date(2016, 12, 30), 4, units.TRAIN[1])
    }
    assert loads == [
        (["NOW", "PRI"], date(2016, 10, 3), date(2016, 12, 31)),
        (["EKS"], date(2016, 12, 30), date(2017, 12, 31)),  # from the S-1 a year back
    ]
    assert (counts["selected"], counts["with_8k"], counts["before_only"]) == (2, 1, 1)


async def test_validation_takes_the_stories_nsn_by_the_cutoff_at_the_fastest_news_lag(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = date(2023, 5, 2)  # cutoff 15:00; the 60 s lag makes items usable 9 minutes sooner
    names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    await liquidity(engine, names, date(2022, 4, 1), date(2023, 5, 1))
    await add_story(engine, "AAA", s, [(1, "analyst_downgrade")], nsn_at=ny(s, 10))
    await add_story(engine, "BBB", s, [(2, "analyst_downgrade")], nsn_at=ny(s, 15, 1))
    await add_story(engine, "CCC", s, [(3, "analyst_downgrade")])  # never NSN
    await add_story(engine, "DDD", s, [(4, "analyst_downgrade")], nsn_at=ny(s, 15, 9))
    await add_story(engine, "EEE", s, [(5, "analyst_downgrade")], nsn_at=ny(s, 15, 10))
    fake_context(monkeypatch, broad=names)

    got = await window_units(engine, "validation")

    assert got == {(n, d) for n in ("AAA", "BBB", "DDD") for d in path(s, 4, units.VALIDATION[1])}


async def test_h1_windows_are_the_train_and_validation_parts(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = date(2023, 5, 2)
    await liquidity(engine, ["AAA"], date(2015, 9, 1), date(2023, 5, 1))
    await add_story(engine, "AAA", S1, [(1, "product")])
    await add_story(engine, "AAA", s, [(2, "analyst_downgrade")], nsn_at=ny(s, 10))
    fake_context(monkeypatch, broad=["AAA"])
    counts: Counter[str] = Counter()

    got = await h1_windows(engine, counts=counts)

    assert got == {
        "train": await window_units(engine, "train"),
        "validation": await window_units(engine, "validation"),
    }
    assert counts["train.selected"] == 1 and counts["validation.selected"] == 1
    assert list(await h1_windows(engine, windows=["validation"])) == ["validation"]


# ── the real point-in-time context ────────────────────────────

VETO = "excluded by SPUS's Shariah index (holdings filed 2016-09-30) although within its size range"


SCREEN = [
    {"s": "AAA", "k": 100, "v": "halal", "r": "[]"},
    {"s": "VET", "k": 200, "v": "not_halal", "r": json.dumps([VETO])},
    {"s": "NOH", "k": 300, "v": "not_halal", "r": '["impermissible business"]'},
]


async def _daily_world(
    engine: AsyncEngine, symbols: Sequence[str], screen: Sequence[dict[str, Any]] = SCREEN
) -> None:
    days = sessions_between(date(2016, 5, 2), date(2016, 12, 30))
    rows = []
    for k, symbol in enumerate(["SPY", *symbols]):
        for j, day in enumerate(days):
            c = 40.0 * (1.0 + 0.01 * math.sin(j * (k + 1))) + 0.02 * j
            for adjustment in ("raw", "all"):
                rows.append(
                    {"s": symbol, "d": day, "a": adjustment, "o": c, "h": c * 1.01, "l": c * 0.99}
                    | {"c": c, "v": 1e6}
                )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES (:s, :d, :a, :o, :h, :l, :c, :v, now())"
            ),
            rows,
        )
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) "
                "VALUES ('2016-09-30', :s, :k, 'SERVICES-PREPACKAGED SOFTWARE', :v, "
                "CAST(:r AS JSONB), '{}', 'v12', now())"
            ),
            list(screen),
        )


async def test_train_asks_the_real_context_for_broad(engine: AsyncEngine) -> None:
    await _daily_world(engine, ["AAA", "VET", "NOH"])
    await liquidity(engine, ["AAA", "VET", "NOH"], date(2015, 10, 1), date(2016, 11, 1))
    for n, symbol in enumerate(["AAA", "VET", "NOH"]):
        await add_story(engine, symbol, S1, [(n + 1, "analyst_downgrade")])

    got = await window_units(engine, "train")

    days = path(S1, 4, units.TRAIN[1])
    assert got == {(s, d) for s in ("AAA", "VET") for d in days}  # NOH fails the screen


async def test_train_holds_a_share_class_primary_keeps_and_broad_does_not(
    engine: AsyncEngine,
) -> None:
    # One CIK, two classes: VCL, the more liquid, is out of PRIMARY on an index
    # veto only, so PRIMARY keeps HCL while BROAD keeps VCL (HCL: share_class).
    screen = [
        {"s": "VCL", "k": 400, "v": "not_halal", "r": json.dumps([VETO])},
        {"s": "HCL", "k": 400, "v": "halal", "r": "[]"},
    ]
    await _daily_world(engine, ["VCL", "HCL"], screen)
    await liquidity(engine, ["VCL", "HCL"], date(2015, 10, 1), date(2016, 11, 1))
    await add_story(engine, "VCL", S1, [(1, "analyst_downgrade")])
    await add_story(engine, "HCL", S1, [(2, "analyst_downgrade")])
    ctx = await PitContext.load(engine, symbols=["VCL", "HCL"], start=S1, end=S1)
    at = session_bounds(S1)[0]
    assert ctx.eligibility("HCL", S1, at_news=at, universe="primary").eligible
    assert ctx.eligibility("HCL", S1, at_news=at, universe="broad").reason == "share_class"
    assert ctx.eligibility("VCL", S1, at_news=at, universe="broad").eligible

    got = await window_units(engine, "train")

    days = path(S1, 4, units.TRAIN[1])
    assert got == {(s, d) for s in ("VCL", "HCL") for d in days}


async def test_the_g1_price_is_the_contexts_previous_close(engine: AsyncEngine) -> None:
    await _daily_world(engine, ["AAA"])
    # SPL splits 2:1 at 2016-08-15's open (raw closes halve, all-adjusted ones
    # do not); GAP has no bar on 2016-08-10.
    split, gap = date(2016, 8, 15), date(2016, 8, 10)
    rows = []
    for j, day in enumerate(sessions_between(date(2016, 5, 2), date(2016, 12, 30))):
        raw = (40.0 if day < split else 20.0) + 0.01 * j
        adjusted = raw * (0.5 if day < split else 1.0)
        for symbol, adjustment, c in [
            ("SPL", "raw", raw),
            ("SPL", "all", adjusted),
            *([] if day == gap else [("GAP", "raw", raw), ("GAP", "all", raw * 0.9)]),
        ]:
            rows.append({"s": symbol, "d": day, "a": adjustment, "c": c})
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES (:s, :d, :a, :c, :c, :c, :c, 1e6, now())"
            ),
            rows,
        )
    first, last = date(2016, 8, 1), date(2016, 8, 31)
    names = ["AAA", "SPL", "GAP", "NONE"]
    ctx = await PitContext.load(engine, symbols=names, start=first, end=last)

    for day in sessions_between(first, last):
        for symbol in names:
            series = ctx._loaded(symbol)
            want = None if series is None else context._prev_close_s(series, ctx._session(day))
            assert prev_close_s(ctx, symbol, day) == want, (symbol, day)
    before_split = 40.0 + 0.01 * (len(sessions_between(date(2016, 5, 2), split)) - 2)
    assert prev_close_s(ctx, "SPL", split) == pytest.approx(before_split * 0.5)
    assert prev_close_s(ctx, "GAP", date(2016, 8, 11)) is None
    assert prev_close_s(ctx, "GAP", gap) is None


# ── gate_g1 ───────────────────────────────────────────────────


async def test_g1_takes_liquid_stories_priced_at_5_or_more_nsn_first(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = date(2016, 3, 1)
    await liquidity(
        engine, ["NSN", "NEG", "POS", "CHEAP", "NOBAR"], date(2015, 1, 1), date(2016, 9, 1)
    )
    await add_story(engine, "NSN", s, [(1, "analyst_downgrade")], nsn_at=ny(s, 10))
    await add_story(engine, "NEG", s, [(2, "guidance_cut")], type_close="guidance_cut")
    await add_story(engine, "POS", s, [(3, "earnings_beat")], type_close="earnings_beat")
    await add_story(engine, "CHEAP", s, [(4, "analyst_downgrade")], nsn_at=ny(s, 10))
    await add_story(engine, "NOBAR", s, [(5, "analyst_downgrade")], nsn_at=ny(s, 10))
    await add_story(engine, "ILLIQ", s, [(6, "analyst_downgrade")], nsn_at=ny(s, 10))
    await add_story(engine, "NSN", date(2016, 10, 3), [(7, "analyst_downgrade")], nsn_at=ny(s, 10))
    loads, _ = fake_context(monkeypatch, closes={"NSN": 30.0, "NEG": 5.0, "CHEAP": 4.99})

    chosen = await g1_stories(engine)

    assert [(c.symbol, c.nsn) for c in chosen] == [("NSN", True), ("NEG", False)]
    assert loads == [(["CHEAP", "NEG", "NOBAR", "NSN"], *units.G1_RANGE)]


# ── SUE ───────────────────────────────────────────────────────


def _obs(symbol: str, at: datetime) -> Observation:
    return Observation(symbol, at.astimezone(UTC), 1.0)


async def test_the_sue_complement_is_liquid_outside_the_universe_with_a_clean_entry(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    await liquidity(engine, ["LIQ", "BRD", "PRI", "OVR"], date(2015, 1, 1), date(2019, 12, 1))
    overlap = _obs("OVR", ny(date(2016, 9, 20), 8))  # exits in October 2016, in train
    observations = [
        _obs("LIQ", ny(date(2017, 3, 1), 8)),
        _obs("LIQ", ny(date(2016, 11, 25), 14)),  # after the 13:00 early close
        _obs("LIQ", ny(date(2015, 12, 15), 8)),  # before the range
        _obs("LIQ", ny(date(2020, 1, 2), 8)),  # after it
        _obs("BRD", ny(date(2018, 5, 1), 12)),  # BROAD at its session
        _obs("PRI", ny(date(2018, 5, 1), 12)),  # PRIMARY at its session
        _obs("ILL", ny(date(2018, 5, 1), 12)),  # not in the universe
        _obs("LIQ", ny(date(2016, 3, 1), 12)),
        _obs("LIQ", ny(date(2019, 12, 31), 17)),  # enters 2020-01-02
        overlap,
    ]
    event = sue_event(overlap, sessions_between(*units._SUE_CALENDAR))
    assert not isinstance(event, str) and event.exits[-1] == date(2016, 10, 17)
    h1_units = {("OVR", date(2016, 10, 17)), ("LIQ", date(2016, 10, 17))}  # an H1 train path
    loads, made = fake_context(monkeypatch, broad=["BRD"], primary=["PRI"])
    counts: Counter[str] = Counter()

    got = await sue_complement(engine, h1_units=h1_units, observations=observations, counts=counts)

    assert [(e.symbol, e.session, e.entry) for e in got] == [
        ("LIQ", date(2016, 3, 1), "close"),
        ("LIQ", date(2017, 3, 1), "open"),
        ("LIQ", date(2020, 1, 2), "open"),
    ]
    assert dict(counts) == {
        "published": 8,
        "early_close": 1,
        "rank": 1,
        "universe": 2,
        "h1_overlap": 1,
        "complement": 3,
    }
    assert [lo.year for _, lo, _ in loads] == [2016, 2017, 2018, 2020]
    asked = [(s, d, at) for ctx in made for s, d, at, _ in ctx.asked]
    assert ("BRD", date(2018, 5, 1), ny(date(2018, 5, 1), 12)) in asked  # at the publication
    assert {u for ctx in made for *_, u in ctx.asked} == {"broad", "primary"}
    assert all(lo <= d <= hi for e in got for _, d in e.units() for lo, hi in [GATE_RANGES["sue"]])
    assert all(e.units().isdisjoint(h1_units) for e in got)
    without = await sue_complement(engine, h1_units=frozenset(), observations=observations)
    assert [e.symbol for e in without if e not in got] == ["OVR"]


async def test_calibration_pairs_are_seeded_liquid_and_never_an_earnings_session(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    await liquidity(engine, ["A", "B", "C"], date(2015, 1, 1), date(2016, 8, 1))
    observations = [_obs("A", ny(date(2016, 3, 1), 8))]
    days = sessions_between(*units.CALIB_RANGE)

    every = await calib_pairs(engine, observations=observations)
    assert len(every) == 3 * len(days) - 1  # fewer than 2,000: all of them
    assert ("A", date(2016, 3, 1)) not in every and ("B", date(2016, 3, 1)) in every
    assert every == sorted(every, key=lambda u: (u[1], u[0]))

    monkeypatch.setattr(units, "CALIB_PAIRS", 7)
    drawn = await calib_pairs(engine, observations=observations)
    assert len(drawn) == 7 and set(drawn) <= set(every)
    assert drawn == await calib_pairs(
        engine, observations=observations, ranks=LiquidityRanks(engine)
    )
    assert drawn != await calib_pairs(engine, seed=1, observations=observations)


# ── gate_reactor ──────────────────────────────────────────────


async def test_the_reactor_set_is_the_scores_before_the_pin(engine: AsyncEngine) -> None:
    mon, tue = date(2026, 3, 2), date(2026, 3, 3)
    ids = await store(
        engine,
        [
            news_row(1, "AAPL", ny(mon, 10), "Apple Unveils New Mac"),
            news_row(2, "MSFT", ny(mon, 11), "Microsoft Unveils New Surface"),
            news_row(3, "TSLA", ny(tue, 10), "Tesla Unveils New Car"),  # scored after the pin
            news_row(4, "NVDA", ny(tue, 16, 30), "Nvidia Unveils New Chip"),  # after 15:30
        ],
        facts=False,
    )
    scored = [("alpaca:1", 0.6, mon), ("alpaca:2", 0.1, mon), ("alpaca:3", 0.9, date(2026, 10, 11))]
    scored.append(("alpaca:4", -0.8, mon))
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO event_scores (event_id, scorer, score, scored_at) "
                "VALUES (:e, 'llm-batch:glm', :s, :t)"
            ),
            [{"e": ids[k], "s": s, "t": ny(d, 20)} for k, s, d in scored],
        )

    heads = await reactor_headlines(engine)

    assert sorted(h.symbol for h in heads) == ["AAPL", "MSFT"]
    assert units.reactor_units(heads) == {("AAPL", mon), ("MSFT", mon), ("SPY", mon)}


# ── the whole plan ────────────────────────────────────────────


async def test_the_plan_holds_every_part_and_each_gate_pins_its_set(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    g1_day, train_day, valid_day = date(2016, 3, 1), date(2017, 5, 2), date(2023, 5, 2)
    reactor_day = date(2026, 3, 2)
    # OVR enters the universe on 2017-03-02: its 2017-03-01 SUE event is in the
    # complement, but exits (h=20) on 2017-03-28, a session of its train story.
    ovr_day = date(2017, 3, 27)
    await liquidity(engine, ["AAA", "SUE", "OVR"], date(2015, 1, 1), date(2024, 12, 1))
    await add_story(engine, "AAA", g1_day, [(201, "analyst_downgrade")], nsn_at=ny(g1_day, 10))
    await add_story(engine, "AAA", train_day, [(202, "product")])
    await add_story(engine, "AAA", valid_day, [(203, "earnings_miss")], nsn_at=ny(valid_day, 9))
    await add_story(engine, "OVR", ovr_day, [(204, "product")])
    ids = await store(engine, [news_row(9, "AAA", ny(reactor_day, 10), "AAA Unveils")], facts=False)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO event_scores (event_id, scorer, score, scored_at) "
                "VALUES (:e, 'llm-batch:glm', 0.7, :t)"
            ),
            {"e": ids["alpaca:9"], "t": ny(reactor_day, 20)},
        )
    observations = [
        _obs("SUE", ny(date(2017, 3, 1), 8)),
        _obs("SUE", ny(date(2018, 8, 1), 17)),
        _obs("OVR", ny(date(2017, 3, 1), 8)),
    ]

    async def load_observations(engine: AsyncEngine) -> list[Observation]:
        return observations

    monkeypatch.setattr(units, "load_sue_observations", load_observations)
    on = {"OVR": lambda d: d >= date(2017, 3, 2)}
    fake_context(monkeypatch, broad=["AAA", "OVR"], closes={"AAA": 20.0}, on=on)
    counts: Counter[str] = Counter()
    await mark_built(engine, units.G1_RANGE[0], units.VALIDATION[1])

    plan = await h1_plan(engine, counts=counts)

    assert plan.fetch_order() == list(units.PARTS)
    assert plan.parts["gate_g1"] == {("AAA", d) for d in path(g1_day, 4, date(2016, 9, 30))}
    assert plan.parts["train"] == {
        *(("AAA", d) for d in path(train_day, 4, date(2021, 12, 31))),
        *(("OVR", d) for d in path(ovr_day, 4, date(2021, 12, 31))),
    }
    assert plan.parts["validation"] == {("AAA", d) for d in path(valid_day, 4, date(2024, 12, 31))}
    assert plan.parts["gate_reactor"] == {("AAA", reactor_day), ("SPY", reactor_day)}
    assert len(plan.parts["gate_sue"]) == 6 and {s for s, _ in plan.parts["gate_sue"]} == {"SUE"}
    assert counts["sue.sample"] == 2 and counts["sue.h1_overlap"] == 1
    assert counts["g1.stories"] == 1 and counts["g1.nsn"] == 1
    assert counts["train.selected"] == 2 and counts["validation.selected"] == 1
    plan.check()
    for part, gate in units.GATE_OF_PART.items():
        sha = await register_gate_units(
            engine,
            gate,
            plan.parts[part],
            expected_sha=plan.sha(part),  # type: ignore[arg-type]
        )
        assert sha == plan.sha(part) and await gate_pins(engine, gate) == {sha}

    only = await h1_plan(engine, parts=["gate_reactor", "spy"])
    assert only.fetch_order() == ["spy", "gate_reactor"]
    sue = await h1_plan(engine, parts=["gate_sue"])  # selects the windows, keeps only its part
    assert sue.fetch_order() == ["gate_sue"] and sue.parts["gate_sue"] == plan.parts["gate_sue"]
    with pytest.raises(units.PlanError, match="unknown part"):
        await h1_plan(engine, parts=["holdout"])


async def test_the_plan_reads_only_stories_a_complete_build_covers(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("read stories no complete build covers")

    monkeypatch.setattr(units, "read_stories", never)
    assert (await h1_plan(engine, parts=["spy"])).fetch_order() == ["spy"]  # reads no story
    whole = r"reads stories no complete build covers: 1322 session\(s\) in 2016-10-03\.\.2021-12-31"
    with pytest.raises(units.PlanError, match=whole) as refused:
        await h1_plan(engine)
    assert isinstance(refused.value.__cause__, StoriesNotReady)
    await mark_built(engine, *units.TRAIN)
    with pytest.raises(units.PlanError, match=r"in 2022-01-03\.\.2024-12-31 have no complete"):
        await h1_plan(engine, parts=["gate_sue"])  # Σ_c reads both windows' stories
    await mark_built(engine, *units.VALIDATION)
    with pytest.raises(units.PlanError, match=r"in 2016-01-04\.\.2016-09-30 have no complete"):
        await h1_plan(engine, parts=["gate_g1"])


async def test_the_plan_refuses_reactor_headlines_outside_the_gates_dates(
    engine: AsyncEngine,
) -> None:
    early = date(2025, 6, 2)  # in the 2025 holdout, before the gate's 2025-12 bars
    ids = await store(engine, [news_row(9, "AAA", ny(early, 10), "AAA Unveils")], facts=False)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO event_scores (event_id, scorer, score, scored_at) "
                "VALUES (:e, 'llm-batch:glm', 0.7, :t)"
            ),
            {"e": ids["alpaca:9"], "t": ny(early, 20)},
        )

    with pytest.raises(units.PlanError, match="gate_reactor: 2 unit"):
        await h1_plan(engine, parts=["gate_reactor"])


# ── fetching ──────────────────────────────────────────────────

D1, D2, D3 = date(2016, 3, 1), date(2016, 3, 2), date(2016, 3, 3)


def _bar(day: date, price: float = 10.0) -> dict[str, Any]:
    at = datetime.combine(day, time(9, 30), MARKET_TZ).astimezone(UTC)
    stamp = at.isoformat().replace("+00:00", "Z")
    return {"t": stamp, "o": price, "h": price, "l": price, "c": price, "v": 100.0, "vw": price}


class FakeMarket:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], date]] = []

    async def minute_bars_many(self, symbols, *, start, end):  # type: ignore[no-untyped-def]
        day = start.astimezone(MARKET_TZ).date()
        self.calls.append((tuple(symbols), day))
        return {s: [_bar(day)] for s in symbols if s != "EMPTY"}


PLAN = UnitPlan(
    {
        "train": frozenset({("A", D1), ("EMPTY", D2), ("A", D3)}),
        "spy": frozenset({("SPY", D1), ("SPY", D2), ("SPY", D3)}),
        "gate_g1": frozenset({("A", D1), ("SPY", D1)}),
    }
)


async def test_fetch_takes_the_parts_in_order_and_each_unit_once(engine: AsyncEngine) -> None:
    market = FakeMarket()
    seen: list[tuple[str, int, int]] = []

    report = await fetch(
        engine, market, PLAN, stop=None, chunk_sessions=2, on_part=lambda *a: seen.append(a)
    )

    assert market.calls == [
        (("SPY",), D2),  # backfill goes newest first inside a chunk
        (("SPY",), D1),
        (("SPY",), D3),
        (("A",), D1),
        (("A",), D3),
        (("EMPTY",), D2),
    ]
    assert seen == [("spy", 3, 3), ("gate_g1", 1, 1), ("train", 2, 1)]
    assert report.stopped is None and sum(report.stored.values()) == 5
    assert await minutes.done_units(engine) == {minutes.unit(*u) for u in PLAN.all()}
    again = await fetch(engine, market, PLAN, stop=None)
    assert len(market.calls) == 6 and sum(again.fetched.values()) == 0


async def test_fetch_stops_between_batches_and_resumes(engine: AsyncEngine) -> None:
    market = FakeMarket()
    answers = iter([None, "US market hours (09:30-16:00 ET)"])

    stopped = await fetch(engine, market, PLAN, stop=lambda now: next(answers), chunk_sessions=2)

    assert stopped.stopped == "US market hours (09:30-16:00 ET)"
    assert market.calls == [(("SPY",), D2), (("SPY",), D1)]
    resumed = await fetch(engine, market, PLAN, stop=None, chunk_sessions=2)
    assert resumed.fetched == {"spy": 1, "gate_g1": 1, "train": 2}
    assert await minutes.done_units(engine) == {minutes.unit(*u) for u in PLAN.all()}
