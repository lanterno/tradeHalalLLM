"""Point-in-time context (events/context.py): every answer from data dated before the story.

One synthetic world serves the tests: SPY and a dozen names with daily bars
(raw and all-adjusted) from 2022-10 to 2024-04, a liquidity history, two
screens and a few earnings facts. Expected values are recomputed here from
the world's own series, not read back from the context.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection
from dataclasses import dataclass, field, fields
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data.minutes import session_bounds
from halal_trader.events import context as context_module
from halal_trader.events import renames
from halal_trader.events.context import (
    DailyPoint,
    PitContext,
    PreEvent,
    Universe,
    index_veto_only,
)
from halal_trader.events.earnings_parse import EXTRACTOR
from halal_trader.market_hours import MARKET_TZ, is_trading_day
from halal_trader.signals.indicators import atr

FIRST, LAST = date(2022, 10, 3), date(2024, 4, 12)
SESSIONS = [
    d
    for d in (FIRST + timedelta(days=i) for i in range((LAST - FIRST).days + 1))
    if is_trading_day(d)
]
S = date(2024, 3, 19)  # a Tuesday: the story's reaction session
PREV = date(2024, 3, 18)  # S-1, a Monday
PREV2 = date(2024, 3, 15)  # S-2, a Friday
NEXT = date(2024, 3, 20)
SCREEN = date(2024, 3, 8)  # the screen the story sees
SCREEN_AT_S = S  # dated the session itself: not yet known at S
START, END = date(2023, 11, 1), date(2024, 3, 28)
EARLY = date(2023, 11, 24)  # closes at 13:00
AFTER_EARLY = date(2023, 11, 27)

VETO = (
    "excluded by SPUS's Shariah index (holdings filed 2024-02-29) although within its "
    "size range (market cap >= 12.0B)"
)
SOFTWARE = "SERVICES-PREPACKAGED SOFTWARE"
SEMIS = "SEMICONDUCTORS & RELATED DEVICES"
BANKING = "NATIONAL COMMERCIAL BANKS"

BETAS = {
    "AAA": 1.2,
    "AAB": 1.2,
    "VET": 0.9,
    "UNM": 1.0,
    "CHEAP": 1.0,
    "HALT": 1.1,
    "GAP": 1.0,
    "SPLIT": 1.3,
    "BETA3": 3.0,
    "BETA0": 0.0,
}
# Universe order (most traded first): ranks 0..11. THIN trades only from S's month.
LIQUIDITY = [
    "AAA",
    "AAB",
    "VET",
    "UNM",
    "FND",
    "NOH",
    "CHEAP",
    "HALT",
    "GAP",
    "SPLIT",
    "BETA3",
    "BETA0",
]
SCREEN_ROWS = [
    # symbol, verdict, cik, sic, reasons
    ("AAA", "halal", 100, SOFTWARE, []),
    ("AAB", "halal", 100, SOFTWARE, []),  # AAA's other share class, less traded
    ("VET", "not_halal", 200, SOFTWARE, [VETO]),
    ("VET2", "not_halal", 210, SOFTWARE, [VETO, "interest-bearing debt / market cap 0.41"]),
    ("UNM", "doubtful", None, "not an SEC registrant (or ticker not mapped)", ["unmapped"]),
    ("FND", "doubtful", None, "not an SEC registrant (or ticker not mapped)", ["unmapped"]),
    ("NOH", "not_halal", 300, BANKING, ["impermissible business: banking"]),
    ("CHEAP", "halal", 400, SOFTWARE, []),
    ("THIN", "halal", 500, SOFTWARE, []),
    ("HALT", "halal", 600, SOFTWARE, []),
    ("GAP", "halal", 700, SOFTWARE, []),
    ("SPLIT", "halal", 800, SEMIS, []),
    ("BETA3", "halal", 900, SOFTWARE, []),
    ("BETA0", "halal", 910, SOFTWARE, []),
]
SYMBOLS = sorted({r[0] for r in SCREEN_ROWS})


def et(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), MARKET_TZ)


PRE_OPEN = et(S, 8)  # pre-open news on S: sigma through S-1


def _i(day: date) -> int:
    return SESSIONS.index(day)


def _adj(symbol: str, day: date) -> float:
    """A(day) of the synthetic world."""
    if symbol == "SPLIT":  # 2-for-1, effective at S's open
        return 0.5 if day < S else 1.0
    if symbol == "AAA":  # a dividend going ex on S
        return 0.98 if day < S else 1.0
    if symbol == "SPY":
        return 0.995 if day < PREV2 else 1.0
    return 0.97 if day < date(2023, 6, 1) else 1.0


@dataclass
class World:
    closes: dict[str, dict[date, float]] = field(default_factory=dict)  # all-adjusted
    highs: dict[str, dict[date, float]] = field(default_factory=dict)
    lows: dict[str, dict[date, float]] = field(default_factory=dict)
    volumes: dict[str, dict[date, float]] = field(default_factory=dict)  # raw

    def raw_close(self, symbol: str, day: date) -> float:
        return self.closes[symbol][day] / _adj(symbol, day)


def _missing(symbol: str) -> set[date]:
    if symbol == "GAP":
        return {PREV}
    if symbol == "HALT":  # 25 sessions inside the sigma window: 26 returns lost, 34 left
        i = _i(PREV)
        return set(SESSIONS[i - 50 : i - 25])
    return set()


async def _world(engine: AsyncEngine) -> World:
    rng = np.random.default_rng(20261010)
    n = len(SESSIONS)
    spy_r = rng.normal(0.0004, 0.01, n)
    returns = {"SPY": spy_r}
    for symbol, beta in BETAS.items():
        returns[symbol] = beta * spy_r + rng.normal(0.0, 0.015, n)
    world = World()
    rows = []
    for symbol, r in returns.items():
        path = 40.0 * np.cumprod(1.0 + r)
        if symbol == "CHEAP":
            path *= 4.0 / path[_i(PREV)]  # closes at $4 the evening before S
        gone = _missing(symbol)
        for j, day in enumerate(SESSIONS):
            if day in gone:
                continue
            c = float(path[j])
            o = c * (1.0 + 0.003 * math.sin(j))
            h, lo = max(o, c) * 1.01, min(o, c) * 0.99
            v = 1e6 + 1000.0 * j
            a = _adj(symbol, day)
            world.closes.setdefault(symbol, {})[day] = c
            world.highs.setdefault(symbol, {})[day] = h
            world.lows.setdefault(symbol, {})[day] = lo
            world.volumes.setdefault(symbol, {})[day] = v
            rows.append((symbol, day, "all", o, h, lo, c, v * a))
            rows.append((symbol, day, "raw", o / a, h / a, lo / a, c / a, v))
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES (:s, :d, :a, :o, :h, :l, :c, :v, now())"
            ),
            [
                {"s": s, "d": d, "a": a, "o": o, "h": h, "l": lo, "c": c, "v": v}
                for s, d, a, o, h, lo, c, v in rows
            ],
        )
        months = [date(2022 + (9 + k) // 12, (9 + k) % 12 + 1, 1) for k in range(17)]  # to 2024-02
        monthly = [
            {"s": symbol, "m": m, "c": 50.0, "v": 1e9 / (rank + 1)}
            for rank, symbol in enumerate(LIQUIDITY)
            for m in months
        ]
        # THIN trades heavily, but only from S's month on: unknown at S.
        monthly += [{"s": "THIN", "m": date(2024, 3, 1), "c": 50.0, "v": 1e12}]
        await conn.execute(
            text(
                "INSERT INTO monthly_bars (symbol, month, close, volume, vwap) "
                "VALUES (:s, :m, :c, :v, :c)"
            ),
            monthly,
        )
        screen = [
            {"a": SCREEN, "s": s, "v": v, "k": cik, "d": sic, "r": _json(reasons)}
            for s, v, cik, sic, reasons in SCREEN_ROWS
        ]
        # The screen dated S itself turns AAA away: it may not count at S.
        screen += [
            {"a": SCREEN_AT_S, "s": "AAA", "v": "not_halal", "k": 100, "d": SOFTWARE, "r": "[]"}
        ]
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) "
                "VALUES (:a, :s, :k, :d, :v, CAST(:r AS JSONB), '{}', 'v12', now())"
            ),
            screen,
        )
    from halal_trader.compliance.delisted import Match, store_matches

    await store_matches(engine, [Match("FND", "fund"), Match("UNM", "no_match")])
    return world


def _json(reasons: list[str]) -> str:
    return json.dumps(reasons)


async def _load(engine: AsyncEngine, symbols: list[str] | None = None) -> PitContext:
    return await PitContext.load(engine, symbols=symbols or SYMBOLS, start=START, end=END)


async def _clone(
    engine: AsyncEngine,
    source: str,
    name: str,
    *,
    cik: int | None,
    verdict: str = "halal",
    sic: str = SOFTWARE,
    reasons: list[str] | None = None,
) -> None:
    """A new name trading as ``source`` did, less than every name in LIQUIDITY,
    with a row on SCREEN. No ticker_ciks row is written for it."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) SELECT :n, day, adjustment, open, high, low, close, volume, now() "
                "FROM daily_bars WHERE symbol = :s"
            ),
            {"n": name, "s": source},
        )
        await conn.execute(
            text(
                "INSERT INTO monthly_bars (symbol, month, close, volume, vwap) "
                "SELECT :n, month, close, volume * 0.01, vwap FROM monthly_bars WHERE symbol = :s"
            ),
            {"n": name, "s": source},
        )
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) "
                "VALUES (:a, :s, :k, :d, :v, CAST(:r AS JSONB), '{}', 'v12', now())"
            ),
            {"a": SCREEN, "s": name, "k": cik, "d": sic, "v": verdict, "r": _json(reasons or [])},
        )


async def _drop(
    engine: AsyncEngine, symbol: str, days: Collection[date], *, adjustment: str | None = None
) -> None:
    """Delete ``symbol``'s daily bars on ``days`` (one adjustment, or both)."""
    sql = "DELETE FROM daily_bars WHERE symbol = :s AND day = ANY(:d)"
    params: dict[str, object] = {"s": symbol, "d": list(days)}
    if adjustment is not None:
        sql += " AND adjustment = :a"
        params["a"] = adjustment
    async with engine.begin() as conn:
        await conn.execute(text(sql), params)


def _expected_sigma_beta(world: World, symbol: str, last: date) -> tuple[float, int, float]:
    """sigma, valid returns and clipped beta over the 60 sessions ending ``last``."""
    i = _i(last)
    own, spy = world.closes[symbol], world.closes["SPY"]
    rs, rm = [], []
    for x, px in zip(SESSIONS[i - 59 : i + 1], SESSIONS[i - 60 : i], strict=True):
        if x in own and px in own:
            rs.append(own[x] / own[px] - 1.0)
            rm.append(spy[x] / spy[px] - 1.0)
    abn = np.array(rs) - np.array(rm)
    beta = float(np.polyfit(rm, rs, 1)[0])
    return float(np.std(abn, ddof=1)), len(rs), min(max(beta, 0.5), 2.0)


# ── eligibility ──────────────────────────────────────────────────


async def test_eligibility_names_the_first_rule_a_name_fails(engine: AsyncEngine) -> None:
    await _world(engine)
    ctx = await _load(engine)

    reasons = {s: ctx.eligibility(s, S, at_news=PRE_OPEN).reason for s in SYMBOLS}

    assert reasons == {
        "AAA": "ok",
        "AAB": "share_class",  # same CIK as AAA, less traded
        "VET": "not_halal",  # an index's exclusion is a fail in PRIMARY
        "VET2": "not_halal",
        "UNM": "unmapped",
        "FND": "not_halal",  # a fund is never a company
        "NOH": "not_halal",
        "CHEAP": "price",  # $4 the evening before
        "THIN": "rank",  # not in the universe built from the months before S
        "HALT": "no_sigma",  # 34 valid returns of 60
        "GAP": "no_daily",  # no bar on S-1
        "SPLIT": "ok",
        "BETA3": "ok",
        "BETA0": "ok",
    }


async def test_broad_adds_index_veto_only_and_unmapped_names(engine: AsyncEngine) -> None:
    await _world(engine)
    ctx = await _load(engine)

    broad = {s: ctx.eligibility(s, S, at_news=PRE_OPEN, universe="broad").reason for s in SYMBOLS}

    assert broad["VET"] == "ok"
    assert broad["UNM"] == "ok"
    assert broad["VET2"] == "not_halal"  # the veto was not its only reason
    assert broad["FND"] == "not_halal"
    assert broad["AAA"] == "ok" and broad["AAB"] == "share_class"
    assert ctx.eligibility("VET", S, at_news=PRE_OPEN, universe="broad").universe == "broad"


async def test_eligibility_reports_what_was_known(engine: AsyncEngine) -> None:
    await _world(engine)
    ctx = await _load(engine)

    aaa = ctx.eligibility("AAA", S, at_news=PRE_OPEN)
    split = ctx.eligibility("SPLIT", S, at_news=PRE_OPEN)
    noh = ctx.eligibility("NOH", S, at_news=PRE_OPEN)

    assert aaa.eligible
    assert (aaa.screen_as_of, aaa.verdict, aaa.cik) == (SCREEN, "halal", 100)
    assert (aaa.sector, aaa.tech) == ("Technology", True)
    assert (aaa.liquidity_rank, aaa.cost_bps) == (0, 7.0)
    assert (split.liquidity_rank, split.tech) == (9, True)
    assert not noh.eligible and not noh.tech and noh.liquidity_rank == 5
    thin = ctx.eligibility("THIN", S, at_news=PRE_OPEN)
    assert (thin.liquidity_rank, thin.cost_bps) == (None, 30.0)


async def test_a_screen_counts_only_after_its_date(engine: AsyncEngine) -> None:
    await _world(engine)
    ctx = await _load(engine)
    dec1, mar11 = date(2023, 12, 1), date(2024, 3, 11)

    # Dated that day: not yet known.
    assert ctx.eligibility("AAA", SCREEN, at_news=et(SCREEN, 8)).reason == "no_screen"
    assert ctx.eligibility("AAA", dec1, at_news=et(dec1, 8)).reason == "no_screen"
    assert ctx.eligibility("AAA", mar11, at_news=et(mar11, 8)).reason == "ok"
    # The screen dated S is not known at S.
    assert ctx.eligibility("AAA", S, at_news=PRE_OPEN).reason == "ok"
    assert ctx.eligibility("AAA", NEXT, at_news=et(NEXT, 8)).reason == "not_halal"
    assert ctx.screen_verdict("AAA", SCREEN) == "no_screen"
    assert ctx.screen_verdict("AAA", S) == "halal"
    assert ctx.screen_verdict("AAA", NEXT) == "not_halal"
    assert ctx.screen_verdict("ZZZ", NEXT) == "not_halal"  # absent from the screen


async def test_a_rank_past_999_is_out(engine: AsyncEngine) -> None:
    await _world(engine)
    months = [date(2023, 3 + k, 1) if k < 10 else date(2024, k - 9, 1) for k in range(12)]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO monthly_bars (symbol, month, close, volume, vwap) "
                "VALUES (:s, :m, 50, 1e10, 50)"
            ),
            [{"s": f"F{k:04d}", "m": m} for k in range(1000) for m in months],
        )
    ctx = await _load(engine)

    aaa = ctx.eligibility("AAA", S, at_news=PRE_OPEN)

    assert (aaa.reason, aaa.liquidity_rank, aaa.cost_bps) == ("rank", 1000, 30.0)


async def test_sigma_can_be_judged_at_the_news(engine: AsyncEngine) -> None:
    await _world(engine)
    ctx = await _load(engine)

    # HALT's 60 sessions back from S-2 still lose 25 returns; from much later news, more.
    assert ctx.eligibility("HALT", S, at_news=et(PREV, 15)).reason == "no_sigma"
    assert ctx.eligibility("AAA", S, at_news=et(PREV, 15)).reason == "ok"
    assert ctx.pre_event("HALT", S, et(S, 8)) is None


async def test_eligibility_and_pre_event_judge_sigma_on_one_window(engine: AsyncEngine) -> None:
    await _world(engine)
    i = _i(PREV)  # S's index less one
    # EDGE misses S-2's bar (the returns on S-2 and S-1) and 18 bars further back
    # (19 returns). The window of late news on S-1 (returns S-61..S-2) keeps 40
    # valid; the pre-open window (S-60..S-1) keeps 39.
    await _clone(engine, "UNM", "EDGE", cik=990)
    await _drop(engine, "EDGE", {PREV2, *SESSIONS[i - 39 : i - 21]})
    ctx = await _load(engine, [*SYMBOLS, "EDGE"])
    late = et(PREV, 15)

    assert ctx.eligibility("EDGE", S, at_news=late).reason == "ok"
    pe = ctx.pre_event("EDGE", S, late)
    assert pe is not None and pe.sigma_n == 40
    assert ctx.eligibility("EDGE", S, at_news=PRE_OPEN).reason == "no_sigma"
    assert ctx.pre_event("EDGE", S, PRE_OPEN) is None
    with pytest.raises(TypeError):
        ctx.eligibility("EDGE", S)  # type: ignore[call-arg]  # no default window


@pytest.mark.parametrize("failure", ["no_daily", "price", "no_sigma"])
async def test_a_share_class_winner_that_fails_later_keeps_the_other_class_out(
    engine: AsyncEngine, failure: str
) -> None:
    world = await _world(engine)
    i = _i(PREV)
    if failure == "no_daily":
        await _drop(engine, "AAA", {PREV})
    elif failure == "price":  # $4 the evening before, A kept
        await _scale(engine, ["AAA"], since=PREV, until=PREV, by=4.0 / world.closes["AAA"][PREV])
    else:  # HALT's gap: 25 bars inside the sigma window
        await _drop(engine, "AAA", set(SESSIONS[i - 50 : i - 25]))
    ctx = await _load(engine)

    aaa = ctx.eligibility("AAA", S, at_news=PRE_OPEN)
    aab = ctx.eligibility("AAB", S, at_news=PRE_OPEN)

    # The winner is chosen on screen and rank, before any bar is read: AAB does not step up.
    assert aaa.reason == failure
    assert (aab.reason, aab.eligible, aab.cik) == ("share_class", False, 100)
    assert ctx.eligibility("AAB", S, at_news=PRE_OPEN, universe="broad").reason == "share_class"


async def test_a_renamed_companys_later_ticker_wins_its_cik_whatever_the_ranks(
    engine: AsyncEngine,
) -> None:
    # Live screens hold IAC and PPLI under one CIK, with the same bars: they
    # tie on liquidity and the universe's symbol order ranks IAC first. The
    # story builder files the company's news under PPLI (renames.owner).
    await _world(engine)
    for symbol in ("IAC", "PPLI"):
        await _clone(engine, "UNM", symbol, cik=1800227)
    ctx = await _load(engine, [*SYMBOLS, "IAC", "PPLI"])

    iac = ctx.eligibility("IAC", S, at_news=PRE_OPEN)
    ppli = ctx.eligibility("PPLI", S, at_news=PRE_OPEN)

    assert (iac.liquidity_rank, ppli.liquidity_rank) == (12, 13)  # IAC is the more liquid
    assert (ppli.reason, ppli.eligible, ppli.cik) == ("ok", True, 1800227)
    assert (iac.reason, iac.eligible) == ("share_class", False)
    broad = {s: ctx.eligibility(s, S, at_news=PRE_OPEN, universe="broad") for s in ("IAC", "PPLI")}
    assert (broad["IAC"].reason, broad["PPLI"].reason) == ("share_class", "ok")
    # As for any winner, one that then fails its bars does not let the other step up.
    await _drop(engine, "PPLI", {PREV})
    ctx = await _load(engine, [*SYMBOLS, "IAC", "PPLI"])
    assert ctx.eligibility("PPLI", S, at_news=PRE_OPEN).reason == "no_daily"
    assert ctx.eligibility("IAC", S, at_news=PRE_OPEN).reason == "share_class"


async def test_an_old_ticker_wins_where_the_screen_keeps_its_later_one_out(
    engine: AsyncEngine,
) -> None:
    # Live 2021-10 to 2022-09: an index veto alone fails PPLI (the index held
    # the company as IAC). PRIMARY admits IAC only, so IAC wins there; BROAD
    # admits both, and PPLI wins.
    await _world(engine)
    await _clone(engine, "UNM", "IAC", cik=1800227)
    await _clone(engine, "UNM", "PPLI", cik=1800227, verdict="not_halal", reasons=[VETO])
    ctx = await _load(engine, [*SYMBOLS, "IAC", "PPLI"])

    def reasons(universe: Universe) -> dict[str, str]:
        return {
            s: ctx.eligibility(s, S, at_news=PRE_OPEN, universe=universe).reason
            for s in ("IAC", "PPLI")
        }

    assert reasons("primary") == {"IAC": "ok", "PPLI": "not_halal"}
    assert reasons("broad") == {"IAC": "share_class", "PPLI": "ok"}


async def test_a_chain_of_renames_keeps_the_last_ticker_and_a_recycled_one_is_no_chain(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        renames,
        "TICKER_RENAMES",
        {
            "ABLE": ("MIDL", date(2016, 2, 10)),  # one company: ABLE, MIDL, then ZEST
            "MIDL": ("ZEST", date(2016, 3, 4)),
            "CORE": ("RCY", date(2016, 3, 4)),  # took RCY the day RCY's company became ZOOM
            "RCY": ("ZOOM", date(2016, 3, 4)),
        },
    )
    await _world(engine)
    for symbol in ("ABLE", "ZEST"):  # the screen holds the chain's ends only
        await _clone(engine, "UNM", symbol, cik=991)
    for symbol in ("CORE", "ZOOM"):
        await _clone(engine, "UNM", symbol, cik=992)
    names = ["ABLE", "CORE", "ZEST", "ZOOM"]
    ctx = await _load(engine, [*SYMBOLS, *names])

    got = {s: ctx.eligibility(s, S, at_news=PRE_OPEN) for s in names}

    assert [got[s].liquidity_rank for s in names] == [12, 13, 14, 15]  # equal bars: by symbol
    assert {s: e.reason for s, e in got.items()} == {
        "ABLE": "share_class",
        "ZEST": "ok",  # ABLE's company took it through MIDL
        "CORE": "ok",  # CORE's company never traded as ZOOM: the rank decides
        "ZOOM": "share_class",
    }


async def test_broad_counts_a_name_without_a_ticker_ciks_row_as_no_fund(
    engine: AsyncEngine,
) -> None:
    await _world(engine)
    unmapped = "not an SEC registrant (or ticker not mapped)"
    await _clone(engine, "UNM", "ORPH", cik=None, verdict="doubtful", sic=unmapped)
    await _clone(engine, "UNM", "ETF2", cik=None, verdict="doubtful", sic=unmapped)
    from halal_trader.compliance.delisted import Match, store_matches

    await store_matches(engine, [Match("ETF2", "fund")])
    async with engine.connect() as conn:
        rows = (
            await conn.execute(text("SELECT count(*) FROM ticker_ciks WHERE symbol = 'ORPH'"))
        ).scalar_one()
    assert rows == 0
    ctx = await _load(engine, [*SYMBOLS, "ORPH", "ETF2"])

    # No ticker_ciks row: not known to be a fund, so BROAD admits it (spec's
    # SQL `status != 'fund'` would not); PRIMARY still wants a CIK.
    assert ctx.eligibility("ORPH", S, at_news=PRE_OPEN).reason == "unmapped"
    assert ctx.eligibility("ORPH", S, at_news=PRE_OPEN, universe="broad").reason == "ok"
    assert ctx.eligibility("ETF2", S, at_news=PRE_OPEN, universe="broad").reason == "not_halal"


async def test_a_name_without_a_bar_on_the_session_is_no_daily(engine: AsyncEngine) -> None:
    # A(S) reads S's own bar: a name halted all of S is refused here, before the
    # simulator could call it halted_all_day. Either half of the bar missing is enough.
    await _world(engine)
    await _drop(engine, "BETA0", {S})
    await _drop(engine, "BETA3", {S}, adjustment="all")
    ctx = await _load(engine)

    for symbol in ("BETA0", "BETA3"):
        assert ctx.eligibility(symbol, S, at_news=PRE_OPEN).reason == "no_daily"
        assert ctx.pre_event(symbol, S, PRE_OPEN) is None
        assert ctx.daily(symbol, S) is None and ctx.adj(symbol, S) is None
        assert ctx.daily(symbol, PREV) is not None  # S-1 is there: only S's own bar is missing


async def test_news_from_before_the_previous_session_is_refused(engine: AsyncEngine) -> None:
    await _world(engine)
    ctx = await _load(engine)

    # Late in S-2's session, or at its close: nothing closed in between, so the
    # news cannot belong to a story reacting in S. Refused before any rule is read.
    for at in (et(PREV2, 15), et(PREV2, 16)):
        with pytest.raises(ValueError, match="too early"):
            ctx.pre_event("AAA", S, at)
        with pytest.raises(ValueError, match="too early"):
            ctx.eligibility("NOH", S, at_news=at)
    with pytest.raises(ValueError, match="too early"):
        ctx.pre_event("GAP", S, et(PREV2, 15))  # even without the bars to answer
    # A minute after S-2's close, S-1 is still to close: one session back is allowed.
    assert ctx.pre_event("AAA", S, et(PREV2, 16, 1)) is not None
    assert ctx.pre_event("AAA", S, et(PREV, 8)) is not None


# ── pre-event state ──────────────────────────────────────────────


async def test_pre_event_known_answers(engine: AsyncEngine) -> None:
    world = await _world(engine)
    ctx = await _load(engine)

    pe = ctx.pre_event("AAA", S, et(S, 8))  # pre-open news

    assert pe is not None
    i = _i(PREV)
    sigma, n, beta = _expected_sigma_beta(world, "AAA", PREV)
    closes, spy = world.closes["AAA"], world.closes["SPY"]
    window20 = SESSIONS[i - 19 : i + 1]
    window252 = SESSIONS[i - 251 : i + 1]
    window60 = SESSIONS[i - 59 : i + 1]
    assert (pe.session, pe.prev_session) == (S, PREV)
    # A dividend goes ex on S: A(S-1)/A(S) = 0.98 carries the raw close into S's units.
    assert pe.prev_close_s == pytest.approx(world.raw_close("AAA", PREV) * 0.98, rel=1e-12)
    assert pe.prev_close_s == pytest.approx(closes[PREV], rel=1e-12)
    assert pe.spy_prev_close_s == pytest.approx(world.raw_close("SPY", PREV), rel=1e-12)
    assert (pe.sigma, pe.sigma_n) == (pytest.approx(sigma, rel=1e-10), n)
    assert n == 60
    assert pe.beta == pytest.approx(beta, rel=1e-9)
    expected_atr = atr(
        np.array([world.highs["AAA"][d] for d in window60]),
        np.array([world.lows["AAA"][d] for d in window60]),
        np.array([closes[d] for d in window60]),
        14,
    )
    assert pe.atr_pct == pytest.approx(expected_atr / closes[PREV], rel=1e-12)
    dollar = [world.raw_close("AAA", d) * world.volumes["AAA"][d] for d in window20]
    assert pe.adv20_usd == pytest.approx(sum(dollar) / 20, rel=1e-12)
    for h, got in ((5, pe.ret5_vs_spy), (20, pe.ret20_vs_spy)):
        back = SESSIONS[i - h]
        want = (closes[PREV] / closes[back] - 1) - (spy[PREV] / spy[back] - 1)
        assert got == pytest.approx(want, rel=1e-10)
    # Levels: raw high * A(d) / A(S), i.e. the adjusted high since A(S) = 1.
    assert pe.hi20_s == pytest.approx(max(world.highs["AAA"][d] for d in window20), rel=1e-12)
    assert pe.lo20_s == pytest.approx(min(world.lows["AAA"][d] for d in window20), rel=1e-12)
    assert pe.hi252_s == pytest.approx(max(world.highs["AAA"][d] for d in window252), rel=1e-12)
    assert pe.lo252_s == pytest.approx(min(world.lows["AAA"][d] for d in window252), rel=1e-12)


async def test_a_split_on_the_session_is_carried_by_the_a_ratio(engine: AsyncEngine) -> None:
    world = await _world(engine)
    ctx = await _load(engine)

    pe = ctx.pre_event("SPLIT", S, et(S, 8))

    assert pe is not None
    raw_prev = world.raw_close("SPLIT", PREV)  # twice the post-split price
    assert ctx.adj("SPLIT", PREV) == pytest.approx(0.5)
    assert ctx.adj("SPLIT", S) == pytest.approx(1.0)
    assert pe.prev_close_s == pytest.approx(raw_prev * 0.5, rel=1e-12)
    assert pe.hi20_s < raw_prev  # levels are in post-split units too
    point = ctx.daily("SPLIT", S)
    assert point is not None
    assert (point.day, point.adj) == (S, pytest.approx(1.0))
    assert point.close == pytest.approx(world.raw_close("SPLIT", S), rel=1e-12)
    before = ctx.daily("SPLIT", PREV)
    assert before is not None and before.close == pytest.approx(raw_prev, rel=1e-12)
    assert before.volume == world.volumes["SPLIT"][PREV]


@pytest.mark.parametrize(
    ("story", "at_news", "last"),
    [
        (S, et(S, 8), PREV),  # pre-open news: S-1 back
        (S, et(S, 11), PREV),  # news in S's session: S-1 back
        (S, et(PREV, 17), PREV),  # after N's close: N back
        (S, et(PREV, 15), PREV2),  # late in N's session: N-1 back
        (S, et(PREV, 16), PREV2),  # at N's close exactly: not strictly before
        (PREV, et(date(2024, 3, 17), 20), PREV2),  # Sunday news for a Monday story
        (AFTER_EARLY, et(EARLY, 13, 30), EARLY),  # after a 13:00 early close
        (AFTER_EARLY, et(EARLY, 12, 30), date(2023, 11, 22)),  # before it (23rd: Thanksgiving)
    ],
)
async def test_sigma_uses_the_sessions_closed_strictly_before_the_news(
    engine: AsyncEngine, story: date, at_news: datetime, last: date
) -> None:
    world = await _world(engine)
    ctx = await _load(engine)

    pe = ctx.pre_event("AAA", story, at_news)

    assert pe is not None
    sigma, n, beta = _expected_sigma_beta(world, "AAA", last)
    assert pe.sigma == pytest.approx(sigma, rel=1e-10)
    assert pe.sigma_n == n
    assert pe.beta == pytest.approx(beta, rel=1e-9)


async def test_beta_is_clipped_to_half_and_two(engine: AsyncEngine) -> None:
    await _world(engine)
    ctx = await _load(engine, [*SYMBOLS])

    high = ctx.pre_event("BETA3", S, et(S, 8))
    low = ctx.pre_event("BETA0", S, et(S, 8))

    assert high is not None and low is not None
    assert (high.beta, low.beta) == (2.0, 0.5)


@pytest.mark.parametrize(
    ("story", "at_news", "last"),
    [
        # 2024-03-11, the Monday after clocks went forward, closed at 20:00 UTC:
        # 20:30 UTC is after its close (a fixed EST offset would call it 15:30).
        (date(2024, 3, 12), datetime(2024, 3, 11, 20, 30, tzinfo=UTC), date(2024, 3, 11)),
        # 2023-11-06, the Monday after clocks went back, closed at 21:00 UTC:
        # 20:30 UTC is 15:30, late in its session (a fixed EDT offset: 16:30, after).
        (date(2023, 11, 7), datetime(2023, 11, 6, 20, 30, tzinfo=UTC), date(2023, 11, 3)),
        # News on the change days themselves, for the Monday stories.
        (date(2024, 3, 11), datetime(2024, 3, 10, 7, 30, tzinfo=UTC), date(2024, 3, 8)),
        (date(2023, 11, 6), datetime(2023, 11, 5, 6, 30, tzinfo=UTC), date(2023, 11, 3)),
    ],
)
async def test_the_sigma_cutoff_follows_the_clock_change(
    engine: AsyncEngine, story: date, at_news: datetime, last: date
) -> None:
    world = await _world(engine)
    ctx = await _load(engine)

    pe = ctx.pre_event("AAA", story, at_news)

    assert pe is not None
    sigma, n, beta = _expected_sigma_beta(world, "AAA", last)
    assert (pe.sigma, pe.sigma_n) == (pytest.approx(sigma, rel=1e-10), n)
    assert pe.beta == pytest.approx(beta, rel=1e-9)


async def test_levels_cover_every_session_they_name_or_are_nan(engine: AsyncEngine) -> None:
    world = await _world(engine)
    i = _i(PREV)
    window = SESSIONS[i - 251 : i + 1]
    await _clone(engine, "UNM", "EXACT", cik=991)  # first bar on the window's first session
    await _drop(engine, "EXACT", set(SESSIONS[: i - 251]))
    await _clone(engine, "UNM", "LATE", cik=992)  # first bar one session later
    await _drop(engine, "LATE", set(SESSIONS[: i - 250]))
    await _clone(engine, "UNM", "HOLE", cik=993)  # listed before, halted inside the window
    hole = set(SESSIONS[i - 200 : i - 195])
    await _drop(engine, "HOLE", hole)
    ctx = await _load(engine, [*SYMBOLS, "EXACT", "LATE", "HOLE"])

    exact, late, gap = (ctx.pre_event(s, S, PRE_OPEN) for s in ("EXACT", "LATE", "HOLE"))

    assert exact is not None and late is not None and gap is not None
    highs, lows = world.highs["UNM"], world.lows["UNM"]  # A(S) = 1: adjusted = S units
    assert exact.hi252_s == pytest.approx(max(highs[d] for d in window), rel=1e-12)
    assert exact.lo252_s == pytest.approx(min(lows[d] for d in window), rel=1e-12)
    assert math.isnan(late.hi252_s) and math.isnan(late.lo252_s)
    assert late.hi20_s == exact.hi20_s and late.lo20_s == exact.lo20_s
    kept = [d for d in window if d not in hole]
    assert gap.hi252_s == pytest.approx(max(highs[d] for d in kept), rel=1e-12)
    assert gap.lo252_s == pytest.approx(min(lows[d] for d in kept), rel=1e-12)


async def test_levels_are_nan_when_the_calendar_is_shorter_than_their_window(
    engine: AsyncEngine,
) -> None:
    world = await _world(engine)
    story = date(2023, 3, 15)  # about 110 sessions after the first daily bar
    ctx = await PitContext.load(
        engine, symbols=["AAA"], start=date(2023, 3, 1), end=date(2023, 3, 31)
    )

    pe = ctx.pre_event("AAA", story, et(story, 8))

    assert pe is not None
    assert ctx.sessions[0] == FIRST
    assert math.isnan(pe.hi252_s) and math.isnan(pe.lo252_s)
    p = _i(story) - 1
    window20 = SESSIONS[p - 19 : p + 1]
    # A(d) = A(story) = 0.98 here: S units are raw, the adjusted high / 0.98.
    assert pe.hi20_s == pytest.approx(max(world.highs["AAA"][d] for d in window20) / 0.98)
    assert pe.lo20_s == pytest.approx(min(world.lows["AAA"][d] for d in window20) / 0.98)


def _comparable(pe: PreEvent | None) -> tuple[object, ...] | None:
    """``pe``'s fields, NaN as a marker: dataclass equality holds NaN unequal to itself."""
    if pe is None:
        return None
    values = (getattr(pe, f.name) for f in fields(pe))
    return tuple("NaN" if isinstance(v, float) and math.isnan(v) else v for v in values)


async def test_levels_do_not_depend_on_where_the_load_starts(engine: AsyncEngine) -> None:
    world = await _world(engine)
    i = _i(PREV)
    window = SESSIONS[i - 251 : i + 1]
    # Listed with the world, then halted from before a load starting at S
    # reaches (S - 380 days) through the 252-session window's first session.
    await _clone(engine, "UNM", "PAUSED", cik=994)
    halted = {d for d in SESSIONS if date(2023, 2, 1) <= d <= window[0]}
    await _drop(engine, "PAUSED", halted)
    await _clone(engine, "UNM", "LATE", cik=992)  # first bar one session into the window
    await _drop(engine, "LATE", set(SESSIONS[: i - 250]))
    names = [*SYMBOLS, "PAUSED", "LATE"]

    short = await PitContext.load(engine, symbols=names, start=S, end=S)
    long = await PitContext.load(engine, symbols=names, start=START, end=END)

    # The short load holds none of PAUSED's bars on or before the window's start.
    assert short.sessions[0] > min(halted)
    assert all(short.daily("PAUSED", d) is None for d in short.sessions if d <= window[0])
    assert {s: _comparable(short.pre_event(s, S, PRE_OPEN)) for s in names} == {
        s: _comparable(long.pre_event(s, S, PRE_OPEN)) for s in names
    }
    assert {s: short.eligibility(s, S, at_news=PRE_OPEN) for s in names} == {
        s: long.eligibility(s, S, at_news=PRE_OPEN) for s in names
    }
    paused, late = (short.pre_event(s, S, PRE_OPEN) for s in ("PAUSED", "LATE"))
    assert paused is not None and late is not None
    kept = [d for d in window if d not in halted]
    assert paused.hi252_s == pytest.approx(max(world.highs["UNM"][d] for d in kept), rel=1e-12)
    assert paused.lo252_s == pytest.approx(min(world.lows["UNM"][d] for d in kept), rel=1e-12)
    assert math.isnan(late.hi252_s) and math.isnan(late.lo252_s)


async def test_a_level_needs_a_bar_on_nine_in_ten_of_its_sessions(engine: AsyncEngine) -> None:
    world = await _world(engine)
    i = _i(PREV)
    window = SESSIONS[i - 251 : i + 1]
    window20 = SESSIONS[i - 19 : i + 1]
    # Holes before the sigma window (the 60 sessions through S-1) leave sigma whole.
    gaps = {
        "MOST": set(SESSIONS[i - 200 : i - 175]),  # 25 missing: 227 of 252
        "FEW": set(SESSIONS[i - 200 : i - 174]),  # 26 missing: 226 of 252
        "TWO": {SESSIONS[i - 10], SESSIONS[i - 5]},  # 18 of 20
        "THREE": {SESSIONS[i - 12], SESSIONS[i - 10], SESSIONS[i - 5]},  # 17 of 20
    }
    for cik, (name, gone) in enumerate(gaps.items(), start=980):
        await _clone(engine, "UNM", name, cik=cik)
        await _drop(engine, name, gone)
    ctx = await _load(engine, [*SYMBOLS, *gaps])

    got = {s: ctx.pre_event(s, S, PRE_OPEN) for s in gaps}

    highs, lows = world.highs["UNM"], world.lows["UNM"]  # A(S) = 1: adjusted = S units

    def level(days: list[date], gone: set[date]) -> tuple[float, float]:
        kept = [d for d in days if d not in gone]
        return max(highs[d] for d in kept), min(lows[d] for d in kept)

    most, few, two, three = (got[s] for s in gaps)
    assert most is not None and few is not None and two is not None and three is not None
    assert (most.hi252_s, most.lo252_s) == pytest.approx(level(window, gaps["MOST"]), rel=1e-12)
    assert math.isnan(few.hi252_s) and math.isnan(few.lo252_s)
    assert few.hi20_s == most.hi20_s and few.lo20_s == most.lo20_s
    assert (two.hi20_s, two.lo20_s) == pytest.approx(level(window20, gaps["TWO"]), rel=1e-12)
    assert math.isnan(three.hi20_s) and math.isnan(three.lo20_s)
    assert (three.hi252_s, three.lo252_s) == pytest.approx(level(window, gaps["THREE"]), rel=1e-12)


async def test_a_recycled_ticker_gets_no_252_session_level_from_a_few_bars(
    engine: AsyncEngine,
) -> None:
    world = await _world(engine)
    i = _i(PREV)
    window = SESSIONS[i - 251 : i + 1]
    # An earlier company traded as RCY to the end of 2022; the ticker's new
    # company starts 100 sessions before S. The first raw bar is the old one's.
    await _clone(engine, "UNM", "RCY", cik=995)
    await _drop(engine, "RCY", {d for d in SESSIONS if date(2023, 1, 1) <= d < SESSIONS[i - 99]})
    names = [*SYMBOLS, "RCY"]
    loads = [
        await PitContext.load(engine, symbols=names, start=start, end=end)
        for start, end in ((S, S), (START, END), (date(2022, 11, 1), END))
    ]

    answers = [ctx.pre_event("RCY", S, PRE_OPEN) for ctx in loads]

    assert date(2023, 1, 1) < window[0] < SESSIONS[i - 99]  # the window holds 100 bars
    assert len({_comparable(pe) for pe in answers}) == 1
    pe = answers[0]
    assert pe is not None
    assert math.isnan(pe.hi252_s) and math.isnan(pe.lo252_s)
    window20 = SESSIONS[i - 19 : i + 1]
    assert pe.hi20_s == pytest.approx(max(world.highs["UNM"][d] for d in window20), rel=1e-12)
    assert pe.lo20_s == pytest.approx(min(world.lows["UNM"][d] for d in window20), rel=1e-12)


# ── look-ahead ───────────────────────────────────────────────────


async def _scale(
    engine: AsyncEngine, symbols: list[str], *, since: date, until: date, by: float
) -> None:
    """Move raw and adjusted bars together (A kept), volume too."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE daily_bars SET open = open * :f, high = high * :f, low = low * :f, "
                "close = close * :f, volume = volume * :f WHERE symbol = ANY(:s) "
                "AND day >= :lo AND day <= :hi"
            ),
            {"f": by, "s": symbols, "lo": since, "hi": until},
        )


async def test_bars_from_the_session_on_change_nothing(engine: AsyncEngine) -> None:
    await _world(engine)
    names = ["AAA", "SPLIT", "VET", "UNM"]
    before = await _load(engine)
    pre = {s: before.pre_event(s, S, et(S, 8)) for s in names}
    elig = {
        (s, u): before.eligibility(s, S, at_news=PRE_OPEN, universe=u)
        for s in SYMBOLS
        for u in ("primary", "broad")
    }

    # A power of two scales raw and adjusted prices exactly, so A(S) keeps every bit.
    await _scale(engine, ["SPY", *SYMBOLS], since=S, until=LAST, by=2.0)
    after = await _load(engine)

    assert {s: after.pre_event(s, S, et(S, 8)) for s in names} == pre
    assert {
        (s, u): after.eligibility(s, S, at_news=PRE_OPEN, universe=u)
        for s in SYMBOLS
        for u in ("primary", "broad")
    } == elig

    # The control: the evening before is information, and moves the answer.
    await _scale(engine, ["AAA"], since=PREV, until=PREV, by=1.1)
    moved = await _load(engine)
    assert moved.pre_event("AAA", S, et(S, 8)) != pre["AAA"]


async def test_late_session_news_does_not_see_that_session_in_sigma(engine: AsyncEngine) -> None:
    await _world(engine)
    late = et(PREV, 15)
    before = (await _load(engine)).pre_event("AAA", S, late)

    await _scale(engine, ["AAA"], since=PREV, until=PREV, by=1.25)
    after = (await _load(engine)).pre_event("AAA", S, late)

    assert before is not None and after is not None
    assert (after.sigma, after.sigma_n, after.beta) == (before.sigma, before.sigma_n, before.beta)
    assert after.prev_close_s == pytest.approx(before.prev_close_s * 1.25)  # known by S's open


# ── facts ────────────────────────────────────────────────────────


async def _fact(
    engine: AsyncEngine, sid: str, at: datetime, kind: str, *, extractor: str = EXTRACTOR
) -> None:
    async with engine.begin() as conn:
        event_id = (
            await conn.execute(
                text(
                    "INSERT INTO events (source, source_id, kind, symbol, published_at, seen_at, "
                    "payload) VALUES ('benzinga', :s, 'news', 'AAA', :t, :t, '{}') RETURNING id"
                ),
                {"s": sid, "t": at},
            )
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO event_facts (event_id, extractor, kind, fields) "
                "VALUES (:e, :x, :k, CAST(:f AS JSONB))"
            ),
            {"e": event_id, "x": extractor, "k": kind, "f": f'{{"id": "{sid}"}}'},
        )


async def test_facts_are_the_ones_published_strictly_before(engine: AsyncEngine) -> None:
    await _world(engine)
    at = et(S, 8)
    await _fact(engine, "old", at - timedelta(days=250), "result")  # beyond 200 days
    await _fact(engine, "q4", et(PREV2, 16, 5), "result")
    await _fact(engine, "guide", et(PREV, 7), "guidance")
    await _fact(engine, "nothing", et(PREV, 7, 30), "none")  # the parser's no-fact marker
    await _fact(engine, "other", et(PREV, 7, 45), "result", extractor="some-other-parser")
    await _fact(engine, "same-moment", at, "result")
    await _fact(engine, "later", et(S, 9), "result")
    ctx = await _load(engine)

    ids = [f.fields["id"] for f in ctx.facts_before("AAA", at)]
    recent = [f.fields["id"] for f in ctx.facts_before("AAA", at, lookback_days=3)]

    assert ids == ["q4", "guide"]
    assert [f.kind for f in ctx.facts_before("AAA", at)] == ["result", "guidance"]
    assert recent == ["guide"]
    assert ctx.facts_before("SPLIT", at) == []
    with pytest.raises(ValueError):
        ctx.facts_before("AAA", datetime(2024, 3, 19, 8))  # naive
    with pytest.raises(ValueError):
        ctx.facts_before("AAA", et(START, 8), lookback_days=400)  # before what was loaded
    # Loaded up to a day after END's close: that moment is answered, a moment later is not.
    loaded_to = session_bounds(END)[1] + timedelta(days=1)
    assert [f.fields["id"] for f in ctx.facts_before("AAA", loaded_to)] == [
        "q4",
        "guide",
        "same-moment",
        "later",
    ]
    with pytest.raises(ValueError, match="facts were loaded for"):
        ctx.facts_before("AAA", loaded_to + timedelta(microseconds=1))


# ── data access and guards ───────────────────────────────────────


def test_a_daily_point_names_its_fields() -> None:
    # The simulator reads .open and .close (halabot.playbooks.interfaces.DailyPointLike).
    assert [f.name for f in fields(DailyPoint)] == [
        "day",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj",
    ]


async def test_a_daily_point_is_the_raw_bar_and_its_a(engine: AsyncEngine) -> None:
    world = await _world(engine)
    ctx = await _load(engine)

    point = ctx.daily("AAA", PREV)

    assert point is not None
    c, a = world.closes["AAA"][PREV], _adj("AAA", PREV)
    o = c * (1.0 + 0.003 * math.sin(_i(PREV)))
    assert point.day == PREV
    assert (point.open, point.high, point.low, point.close, point.volume, point.adj) == (
        pytest.approx(o / a, rel=1e-12),
        pytest.approx(max(o, c) * 1.01 / a, rel=1e-12),
        pytest.approx(min(o, c) * 0.99 / a, rel=1e-12),
        pytest.approx(c / a, rel=1e-12),
        world.volumes["AAA"][PREV],
        pytest.approx(0.98, rel=1e-12),
    )


async def test_daily_bars_are_answered_up_to_seven_days_after_the_last_session(
    engine: AsyncEngine,
) -> None:
    await _world(engine)
    ctx = await _load(engine)
    edge = END + timedelta(days=7)  # 2024-04-04, a Thursday
    beyond = END + timedelta(days=8)  # a session with bars in the database, never read

    assert ctx.sessions[-1] == edge  # every session the simulator walks is answerable
    assert ctx.daily("AAA", edge) is not None and ctx.adj("AAA", edge) is not None
    # Outside what was loaded is an error, not None (which would claim a missing bar).
    with pytest.raises(ValueError, match="outside the loaded daily bars"):
        ctx.daily("AAA", beyond)
    with pytest.raises(ValueError, match="outside the loaded daily bars"):
        ctx.adj("AAA", beyond)
    with pytest.raises(ValueError, match="outside the loaded daily bars"):
        ctx.adj("AAA", START - timedelta(days=381))


async def test_daily_points_and_a_ratios(engine: AsyncEngine) -> None:
    world = await _world(engine)
    ctx = await _load(engine)

    spy = ctx.daily("SPY", S)

    assert spy is not None
    assert spy.close == pytest.approx(world.raw_close("SPY", S), rel=1e-12)
    assert ctx.adj("SPY", date(2024, 3, 14)) == pytest.approx(0.995)
    assert ctx.daily("GAP", PREV) is None and ctx.adj("GAP", PREV) is None
    assert ctx.daily("AAA", date(2024, 3, 16)) is None  # a Saturday
    assert ctx.daily("THIN", S) is None  # loaded, no bars
    assert ctx.sessions == [
        d for d in SESSIONS if START - timedelta(days=380) <= d <= END + timedelta(days=7)
    ]


async def test_questions_beyond_what_was_loaded_are_errors(engine: AsyncEngine) -> None:
    await _world(engine)
    ctx = await _load(engine, ["AAA"])

    with pytest.raises(ValueError, match="not loaded"):
        ctx.eligibility("SPLIT", S, at_news=PRE_OPEN)
    with pytest.raises(ValueError, match="outside the loaded sessions"):
        ctx.pre_event("AAA", date(2024, 4, 2), et(date(2024, 4, 2), 8))
    with pytest.raises(ValueError, match="not a session"):
        ctx.eligibility("AAA", date(2024, 3, 16), at_news=PRE_OPEN)
    with pytest.raises(ValueError, match="outside the loaded daily bars"):
        ctx.daily("AAA", date(2022, 1, 3))
    with pytest.raises(ValueError, match="timezone"):
        ctx.pre_event("AAA", S, datetime(2024, 3, 19, 8))
    # News after S's close cannot react in S: refused, never let into S's sigma.
    with pytest.raises(ValueError, match="after the close"):
        ctx.pre_event("AAA", S, et(S, 16))
    with pytest.raises(ValueError, match="after the close"):
        ctx.eligibility("AAA", PREV, at_news=et(S, 8))
    assert ctx.pre_event("AAA", S, et(S, 15, 59)) is not None


async def test_loading_in_small_batches_gives_the_same_answers(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _world(engine)
    whole = await _load(engine)
    monkeypatch.setattr(context_module, "BATCH_SYMBOLS", 3)
    batched = await _load(engine)

    for symbol in SYMBOLS:
        assert batched.eligibility(symbol, S, at_news=PRE_OPEN) == whole.eligibility(
            symbol, S, at_news=PRE_OPEN
        )
        assert batched.pre_event(symbol, S, et(S, 8)) == whole.pre_event(symbol, S, et(S, 8))


def test_only_a_lone_index_exclusion_is_an_index_veto() -> None:
    assert index_veto_only([VETO])
    assert index_veto_only(["excluded by HLAL's Shariah index"])
    assert not index_veto_only([VETO, "debt"])
    assert not index_veto_only([])
    assert not index_veto_only(
        ["business activity unverified: x; no Shariah index (SPUS, HLAL) holds the company"]
    )
