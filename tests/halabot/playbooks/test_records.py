"""Records: daily legs, the daily book, the canonical hash, and the hb_playbook_* sink."""

from __future__ import annotations

import math
import uuid
from dataclasses import replace
from datetime import date, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.platform.db import bootstrap_schema
from halabot.playbooks.records import (
    Leg,
    MemorySink,
    PgOutcomeSink,
    RunInfo,
    StoryOutcome,
    TradeRecord,
    daily_book,
    outcomes_sha256,
)
from halal_trader.db.repos.quant_trials import config_hash
from tests.halabot.playbooks._support import (
    MON,
    TUE,
    WED,
    Context,
    daily,
    downgrade,
    et,
    path,
    run_one,
    session_bars,
    story,
)

SESSIONS = [date(2016, 3, 3), date(2016, 3, 4), MON, TUE, WED]


def _trade() -> TradeRecord:
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON, TUE, WED], [session_bars(d) for d in (MON, TUE, WED)])
    ctx = Context(
        daily={("AAA", d): daily(100.0) for d in (MON, TUE)}
        | {("SPY", d): daily(200.0) for d in (MON, TUE)}
    )
    out = run_one(st, pd, ctx=ctx, toy={"sessions": 3, "target": 1.0})
    assert out.trade is not None
    return out.trade


def make(
    sid: str,
    day: date,
    decided: tuple[int, int],
    exit_at: tuple[int, int],
    *,
    exit_day: date | None = None,
) -> TradeRecord:
    exit_day = exit_day or day
    return replace(
        _TEMPLATE,
        story_id=sid,
        session=day,
        exit_session=exit_day,
        entry_decided_at=et(day, *decided),
        exit_bar_ts=et(exit_day, *exit_at) - timedelta(seconds=60),  # exit fills at exit_at
    )


_TEMPLATE = _trade()


def test_a_leg_is_abnormal_and_net_of_that_days_cost() -> None:
    leg = Leg(MON, 100.0, 102.0, 200.0, 201.0, 0.003)
    assert math.isclose(leg.stock, 0.02 - 0.003)
    assert math.isclose(leg.spy, 0.005)
    assert math.isclose(leg.abn, 0.02 - 0.003 - 0.005)


def test_one_trade_in_the_book() -> None:
    t = make("a", MON, (9, 50), (10, 30))
    legs = {"a": [Leg(MON, 100.0, 101.0, 200.0, 202.0, 0.003)]}
    book = daily_book([t], legs, SESSIONS)
    assert book.days == SESSIONS[:3]  # from the window start to the last exit
    assert list(book.returns) == [0.0, 0.0, pytest.approx(0.125 * (0.01 - 0.003))]
    assert list(book.benchmark) == [0.0, 0.0, pytest.approx(0.125 * 0.01)]
    assert list(book.exposure) == [0.0, 0.0, 0.125]
    assert (book.skipped_full, book.skipped_loss_limit) == (0, 0)


def test_the_book_has_eight_slots_first_come_first_served() -> None:
    trades = [make(f"t{i}", MON, (10, i), (15, 0)) for i in range(9)]
    legs = {t.story_id: [Leg(MON, 100.0, 100.0, 200.0, 200.0, 0.0)] for t in trades}
    book = daily_book(list(reversed(trades)), legs, SESSIONS)
    assert book.skipped_full == 1 and book.exposure[-1] == 1.0
    # A slot frees when a trade exits: a ninth trade after an exit is taken.
    early = make("t0", MON, (10, 0), (10, 5))
    later = make("t8", MON, (10, 8), (15, 0))
    book = daily_book([early, *trades[1:8], later], legs, SESSIONS)
    assert book.skipped_full == 0


def test_the_daily_loss_limit_counts_trades_closed_earlier_that_day() -> None:
    loser = make("loser", MON, (9, 50), (10, 0))
    after = make("after", MON, (10, 30), (15, 0))
    before = make("before", MON, (9, 55), (15, 0))
    legs = {
        "loser": [Leg(MON, 100.0, 80.0, 200.0, 200.0, 0.0)],  # stake 1/8 x -20% = -2.5% NAV
        "after": [Leg(MON, 100.0, 110.0, 200.0, 200.0, 0.0)],
        "before": [Leg(MON, 100.0, 110.0, 200.0, 200.0, 0.0)],
    }
    book = daily_book([loser, after, before], legs, SESSIONS)
    assert book.skipped_loss_limit == 1  # "after" was decided after the loss closed
    assert book.returns[-1] == pytest.approx(0.125 * -0.2 + 0.125 * 0.1)


def test_multi_day_legs_and_compounding() -> None:
    a = make("a", MON, (9, 50), (10, 0), exit_day=TUE)
    b = make("b", TUE, (11, 0), (12, 0))
    legs = {
        "a": [Leg(MON, 100.0, 110.0, 200.0, 200.0, 0.0), Leg(TUE, 110.0, 121.0, 200.0, 210.0, 0.0)],
        "b": [Leg(TUE, 50.0, 55.0, 200.0, 200.0, 0.0)],
    }
    book = daily_book([a, b], legs, SESSIONS)
    nav1 = 1.0 + 0.125 * 0.1
    a_tue = 0.125 * 1.1  # a, marked to market at MON's close
    assert book.days[-2:] == [MON, TUE]
    assert book.returns[-2] == pytest.approx(0.125 * 0.1)
    # TUE: a earns on its marked value; b's stake is NAV_MON / 8.
    assert book.returns[-1] == pytest.approx((a_tue * 0.1 + nav1 / 8 * 0.1) / nav1)
    assert book.benchmark[-1] == pytest.approx(a_tue * 0.05 / nav1)
    assert book.exposure[-1] == pytest.approx((a_tue + nav1 / 8) / nav1)


def test_an_md3_trade_is_marked_to_market_day_by_day() -> None:
    """v_d = v_{d-1} (1 + leg_d.stock): the book compounds exactly as the trade does."""
    t = make("md3", MON, (9, 50), (15, 56), exit_day=WED)
    legs = {
        "md3": [
            Leg(MON, 100.0, 110.0, 200.0, 202.0, 0.0015),  # +10% less the entry cost
            Leg(TUE, 110.0, 99.0, 202.0, 200.0, 0.0),  # -10%
            Leg(WED, 99.0, 103.95, 200.0, 201.0, 0.0015),  # +5% less the exit cost
        ]
    }
    book = daily_book([t], legs, SESSIONS)
    s1, s2, s3 = (leg.stock for leg in legs["md3"])
    v0 = 0.125
    v1 = v0 * (1 + s1)
    v2 = v1 * (1 + s2)
    nav1 = 1 + v0 * s1
    nav2 = nav1 + v1 * s2
    assert book.days[-3:] == [MON, TUE, WED]
    assert list(book.returns[-3:]) == pytest.approx([v0 * s1, v1 * s2 / nav1, v2 * s3 / nav2])
    assert list(book.exposure[-3:]) == pytest.approx([v0, v1 / nav1, v2 / nav2])
    spy = [leg.spy for leg in legs["md3"]]
    assert list(book.benchmark[-3:]) == pytest.approx(
        [v0 * spy[0], v1 * spy[1] / nav1, v2 * spy[2] / nav2]
    )
    # The NAV gains exactly the stake times the trade's compounded return.
    growth = float(np.prod(1 + book.returns))
    assert growth - 1 == pytest.approx(v0 * ((1 + s1) * (1 + s2) * (1 + s3) - 1), abs=1e-15)


def test_an_empty_book() -> None:
    book = daily_book([], {}, SESSIONS)
    assert book.days == [] and len(book.returns) == 0


def test_the_hash_ignores_order_and_sees_any_change() -> None:
    o1 = StoryOutcome("a", "AAA", MON, "EXITED", "target", None, None, None, _TEMPLATE)
    o2 = StoryOutcome("b", "BBB", MON, "EXPIRED", "cutoff", None, None, None, None)
    assert outcomes_sha256([o1, o2]) == outcomes_sha256([o2, o1])
    nudged = replace(o1, trade=replace(_TEMPLATE, exit_px=_TEMPLATE.exit_px + 1e-12))
    assert outcomes_sha256([nudged, o2]) != outcomes_sha256([o1, o2])


async def test_memory_sink() -> None:
    sink = MemorySink(run_id="r1")
    info = RunInfo("gate", WED, "sip-rt", "end", "toy", "1")
    assert await sink.begin(info) == "r1" and sink.info == info
    await sink.write([StoryOutcome("b", "BBB", MON, "EXPIRED", "cutoff", None, None, None, None)])
    await sink.finish({"n": 1})
    assert len(sink.outcomes) == 1 and sink.summary == {"n": 1}


async def test_the_pg_sink_writes_runs_stories_and_trades(halabot_engine: AsyncEngine) -> None:
    await bootstrap_schema(halabot_engine)  # idempotent with the playbook tables in place
    cfg = {"prereg": "abc", "cell": ["NSN_CORE", "MD3", 3]}
    sink = PgOutcomeSink(
        halabot_engine, cell="NSN_CORE/MD3", config=cfg, prereg_trial_id=7, code_sha="deadbeef"
    )
    run_id = await sink.begin(
        RunInfo("train", date(2021, 12, 31), "sip-rt", "end", "toy", "1", {"feed": "sip-rt"})
    )
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON, TUE, WED], [session_bars(d) for d in (MON, TUE, WED)])
    ctx = Context(
        daily={("AAA", d): daily(100.0) for d in (MON, TUE)}
        | {("SPY", d): daily(200.0) for d in (MON, TUE)}
    )
    traded = run_one(st, pd, ctx=ctx, toy={"sessions": 3, "target": 1.0})
    blocked = StoryOutcome(
        "BBB:2016-03-07",
        "BBB",
        MON,
        "DISMISSED",
        "blocked_open",
        None,
        None,
        None,
        None,
        start_at=et(MON, 9, 30),
    )
    await sink.write([traded, blocked])
    await sink.write([])
    await sink.finish({"outcomes": 2, "when": et(MON, 9, 30)})

    async with halabot_engine.connect() as conn:
        run = (await conn.execute(text("SELECT * FROM hb_playbook_run"))).one()
        assert run.run_id == uuid.UUID(run_id) and run.mode == "sim" and run.cell == "NSN_CORE/MD3"
        assert (run.feed, run.window, run.window_end, run.stop_at) == (
            "sip-rt",
            "train",
            date(2021, 12, 31),
            "end",
        )
        assert run.config == {**cfg, "sim": {"feed": "sip-rt"}} and run.config_hash == config_hash(
            cfg
        )
        assert (run.prereg_trial_id, run.code_sha) == (7, "deadbeef")
        assert run.summary == {"outcomes": 2, "when": et(MON, 9, 30).isoformat()}
        stories = (
            await conn.execute(text("SELECT * FROM hb_playbook_story ORDER BY story_id"))
        ).all()
        assert [(s.story_id, s.terminal_state, s.reason) for s in stories] == [
            ("AAA:2016-03-07", "EXITED", "time_stop"),
            ("BBB:2016-03-07", "DISMISSED", "blocked_open"),
        ]
        assert stories[0].entry_bar_ts == et(MON, 9, 51) and stories[1].entry_bar_ts is None
        trade = (await conn.execute(text("SELECT * FROM hb_playbook_trade"))).one()
    t = traded.trade
    assert t is not None
    assert trade.story_id == t.story_id and trade.exit_session == t.exit_session == WED
    assert trade.r_net_abn == t.r_net_abn and trade.entry_px == t.entry_px
    assert trade.p0 is None  # NaN levels are stored as NULL
    assert trade.flags == list(t.flags)
    assert [leg["day"] for leg in trade.legs] == [MON.isoformat(), TUE.isoformat(), WED.isoformat()]
    with pytest.raises(ValueError):
        PgOutcomeSink(halabot_engine, mode="backtest")


def test_trade_exit_time() -> None:
    t = _TEMPLATE
    assert t.exit_bar_ts is not None
    assert t.exit_time == t.exit_bar_ts + timedelta(seconds=60)
    assert replace(t, exit_bar_ts=None).exit_time == et(t.exit_session, 16, 0)
    assert isinstance(t.entry_decided_at, datetime)
