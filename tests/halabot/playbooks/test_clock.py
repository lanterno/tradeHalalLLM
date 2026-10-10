"""The clock: event order, time conversions, bar visibility and the spec's worked examples."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halabot.playbooks.clock import (
    HARNESS,
    ORDER_LAG,
    SIP_DELAYED,
    SIP_RT,
    EventHeap,
    Priority,
    from_us,
    span_us,
    to_us,
    visible_at,
)
from halabot.playbooks.playbook import Ctx
from halabot.playbooks.sim import simulate_symbol
from halabot.playbooks.types import BarIn, FillIn, Input, Intent, Session, SimConfig
from tests.halabot.playbooks._support import (
    FACTS,
    MON,
    Context,
    Toy,
    downgrade,
    et,
    path,
    run_one,
    session_bars,
    spy_data,
    story,
)


def test_heap_orders_by_time_then_priority_then_insertion() -> None:
    heap: EventHeap[str] = EventHeap()
    heap.push(10, Priority.NEWS, "news@10")
    heap.push(10, Priority.BAR, "bar@10 first")
    heap.push(5, Priority.TIMER, "timer@5")
    heap.push(10, Priority.EXCHANGE, "exchange@10")
    heap.push(10, Priority.BAR, "bar@10 second")
    heap.push(10, Priority.FILL, "fill@10")
    heap.push(10, Priority.SESSION, "session@10")
    heap.push(10, Priority.SPY_BAR, "spy@10")
    assert heap.peek_time() == 5 and len(heap) == 8
    out = []
    while heap:
        out.append(heap.pop()[2])
    assert out == [
        "timer@5",
        "exchange@10",
        "fill@10",
        "session@10",
        "spy@10",
        "bar@10 first",
        "bar@10 second",
        "news@10",
    ]


def test_priorities_are_the_spec_table() -> None:
    assert [p.name for p in sorted(Priority)] == [
        "EXCHANGE",
        "FILL",
        "SESSION",
        "SPY_BAR",
        "BAR",
        "NEWS",
        "TIMER",
    ]
    assert [int(p) for p in sorted(Priority)] == list(range(7))


def test_time_conversions_are_exact() -> None:
    t = datetime(2016, 3, 7, 14, 50, 5, 123456, tzinfo=UTC)
    assert from_us(to_us(t)) == t
    assert to_us(datetime(1970, 1, 1, tzinfo=UTC)) == 0
    assert span_us(timedelta(minutes=17)) == 17 * 60 * 1_000_000
    assert span_us(ORDER_LAG) == 3_000_000


def test_a_bar_is_visible_a_minute_after_its_start_plus_the_feed_lag() -> None:
    ts = int(et(MON, 10, 14).timestamp())
    assert visible_at(ts, SIP_RT) == int(et(MON, 10, 15, 5).timestamp())
    assert visible_at(ts, SIP_DELAYED) == int(et(MON, 10, 32).timestamp())
    assert visible_at(ts, HARNESS) == int(et(MON, 10, 15).timestamp())


def test_session_markers_and_early_close() -> None:
    s = Session.of(MON)
    assert (s.pre_open, s.open, s.entry_start) == (et(MON, 9, 20), et(MON, 9, 30), et(MON, 9, 50))
    assert (s.entry_cutoff, s.flatten, s.close) == (et(MON, 15, 0), et(MON, 15, 55), et(MON, 16, 0))
    from tests.halabot.playbooks._support import HALF

    h = Session.of(HALF)
    assert h.early and (h.entry_cutoff, h.flatten, h.close) == (
        et(HALF, 12, 0),
        et(HALF, 12, 55),
        et(HALF, 13, 0),
    )
    with pytest.raises(ValueError):
        Session.of(datetime(2016, 3, 6).date())  # a Sunday


@pytest.mark.parametrize(
    ("feed", "decided", "active", "bar"),
    [
        (SIP_RT, (10, 15, 5), (10, 15, 8), (10, 16)),
        (SIP_DELAYED, (10, 32, 0), (10, 32, 3), (10, 33)),
        (HARNESS, (10, 15, 0), (10, 15, 3), (10, 16)),
    ],
)
def test_the_worked_example(feed, decided, active, bar) -> None:  # type: ignore[no-untyped-def]
    """Bar 10:14 visible -> decision -> active ORDER_LAG later -> first bar with ts >= active."""
    rows = {
        (10, 16): (100.0, 100.9, 99.9, 100.5, 1_000.0, 100.7),
        (10, 33): (100.0, 101.0, 99.9, 100.5, 1_000.0, 100.9),
    }
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON, rows=rows)])
    out = run_one(st, pd, cfg=SimConfig(feed=feed), toy={"entry_bar": (10, 14), "target": 1.0})
    t = out.trade
    assert t is not None
    assert t.entry_decided_at == et(MON, *decided)
    assert t.entry_active_at == et(MON, *active)
    assert t.entry_bar_ts == et(MON, *bar)
    assert t.entry_px == rows[bar][5]  # that bar's VWAP
    assert t.feed == feed.name


def test_the_exchange_fills_before_the_same_instant_bar_reaches_the_playbook() -> None:
    """HARNESS: bar 10:16 completes and is visible at 10:17:00; the fill comes first."""
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON)])
    held_at_bar: dict[datetime, bool] = {}
    fills: list[datetime] = []

    class Watch(Toy):
        def on(self, ev: Input, ctx: Ctx) -> list[Intent]:
            if isinstance(ev, BarIn) and ev.symbol == "AAA":
                held_at_bar[ctx.market.bars("AAA").bar_time(ev.i)] = ctx.position.held
            if isinstance(ev, FillIn):
                fills.append(ev.at)
            return super().on(ev, ctx)

    simulate_symbol(
        "AAA",
        [st],
        lambda s: Watch(s, entry_bar=(10, 14), target=1.0),
        {st.story_id: pd},
        spy_data([MON]),
        Context(),
        SimConfig(feed=HARNESS),
    )
    assert held_at_bar[et(MON, 10, 15)] is False
    assert held_at_bar[et(MON, 10, 16)] is True  # executed at 10:17:00 (priority 0) before it
    assert fills[0] == et(MON, 10, 17, 1)  # the FillIn: the filling bar's end + 1 s


def test_bars_visible_at_the_start_are_history_not_inputs() -> None:
    """A story starting at 11:17:00 sees the 11:15 bar (visible 11:16:05) in its history."""
    from tests.halabot.playbooks._support import Item

    st = story("AAA", MON, Item(et(MON, 11, 17)))
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON)])
    first_bar: list[tuple[datetime, int]] = []

    class First(Toy):
        def start(self, ctx: Ctx) -> list[Intent]:
            first_bar.append((ctx.now, len(ctx.market.bars("AAA"))))
            return super().start(ctx)

        def on(self, ev: Input, ctx: Ctx) -> list[Intent]:
            if isinstance(ev, BarIn) and ev.symbol == "AAA" and len(first_bar) == 1:
                first_bar.append((ev.at, ev.i))
            return super().on(ev, ctx)

    simulate_symbol(
        "AAA",
        [st],
        lambda s: First(s, target=1.0),
        {st.story_id: pd},
        spy_data([MON]),
        Context(),
        SimConfig(),
    )
    start, history = first_bar[0]
    assert start == et(MON, 11, 17) and history == 106  # 09:30 .. 11:15
    assert first_bar[1] == (et(MON, 11, 17, 5), 106)  # the 11:16 bar is the first BarIn


def test_facts_are_carried_to_the_record() -> None:
    st = story("AAA", MON, downgrade(MON))
    pd = path(st.story_id, "AAA", [MON], [session_bars(MON)])
    t = run_one(st, pd, toy={"target": 1.0}).trade
    assert t is not None
    assert (t.family_type, t.cell, t.variant, t.rank, t.tech) == (
        FACTS.family_type,
        FACTS.cell,
        FACTS.variant,
        FACTS.rank,
        FACTS.tech,
    )
