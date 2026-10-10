"""The playbook contract as the simulator reads it: live(), the factory, overlap, visible bars."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum, auto

from halabot.playbooks.playbook import TRIGGERED, Ctx, Factory
from halabot.playbooks.sim import PitStory, simulate_symbol
from halabot.playbooks.types import (
    BarIn,
    FillIn,
    Finish,
    Input,
    Intent,
    SessionIn,
    SimConfig,
    Submit,
    Transition,
)
from halal_trader.market_hours import next_trading_day
from tests.halabot.playbooks._support import (
    FACTS,
    MON,
    TUE,
    WED,
    Context,
    Item,
    Recorder,
    Toy,
    downgrade,
    et,
    path,
    session_bars,
    spy_data,
    story,
)

THU = next_trading_day(WED)
FRI = next_trading_day(THU)


def _paths(*stories, sessions: int):  # type: ignore[no-untyped-def]
    out = {}
    for st in stories:
        days = [st.session]
        while len(days) < sessions:
            days.append(next_trading_day(days[-1]))
        out[st.story_id] = path(st.story_id, st.symbol, days, [session_bars(d) for d in days])
    return out


# ── liveness is the playbook's own answer ──


class Low(StrEnum):
    """States as a StrEnum with auto(): lowercase values."""

    WATCHING = auto()
    ARMED = auto()
    ENTERED = auto()
    EXITED = auto()


class Lowercase:
    """Buys at entry_start, holds to the simulator's time stop; its states are lowercase."""

    name, version = "lower", "1"

    def __init__(self, sessions: int) -> None:
        self.path_sessions = sessions
        self._s = Low.WATCHING

    def state(self) -> str:
        return self._s

    def live(self) -> bool:
        return self._s in (Low.WATCHING, Low.ARMED, Low.ENTERED)

    def start(self, ctx: Ctx) -> list[Intent]:
        return [Transition(Low.WATCHING, "start")]

    def on(self, ev: Input, ctx: Ctx) -> list[Intent]:
        if isinstance(ev, SessionIn) and ev.kind == "entry_start" and self._s is Low.WATCHING:
            self._s = Low.ARMED
            return [Transition(Low.ARMED, TRIGGERED), Submit("buy", facts=FACTS)]
        if isinstance(ev, FillIn) and ev.side == "buy":
            self._s = Low.ENTERED
            return [Transition(Low.ENTERED, "filled")]
        if isinstance(ev, FillIn):
            self._s = Low.EXITED
            return [Transition(Low.EXITED, ev.reason), Finish(ev.reason)]
        return []


def test_lowercase_states_still_block_the_symbol_and_set_armed_at() -> None:
    mon = story("AAA", MON, downgrade(MON))
    tue = story("AAA", TUE, downgrade(TUE))
    made: list[Lowercase] = []

    def make(s):  # type: ignore[no-untyped-def]
        made.append(Lowercase(3))
        return made[-1]

    first, second = simulate_symbol(
        "AAA",
        [mon, tue],
        make,
        _paths(mon, tue, sessions=3),
        spy_data([MON, TUE, WED, THU]),
        Context(),
        SimConfig(),
        keep_transitions=True,
    )
    assert (second.terminal_state, second.reason) == ("DISMISSED", "blocked_open")
    assert len(made) == 1  # the blocked story never got a playbook
    assert first.armed_at == et(MON, 9, 50) and first.triggered_at == et(MON, 9, 50)
    assert first.terminal_state == "exited"
    t = first.trade
    assert t is not None and t.exit_reason == "time_stop" and t.exit_session == WED
    assert [to for _, _, to, _ in first.transitions] == ["watching", "armed", "entered", "exited"]


# ── a blocked story is blocked before any eligibility is judged ──


class Picky(Toy):
    """The toy, but a story named ineligible is dismissed by its own playbook at the start."""

    def __init__(self, story: object, *, ineligible: frozenset[str], **kw: object) -> None:
        super().__init__(story, **kw)  # type: ignore[arg-type]
        self._out = getattr(story, "story_id") in ineligible  # noqa: B009

    def start(self, ctx: Ctx) -> list[Intent]:
        if self._out:
            return [self._go("DISMISSED", "not_halal"), Finish("not_halal")]
        return super().start(ctx)


class PickyFactory:
    def __init__(self, ineligible: set[str], sessions: int) -> None:
        self.ineligible = frozenset(ineligible)
        self.sessions = sessions
        self.built: list[str] = []

    def __call__(self, s):  # type: ignore[no-untyped-def]
        self.built.append(s.story_id)
        return Picky(s, ineligible=self.ineligible, sessions=self.sessions, target=1.0)


def test_an_ineligible_story_starting_while_another_is_live_counts_as_blocked_open() -> None:
    mon = story("AAA", MON, downgrade(MON))  # eligible, held MON..WED
    tue = story("AAA", TUE, downgrade(TUE))  # ineligible, starts while MON's is live
    factory = PickyFactory({tue.story_id}, sessions=3)
    out = simulate_symbol(
        "AAA",
        [mon, tue],
        factory,
        _paths(mon, tue, sessions=3),
        spy_data([MON, TUE, WED, THU]),
        Context(),
        SimConfig(),
    )
    by_id = {o.story_id: o for o in out}
    assert (by_id[tue.story_id].terminal_state, by_id[tue.story_id].reason) == (
        "DISMISSED",
        "blocked_open",
    )
    assert factory.built == [mon.story_id]  # its eligibility was never asked


def test_an_ineligible_story_with_nothing_live_dismisses_itself_and_blocks_nobody() -> None:
    mon = story("AAA", MON, downgrade(MON))  # eligible, intraday: done by MON's close
    tue = story("AAA", TUE, downgrade(TUE))  # ineligible: nothing is live at its start
    wed = story("AAA", WED, downgrade(WED))  # eligible
    factory = PickyFactory({tue.story_id}, sessions=1)
    out = simulate_symbol(
        "AAA",
        [mon, tue, wed],
        factory,
        _paths(mon, tue, wed, sessions=1),
        spy_data([MON, TUE, WED]),
        Context(),
        SimConfig(),
    )
    by_id = {o.story_id: o for o in out}
    assert (by_id[tue.story_id].terminal_state, by_id[tue.story_id].reason) == (
        "DISMISSED",
        "not_halal",
    )
    assert by_id[wed.story_id].reason != "blocked_open" and by_id[wed.story_id].entered
    assert factory.built == [mon.story_id, tue.story_id, wed.story_id]


# ── the factory ──


def test_the_factory_gets_the_story_as_of_the_start_once_per_story_that_runs() -> None:
    st = story(
        "AAA",
        MON,
        downgrade(MON),
        Item(et(MON, 11, 0), "offering", structural=True),  # known only at 11:00
    )
    seen: list[tuple[object, bool, list[datetime]]] = []

    def make(s):  # type: ignore[no-untyped-def]
        seen.append((s, s.card_at(et(MON, 16, 0)).structural, list(s.news_times())))
        return Toy(s, target=1.0)

    factory = Factory(make, "toy", "1", 1)
    assert (factory.name, factory.version, factory.path_sessions) == ("toy", "1", 1)
    simulate_symbol(
        "AAA", [st], factory, _paths(st, sessions=1), spy_data([MON]), Context(), SimConfig()
    )
    ((view, structural, news),) = seen
    assert isinstance(view, PitStory)
    assert structural is False  # card_at(16:00) asked at the 09:30 start: the 11:00 item unknown
    assert news == [et(MON, 8, 0)]


def test_a_skipped_or_blocked_story_never_builds_a_playbook() -> None:
    mon = story("AAA", MON, downgrade(MON))
    tue = story("AAA", TUE, downgrade(TUE))
    thu = story("AAA", THU, downgrade(THU))  # after MON's deadline, but without a path
    rec = Recorder(sessions=3, target=1.0)
    paths = _paths(mon, tue, sessions=3)
    out = simulate_symbol(
        "AAA", [mon, tue, thu], rec, paths, spy_data([MON, TUE, WED, THU]), Context(), SimConfig()
    )
    assert [o.reason for o in out[1:]] == ["blocked_open", "units_missing"]
    assert len(rec.made) == 1 and rec.path_sessions == 3


# ── what a playbook can reach ──


def test_visible_bars_are_copies_with_nothing_behind_them() -> None:
    st = story("AAA", MON, downgrade(MON))
    checks: list[tuple[int, int, bool, bool]] = []

    class Look(Toy):
        def on(self, ev: Input, ctx: Ctx) -> list[Intent]:
            if isinstance(ev, BarIn):
                bars = ctx.market.bars(ev.symbol)
                again = ctx.market.bars(ev.symbol)
                arrays = (bars.ts, bars.o, bars.h, bars.l, bars.c, bars.v, bars.vw)
                arrays += (bars.visible_at, bars.k, bars.scale)
                checks.append(
                    (
                        ev.i,
                        len(bars),
                        all(a.base is None and not a.flags.writeable for a in arrays),
                        again is bars,  # one copy per symbol and n
                    )
                )
            return super().on(ev, ctx)

    simulate_symbol(
        "AAA",
        [st],
        lambda s: Look(s, target=1.0),
        _paths(st, sessions=1),
        spy_data([MON]),
        Context(),
        SimConfig(),
    )
    assert len(checks) > 300
    assert all(n == i + 1 and copied and cached for i, n, copied, cached in checks)
