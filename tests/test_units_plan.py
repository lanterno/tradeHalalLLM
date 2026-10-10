"""Plan H (events/units.py), pure parts: paths, the plan, seeded orders, the clock, estimates.

Synthetic stories and observations only; no database and no price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta

import pytest

from halabot.playbooks.loader import GATE_RANGES, unit_set_sha
from halal_trader.data.minutes import session_bounds
from halal_trader.events import units
from halal_trader.events.intraday import Headline
from halal_trader.events.study import Observation
from halal_trader.events.units import (
    GateStory,
    StoryRef,
    SueEvent,
    UnitPlan,
    broad_eligible,
    busy,
    calib_units,
    estimate,
    g1_units,
    hours_at,
    news_times,
    order_g1,
    path,
    prev_close_s,
    reactor_units,
    requests_for,
    sessions_between,
    spy_units,
    story_units,
    sue_event,
    sue_sample,
    sue_units,
)
from halal_trader.market_hours import MARKET_TZ


def ny(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), MARKET_TZ)


def ref(
    symbol: str,
    session: date,
    *,
    nsn: bool = False,
    type_close: str = "analyst_downgrade",
    substantive: bool = True,
    has_8k: bool = False,
) -> StoryRef:
    return StoryRef(
        story_id=f"{symbol}:{session.isoformat()}",
        symbol=symbol,
        session=session,
        nsn=nsn,
        type_close=type_close,
        substantive=substantive,
        has_8k=has_8k,
    )


# ── paths ─────────────────────────────────────────────────────


def test_a_path_is_n_sessions_by_the_calendar_cut_at_the_cap() -> None:
    # Thanksgiving 2016 (Thu 11-24) is a holiday; Fri 11-25 closes early but is a session.
    assert path(date(2016, 11, 22), 4, date(2016, 12, 31)) == [
        date(2016, 11, 22),
        date(2016, 11, 23),
        date(2016, 11, 25),
        date(2016, 11, 28),
    ]
    assert path(date(2021, 12, 29), 4, date(2021, 12, 31)) == [
        date(2021, 12, 29),
        date(2021, 12, 30),
        date(2021, 12, 31),
    ]
    assert path(date(2016, 9, 30), 4, date(2016, 9, 30)) == [date(2016, 9, 30)]


def test_a_path_starts_on_a_session() -> None:
    with pytest.raises(ValueError, match="not a trading session"):
        path(date(2016, 10, 1), 4, date(2016, 12, 31))  # a Saturday
    with pytest.raises(ValueError, match="at least one"):
        path(date(2016, 10, 3), 0, date(2016, 12, 31))


def test_spy_holds_every_session_of_2016_to_2024() -> None:
    spy = spy_units()
    assert len(spy) == 2264  # spec §H's count
    assert {s for s, _ in spy} == {"SPY"}
    days = sorted(d for _, d in spy)
    assert days[0] == date(2016, 1, 4) and days[-1] == date(2024, 12, 31)
    assert days == sessions_between(date(2016, 1, 1), date(2024, 12, 31))


# ── the plan ──────────────────────────────────────────────────


def test_the_plan_pins_each_part_with_the_loaders_sha() -> None:
    a = frozenset({("AAPL", date(2016, 3, 1)), ("MSFT", date(2016, 3, 2))})
    plan = UnitPlan({"gate_g1": a, "spy": frozenset({("SPY", date(2016, 3, 1))})})
    assert plan.sha("gate_g1") == unit_set_sha(a)
    assert len(plan.sha("gate_g1")) == 64  # the digest register_gate_units compares
    assert plan.fetch_order() == ["spy", "gate_g1"]
    assert plan.all() == a | {("SPY", date(2016, 3, 1))}
    plan.check()


def test_nothing_after_2024_except_the_reactors_set() -> None:
    late = frozenset({("AAPL", date(2025, 1, 2))})
    UnitPlan({"gate_reactor": late}).check()
    with pytest.raises(ValueError, match="validation: 1 unit"):
        UnitPlan({"validation": late}).check()
    with pytest.raises(ValueError, match="unknown part"):
        UnitPlan({"holdout": frozenset()}).check()


def test_the_plan_reports_gate_units_outside_their_gates_dates() -> None:
    plan = UnitPlan(
        {
            "gate_g1": frozenset({("A", date(2016, 9, 30)), ("A", date(2016, 10, 3))}),
            "gate_reactor": frozenset({("B", date(2026, 3, 2))}),
            "train": frozenset({("C", date(2018, 1, 2))}),
        }
    )
    assert plan.outside_gate_ranges() == {"gate_g1": 1}
    assert set(units.GATE_OF_PART.values()) == set(GATE_RANGES)


# ── stories ───────────────────────────────────────────────────


def test_a_story_holds_its_path_and_an_8k_story_also_the_session_before() -> None:
    s = date(2018, 3, 6)  # a Tuesday
    plain = story_units(ref("AAPL", s), units.TRAIN[0], units.TRAIN[1])
    assert plain == {("AAPL", d) for d in path(s, 4, units.TRAIN[1])}
    with_8k = story_units(ref("AAPL", s, has_8k=True), units.TRAIN[0], units.TRAIN[1])
    assert with_8k == plain | {("AAPL", date(2018, 3, 5))}


def test_the_session_before_stays_inside_the_window() -> None:
    first = units.VALIDATION[0]  # 2022-01-03; the session before is train's last
    got = story_units(ref("AAPL", first, has_8k=True), *units.VALIDATION)
    assert ("AAPL", date(2021, 12, 31)) not in got
    assert min(d for _, d in got) == first


def test_g1_puts_the_nsn_stories_first_then_the_other_negative_ones() -> None:
    day = date(2016, 3, 1)
    pool = (
        [ref(f"N{i}", day, nsn=True) for i in range(5)]
        + [ref(f"P{i}", day, type_close="earnings_beat") for i in range(5)]
        + [ref(f"G{i}", day, type_close="guidance_cut") for i in range(5)]
        + [ref(f"X{i}", day, type_close="noise_only") for i in range(5)]
    )
    ordered = order_g1(pool, seed=7)
    assert [s.symbol[0] for s in ordered] == ["N"] * 5 + ["G"] * 5
    assert all(s.nsn for s in ordered[:5]) and not any(s.nsn for s in ordered[5:])
    assert ordered == order_g1(list(reversed(pool)), seed=7)  # seeded, not input order
    assert [s.symbol for s in ordered] != [s.symbol for s in order_g1(pool, seed=8)]


def test_g1_keeps_at_most_500_stories_on_paths_inside_its_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(units, "G1_STORIES", 3)
    pool = [ref(f"N{i}", date(2016, 9, 29), nsn=True) for i in range(5)]
    chosen = order_g1(pool)
    assert len(chosen) == 3
    got = g1_units(chosen)
    assert {d for _, d in got} == {date(2016, 9, 29), date(2016, 9, 30)}  # cut at 09-30
    assert got == g1_units([GateStory(s.story_id, s.symbol, s.session, s.nsn) for s in chosen])


# ── the context's questions ───────────────────────────────────


@dataclass(frozen=True)
class Elig:
    eligible: bool


@dataclass(frozen=True)
class Daily:
    close: float
    adj: float


@dataclass
class Ctx:
    """Eligible for news at S's open only (``late``), or at any time (``broad``)."""

    broad: set[str] = field(default_factory=set)
    late: set[str] = field(default_factory=set)
    asked: list[tuple[str, date, datetime, str]] = field(default_factory=list)

    def eligibility(self, symbol: str, session: date, *, at_news: datetime, universe: str) -> Elig:
        self.asked.append((symbol, session, at_news, universe))
        late_ok = symbol in self.late and at_news >= session_bounds(session)[0]
        return Elig(symbol in self.broad or late_ok)

    def daily(self, symbol: str, day: date) -> Daily | None:
        return Daily(10.0, 0.5) if symbol != "NONE" else None

    def adj(self, symbol: str, day: date) -> float | None:
        return 1.0


def test_broad_is_asked_at_both_news_times_a_story_can_have() -> None:
    s = date(2018, 3, 6)
    before, opening = news_times(s)
    assert before == session_bounds(date(2018, 3, 5))[1] - timedelta(seconds=1)
    assert opening == session_bounds(s)[0]
    ctx = Ctx(broad={"A"}, late={"B"})
    assert broad_eligible(ctx, "A", s) and broad_eligible(ctx, "B", s)
    assert not broad_eligible(ctx, "C", s)
    assert {u for *_, u in ctx.asked} == {"broad"}
    assert [t for sym, _, t, _ in ctx.asked if sym == "C"] == [before, opening]


def test_the_previous_close_is_in_session_units() -> None:
    ctx = Ctx()
    assert prev_close_s(ctx, "A", date(2016, 3, 2)) == 5.0  # 10 * 0.5 / 1.0
    assert prev_close_s(ctx, "NONE", date(2016, 3, 2)) is None


# ── SUE ───────────────────────────────────────────────────────

CAL = sessions_between(date(2015, 12, 1), date(2020, 3, 31))


def test_a_sue_event_follows_the_studys_clock() -> None:
    wed = date(2017, 3, 1)
    pre = sue_event(Observation("A", ny(wed, 8).astimezone(UTC), 1.5), CAL)
    assert isinstance(pre, SueEvent)
    assert (pre.session, pre.entry) == (wed, "open")
    assert pre.exits == (
        path(wed, 5, wed + timedelta(days=60))[-1],
        path(wed, 20, date(2018, 1, 1))[-1],
    )
    during = sue_event(Observation("A", ny(wed, 12), 1.5), CAL)
    assert isinstance(during, SueEvent)
    assert (during.session, during.entry) == (wed, "close")
    assert during.exits == (path(wed, 6, date(2018, 1, 1))[-1], path(wed, 21, date(2018, 1, 1))[-1])
    after = sue_event(Observation("A", ny(wed, 16, 5), 1.5), CAL)
    assert isinstance(after, SueEvent)
    assert (after.session, after.entry) == (date(2017, 3, 2), "open")
    assert after.units() == {("A", after.session), *(("A", d) for d in after.exits)}


def test_news_after_an_early_close_has_no_event() -> None:
    half = date(2016, 11, 25)  # closes at 13:00
    assert sue_event(Observation("A", ny(half, 14), 1.0), CAL) == "early_close"
    early = sue_event(Observation("A", ny(half, 12), 1.0), CAL)
    assert isinstance(early, SueEvent) and early.entry == "close"
    assert sue_event(Observation("A", ny(date(2020, 3, 30), 8), 1.0), CAL) == "no_entry"


def _event(i: int) -> SueEvent:
    day = CAL[40 + i % 100]
    return SueEvent(f"S{i}", ny(day, 8).astimezone(UTC), float(i), day, "open", (day, day))


def test_the_sue_sample_is_seeded_and_drawn_from_the_complement() -> None:
    pool = [_event(i) for i in range(50)]
    a = sue_sample(pool, seed=1, n=10)
    assert len(a) == 10 and set(a) <= set(pool)
    assert a == sue_sample(list(reversed(pool)), seed=1, n=10)
    assert a != sue_sample(pool, seed=2, n=10)
    assert sue_sample(pool[:5], n=10) == sorted(pool[:5], key=lambda e: e.published_at)
    assert sue_units(a) == {u for e in a for u in e.units()}


def test_a_calibration_pair_holds_its_session_and_the_fifth() -> None:
    got = calib_units([("A", date(2016, 8, 31))])
    assert got == {("A", date(2016, 8, 31)), ("A", date(2016, 9, 7))}  # Labor Day 09-05


def test_the_reactor_set_holds_each_session_headline_and_spys_day() -> None:
    heads = [
        Headline("AAPL", ny(date(2026, 3, 2), 10).astimezone(UTC), 0.6),
        Headline("MSFT", ny(date(2026, 3, 2), 11), -0.5),
        Headline("TSLA", ny(date(2026, 4, 3), 10), 0.7),  # Good Friday: no session
    ]
    assert reactor_units(heads) == {
        ("AAPL", date(2026, 3, 2)),
        ("MSFT", date(2026, 3, 2)),
        ("SPY", date(2026, 3, 2)),
    }


# ── the clock and the estimate ────────────────────────────────


@pytest.mark.parametrize(
    ("at", "why"),
    [
        (ny(date(2026, 10, 6), 9, 29), None),
        (ny(date(2026, 10, 6), 9, 30), "US market hours"),
        (ny(date(2026, 10, 6), 15, 59), "US market hours"),
        (ny(date(2026, 10, 6), 16, 0), None),
        (ny(date(2026, 11, 27), 13, 0), None),  # an early close
        (ny(date(2026, 10, 6), 20, 30), "research job"),
        (ny(date(2026, 10, 6), 23, 29), "research job"),
        (ny(date(2026, 10, 6), 23, 30), None),
        (ny(date(2026, 10, 10), 12, 0), None),  # a Saturday
    ],
)
def test_busy_names_market_hours_and_the_research_job(at: datetime, why: str | None) -> None:
    got = busy(at.astimezone(UTC))
    assert (got is None) if why is None else (got is not None and why in got)


def test_requests_group_units_by_session_and_page() -> None:
    day, other = date(2016, 3, 1), date(2016, 3, 2)
    assert requests_for([]) == 0
    assert requests_for([("A", day)]) == 1
    # 26 units: 10,140 bars, two pages; 150 units: a request of 100 (4 pages) and of 50 (2)
    assert requests_for([(f"S{i}", day) for i in range(26)]) == 2
    assert requests_for([(f"S{i}", day) for i in range(150)]) == 6
    assert requests_for([("A", day), ("A", other)]) == 2
    assert hours_at(6000, 100) == pytest.approx(1.0)


def test_the_estimate_counts_new_done_and_to_fetch_in_fetch_order() -> None:
    d1, d2 = date(2016, 3, 1), date(2016, 3, 2)
    plan = UnitPlan(
        {
            "train": frozenset({("A", d1), ("B", d1), ("C", d2)}),
            "spy": frozenset({("SPY", d1), ("SPY", d2)}),
            "gate_g1": frozenset({("A", d1), ("SPY", d1)}),
        }
    )
    rows = {r.part: r for r in estimate(plan, {"SPY:2016-03-02", "B:2016-03-01"})}
    assert [r.part for r in estimate(plan, set())] == ["spy", "gate_g1", "train"]
    assert (rows["spy"].new, rows["spy"].done, rows["spy"].to_fetch) == (2, 1, 1)
    assert (rows["gate_g1"].units, rows["gate_g1"].new, rows["gate_g1"].to_fetch) == (2, 1, 1)
    assert (rows["train"].new, rows["train"].done, rows["train"].to_fetch) == (2, 1, 1)
    assert rows["train"].sessions == 1 and rows["train"].requests == 1
    assert (rows["train"].first, rows["train"].last) == (d1, d2)
    assert rows["train"].sha == plan.sha("train")
