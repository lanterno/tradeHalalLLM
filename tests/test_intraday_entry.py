"""The day-trader's entry-rule test: sessions, universes, observations, the rule."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import pytest

from halal_trader.research import intraday_entry as ie

MON = date(2018, 3, 5)  # EST: 09:30 New York = 14:30 UTC


def _raw(day: date, k: int, o: float, c: float, v: float = 100.0) -> dict[str, object]:
    ts = datetime(day.year, day.month, day.day, 14, 30, tzinfo=UTC) + timedelta(minutes=15 * k)
    return {"t": ts.isoformat().replace("+00:00", "Z"), "o": o, "c": c, "v": v}


def _flat(price: float = 100.0, volume: float = 100.0) -> ie.Session:
    return ie.Session(
        (price,) * ie.BARS,
        (price,) * ie.BARS,
        tuple(volume * (k + 1) for k in range(ie.BARS)),
    )


def test_bar_index_counts_regular_session_quarter_hours_only() -> None:
    at = datetime(2018, 3, 5, 14, 30, tzinfo=UTC)
    assert ie.bar_index(at) == 0
    assert ie.bar_index(at + timedelta(minutes=15 * 25)) == 25
    assert ie.bar_index(at + timedelta(minutes=15 * 26)) is None  # 16:00, after the close
    assert ie.bar_index(at - timedelta(minutes=15)) is None  # pre-market
    assert ie.bar_index(at + timedelta(minutes=5)) is None


def test_sessions_keep_only_complete_days_and_accumulate_volume() -> None:
    full = [_raw(MON, k, 10.0 + k, 11.0 + k, v=2.0) for k in range(ie.BARS)]
    short = [_raw(MON + timedelta(days=1), k, 10.0, 10.0) for k in range(ie.BARS - 1)]
    pre = [{"t": "2018-03-05T13:00:00Z", "o": 1.0, "c": 1.0, "v": 9.0}]
    got = ie.sessions_of(full + short + pre)
    assert list(got) == [MON]
    s = got[MON]
    assert s.open[0] == 10.0 and s.close[-1] == 11.0 + 25
    assert s.cumvol[0] == 2.0 and s.cumvol[-1] == 2.0 * ie.BARS


def test_universe_is_the_largest_halal_names_on_the_screen_dated_before_the_day() -> None:
    screens = {
        date(2018, 3, 31): [("A", 3.0), ("B", 2.0), ("C", None)],
        date(2018, 6, 30): [("D", 9.0), ("A", 1.0)],
    }
    got = ie.universes(screens, [date(2018, 3, 31), date(2018, 4, 2), date(2018, 7, 2)], limit=2)
    assert date(2018, 3, 31) not in got  # a screen is known only after its date
    assert got[date(2018, 4, 2)] == ["A", "B"]
    assert got[date(2018, 7, 2)] == ["D", "A"]


def test_observations_need_a_volume_baseline_and_measure_against_spy() -> None:
    days = [MON + timedelta(days=i) for i in range(ie.VOLUME_LOOKBACK + 1)]
    last = days[-1]
    # The traded day: up 1% by 09:45 on double volume, then 2% higher at the close.
    opens = [100.0] + [101.0] * (ie.BARS - 1)
    closes = [101.0] * (ie.BARS - 1) + [103.02]
    traded = ie.Session(tuple(opens), tuple(closes), tuple(200.0 * (k + 1) for k in range(ie.BARS)))
    stock = {d: _flat() for d in days[:-1]} | {last: traded}
    spy_last = ie.Session((100.0,) * ie.BARS, (100.0,) * (ie.BARS - 1) + (100.5,), _flat().cumvol)
    spy = {d: _flat() for d in days[:-1]} | {last: spy_last}
    obs = ie.observations({"AAA": stock, "SPY": spy}, {d: ["AAA"] for d in days})

    assert {o.day for o in obs} == {last}  # earlier days lack 20 sessions of history
    first = min(obs, key=lambda o: o.bar)
    assert first.bar == ie.FIRST_DECISION
    assert first.move == pytest.approx(0.01)
    assert first.rvol == pytest.approx(2.0)
    assert first.excess == pytest.approx(103.02 / 101.0 - 1 - 0.005)
    assert len(obs) == ie.LAST_DECISION - ie.FIRST_DECISION + 1
    assert ie.in_group(first, "CHASE") and not ie.in_group(first, "DIP")


def test_observations_skip_names_outside_the_day_universe() -> None:
    days = [MON + timedelta(days=i) for i in range(ie.VOLUME_LOOKBACK + 1)]
    bars = {"AAA": {d: _flat() for d in days}, "SPY": {d: _flat() for d in days}}
    assert ie.observations(bars, {days[-1]: ["BBB"]}) == []


def _obs(day: date, excess: float) -> ie.Observation:
    return ie.Observation(day, 1, "AAA", 0.01, 2.0, excess)


def test_summary_treats_each_session_as_one_draw_net_of_cost() -> None:
    d1, d2 = date(2018, 1, 2), date(2018, 1, 3)
    obs = [_obs(d1, 0.010), _obs(d1, 0.010), _obs(d1, 0.010), _obs(d2, -0.002)]
    s = ie.summarise(obs, "CHASE", cost=0.001)
    assert (s.observations, s.sessions) == (4, 2)
    assert s.mean == pytest.approx(((0.010 - 0.001) + (-0.002 - 0.001)) / 2)
    sd = math.sqrt(2 * (0.006) ** 2)
    assert s.t == pytest.approx(0.003 / (sd / math.sqrt(2)))


def test_verdict_needs_both_windows() -> None:
    def s(mean: float, t: float) -> ie.Summary:
        return ie.Summary("CHASE", 100, 50, mean, t)

    assert ie.verdict(s(-0.001, -3.0), s(-0.002, -2.1)) == "refuted"
    assert ie.verdict(s(0.001, 2.5), s(0.001, 2.0)) == "supported"
    assert ie.verdict(s(-0.001, -3.0), s(-0.001, -1.5)) == "inconclusive"
    assert ie.verdict(s(0.001, 2.5), s(-0.001, -2.5)) == "inconclusive"


async def test_load_fetches_each_universe_name_and_spy_in_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from halal_trader.halal import strict

    async def screens(engine: object) -> dict[date, list[object]]:
        row = SimpleNamespace
        return {
            date(2018, 2, 28): [
                row(symbol="AAA", market_cap=5.0, verdict="halal"),
                row(symbol="BAD", market_cap=9.0, verdict="not_halal"),
            ]
        }

    monkeypatch.setattr(strict, "all_screens", screens)
    calls: list[tuple[list[str], datetime, datetime, str]] = []

    class Market:
        async def minute_bars_many(
            self, symbols: object, *, start: datetime, end: datetime, timeframe: str
        ) -> dict[str, list[dict[str, object]]]:
            calls.append((list(symbols), start, end, timeframe))  # type: ignore[call-overload]
            if start <= datetime(2018, 3, 5, tzinfo=UTC) < end:
                return {"AAA": [_raw(MON, k, 1.0, 1.0) for k in range(ie.BARS)]}
            return {}

    bars, universe = await ie.load(
        object(),  # type: ignore[arg-type]
        Market(),
        start=date(2018, 3, 1),
        end=date(2018, 9, 30),
    )
    assert universe[MON] == ["AAA"]
    assert {c[0] == ["AAA", "SPY"] and c[3] == "15Min" for c in calls} == {True}
    assert calls[0][1] == datetime(2018, 3, 1, tzinfo=UTC) - ie.WARM_UP
    assert calls[-1][2] == datetime(2018, 10, 1, tzinfo=UTC)
    assert len(calls) == 3  # 45 + 214 days in 92-day chunks
    assert list(bars["AAA"]) == [MON] and bars["SPY"] == {}
