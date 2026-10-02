"""Event portfolios: slots, clock entries, exits, eligibility, costs; Phase C signals."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from halal_trader.events.portfolio import Candidate, simulate
from halal_trader.events.signals import (
    above_trailing_percentile,
    announcement_returns,
    clusters,
)
from halal_trader.events.study import Bars

DAYS = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]  # treat all as sessions


def _bars(prices: dict[str, list[float]], opens: dict[str, list[float]] | None = None) -> Bars:
    opens = opens or prices
    return Bars(
        DAYS,
        open={s: dict(zip(DAYS, v, strict=False)) for s, v in opens.items()},
        close={s: dict(zip(DAYS, v, strict=False)) for s, v in prices.items()},
    )


def _at(day: int, hour: int) -> datetime:  # hour in UTC; 14:00 UTC = 09:00/10:00 ET
    return datetime(2024, 1, 1 + day, hour, tzinfo=UTC)


def _run(bars, cands, *, slots=2, hold=2, eligible=None, cost=0.0):
    return simulate(
        bars,
        cands,
        hold=hold,
        slots=slots,
        start=DAYS[0],
        eligible_from={DAYS[0]: eligible if eligible is not None else {"A", "B", "C"}},
        cost_bps=lambda s, d: cost,
    )


def test_a_pre_open_event_buys_at_that_open_and_holds_to_the_close_hold_sessions_later() -> None:
    bars = _bars(
        {"A": [10, 10, 11, 12, 12, 12, 12, 12, 12, 12]}, {"A": [10, 10, 10, 12] + [12] * 6}
    )
    book = _run(bars, [Candidate("A", _at(2, 12), 1.0)], slots=2, hold=2)  # 07:00 ET on day 2
    nav = list(_nav(book.returns))
    # Half the book in A from day 2's open (10) to day 3's close (12): +20% on half.
    assert nav[3] == pytest.approx(1.10)
    assert nav[-1] == pytest.approx(1.10)  # exited, back to cash
    assert book.trades == 1


def _nav(returns):
    v = 1.0
    for r in returns:
        v *= 1 + r
        yield v


def test_slots_cap_positions_and_the_strongest_signal_goes_first() -> None:
    bars = _bars({s: [10.0] * 10 for s in "ABC"})
    cands = [Candidate(s, _at(1, 12), score) for s, score in (("A", 1), ("B", 3), ("C", 2))]
    book = _run(bars, cands, slots=2, hold=3)
    assert book.trades == 2 and book.skipped_full == 1


def test_ineligible_names_are_never_bought_and_costs_are_paid_both_ways() -> None:
    bars = _bars({s: [10.0] * 10 for s in "AB"})
    book = _run(
        bars,
        [Candidate("A", _at(1, 12), 1), Candidate("B", _at(1, 12), 1)],
        eligible={"A"},
        cost=50.0,
    )
    assert book.trades == 1 and book.skipped_ineligible == 1
    nav = list(_nav(book.returns))
    # Half the book pays 50 bps in and 50 bps out on a flat price: about -0.5% of NAV.
    assert nav[-1] == pytest.approx(1 - 0.5 * (1 - (1 - 0.005) / 1.005), abs=1e-6)


def test_trailing_percentiles_never_see_the_future() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [(t0 + timedelta(hours=i), "X", float(i % 100)) for i in range(400)]
    rows.append((t0 + timedelta(hours=401), "Y", 95.0))
    picked = above_trailing_percentile(rows)
    assert any(c.symbol == "Y" for c in picked)  # 95 >= the trailing 90th (~89)
    # A huge later value does not change what was picked before it.
    later = rows + [(t0 + timedelta(hours=500), "Z", 1e9)]
    assert [c for c in above_trailing_percentile(later) if c.symbol != "Z"] == picked


def test_announcement_return_spans_the_close_before_to_the_close_after() -> None:
    bars = _bars({"A": [10, 10, 10, 11, 12, 12, 12, 12, 12, 12], "SPY": [100.0] * 10})
    (row,) = announcement_returns([("A", _at(3, 12))], bars)  # pre-open day 3
    known, symbol, abn = row
    assert symbol == "A" and abn == pytest.approx(0.2)  # day 2 close 10 -> day 4 close 12
    assert known.date() == DAYS[4]


def test_insider_clusters_need_two_insiders_and_100k_within_30_days() -> None:
    t = datetime(2024, 3, 1, tzinfo=UTC)
    buys = [
        (t, "A", "ceo", 60_000.0),
        (t + timedelta(days=10), "A", "cfo", 50_000.0),  # completes the cluster
        (t + timedelta(days=12), "A", "dir", 500_000.0),  # quiet period: no second signal
        (t, "B", "ceo", 90_000.0),
        (t + timedelta(days=40), "B", "cfo", 90_000.0),  # too far apart
        (t, "C", "ceo", 200_000.0),  # one insider only
    ]
    (c,) = clusters(buys)
    assert c.symbol == "A" and c.published_at == t + timedelta(days=10) and c.score == 110_000.0


def test_combo_averages_a_releases_percentiles_and_waits_for_the_latest() -> None:
    from halal_trader.events.signals import combine

    t = datetime(2024, 5, 1, 20, tzinfo=UTC)
    sue = {("A", t): 0.9, ("B", t): 0.2}
    headline = {("A", t - timedelta(minutes=5)): 0.7}
    ear = {("A", t + timedelta(days=1)): 0.5, ("Z", t): 0.99}
    (row,) = combine(sue, headline, ear)  # B has one reading only: not combined
    when, symbol, value = row
    assert symbol == "A" and value == pytest.approx(0.7) and when == t + timedelta(days=1)


def test_trailing_rank_is_a_past_only_percentile() -> None:
    from halal_trader.events.signals import trailing_rank

    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [(t0 + timedelta(hours=i), "X", float(i)) for i in range(300)]
    ranks = trailing_rank(rows)
    assert ranks[("X", t0 + timedelta(hours=299))] == 1.0  # above everything before it
    assert ("X", t0) not in ranks  # no history yet
