"""The exchange's fill rules, SPY legs and costs, on hand-built bar series (pure)."""

from __future__ import annotations

import math

import pytest

from halabot.playbooks.clock import US, to_us
from halabot.playbooks.exchange import (
    Exchange,
    first_eligible,
    in_surcharge_window,
    market_fill,
    one_way_bps,
    spy_fill,
    unfilled_exit,
)
from halabot.playbooks.types import BarSeries, OrderKind, SimConfig, WorkingOrder
from tests.halabot.playbooks._support import MON, TUE, et, session_bars

GAP_US = 5 * 60 * US


def series(*parts, scales=None):  # type: ignore[no-untyped-def]
    return BarSeries.build(list(parts), scales or [1.0] * len(parts), 5)


def idx(s: BarSeries, day, hh, mm) -> int:  # type: ignore[no-untyped-def]
    return [int(t) for t in s.ts].index(int(et(day, hh, mm).timestamp()))


def order(oid="o", side="buy", k=0, active=None, qty=None):  # type: ignore[no-untyped-def]
    return WorkingOrder(oid, side, OrderKind.MARKET, k, 0, active, qty)


def test_first_eligible_is_the_first_bar_at_or_after_active_in_the_session() -> None:
    s = series(session_bars(MON), session_bars(TUE))
    assert first_eligible(s, to_us(et(MON, 10, 15, 8)), 0) == idx(s, MON, 10, 16)
    assert first_eligible(s, to_us(et(MON, 10, 16)), 0) == idx(s, MON, 10, 16)  # ts == active
    assert first_eligible(s, to_us(et(MON, 15, 59, 1)), 0) is None  # next bar is TUE's
    assert first_eligible(s, to_us(et(TUE, 9, 30, 3)), 1) == idx(s, TUE, 9, 31)
    assert first_eligible(s, to_us(et(TUE, 16, 0)), 1) is None


@pytest.mark.parametrize(
    ("row", "price"),
    [
        ((100.0, 101.0, 99.0, 100.5, 1e3, 100.4), 100.4),  # inside the range
        ((100.0, 101.0, 99.0, 100.5, 1e3, 101.7), 101.0),  # clamped to the high
        ((100.0, 101.0, 99.0, 100.5, 1e3, 98.2), 99.0),  # clamped to the low
        ((100.0, 101.0, 99.0, 100.5, 1e3, math.nan), 100.0),  # no VWAP: the open
    ],
)
def test_market_fill_is_the_clamped_vwap(row, price) -> None:  # type: ignore[no-untyped-def]
    s = series(session_bars(MON, rows={(10, 16): row}))
    i = idx(s, MON, 10, 16)
    got, flags = market_fill(
        s, i, to_us(et(MON, 10, 15, 8)), gap_us=GAP_US, session_open_us=to_us(et(MON, 9, 30))
    )
    assert (got, flags) == (price, ())


def test_market_fill_takes_the_open_after_a_gap_a_wait_or_a_late_open() -> None:
    row = (100.0, 101.0, 99.0, 100.5, 1e3, 100.4)
    open_us = to_us(et(MON, 9, 30))
    gapped = series(session_bars(MON, rows={(10, 5): row}, skip=[(10, m) for m in range(1, 5)]))
    i = idx(gapped, MON, 10, 5)
    assert market_fill(
        gapped, i, to_us(et(MON, 10, 0, 30)), gap_us=GAP_US, session_open_us=open_us
    ) == (
        100.0,
        ("gap_fill",),
    )
    four = series(session_bars(MON, rows={(10, 5): row}, skip=[(10, m) for m in range(2, 5)]))
    j = idx(four, MON, 10, 5)  # 10:01 -> 10:05 is 4 minutes: no gap
    assert market_fill(
        four, j, to_us(et(MON, 10, 1, 30)), gap_us=GAP_US, session_open_us=open_us
    ) == (
        100.4,
        (),
    )
    flat = series(session_bars(MON, rows={(10, 6): row}))
    k = idx(flat, MON, 10, 6)  # waited 5.5 minutes for this bar
    assert market_fill(flat, k, to_us(et(MON, 10, 0, 30)), gap_us=GAP_US, session_open_us=open_us)[
        1
    ] == ("gap_fill",)
    late = series(session_bars(MON, first=(9, 35), rows={(9, 35): row}))
    assert market_fill(
        late, 0, to_us(et(MON, 9, 30, 3)), gap_us=GAP_US, session_open_us=open_us
    ) == (
        100.0,
        ("gap_fill", "late_open"),
    )
    on_time = series(session_bars(MON, first=(9, 34), rows={(9, 34): row}))
    assert market_fill(
        on_time, 0, to_us(et(MON, 9, 30, 3)), gap_us=GAP_US, session_open_us=open_us
    ) == (
        100.4,
        (),
    )


def test_a_sessions_first_bar_is_not_after_a_gap_from_the_previous_session() -> None:
    row = (100.0, 101.0, 99.0, 100.5, 1e3, 100.4)
    s = series(session_bars(MON), session_bars(TUE, rows={(9, 30): row}))
    i = idx(s, TUE, 9, 30)
    assert market_fill(
        s, i, to_us(et(TUE, 9, 29, 59)), gap_us=GAP_US, session_open_us=to_us(et(TUE, 9, 30))
    ) == (100.4, ())


def test_spy_fills_at_the_same_bar_start_by_the_same_rule() -> None:
    rows = {
        (10, 16): (200.0, 201.0, 199.0, 200.5, 1e5, 200.4),
        (10, 20): (200.0, 201.0, 199.0, 200.5, 1e5, math.nan),
    }
    spy = series(session_bars(MON, price=200.0, rows=rows, skip=[(10, 17), (10, 18)]))
    t16 = int(et(MON, 10, 16).timestamp())
    assert spy_fill(spy, t16, 0, use_open=False) == (200.4, ())
    assert spy_fill(spy, t16, 0, use_open=True) == (200.0, ())
    assert spy_fill(spy, int(et(MON, 10, 20).timestamp()), 0, use_open=False) == (200.0, ())
    # No SPY bar at 10:17: its last print before (the 10:16 close), flagged.
    assert spy_fill(spy, int(et(MON, 10, 17).timestamp()), 0, use_open=False) == (
        200.5,
        ("spy_proxy",),
    )
    later = series(session_bars(MON, price=200.0, first=(9, 40)))
    assert spy_fill(later, int(et(MON, 9, 31).timestamp()), 0, use_open=False) == (
        200.0,
        ("spy_proxy",),
    )
    price, flags = spy_fill(series(), int(et(MON, 9, 31).timestamp()), 0, use_open=False)
    assert math.isnan(price) and flags == ("spy_proxy",)


def test_the_surcharge_window() -> None:
    s = series(session_bars(MON, skip=[(11, m) for m in range(10)]))
    assert in_surcharge_window(s, idx(s, MON, 9, 44))  # 14 minutes after the first bar
    assert not in_surcharge_window(s, idx(s, MON, 9, 45))
    assert in_surcharge_window(s, idx(s, MON, 11, 10))  # the bar ending a 10-minute gap
    assert in_surcharge_window(s, idx(s, MON, 11, 14))
    assert not in_surcharge_window(s, idx(s, MON, 11, 15))


def test_costs_by_mode() -> None:
    assert one_way_bps(15.0, "study", rank=500) == 15.0
    assert one_way_bps(15.0, "study_x1.5", rank=500) == 22.5
    assert one_way_bps(7.0, "study_x2", rank=10) == 14.0
    assert one_way_bps(7.0, "surcharge", rank=10) == 2.0 + 5.0
    assert one_way_bps(15.0, "surcharge", rank=500, surcharge_window=True) == 20.0 + 5.0
    # impact: max(5, sigma * 1e4 * sqrt(notional / ADV)) = 0.02 * 1e4 * sqrt(1e4 / 1e6) = 20
    got = one_way_bps(15.0, "surcharge", rank=500, sigma=0.02, adv20_usd=1e6)
    assert abs(got - (10.0 + 20.0)) < 1e-12


def test_the_exchange_fills_sells_first_and_expires_day_orders() -> None:
    s = series(session_bars(MON, rows={(10, 16): (100.0, 101.0, 99.0, 100.5, 500.0, 100.4)}))
    ex = Exchange(SimConfig())
    active = to_us(et(MON, 10, 15, 8))
    ex.submit(order("buy", "buy", active=active))
    ex.submit(order("sell", "sell", active=active, qty=3.0))
    ex.submit(order("later", "buy", active=to_us(et(MON, 10, 30))))
    with pytest.raises(ValueError):
        ex.submit(order("buy", "buy", active=active))
    i = idx(s, MON, 10, 16)
    fills = ex.on_bar(i, s, session_open_us=to_us(et(MON, 9, 30)))
    assert [(f.order.order_id, f.price, f.bar_ts, f.filled_at_us) for f in fills] == [
        ("sell", 100.4, int(et(MON, 10, 16).timestamp()), to_us(et(MON, 10, 17))),
        ("buy", 100.4, int(et(MON, 10, 16).timestamp()), to_us(et(MON, 10, 17))),
    ]
    assert abs(fills[1].participation - (10_000 / 100.4) / 500.0) < 1e-12
    assert [o.order_id for o in ex.working] == ["later"]
    with pytest.raises(AssertionError):  # tested on a bar after the one it was due on
        ex.on_bar(idx(s, MON, 10, 31), s, session_open_us=to_us(et(MON, 9, 30)))
    assert [o.order_id for o in ex.expire(0)] == ["later"] and ex.working == ()
    ex.submit(order("x", "buy", active=active))
    assert ex.cancel("x") is not None and ex.cancel("x") is None


def test_an_unfilled_exit_falls_back_close_then_market_then_last_trade() -> None:
    spare = series(
        session_bars(TUE, first=(9, 41), rows={(9, 41): (99.0, 99.5, 98.5, 99.2, 1e3, 99.1)})
    )
    spy_spare = series(
        session_bars(TUE, price=200.0, rows={(9, 41): (201.0, 202.0, 200.0, 201.5, 1e5, 201.2)})
    )
    common = dict(
        lag_us=3 * US,
        gap_us=GAP_US,
        last_close=100.5,
        spy_last=(200.2, ()),
        spare_open_us=to_us(et(TUE, 9, 30)),
    )
    close = unfilled_exit(
        official_close=100.7, spy_close=200.5, spare=spare, spy_spare=spy_spare, **common
    )
    assert (close.rule, close.price, close.spy_price, close.flags) == (
        "close_fallback",
        100.7,
        200.5,
        ("close_fallback",),
    )
    market = unfilled_exit(
        official_close=None, spy_close=200.5, spare=spare, spy_spare=spy_spare, **common
    )
    # The spare's first bar is 09:41, 11 minutes after the open: a late open at its open,
    # and SPY's open at the same bar start.
    assert (market.rule, market.price, market.spy_price, market.bar_ts, market.i) == (
        "no_market",
        99.0,
        201.0,
        int(et(TUE, 9, 41).timestamp()),
        0,
    )
    assert market.flags == ("no_market", "gap_fill", "late_open")
    last = unfilled_exit(official_close=None, spy_close=200.5, spare=None, spy_spare=None, **common)
    assert (last.rule, last.price, last.spy_price, last.flags) == (
        "unresolved",
        100.5,
        200.2,
        ("unresolved",),
    )
