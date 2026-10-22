"""The default-stop test: where a stop sells, what it is worth against the close, the rule."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.research import intraday_entry as ie
from halal_trader.research import intraday_stop as st
from halal_trader.trading.decision_quality import Stat

DAY = date(2018, 3, 6)


def _session(
    *,
    low: dict[int, float] | None = None,
    open_: dict[int, float] | None = None,
    close: float = 100.0,
) -> ie.Session:
    opens = [100.0] * ie.BARS
    for k, v in (open_ or {}).items():
        opens[k] = v
    lows = [min(o, 100.0) for o in opens]
    for k, v in (low or {}).items():
        lows[k] = v
    closes = [100.0] * (ie.BARS - 1) + [close]
    return ie.Session(
        tuple(opens), tuple(closes), tuple(float(k) for k in range(ie.BARS)), tuple(lows)
    )


def _obs(symbol: str, bar: int, move: float = 0.0) -> ie.Observation:
    return ie.Observation(DAY, bar, symbol, move, 1.0, 0.0)


def test_a_stop_never_reached_sells_nothing() -> None:
    assert st.stop_sale(_session(low={10: 99.1}), 1, 0.01) is None


def test_a_stop_sells_at_its_price_less_slippage() -> None:
    sale = st.stop_sale(_session(low={10: 98.5}), 1, 0.01)
    assert sale == pytest.approx(99.0 * (1 - st.SLIPPAGE))


def test_a_stop_sells_at_the_open_after_a_gap_below_it() -> None:
    sale = st.stop_sale(_session(open_={10: 97.0}), 1, 0.01)
    assert sale == pytest.approx(97.0 * (1 - st.SLIPPAGE))


def test_a_stop_counts_from_the_entry_bar_on_only() -> None:
    before = _session(low={3: 90.0})
    assert st.stop_sale(before, 4, 0.01) is None
    assert st.stop_sale(before, 3, 0.01) == pytest.approx(99.0 * (1 - st.SLIPPAGE))


def test_the_value_is_the_stop_sale_against_the_close_per_unit_bought() -> None:
    # Stopped at 99 and the close recovered to 101: the stop cost about 2%.
    reverted = _session(low={10: 98.0}, close=101.0)
    held = _session(close=105.0)  # never stopped: worth zero, however the day ended
    bars = {"AAA": {DAY: reverted}, "BBB": {DAY: held}}
    s = st.summarise([_obs("AAA", 1), _obs("BBB", 1)], bars, "ALL", 0.01)
    sale = 99.0 * (1 - st.SLIPPAGE)
    assert s.value.trips == 2 and s.value.sessions == 1
    assert s.value.mean == pytest.approx((sale - 101.0) / 100.0 / 2)
    assert s.stopped == pytest.approx(0.5)
    assert s.when_stopped.trips == 1
    assert s.when_stopped.mean == pytest.approx((sale - 101.0) / 100.0)


def test_groups_pick_their_entries() -> None:
    bars = {"AAA": {DAY: _session(low={10: 98.0}, close=97.0)}}
    obs = [_obs("AAA", 1, move=-0.01), _obs("AAA", 2, move=0.0)]
    assert st.summarise(obs, bars, "DIP", 0.01).value.trips == 1
    assert st.summarise(obs, bars, "ALL", 0.01).value.trips == 2
    assert st.summarise(obs, bars, "CHASE", 0.01).value.trips == 0


def test_the_rule_needs_both_windows() -> None:
    def stat(t: float | None) -> Stat:
        return Stat(100, 50, None if t is None else t / 1000, t)

    assert st.verdict(stat(-2.5), stat(-2.0)) == "refuted"
    assert st.verdict(stat(2.1), stat(3.0)) == "supported"
    assert st.verdict(stat(-2.5), stat(-1.9)) == "inconclusive"
    assert st.verdict(stat(2.5), stat(-2.5)) == "inconclusive"
    assert st.verdict(stat(None), stat(-3.0)) == "inconclusive"
