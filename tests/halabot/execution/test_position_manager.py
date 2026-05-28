"""Exit ladder (Appendix H) — precedence + trailing ratchet + monitor actions."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.execution.position_manager import HoldContext, PositionMonitor, decide_exit
from halabot.execution.venue import FakeVenue, Order
from halabot.platform.bus import InProcessEventBus
from halabot.platform.clock import FakeClock
from halabot.platform.event_log import InMemoryEventLog

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


def _ctx(**kw) -> HoldContext:
    base = dict(asset="NVDA", price=100.0, target_weight=0.1)
    base.update(kw)
    return HoldContext(**base)  # type: ignore[arg-type]


# ── precedence (first match wins) ──
def test_risk_halt_flattens_first():
    # Even with a healthy hold, a risk halt flattens (rung 1 beats all).
    d = decide_exit(_ctx(risk_halted=True, target_weight=0.2, compliance_lapsed=True))
    assert d.action == "exit" and d.reason == "risk_halt"


def test_compliance_beats_stop_and_target():
    # A held name turned non-compliant exits ANY P&L, above stop/trend (rung 2).
    d = decide_exit(_ctx(compliance_lapsed=True, price=200.0, stop=50.0, target_weight=0.3))
    assert d.action == "exit" and d.reason == "compliance_lapsed"


def test_belief_invalidated_exits():
    d = decide_exit(_ctx(belief_invalidated=True, target_weight=0.3))
    assert d.action == "exit" and d.reason == "belief_invalidated"


def test_hard_stop_exit():
    d = decide_exit(_ctx(price=94.0, stop=95.0, target_weight=0.3))
    assert d.action == "exit" and d.reason == "stop_loss"


def test_trend_break_only_for_winners():
    # Winner closing below its SMA → trend_break (rung 5).
    d = decide_exit(_ctx(price=110.0, sma=112.0, is_winner=True, target_weight=0.3))
    assert d.action == "exit" and d.reason == "trend_break"
    # A loser below SMA does NOT trend-break (it'd exit on stop/target instead).
    d2 = decide_exit(_ctx(price=90.0, sma=112.0, is_winner=False, target_weight=0.3))
    assert d2.reason != "trend_break"


def test_trailing_ratchet_tightens_without_exit():
    d = decide_exit(_ctx(price=120.0, trailing_high=120.0, trailing_pct=0.05, target_weight=0.3))
    assert d.action == "tighten" and d.new_stop == pytest.approx(114.0)


def test_target_zero_exits_when_nothing_else_fires():
    d = decide_exit(_ctx(price=100.0, target_weight=0.0))
    assert d.action == "exit" and d.reason == "target_zero"


def test_target_zero_not_starved_by_trailing_ratchet():
    # A decayed-conviction position making new highs must still exit on
    # target_zero, not ratchet forever (audit #2).
    d = decide_exit(
        _ctx(price=130.0, trailing_high=130.0, trailing_pct=0.05, target_weight=0.0)
    )
    assert d.action == "exit" and d.reason == "target_zero"


def test_trailing_ratchet_still_active_when_target_positive():
    d = decide_exit(
        _ctx(price=130.0, trailing_high=130.0, trailing_pct=0.05, target_weight=0.2)
    )
    assert d.action == "tighten"  # policy still wants it → slow-out via trailing


def test_healthy_hold_holds():
    d = decide_exit(_ctx(price=100.0, stop=90.0, target_weight=0.2))
    assert d.action == "hold"


# ── monitor applies decisions ──
def _monitor():
    venue = FakeVenue(clock_ts=T0, prices={"NVDA": 100.0})
    bus = InProcessEventBus(InMemoryEventLog())
    return PositionMonitor(venue=venue, bus=bus, clock=FakeClock(T0)), venue


@pytest.mark.asyncio
async def test_monitor_closes_on_exit():
    mon, venue = _monitor()
    await venue.place(Order("NVDA", "buy", 5.0, "c1"))  # seed a position
    d = await mon.evaluate(_ctx(compliance_lapsed=True))
    assert d.action == "exit"
    assert await venue.positions() == []  # flattened


@pytest.mark.asyncio
async def test_monitor_ratchets_and_persists_stop():
    mon, _ = _monitor()
    d1 = await mon.evaluate(_ctx(price=120.0, trailing_high=120.0, trailing_pct=0.05))
    assert d1.action == "tighten"
    assert mon.stop_for("NVDA") == pytest.approx(114.0)
    # Next tick at a lower price: the ratcheted stop now triggers a stop_loss exit.
    d2 = await mon.evaluate(_ctx(price=113.0, trailing_pct=0.05, trailing_high=120.0))
    assert d2.action == "exit" and d2.reason == "stop_loss"
