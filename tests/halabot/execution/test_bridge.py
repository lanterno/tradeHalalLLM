"""LiveTradeBridge — routes proposals to the executor; refuses unless ARMED."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.execution.bridge import LiveTradeBridge, wire_live_execution
from halabot.execution.feasibility import FeasibilityConfig
from halabot.execution.live_mode import LiveModeDecision
from halabot.execution.orders import Executor
from halabot.execution.venue import FakeVenue
from halabot.platform.bus import InProcessEventBus
from halabot.platform.clock import FakeClock
from halabot.platform.event_log import InMemoryEventLog
from halabot.platform.events import Event, EventType, new_event

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


class _Account:
    def __init__(self, *, equity=100_000.0, bp=100_000.0, gross=0.0, held=None, halal=None):
        self._equity, self._bp, self._gross = equity, bp, gross
        self._held = held or {}
        self._halal = halal or set()

    def equity(self):
        return self._equity

    def buying_power(self):
        return self._bp

    def gross_exposure_usd(self):
        return self._gross

    def position_qty(self, asset):
        return self._held.get(asset, 0.0)

    def halal_ok(self, asset):
        return asset in self._halal


def _armed(**kw):
    base = dict(armed=True, reason="ARMED", market="stocks", max_order_usd=1000.0,
                max_account_usd=10_000.0, daily_loss_floor_pct=0.05)
    base.update(kw)
    return LiveModeDecision(**base)


def _build(account, decision):
    venue = FakeVenue(clock_ts=T0, prices={"NVDA": 100.0})
    bus = InProcessEventBus(InMemoryEventLog())
    fills: list[Event] = []
    bus.subscribe({EventType.ORDER_FILLED}, lambda e: _cap(fills, e))
    ex = Executor(venue=venue, bus=bus, clock=FakeClock(T0),
                  feasibility=FeasibilityConfig(min_notional_usd=50.0, lot_step=1.0))
    return bus, venue, fills, ex


async def _cap(sink, e):
    sink.append(e)


async def _propose(bus, asset, side, delta):
    await bus.publish(
        new_event(
            FakeClock(T0), EventType.POLICY_TRADE_PROPOSED, source="policy.shadow", asset=asset,
            payload={"side": side, "target_weight": max(0.0, delta), "current_weight": 0.0,
                     "weight_delta": delta, "reason": "test", "belief_version": 1, "shadow": True},
        )
    )


def test_wire_refuses_when_not_armed():
    bus, _, _, ex = _build(_Account(), _armed())
    with pytest.raises(RuntimeError):
        wire_live_execution(
            bus=bus, executor=ex, decision=_armed(armed=False, reason="shadow only"),
            account=_Account(),
        )


@pytest.mark.asyncio
async def test_bridge_routes_buy_to_executor_when_armed():
    acct = _Account(halal={"NVDA"})
    bus, venue, fills, ex = _build(acct, _armed())
    bridge = wire_live_execution(bus=bus, executor=ex, decision=_armed(), account=acct)
    await _propose(bus, "NVDA", "buy", 0.001)  # tiny: 0.001×100k=$100 → within $1000 cap
    assert bridge.routed == 1
    assert len(fills) == 1


@pytest.mark.asyncio
async def test_order_cap_clamps_buying_power():
    # weight_delta would want $50k, but the $1000 SAFEGUARD order cap limits the buy.
    acct = _Account(bp=100_000.0, halal={"NVDA"})
    bus, venue, fills, ex = _build(acct, _armed())
    wire_live_execution(bus=bus, executor=ex, decision=_armed(max_order_usd=1000.0), account=acct)
    await _propose(bus, "NVDA", "buy", 0.5)  # 0.5×100k = $50k desired
    assert fills  # filled, but...
    assert fills[0].payload["filled_quantity"] == 10.0  # capped to $1000 / $100 = 10 shares


@pytest.mark.asyncio
async def test_account_exposure_ceiling_refuses_buy():
    acct = _Account(gross=10_000.0, halal={"NVDA"})  # already at the $10k cap
    bus, venue, fills, ex = _build(acct, _armed())
    bridge = wire_live_execution(bus=bus, executor=ex, decision=_armed(), account=acct)
    await _propose(bus, "NVDA", "buy", 0.001)
    assert fills == []  # refused — account exposure at SAFEGUARD cap
    assert bridge.routed == 0


@pytest.mark.asyncio
async def test_bridge_cannot_be_constructed_unarmed():
    bus, _, _, ex = _build(_Account(), _armed())
    with pytest.raises(RuntimeError):
        LiveTradeBridge(
            bus=bus, executor=ex, decision=_armed(armed=False, reason="x"), account=_Account()
        )
