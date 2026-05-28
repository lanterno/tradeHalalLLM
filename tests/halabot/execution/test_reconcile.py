"""Reconcile — broker-truth plan, engine_owner scoping (fix R-02), events."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.execution.reconcile import Reconciler, reconcile_plan
from halabot.execution.venue import FakeVenue, Order
from halabot.platform.bus import InProcessEventBus
from halabot.platform.clock import FakeClock
from halabot.platform.event_log import InMemoryEventLog
from halabot.platform.events import Event, EventType

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


def test_plan_none_when_matched():
    plan = {a.asset: a for a in reconcile_plan({"NVDA": 5.0}, {"NVDA": 5.0}, lambda a: "belief")}
    assert plan["NVDA"].action == "none"


def test_plan_neutralizes_phantom_db_position():
    # DB thinks we hold 5; broker is flat → neutralize the DB to 0.
    plan = {a.asset: a for a in reconcile_plan({}, {"NVDA": 5.0}, lambda a: "belief")}
    assert plan["NVDA"].action == "neutralize"
    assert plan["NVDA"].adjustment_qty == -5.0


def test_plan_imports_broker_only_position():
    plan = {a.asset: a for a in reconcile_plan({"NVDA": 5.0}, {}, lambda a: "belief")}
    assert plan["NVDA"].action == "import"
    assert plan["NVDA"].adjustment_qty == 5.0


def test_plan_adjusts_quantity_mismatch():
    plan = {a.asset: a for a in reconcile_plan({"NVDA": 7.0}, {"NVDA": 5.0}, lambda a: "belief")}
    assert plan["NVDA"].action == "adjustment"
    assert plan["NVDA"].adjustment_qty == 2.0


def test_plan_skips_other_engine_positions():
    # A legacy-owned broker position must NOT be imported by the belief engine (R-02).
    plan = {
        a.asset: a
        for a in reconcile_plan({"NVDA": 5.0}, {}, lambda a: "legacy", engine_owner="belief")
    }
    assert plan["NVDA"].action == "skip_other_engine"
    assert plan["NVDA"].adjustment_qty == 0.0


@pytest.mark.asyncio
async def test_reconciler_emits_events_for_actions():
    venue = FakeVenue(clock_ts=T0, prices={"NVDA": 100.0})
    await venue.place(Order("NVDA", "buy", 5.0, "c1"))  # broker has 5
    bus = InProcessEventBus(InMemoryEventLog())
    events: list[Event] = []
    bus.subscribe({EventType.POSITION_RECONCILED}, lambda e: _cap(events, e))
    rec = Reconciler(
        venue=venue, bus=bus, clock=FakeClock(T0),
        db_net=lambda: {},  # DB knows nothing → import
        owner_of=lambda a: "belief",
    )
    await rec.run_once()
    assert events and events[0].payload["action"] == "import"
    assert events[0].payload["engine_owner"] == "belief"


async def _cap(sink, e):
    sink.append(e)
