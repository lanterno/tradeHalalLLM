"""Composition root — the full read-only engine on Postgres, end-to-end.

Builds the real engine against the test DB, feeds a compliance verdict + a
stream of uptrend bars through its bus, and asserts the whole stack runs:
belief persisted (perception→cognition→belief) and a shadow buy proposed
(belief→conviction→policy), with NO execution.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halabot.app import build_engine
from halabot.belief.schema import Direction
from halabot.platform.clock import FakeClock
from halabot.platform.events import Event, EventType, new_event

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_engine_builds_and_runs_end_to_end(halabot_engine):
    clock = FakeClock(T0)
    engine = await build_engine(db_engine=halabot_engine, clock=clock)
    proposed: list[Event] = []
    engine.bus.subscribe(
        {EventType.POLICY_TRADE_PROPOSED}, lambda e: _cap(proposed, e)
    )
    try:
        # Halal verdict first (INV-7 — without it the policy's halal gate blocks buys).
        await engine.bus.publish(
            new_event(
                clock, EventType.COMPLIANCE_VERDICT, source="zoya", asset="NVDA",
                payload={"status": "halal", "detail": "ok", "screening_id": 1,
                         "transient_error": False},
            )
        )
        # Uptrend bars → bullish belief.
        for i in range(30):
            clock.advance(timedelta(minutes=1))
            c = 100.0 + i
            await engine.bus.publish(
                new_event(
                    clock, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA",
                    payload={"o": c, "h": c + 1, "low": c - 1, "c": c, "v": 1000.0},
                )
            )

        belief = await engine.store.get("NVDA")
        assert belief is not None
        assert belief.direction == Direction.LONG_BIAS
        assert belief.conviction > 0.0
        assert belief.halal is not None and belief.halal.status == "halal"

        assert engine.shadow.proposals_count >= 1
        assert proposed and proposed[0].payload["side"] == "buy"
        assert proposed[0].payload["shadow"] is True  # never executed
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_blocks_buy_for_non_halal_asset(halabot_engine):
    clock = FakeClock(T0)
    engine = await build_engine(db_engine=halabot_engine, clock=clock)
    proposed: list[Event] = []
    engine.bus.subscribe({EventType.POLICY_TRADE_PROPOSED}, lambda e: _cap(proposed, e))
    try:
        await engine.bus.publish(
            new_event(
                clock, EventType.COMPLIANCE_VERDICT, source="zoya", asset="HOOD",
                payload={"status": "not_halal", "detail": "interest income",
                         "screening_id": 2, "transient_error": False},
            )
        )
        for i in range(30):
            clock.advance(timedelta(minutes=1))
            c = 100.0 + i
            await engine.bus.publish(
                new_event(
                    clock, EventType.OBSERVATION_BAR, source="alpaca", asset="HOOD",
                    payload={"o": c, "h": c + 1, "low": c - 1, "c": c, "v": 1000.0},
                )
            )
        belief = await engine.store.get("HOOD")
        assert belief is not None and belief.direction == Direction.LONG_BIAS  # bullish belief...
        assert proposed == []  # ...but NO buy proposed — halal gate (INV-7)
    finally:
        await engine.stop()


async def _cap(sink: list, e: Event) -> None:
    sink.append(e)
