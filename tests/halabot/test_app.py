"""Composition root — the full read-only engine on Postgres, end-to-end.

Builds the real engine against the test DB, feeds a compliance verdict + a
stream of uptrend bars through its bus, and asserts the whole stack runs:
belief persisted (perception→cognition→belief) and a shadow buy proposed
(belief→conviction→policy), with NO execution.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa

from halabot.app import build_engine
from halabot.belief.schema import Direction
from halabot.platform.clock import FakeClock
from halabot.platform.db import OUTCOME_COHORT, open_position, outcome
from halabot.platform.events import Event, EventType, new_event

T0 = datetime(2026, 5, 28, 15, 0, tzinfo=UTC)  # 11:00 ET, a Thursday: the market is open


@pytest.mark.asyncio
async def test_engine_builds_and_runs_end_to_end(halabot_engine):
    clock = FakeClock(T0)
    engine = await build_engine(db_engine=halabot_engine, clock=clock)
    proposed: list[Event] = []
    engine.bus.subscribe({EventType.POLICY_TRADE_PROPOSED}, lambda e: _cap(proposed, e))
    try:
        # Halal verdict first (INV-7 — without it the policy's halal gate blocks buys).
        await engine.bus.publish(
            new_event(
                clock,
                EventType.COMPLIANCE_VERDICT,
                source="screen",
                asset="NVDA",
                payload={
                    "status": "halal",
                    "detail": "ok",
                    "screening_id": 1,
                    "transient_error": False,
                },
            )
        )
        # Uptrend bars → bullish belief.
        for i in range(30):
            clock.advance(timedelta(minutes=1))
            c = 100.0 + i
            await engine.bus.publish(
                new_event(
                    clock,
                    EventType.OBSERVATION_BAR,
                    source="alpaca",
                    asset="NVDA",
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
                clock,
                EventType.COMPLIANCE_VERDICT,
                source="screen",
                asset="HOOD",
                payload={
                    "status": "not_halal",
                    "detail": "interest income",
                    "screening_id": 2,
                    "transient_error": False,
                },
            )
        )
        for i in range(30):
            clock.advance(timedelta(minutes=1))
            c = 100.0 + i
            await engine.bus.publish(
                new_event(
                    clock,
                    EventType.OBSERVATION_BAR,
                    source="alpaca",
                    asset="HOOD",
                    payload={"o": c, "h": c + 1, "low": c - 1, "c": c, "v": 1000.0},
                )
            )
        belief = await engine.store.get("HOOD")
        assert belief is not None and belief.direction == Direction.LONG_BIAS  # bullish belief...
        assert proposed == []  # ...but NO buy proposed — halal gate (INV-7)
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_coalesce_mode_forms_belief_end_to_end(halabot_engine):
    """The live coalescing worker path: belief writes drain through one worker
    task; after draining, the belief is formed and a buy is proposed."""
    clock = FakeClock(T0)
    engine = await build_engine(db_engine=halabot_engine, clock=clock, coalesce=True)
    proposed: list[Event] = []
    engine.bus.subscribe({EventType.POLICY_TRADE_PROPOSED}, lambda e: _cap(proposed, e))
    try:
        await engine.bus.publish(
            new_event(
                clock,
                EventType.COMPLIANCE_VERDICT,
                source="screen",
                asset="NVDA",
                payload={
                    "status": "halal",
                    "detail": "ok",
                    "screening_id": 1,
                    "transient_error": False,
                },
            )
        )
        for i in range(30):
            clock.advance(timedelta(minutes=1))
            c = 100.0 + i
            await engine.bus.publish(
                new_event(
                    clock,
                    EventType.OBSERVATION_BAR,
                    source="alpaca",
                    asset="NVDA",
                    payload={"o": c, "h": c + 1, "low": c - 1, "c": c, "v": 1000.0},
                )
            )
        assert engine.worker is not None
        await engine.worker.drain()  # flush queued belief writes

        belief = await engine.store.get("NVDA")
        assert belief is not None and belief.direction == Direction.LONG_BIAS
        assert belief.conviction > 0.0
        assert engine.shadow.proposals_count >= 1
        assert proposed and proposed[0].payload["side"] == "buy"
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_resumes_the_shadow_book_after_a_restart(halabot_engine):
    async with halabot_engine.begin() as conn:
        await conn.execute(
            sa.insert(open_position).values(
                asset="NVDA",
                entry_ts=T0,
                entry_vwap=100.0,
                weight=0.08,
                last_price=101.0,
                unrealized_return_pct=0.01,
                belief_version=1,
                entry_belief=None,
                updated_at=T0,
                cohort=OUTCOME_COHORT,
            )
        )
    engine = await build_engine(db_engine=halabot_engine, clock=FakeClock(T0))
    try:
        assert engine.shadow._portfolio.weight("NVDA") == pytest.approx(0.08)
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_refits_the_calibrator_at_start_up(halabot_engine):
    # The refit counter is per process: without a start-up refit, outcomes
    # collected before a restart waited for N new closes in one run.
    async with halabot_engine.begin() as conn:
        for i in range(60):
            raw, label = (0.8, 1) if i % 2 == 0 else (0.2, 0)
            await conn.execute(
                sa.insert(outcome).values(
                    asset="NVDA",
                    entry_ts=T0,
                    exit_ts=T0 + timedelta(minutes=i),
                    entry_price=100.0,
                    exit_price=101.0,
                    closed_weight=0.1,
                    return_pct=0.01,
                    hold_seconds=60,
                    belief_version=1,
                    entry_belief={"conviction_raw": raw},
                    label=label,
                    reason="test",
                    created_at=T0,
                    cohort=OUTCOME_COHORT,
                )
            )
    engine = await build_engine(db_engine=halabot_engine, clock=FakeClock(T0))
    try:
        assert engine.retrainer is not None and engine.retrainer.refits == 1
        assert engine.retrainer.calibrator.fitted is True
    finally:
        await engine.stop()


async def _cap(sink: list, e: Event) -> None:
    sink.append(e)
