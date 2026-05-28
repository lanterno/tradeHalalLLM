"""Decision telemetry writers — conviction.scored + policy.target_changed rows."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import sqlalchemy as sa

from halabot.learning.telemetry import ConvictionScoreWriter, TargetWeightWriter
from halabot.platform.bus import InProcessEventBus
from halabot.platform.clock import FakeClock
from halabot.platform.db import conviction_score, target_weight
from halabot.platform.event_log import PgEventLog
from halabot.platform.events import EventType, new_event

CLOCK = FakeClock(datetime(2026, 5, 28, 12, 0, tzinfo=UTC))


@pytest.mark.asyncio
async def test_conviction_score_row_written(halabot_engine):
    bus = InProcessEventBus(PgEventLog(halabot_engine))
    writer = ConvictionScoreWriter(bus=bus, engine=halabot_engine)
    writer.start()
    try:
        await bus.publish(
            new_event(
                CLOCK, EventType.CONVICTION_SCORED, source="belief.updater", asset="NVDA",
                payload={"raw": 0.42, "calibrated": 0.55, "belief_version": 3,
                         "features": {"regime": "trending_up", "n_evidence": 4}},
            )
        )
    finally:
        writer.stop()
    async with halabot_engine.connect() as conn:
        rows = (await conn.execute(sa.select(conviction_score).where(
            conviction_score.c.asset == "NVDA"))).mappings().all()
    assert len(rows) == 1
    assert rows[0]["raw_score"] == pytest.approx(0.42)
    assert rows[0]["calibrated"] == pytest.approx(0.55)
    assert rows[0]["belief_version"] == 3
    assert rows[0]["features"]["regime"] == "trending_up"


@pytest.mark.asyncio
async def test_target_weight_row_written(halabot_engine):
    bus = InProcessEventBus(PgEventLog(halabot_engine))
    writer = TargetWeightWriter(bus=bus, engine=halabot_engine)
    writer.start()
    try:
        await bus.publish(
            new_event(
                CLOCK, EventType.POLICY_TARGET_CHANGED, source="policy.shadow", asset="AAPL",
                payload={"target_weight": 0.15, "current_weight": 0.0,
                         "reason": "conviction", "belief_version": 7},
            )
        )
    finally:
        writer.stop()
    async with halabot_engine.connect() as conn:
        rows = (await conn.execute(sa.select(target_weight).where(
            target_weight.c.asset == "AAPL"))).mappings().all()
    assert len(rows) == 1
    assert rows[0]["target_weight"] == pytest.approx(0.15)
    assert rows[0]["reason"] == "conviction"
    assert rows[0]["belief_version"] == 7
