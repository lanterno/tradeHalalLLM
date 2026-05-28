"""API query layer — beliefs, decision-chain replay, control toggle."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.api import queries
from halabot.belief.schema import BeliefState, ComplianceVerdict, Direction, Regime
from halabot.belief.store import PgBeliefStore
from halabot.platform.bus import InProcessEventBus
from halabot.platform.clock import FakeClock
from halabot.platform.event_log import PgEventLog
from halabot.platform.events import EventType, new_event

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


async def _seed_belief(engine, asset="NVDA", conviction=0.7):
    b = BeliefState(
        asset=asset, regime=Regime.TRENDING_UP, direction=Direction.LONG_BIAS,
        conviction=conviction, conviction_raw=conviction,
        halal=ComplianceVerdict(asset, "halal", screened_at=T0),
    )
    await PgBeliefStore(engine).put(b)


@pytest.mark.asyncio
async def test_list_and_get_beliefs(halabot_engine):
    await _seed_belief(halabot_engine, "NVDA", 0.7)
    await _seed_belief(halabot_engine, "AAPL", 0.9)
    beliefs = await queries.list_beliefs(halabot_engine)
    assert [b["asset"] for b in beliefs] == ["AAPL", "NVDA"]  # conviction-ranked
    one = await queries.get_belief(halabot_engine, "NVDA")
    assert one is not None and one["direction"] == "long_bias" and one["halal"] == "halal"
    assert await queries.get_belief(halabot_engine, "TSLA") is None


@pytest.mark.asyncio
async def test_decision_chain_replays_by_correlation_id(halabot_engine):
    bus = InProcessEventBus(PgEventLog(halabot_engine))
    clock = FakeClock(T0)
    # An observation starts a chain; downstream events inherit its correlation_id.
    obs = new_event(
        clock, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA",
        payload={"o": 1, "h": 1, "low": 1, "c": 1},
    )
    await bus.publish(obs)
    belief = new_event(
        clock, EventType.BELIEF_UPDATED, source="belief.updater", asset="NVDA",
        payload={"version": 1}, correlation_id=obs.correlation_id,
    )
    await bus.publish(belief)
    policy = new_event(
        clock, EventType.POLICY_TRADE_PROPOSED, source="policy.shadow", asset="NVDA",
        payload={"side": "buy", "target_weight": 0.1, "current_weight": 0.0,
                 "weight_delta": 0.1, "shadow": True},
        causation=belief,
    )
    await bus.publish(policy)

    chain = await queries.decision_chain(halabot_engine, obs.correlation_id)
    assert [e["type"] for e in chain] == [
        "observation.bar", "belief.updated", "policy.trade_proposed"
    ]
    # The proposal is a separate query feed too.
    recent = await queries.recent_decisions(halabot_engine)
    assert recent and recent[0]["type"] == "policy.trade_proposed"


@pytest.mark.asyncio
async def test_control_toggle_roundtrip(halabot_engine):
    assert (await queries.get_halt(halabot_engine))["halted"] is False
    await queries.set_halt(halabot_engine, halted=True, reason="manual stop")
    h = await queries.get_halt(halabot_engine)
    assert h["halted"] is True and h["reason"] == "manual stop"
    await queries.set_halt(halabot_engine, halted=False, reason=None)
    assert (await queries.get_halt(halabot_engine))["halted"] is False


@pytest.mark.asyncio
async def test_system_health_counts(halabot_engine):
    bus = InProcessEventBus(PgEventLog(halabot_engine))
    await bus.publish(
        new_event(FakeClock(T0), EventType.SYSTEM_HEARTBEAT, source="hb")
    )
    health = await queries.system_health(halabot_engine)
    assert health["events"] >= 1
    assert health["halted"] is False
