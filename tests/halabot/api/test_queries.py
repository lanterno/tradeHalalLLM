"""API query layer — beliefs, decision-chain replay."""

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
        asset=asset,
        regime=Regime.TRENDING_UP,
        direction=Direction.LONG_BIAS,
        conviction=conviction,
        conviction_raw=conviction,
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
        clock,
        EventType.OBSERVATION_BAR,
        source="alpaca",
        asset="NVDA",
        payload={"o": 1, "h": 1, "low": 1, "c": 1},
    )
    await bus.publish(obs)
    belief = new_event(
        clock,
        EventType.BELIEF_UPDATED,
        source="belief.updater",
        asset="NVDA",
        payload={"version": 1},
        correlation_id=obs.correlation_id,
    )
    await bus.publish(belief)
    policy = new_event(
        clock,
        EventType.POLICY_TRADE_PROPOSED,
        source="policy.shadow",
        asset="NVDA",
        payload={
            "side": "buy",
            "target_weight": 0.1,
            "current_weight": 0.0,
            "weight_delta": 0.1,
            "shadow": True,
        },
        causation=belief,
    )
    await bus.publish(policy)

    chain = await queries.decision_chain(halabot_engine, obs.correlation_id)
    assert [e["type"] for e in chain] == [
        "observation.bar",
        "belief.updated",
        "policy.trade_proposed",
    ]
    # The proposal is a separate query feed too.
    recent = await queries.recent_decisions(halabot_engine)
    assert recent and recent[0]["type"] == "policy.trade_proposed"


@pytest.mark.asyncio
async def test_system_health_counts(halabot_engine):
    bus = InProcessEventBus(PgEventLog(halabot_engine))
    await bus.publish(new_event(FakeClock(T0), EventType.SYSTEM_HEARTBEAT, source="hb"))
    await _seed_belief(halabot_engine, "NVDA", 0.6)
    await _seed_belief(halabot_engine, "NVDA", 0.7)  # a second version, same asset
    await _seed_belief(halabot_engine, "AAPL", 0.5)
    health = await queries.system_health(halabot_engine)
    assert health["events"] >= 1
    assert health["events_estimated"] is False  # a small table is counted exactly
    assert health["halted"] is False
    assert health["last_event_ts"] == T0.isoformat()
    # active_beliefs counts current beliefs (one per asset, not per version).
    assert health["active_beliefs"] == 2


@pytest.mark.asyncio
async def test_list_beliefs_reads_the_latest_version_of_each_asset(halabot_engine):
    await _seed_belief(halabot_engine, "NVDA", 0.6)
    await _seed_belief(halabot_engine, "NVDA", 0.2)  # the newer version wins
    await _seed_belief(halabot_engine, "AAPL", 0.5)
    beliefs = await queries.list_beliefs(halabot_engine)
    assert [(b["asset"], b["conviction"]) for b in beliefs] == [("AAPL", 0.5), ("NVDA", 0.2)]
    assert await queries.list_beliefs(halabot_engine) == beliefs  # deterministic


async def _bar(engine, asset: str, at: datetime, close: float, *, seen: datetime | None = None):
    bar = new_event(
        FakeClock(seen or at),
        EventType.OBSERVATION_BAR,
        source="alpaca-bars",
        asset=asset,
        payload={"o": close, "h": close, "low": close, "c": close, "bar_ts": at.isoformat()},
    )
    await PgEventLog(engine).append(bar)


@pytest.mark.asyncio
async def test_latest_prices_and_bar_closes(halabot_engine):
    from datetime import timedelta

    h = timedelta(hours=1)
    await _bar(halabot_engine, "NVDA", T0, 100.0)
    await _bar(halabot_engine, "NVDA", T0 + h, 101.0)
    # The same bar seen again later, finalised: its last print is the close.
    await _bar(halabot_engine, "NVDA", T0 + h, 102.0, seen=T0 + 2 * h)
    await _bar(halabot_engine, "AAPL", T0 - timedelta(days=9), 50.0)

    prices = await queries.latest_prices(halabot_engine, ["NVDA", "AAPL", "TSLA"], now=T0 + 3 * h)
    assert prices == {"NVDA": (102.0, T0 + 2 * h)}  # AAPL's bar is outside the window
    assert await queries.last_bar_at(halabot_engine) == T0 + 2 * h

    closes = await queries.bar_closes(halabot_engine, ["NVDA"], since=T0 - h)
    assert closes == {"NVDA": [(T0, 100.0), (T0 + h, 102.0)]}
    assert await queries.latest_prices(halabot_engine, [], now=T0) == {}


@pytest.mark.asyncio
async def test_outcome_stats_and_open_positions_on_an_empty_engine(halabot_engine):
    stats = await queries.outcome_stats(halabot_engine, cohort=2)
    assert stats["current"]["closed"] == 0 and stats["earlier"]["closed"] == 0
    assert stats["started"] is None and stats["median_hold_s"] is None
    assert await queries.open_positions(halabot_engine, cohort=2) == []
    assert await queries.calibration_probe(halabot_engine, ["NVDA"], since=T0) == (0, 0)
