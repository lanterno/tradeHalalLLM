"""PgBeliefStore — versioned persistence + full round-trip (requires PG on :5433)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from halabot.belief.schema import (
    BeliefState,
    Catalyst,
    ComplianceVerdict,
    Direction,
    EvidenceItem,
    Horizon,
    Levels,
    Regime,
)
from halabot.belief.store import PgBeliefStore

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_get_missing_returns_none(halabot_engine):
    store = PgBeliefStore(halabot_engine)
    assert await store.get("NVDA") is None


@pytest.mark.asyncio
async def test_put_increments_version_and_get_returns_latest(halabot_engine):
    store = PgBeliefStore(halabot_engine)
    b = BeliefState.neutral("NVDA")
    assert await store.put(b) == 1
    b.regime = Regime.TRENDING_UP
    assert await store.put(b) == 2
    latest = await store.get("NVDA")
    assert latest is not None
    assert latest.version == 2
    assert latest.regime == Regime.TRENDING_UP


@pytest.mark.asyncio
async def test_old_version_retained(halabot_engine):
    store = PgBeliefStore(halabot_engine)
    b = BeliefState.neutral("NVDA")
    b.regime = Regime.RANGING
    await store.put(b)
    b.regime = Regime.VOLATILE
    await store.put(b)
    v1 = await store.get_version("NVDA", 1)
    assert v1 is not None and v1.regime == Regime.RANGING


@pytest.mark.asyncio
async def test_full_belief_round_trips_through_jsonb(halabot_engine):
    """A belief with evidence, levels, catalysts, and a verdict survives a
    persist → load round-trip intact."""
    store = PgBeliefStore(halabot_engine)
    b = BeliefState(
        asset="NVDA",
        regime=Regime.TRENDING_UP,
        regime_confidence=0.85,
        direction=Direction.LONG_BIAS,
        conviction=0.7,
        conviction_raw=0.66,
        horizon=Horizon.SWING,
        thesis="earnings beat, momentum intact",
        levels=Levels(support=90.0, resistance=120.0, stop=88.0, invalidation=88.0),
        catalysts_pending=[
            Catalyst(kind="earnings", scheduled_for=T0, expected_impact=0.9, detail="Q1")
        ],
        evidence=[
            EvidenceItem(source="news", direction=1.0, weight=0.9, ts=T0, event_id=uuid4()),
            EvidenceItem(source="anomaly", direction=0.0, weight=1.0, ts=T0, directional=False),
        ],
        halal=ComplianceVerdict(asset="NVDA", status="halal", screening_id=42, screened_at=T0),
        opened_trade_ids=[101, 102],
        last_updated=T0,
        last_thesis_refresh=T0,
    )
    await store.put(b)
    got = await store.get("NVDA")
    assert got is not None
    assert got.regime == Regime.TRENDING_UP
    assert got.direction == Direction.LONG_BIAS
    assert got.thesis == "earnings beat, momentum intact"
    assert got.levels.invalidation == 88.0
    assert len(got.catalysts_pending) == 1
    assert got.catalysts_pending[0].kind == "earnings"
    assert len(got.evidence) == 2
    assert {e.source for e in got.evidence} == {"news", "anomaly"}
    assert got.halal is not None and got.halal.status == "halal" and got.halal.screening_id == 42
    assert got.opened_trade_ids == [101, 102]


@pytest.mark.asyncio
async def test_all_active_returns_latest_per_asset(halabot_engine):
    store = PgBeliefStore(halabot_engine)
    await store.put(BeliefState.neutral("NVDA"))
    nvda = BeliefState.neutral("NVDA")
    nvda.regime = Regime.BREAKOUT
    await store.put(nvda)  # NVDA v2
    await store.put(BeliefState.neutral("MSFT"))  # MSFT v1
    actives = await store.all_active()
    by_asset = {b.asset: b for b in actives}
    assert set(by_asset) == {"NVDA", "MSFT"}
    assert by_asset["NVDA"].version == 2
    assert by_asset["NVDA"].regime == Regime.BREAKOUT
