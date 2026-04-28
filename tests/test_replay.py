"""Tests for the replay harness."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.replay import (
    CycleSnapshot,
    ReplayStore,
    diff_snapshots,
    record_snapshot,
    replay_cycle,
)
from halal_trader.domain.models import Kline


def _kline(ts: int = 0, close: float = 100.0) -> Kline:
    return Kline(
        open_time=ts,
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=10.0,
        close_time=ts + 59_999,
    )


def _make_snapshot(cycle_id: str = "cycle-aaa", market: str = "crypto") -> CycleSnapshot:
    return CycleSnapshot.from_inputs(
        cycle_id=cycle_id,
        market=market,
        klines_by_symbol={"BTCUSDT": [_kline(0), _kline(60_000, 100.5)]},
        indicators_cache={"BTCUSDT": {"rsi_14": 55.0}},
        halal_pairs=["BTCUSDT", "ETHUSDT"],
        positions_text="No open positions.",
        sentiment_text="bullish",
        regime_text="trending_up",
        microstructure_text="bid imbalance",
        today_pnl=12.34,
    )


async def test_snapshot_round_trip(engine: AsyncEngine) -> None:
    store = ReplayStore(engine=engine)
    snap = _make_snapshot()
    await store.write(snap)

    back = await store.read(snap.cycle_id)
    assert back.cycle_id == snap.cycle_id
    assert back.halal_pairs == snap.halal_pairs
    assert back.indicators_cache == snap.indicators_cache
    klines = back.klines_native()
    assert "BTCUSDT" in klines
    assert klines["BTCUSDT"][0].close == 100.0


async def test_snapshot_klines_native_preserves_objects(engine: AsyncEngine) -> None:
    store = ReplayStore(engine=engine)
    snap = _make_snapshot()
    await store.write(snap)
    back = await store.read(snap.cycle_id)
    klines = back.klines_native()
    for k in klines["BTCUSDT"]:
        assert isinstance(k, Kline)


async def test_record_snapshot_does_not_raise_on_failure(engine: AsyncEngine) -> None:
    """``record_snapshot`` swallows DB failures so the cycle keeps running."""
    await engine.dispose()  # subsequent writes will fail
    store = ReplayStore(engine=engine)
    await record_snapshot(store, _make_snapshot())  # must not raise


async def test_list_cycle_ids(engine: AsyncEngine) -> None:
    store = ReplayStore(engine=engine)
    for i in range(3):
        await store.write(_make_snapshot(cycle_id=f"cycle-{i:08x}"))
    ids = await store.list_cycle_ids()
    assert set(ids) == {"cycle-00000000", "cycle-00000001", "cycle-00000002"}


async def test_list_cycle_ids_orders_recent_first(engine: AsyncEngine) -> None:
    store = ReplayStore(engine=engine)
    import asyncio

    for i in range(3):
        await store.write(_make_snapshot(cycle_id=f"cycle-{i:08x}"))
        # ensure created_at ordering — sub-millisecond differences are
        # enough but a tiny sleep keeps the assertion deterministic.
        await asyncio.sleep(0.01)
    ids = await store.list_cycle_ids(limit=2)
    assert ids[0] == "cycle-00000002"
    assert ids[1] == "cycle-00000001"


async def test_replay_cycle_invokes_decider(engine: AsyncEngine) -> None:
    store = ReplayStore(engine=engine)
    snap = _make_snapshot()
    await store.write(snap)

    async def decider(s: CycleSnapshot) -> str:
        assert s.cycle_id == snap.cycle_id
        assert "BTCUSDT" in s.klines_by_symbol
        return "decided"

    result = await replay_cycle(store, snap.cycle_id, decider)
    assert result == "decided"


async def test_replay_unknown_cycle_raises(engine: AsyncEngine) -> None:
    store = ReplayStore(engine=engine)
    with pytest.raises(KeyError):
        await store.read("nonexistent-cycle")


async def test_unknown_schema_version_rejected(engine: AsyncEngine) -> None:
    store = ReplayStore(engine=engine)
    snap = _make_snapshot()
    await store.write(snap)

    # Tamper directly with the row's schema_version to simulate a forward-rev.
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from halal_trader.db.models import ReplaySnapshotRow

    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        row = await s.get(ReplaySnapshotRow, snap.cycle_id)
        assert row is not None
        row.schema_version = 99
        s.add(row)
        await s.commit()

    with pytest.raises(ValueError):
        await store.read(snap.cycle_id)


def test_diff_snapshots_detects_changes() -> None:
    a = _make_snapshot()
    b = _make_snapshot()
    b.sentiment_text = "different"
    b.klines_by_symbol["BTCUSDT"].append(_kline(120_000, 101.0).model_dump())
    d = diff_snapshots(a, b)
    assert "sentiment_text" in d
    assert "klines_by_symbol" in d
    assert "BTCUSDT" in d["klines_by_symbol"]


def test_diff_snapshots_no_diff_for_identical() -> None:
    a = _make_snapshot()
    b = _make_snapshot()
    # cycle_started_at differs each call — sync them
    b.cycle_started_at = a.cycle_started_at
    assert diff_snapshots(a, b) == {}
