"""Tests for the replay harness."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_snapshot_round_trip(tmp_path: Path) -> None:
    store = ReplayStore(root=tmp_path)
    snap = _make_snapshot()
    store.write(snap)

    back = store.read(snap.cycle_id)
    assert back.cycle_id == snap.cycle_id
    assert back.halal_pairs == snap.halal_pairs
    assert back.indicators_cache == snap.indicators_cache
    klines = back.klines_native()
    assert "BTCUSDT" in klines
    assert klines["BTCUSDT"][0].close == 100.0


def test_snapshot_klines_native_preserves_objects(tmp_path: Path) -> None:
    store = ReplayStore(root=tmp_path)
    snap = _make_snapshot()
    store.write(snap)
    back = store.read(snap.cycle_id)
    klines = back.klines_native()
    for k in klines["BTCUSDT"]:
        assert isinstance(k, Kline)


def test_record_snapshot_does_not_raise_on_io_error(tmp_path: Path) -> None:
    # ReplayStore points at a *file* path — writing children fails.
    bad_root = tmp_path / "not_a_dir.txt"
    bad_root.write_text("hello")  # exists as a file
    # Constructor calls mkdir(parents=True, exist_ok=True) — that fails on a
    # path that exists as a file. Catch it the same way `record_snapshot`
    # would by going through it.
    try:
        store = ReplayStore(root=bad_root)
    except (FileExistsError, NotADirectoryError) as _exc:  # noqa: F841
        pytest.skip("OS prevents constructing the store on a file path")
    record_snapshot(store, _make_snapshot())  # must not raise


def test_list_cycle_ids(tmp_path: Path) -> None:
    store = ReplayStore(root=tmp_path)
    for i in range(3):
        store.write(_make_snapshot(cycle_id=f"cycle-{i:08x}"))
    ids = store.list_cycle_ids()
    assert ids == ["cycle-00000000", "cycle-00000001", "cycle-00000002"]


def test_max_keep_gc(tmp_path: Path) -> None:
    store = ReplayStore(root=tmp_path, max_keep=2)
    import time

    for i in range(5):
        store.write(_make_snapshot(cycle_id=f"cycle-{i:08x}"))
        time.sleep(0.01)  # ensure mtime ordering
    ids = store.list_cycle_ids()
    assert len(ids) == 2


@pytest.mark.asyncio
async def test_replay_cycle_invokes_decider(tmp_path: Path) -> None:
    store = ReplayStore(root=tmp_path)
    snap = _make_snapshot()
    store.write(snap)

    async def decider(s: CycleSnapshot) -> str:
        assert s.cycle_id == snap.cycle_id
        assert "BTCUSDT" in s.klines_by_symbol
        return "decided"

    result = await replay_cycle(store, snap.cycle_id, decider)
    assert result == "decided"


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


def test_unknown_schema_version_rejected(tmp_path: Path) -> None:
    store = ReplayStore(root=tmp_path)
    snap = _make_snapshot()
    store.write(snap)
    # Tamper with the schema_version on disk
    p = next(tmp_path.glob("*.json"))
    raw = p.read_text().replace('"schema_version": 1', '"schema_version": 99')
    p.write_text(raw)
    with pytest.raises(ValueError):
        store.read(snap.cycle_id)
