"""Backtest bar-cache round-trip + out-of-sample window partitioning (CLI helpers)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from halabot.cli import _load_bars_cache, _oos_windows, _save_bars_cache
from halabot.cognition.bars import Bar

T0 = datetime(2026, 1, 1, 14, 0, tzinfo=UTC)


def _bars(n: int, *, start: datetime = T0, step_min: int = 60) -> list[Bar]:
    return [
        Bar(o=100.0 + i, h=101.0 + i, low=99.0 + i, c=100.5 + i, v=1000.0 + i,
            ts=start + timedelta(minutes=i * step_min))
        for i in range(n)
    ]


def test_cache_round_trip_preserves_bars(tmp_path):
    path = str(tmp_path / "bars.json")
    original = {"NVDA": _bars(5), "SPY": _bars(3)}
    _save_bars_cache(path, original)
    loaded = _load_bars_cache(path)
    assert set(loaded) == {"NVDA", "SPY"}
    assert len(loaded["NVDA"]) == 5 and len(loaded["SPY"]) == 3
    a, b = original["NVDA"][2], loaded["NVDA"][2]
    assert (a.o, a.h, a.low, a.c, a.v) == (b.o, b.h, b.low, b.c, b.v)
    assert a.ts == b.ts  # timestamp survives the isoformat round-trip


def test_oos_windows_single_when_n_below_two():
    bbs = {"NVDA": _bars(10)}
    out = list(_oos_windows(bbs, 1))
    assert len(out) == 1 and out[0][0] == ""  # no label, whole set


def test_oos_windows_partitions_disjoint_and_complete():
    # 30 hourly bars split into 3 windows → disjoint, and every bar lands in
    # exactly one window (no loss, no overlap).
    bbs = {"NVDA": _bars(30)}
    windows = list(_oos_windows(bbs, 3))
    assert len(windows) == 3
    counts = [len(w[1]["NVDA"]) for w in windows]
    assert sum(counts) == 30  # complete partition
    # Disjoint: max ts of an earlier window < min ts of the next.
    spans = [[b.ts for b in w[1]["NVDA"]] for w in windows]
    assert max(spans[0]) < min(spans[1])
    assert max(spans[1]) < min(spans[2])


def test_oos_windows_drops_empty_symbols():
    # A symbol whose bars all fall outside a window is omitted from that window.
    early = {"A": _bars(5, start=T0)}
    late = {"B": _bars(5, start=T0 + timedelta(days=10))}
    bbs = {**early, **late}
    windows = list(_oos_windows(bbs, 2))
    # The first window should contain A but not B (B's bars are all later).
    assert "A" in windows[0][1] and "B" not in windows[0][1]
