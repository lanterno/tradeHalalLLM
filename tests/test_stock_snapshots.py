"""Tests for trading/snapshots.py — stock-side IndicatorSnapshot write-on-buy."""

from __future__ import annotations

from typing import Any

import pytest

from halal_trader.trading.snapshots import _FEATURE_KEYS, record_stock_snapshot


class _FakeRepo:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def record_indicator_snapshot(
        self, *, trade_id: int, pair: str, indicators: dict[str, float]
    ) -> int:
        self.calls.append({"trade_id": trade_id, "pair": pair, "indicators": indicators})
        if self.fail:
            raise RuntimeError("DB locked")
        return 99


def _bar(o: float, h: float, low: float, c: float) -> dict:
    return {"o": o, "h": h, "l": low, "c": c, "v": 1_000.0}


def _series(start: float, n: int, step: float) -> list[dict]:
    return [
        _bar(start + i * step, start + i * step + 0.5, start + i * step - 0.5, start + i * step)
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_record_stock_snapshot_skips_when_too_few_bars():
    repo = _FakeRepo()
    result = await record_stock_snapshot(
        repo=repo, trade_id=1, symbol="AAPL", bars=_series(180, 5, 0.5)
    )
    assert result is None
    assert repo.calls == []


@pytest.mark.asyncio
async def test_record_stock_snapshot_writes_with_enough_bars():
    repo = _FakeRepo()
    bars = _series(180.0, 60, 0.3)
    result = await record_stock_snapshot(
        repo=repo, trade_id=42, symbol="AAPL", bars=bars
    )
    assert result == 99
    assert len(repo.calls) == 1
    call = repo.calls[0]
    assert call["trade_id"] == 42
    assert call["pair"] == "AAPL"
    # At least one of the 9 indicator features should be populated.
    assert any(k in call["indicators"] for k in _FEATURE_KEYS)


@pytest.mark.asyncio
async def test_record_stock_snapshot_handles_alpaca_dict_form():
    repo = _FakeRepo()
    bars = {"bars": _series(50.0, 60, 0.5)}
    result = await record_stock_snapshot(
        repo=repo, trade_id=7, symbol="MSFT", bars=bars
    )
    assert result == 99


@pytest.mark.asyncio
async def test_record_stock_snapshot_swallows_repo_failure():
    repo = _FakeRepo(fail=True)
    bars = _series(100.0, 60, 0.5)
    # Must not raise — snapshotting is best-effort.
    result = await record_stock_snapshot(
        repo=repo, trade_id=1, symbol="AAPL", bars=bars
    )
    assert result is None


@pytest.mark.asyncio
async def test_record_stock_snapshot_handles_empty_bars():
    repo = _FakeRepo()
    result = await record_stock_snapshot(
        repo=repo, trade_id=1, symbol="AAPL", bars={}
    )
    assert result is None


@pytest.mark.asyncio
async def test_record_stock_snapshot_skips_on_indicator_error():
    """`compute_all` returns {"error": ...} for degenerate series."""
    repo = _FakeRepo()
    # All-zero closes are filtered by _bars_to_klines (zero-close skip),
    # leaving < 30 klines → indicator pass returns no row.
    bars = [{"o": 1, "h": 1, "l": 1, "c": 0} for _ in range(50)]
    result = await record_stock_snapshot(
        repo=repo, trade_id=1, symbol="AAPL", bars=bars
    )
    assert result is None
