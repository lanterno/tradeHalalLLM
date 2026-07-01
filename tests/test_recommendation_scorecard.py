"""Tests for the recommendation forward-return labeling + scorecard."""

from __future__ import annotations

from typing import Any

import pytest

from halal_trader.recommendation.scorecard import (
    _closes_by_date,
    _forward_returns,
    backfill_outcomes,
    compute_scorecard,
    whatif_equity_curve,
)


def _bars(start_close: float, daily_step: float, dates: list[str]) -> list[dict[str, Any]]:
    return [
        {"t": f"{d}T05:00:00Z", "c": start_close + daily_step * i}
        for i, d in enumerate(dates)
    ]


DATES = [f"2026-05-{day:02d}" for day in range(1, 26)]  # 25 consecutive (test) days


def test_closes_by_date_handles_envelopes():
    flat = _bars(100.0, 1.0, ["2026-05-02", "2026-05-01"])  # out of order
    out = _closes_by_date(flat)
    assert [d for d, _ in out] == ["2026-05-01", "2026-05-02"]  # sorted ascending
    # symbol-keyed envelope
    wrapped = {"bars": {"NVDA": _bars(50.0, 0.0, ["2026-05-01"])}}
    assert _closes_by_date(wrapped) == [("2026-05-01", 50.0)]
    # long-key 'close' variant
    assert _closes_by_date([{"timestamp": "2026-05-01", "close": 7.0}]) == [("2026-05-01", 7.0)]


def test_forward_returns_from_entry_date():
    cbd = _closes_by_date(_bars(100.0, 10.0, DATES))  # 100,110,120,... +10/day
    fr = _forward_returns(cbd, "2026-05-01")  # entry idx 0, entry_close 100
    assert fr["entry_close"] == 100.0
    assert fr[1] == pytest.approx(10.0)  # 110/100-1 = 10%
    assert fr[5] == pytest.approx(50.0)  # 150/100-1 = 50%
    assert fr[20] == pytest.approx(200.0)  # 300/100-1 = 200%


def test_forward_returns_entry_on_or_after_rec_date():
    cbd = _closes_by_date(_bars(100.0, 10.0, DATES))
    # rec dated on a weekend/gap before the first bar present after it
    fr = _forward_returns(cbd, "2026-05-03")  # first bar >= is idx 2 (close 120)
    assert fr["entry_close"] == 120.0
    assert fr[1] == pytest.approx((130.0 / 120.0 - 1) * 100, abs=1e-3)  # rounded 4dp


def test_forward_returns_partial_maturity():
    # only 6 bars → 1d & 5d available, 20d not yet
    cbd = _closes_by_date(_bars(100.0, 10.0, DATES[:6]))
    fr = _forward_returns(cbd, "2026-05-01")
    assert fr[1] is not None
    assert fr[5] is not None
    assert fr[20] is None


def test_forward_returns_none_when_no_bar_after_rec():
    cbd = _closes_by_date(_bars(100.0, 10.0, DATES[:3]))
    assert _forward_returns(cbd, "2026-12-01") is None


class _FakeBroker:
    def __init__(self, by_symbol: dict[str, list]):
        self._by = by_symbol

    async def get_stock_bars(self, symbol: str, days: int = 90, timeframe: str = "1Day"):
        return self._by.get(symbol, [])


class _FakeRepo:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = {r["id"]: r for r in rows}

    async def get_recommendations_to_score(self, limit: int = 500):
        return [r for r in self.rows.values() if r.get("outcome_status") != "scored"]

    async def get_recent_recommendations(self, limit: int = 500):
        return list(self.rows.values())

    async def update_recommendation_outcome(self, rec_id: int, **fields):
        self.rows[rec_id].update(fields)
        return True


@pytest.mark.asyncio
async def test_backfill_marks_scored_when_20d_available():
    broker = _FakeBroker({
        "NVDA": _bars(100.0, 10.0, DATES),     # full 25 days → 20d matures
        "SPUS": _bars(50.0, 1.0, DATES),       # benchmark
    })
    repo = _FakeRepo([
        {"id": 1, "symbol": "NVDA", "date": "2026-05-01", "outcome_status": "pending"},
    ])
    res = await backfill_outcomes(broker, repo)
    assert res == {"updated": 1, "scored": 1}
    row = repo.rows[1]
    assert row["outcome_status"] == "scored"
    assert row["entry_close"] == 100.0
    assert row["fwd_return_5d"] == pytest.approx(50.0)
    assert row["fwd_return_20d"] == pytest.approx(200.0)
    assert row["benchmark_return_5d"] is not None  # SPUS 5d filled


@pytest.mark.asyncio
async def test_backfill_progressive_stays_pending_until_20d():
    broker = _FakeBroker({"NVDA": _bars(100.0, 10.0, DATES[:6])})  # only 6 days
    repo = _FakeRepo(
        [{"id": 1, "symbol": "NVDA", "date": "2026-05-01", "outcome_status": "pending"}]
    )
    res = await backfill_outcomes(broker, repo)
    assert res == {"updated": 1, "scored": 0}
    row = repo.rows[1]
    assert row["outcome_status"] == "pending"  # not fully scored yet
    assert row["fwd_return_5d"] is not None     # but 5d already labeled
    assert row["fwd_return_20d"] is None


@pytest.mark.asyncio
async def test_compute_scorecard_aggregates():
    repo = _FakeRepo([
        {"id": 1, "symbol": "NVDA", "date": "2026-05-01", "fwd_return_1d": 1.0,
         "fwd_return_5d": 5.0, "fwd_return_20d": 20.0, "benchmark_return_5d": 2.0},
        {"id": 2, "symbol": "AAPL", "date": "2026-05-02", "fwd_return_1d": -1.0,
         "fwd_return_5d": -3.0, "fwd_return_20d": None, "benchmark_return_5d": 1.0},
        {"id": 3, "symbol": "MSFT", "date": "2026-05-03", "fwd_return_5d": None},  # unlabeled
    ])
    sc = await compute_scorecard(repo)
    assert sc["available"] is True
    assert sc["n_total"] == 3
    assert sc["n_scored"] == 2  # only the two with fwd_return_5d
    assert sc["sufficient"] is False  # 2 picks is far below the trust threshold
    assert sc["min_samples"] >= 2
    assert sc["conviction_ic"] is None  # not enough samples to claim an IC
    assert sc["hit_rate_5d"] == pytest.approx(0.5)  # 1 of 2 positive
    assert sc["avg_fwd_5d"] == pytest.approx(1.0)  # (5 + -3)/2
    assert sc["avg_excess_5d"] == pytest.approx(((5.0 - 2.0) + (-3.0 - 1.0)) / 2)
    assert sc["best"]["symbol"] == "NVDA"
    assert sc["worst"]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_compute_scorecard_empty():
    repo = _FakeRepo([{"id": 1, "symbol": "NVDA", "date": "2026-05-01", "fwd_return_5d": None}])
    sc = await compute_scorecard(repo)
    assert sc["available"] is False
    assert sc["n_scored"] == 0


@pytest.mark.asyncio
async def test_whatif_equity_curve_compounds_in_date_order():
    repo = _FakeRepo([
        {"id": 1, "symbol": "AAPL", "date": "2026-05-01", "fwd_return_5d": 10.0,
         "benchmark_return_5d": 2.0},
        {"id": 2, "symbol": "MSFT", "date": "2026-05-02", "fwd_return_5d": -5.0,
         "benchmark_return_5d": 1.0},
        {"id": 3, "symbol": "NVDA", "date": "2026-05-03", "fwd_return_5d": None},  # unscored
    ])
    wc = await whatif_equity_curve(repo, start=100.0)
    assert wc["available"] is True
    assert wc["n"] == 2  # only the two scored picks
    # 100 * 1.10 * 0.95 = 104.5
    assert wc["final_equity"] == pytest.approx(104.5)
    assert wc["total_return_pct"] == pytest.approx(4.5)
    # benchmark: 100 * 1.02 * 1.01 = 103.02
    assert wc["benchmark_return_pct"] == pytest.approx(3.02)
    assert [p["symbol"] for p in wc["points"]] == ["AAPL", "MSFT"]  # date order
    assert wc["points"][-1]["equity"] == pytest.approx(104.5)


@pytest.mark.asyncio
async def test_whatif_equity_curve_empty():
    repo = _FakeRepo([{"id": 1, "symbol": "NVDA", "date": "2026-05-01", "fwd_return_5d": None}])
    wc = await whatif_equity_curve(repo)
    assert wc["available"] is False
    assert wc["points"] == []
