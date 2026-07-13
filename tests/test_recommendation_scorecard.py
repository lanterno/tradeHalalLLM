"""Tests for the recommendation outcome labeling (returns + path) + scorecard."""

from __future__ import annotations

from typing import Any

import pytest

from halal_trader.recommendation.scorecard import (
    _forward_outcomes,
    _ohlc_by_date,
    audit_scored_outcomes,
    backfill_outcomes,
    compute_scorecard,
    whatif_equity_curve,
)


def _bars(
    start_close: float,
    daily_step: float,
    dates: list[str],
    *,
    spread: float = 0.0,
) -> list[dict[str, Any]]:
    """Synthetic daily bars: close walks by ``daily_step``; high/low = close ± spread."""
    out = []
    for i, d in enumerate(dates):
        c = start_close + daily_step * i
        out.append({"t": f"{d}T05:00:00Z", "c": c, "h": c + spread, "l": max(c - spread, 0.01)})
    return out


DATES = [f"2026-05-{day:02d}" for day in range(1, 26)]  # 25 consecutive (test) days


def test_ohlc_by_date_handles_envelopes():
    flat = _bars(100.0, 1.0, ["2026-05-02", "2026-05-01"])  # out of order
    out = _ohlc_by_date(flat)
    assert [d for d, *_ in out] == ["2026-05-01", "2026-05-02"]  # sorted ascending
    # symbol-keyed envelope
    wrapped = {"bars": {"NVDA": _bars(50.0, 0.0, ["2026-05-01"])}}
    assert _ohlc_by_date(wrapped) == [("2026-05-01", 50.0, 50.0, 50.0, 50.0)]
    # long-key 'close' variant; missing h/l falls back to close
    assert _ohlc_by_date([{"timestamp": "2026-05-01", "close": 7.0}]) == [
        ("2026-05-01", 7.0, 7.0, 7.0, 7.0)
    ]


def test_ohlc_by_date_sanitizes_bad_prints():
    # high < low is a bad print — sanitized so low <= close <= high
    rows = _ohlc_by_date([{"t": "2026-05-01", "c": 10.0, "h": 8.0, "l": 12.0}])
    _, open_, high, low, close = rows[0]
    assert low <= open_ <= high
    assert low <= close <= high


def test_forward_outcomes_from_entry_date():
    ohlc = _ohlc_by_date(_bars(100.0, 10.0, DATES))  # 100,110,120,... +10/day
    fr = _forward_outcomes(ohlc, "2026-05-01")  # entry idx 0, entry_close 100
    assert fr["entry_close"] == 100.0
    assert fr[1] == pytest.approx(10.0)  # 110/100-1 = 10%
    assert fr[5] == pytest.approx(50.0)  # 150/100-1 = 50%
    assert fr[20] == pytest.approx(200.0)  # 300/100-1 = 200%


def test_forward_outcomes_entry_on_or_after_rec_date():
    ohlc = _ohlc_by_date(_bars(100.0, 10.0, DATES))
    # rec dated on a weekend/gap before the first bar present after it —
    # legitimate because the window still starts on/before the rec date
    fr = _forward_outcomes(ohlc, "2026-05-03")  # first bar >= is idx 2 (close 120)
    assert fr["entry_close"] == 120.0
    assert fr[1] == pytest.approx((130.0 / 120.0 - 1) * 100, abs=1e-3)  # rounded 4dp


def test_forward_outcomes_window_missed_guard():
    # Rec date BEFORE the window's first bar: the true entry bar may be
    # outside the fetch — must refuse to label, not anchor on bar 0.
    ohlc = _ohlc_by_date(_bars(100.0, 10.0, DATES[5:]))  # window starts 05-06
    fr = _forward_outcomes(ohlc, "2026-05-01")
    assert fr == {"window_missed": True}


def test_forward_outcomes_partial_maturity():
    # only 6 bars → 1d & 5d available, 20d not yet
    ohlc = _ohlc_by_date(_bars(100.0, 10.0, DATES[:6]))
    fr = _forward_outcomes(ohlc, "2026-05-01")
    assert fr[1] is not None
    assert fr[5] is not None
    assert fr[20] is None
    # 5d path needs bar index entry+5 → not present with 6 bars? idx 0..5,
    # path_end = 5 < 6 → matured exactly at the boundary.
    assert fr["realized_high_5d"] is not None


def test_forward_outcomes_none_when_no_bar_after_rec():
    ohlc = _ohlc_by_date(_bars(100.0, 10.0, DATES[:3]))
    assert _forward_outcomes(ohlc, "2026-12-01") is None


def test_path_stats_realized_extremes_and_mfe_mae():
    # closes 100,110,...; spread 5 → highs c+5, lows c-5. Entry 100 (idx 0);
    # path = bars 1..5: highs 115..155, lows 105..145.
    ohlc = _ohlc_by_date(_bars(100.0, 10.0, DATES, spread=5.0))
    fr = _forward_outcomes(ohlc, "2026-05-01")
    assert fr["realized_high_5d"] == pytest.approx(155.0)
    assert fr["realized_low_5d"] == pytest.approx(105.0)
    assert fr["mfe_pct"] == pytest.approx(55.0)
    assert fr["mae_pct"] == pytest.approx(5.0)  # rising tape: even the low gained


def test_path_stats_entry_bar_excluded():
    # The entry bar's own high must not count toward the path extremes:
    # entry day has a huge high, following days are flat below it.
    bars = [
        {"t": "2026-05-01", "c": 100.0, "h": 999.0, "l": 99.0},
        *_bars(100.0, 0.0, DATES[1:7]),
    ]
    fr = _forward_outcomes(_ohlc_by_date(bars), "2026-05-01")
    assert fr["realized_high_5d"] == pytest.approx(100.0)


def test_target_stop_hit_detection_and_sequencing():
    ohlc = _ohlc_by_date(_bars(100.0, 10.0, DATES, spread=5.0))
    # Target 130 first touched by high on bar 2 (high 125? no: bar1 high 115,
    # bar2 high 125, bar3 high 135) → target bar idx 2 of path. Stop 90 never
    # touched on a rising tape.
    fr = _forward_outcomes(ohlc, "2026-05-01", target=130.0, stop=90.0)
    assert fr["target_hit"] is True
    assert fr["stop_hit"] is False
    assert fr["first_hit"] == "target"
    # Falling tape mirrors to a stop-first outcome
    down = _ohlc_by_date(_bars(100.0, -5.0, DATES, spread=2.0))
    fr2 = _forward_outcomes(down, "2026-05-01", target=200.0, stop=90.0)
    assert fr2["target_hit"] is False
    assert fr2["stop_hit"] is True
    assert fr2["first_hit"] == "stop"


def test_target_stop_same_bar_is_honest_ignorance():
    # One wide bar touches both levels on the same day → both_same_bar.
    bars = [
        {"t": "2026-05-01", "c": 100.0, "h": 100.0, "l": 100.0},
        {"t": "2026-05-02", "c": 100.0, "h": 120.0, "l": 80.0},
        *_bars(100.0, 0.0, DATES[2:8]),
    ]
    fr = _forward_outcomes(_ohlc_by_date(bars), "2026-05-01", target=110.0, stop=90.0)
    assert fr["target_hit"] is True
    assert fr["stop_hit"] is True
    assert fr["first_hit"] == "both_same_bar"


def test_plan_bracket_target_exit_and_time_exit():
    # Rising tape, o == c per bar: entry open 100 (bar 0), target 130 gapped
    # through at bar 3's open (o=130 >= target) → exit at the open price.
    ohlc = _ohlc_by_date(_bars(100.0, 10.0, DATES))
    fr = _forward_outcomes(ohlc, "2026-05-01", target=130.0, stop=50.0)
    assert fr["entry_open"] == 100.0
    assert fr["plan_exit"] == "target"
    assert fr["plan_return_5d"] == pytest.approx(30.0)
    # No levels → pure 5-session hold from the open (time exit at bar 4).
    fr2 = _forward_outcomes(ohlc, "2026-05-01")
    assert fr2["plan_exit"] == "time"
    assert fr2["plan_return_5d"] == pytest.approx(40.0)  # close 140 vs open 100


def test_plan_bracket_same_bar_tie_resolves_to_stop():
    # One wide bar (bar 1) touches both levels intrabar → pessimistic stop.
    bars = [
        {"t": "2026-05-01", "o": 100.0, "c": 100.0, "h": 100.0, "l": 100.0},
        {"t": "2026-05-02", "o": 100.0, "c": 100.0, "h": 120.0, "l": 80.0},
        *_bars(100.0, 0.0, DATES[2:8]),
    ]
    fr = _forward_outcomes(_ohlc_by_date(bars), "2026-05-01", target=110.0, stop=90.0)
    assert fr["plan_exit"] == "stop"
    assert fr["plan_return_5d"] == pytest.approx(-10.0)  # exit at the stop price


def test_plan_bracket_gap_through_stop_exits_at_open():
    # Bar 1 gaps open far below the stop: exit at the open, not the stop.
    bars = [
        {"t": "2026-05-01", "o": 100.0, "c": 100.0, "h": 101.0, "l": 99.0},
        {"t": "2026-05-02", "o": 80.0, "c": 82.0, "h": 83.0, "l": 79.0},
        *_bars(85.0, 0.0, DATES[2:8]),
    ]
    fr = _forward_outcomes(_ohlc_by_date(bars), "2026-05-01", target=150.0, stop=95.0)
    assert fr["plan_exit"] == "stop"
    assert fr["plan_return_5d"] == pytest.approx(-20.0)  # 80/100 - 1


def test_no_levels_means_unknown_not_false():
    ohlc = _ohlc_by_date(_bars(100.0, 10.0, DATES, spread=5.0))
    fr = _forward_outcomes(ohlc, "2026-05-01")  # no target/stop supplied
    assert fr["target_hit"] is None
    assert fr["stop_hit"] is None
    assert fr["first_hit"] is None


def test_levels_present_but_untouched_reports_none_hit():
    flat = _ohlc_by_date(_bars(100.0, 0.0, DATES, spread=1.0))
    fr = _forward_outcomes(flat, "2026-05-01", target=200.0, stop=50.0)
    assert fr["target_hit"] is False
    assert fr["stop_hit"] is False
    assert fr["first_hit"] == "none"


class _FakeBroker:
    def __init__(self, by_symbol: dict[str, list]):
        self._by = by_symbol
        self.requested_days: list[int] = []

    async def get_stock_bars(self, symbol: str, days: int = 90, timeframe: str = "1Day"):
        self.requested_days.append(days)
        return self._by.get(symbol, [])


class _FakeRepo:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = {r["id"]: r for r in rows}

    async def get_recommendations_to_score(self, limit: int = 500):
        return [
            r for r in self.rows.values() if r.get("outcome_status") not in ("scored", "skipped")
        ]

    async def get_recent_recommendations(self, limit: int = 500):
        return list(self.rows.values())

    async def update_recommendation_outcome(self, rec_id: int, **fields):
        self.rows[rec_id].update(fields)
        return True


@pytest.mark.asyncio
async def test_backfill_marks_scored_when_20d_available():
    broker = _FakeBroker(
        {
            "NVDA": _bars(100.0, 10.0, DATES, spread=5.0),  # full 25 days → 20d matures
            "SPUS": _bars(50.0, 1.0, DATES),  # benchmark
        }
    )
    repo = _FakeRepo(
        [
            {
                "id": 1,
                "symbol": "NVDA",
                "date": "2026-05-01",
                "outcome_status": "pending",
                "suggested_target": 130.0,
                "suggested_stop": 90.0,
            },
        ]
    )
    res = await backfill_outcomes(broker, repo)
    assert res == {"updated": 1, "scored": 1, "skipped": 0}
    row = repo.rows[1]
    assert row["outcome_status"] == "scored"
    assert row["entry_close"] == 100.0
    assert row["fwd_return_5d"] == pytest.approx(50.0)
    assert row["fwd_return_20d"] == pytest.approx(200.0)
    assert row["benchmark_return_5d"] is not None  # SPUS 5d filled
    assert row["realized_high_5d"] == pytest.approx(155.0)
    assert row["target_hit"] is True
    assert row["stop_hit"] is False
    assert row["first_hit"] == "target"
    # Plan-anchored outcome persisted too: entry open 100, target 130 gapped
    # through at bar 3's open (o=130) → exit at the open.
    assert row["entry_open"] == 100.0
    assert row["plan_exit"] == "target"
    assert row["plan_return_5d"] == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_backfill_progressive_stays_pending_until_20d():
    broker = _FakeBroker({"NVDA": _bars(100.0, 10.0, DATES[:6])})  # only 6 days
    repo = _FakeRepo(
        [{"id": 1, "symbol": "NVDA", "date": "2026-05-01", "outcome_status": "pending"}]
    )
    res = await backfill_outcomes(broker, repo)
    assert res == {"updated": 1, "scored": 0, "skipped": 0}
    row = repo.rows[1]
    assert row["outcome_status"] == "pending"  # not fully scored yet
    assert row["fwd_return_5d"] is not None  # but 5d already labeled
    assert row["fwd_return_20d"] is None


@pytest.mark.asyncio
async def test_backfill_skips_pick_older_than_window():
    # Window starts 05-06 but the pick is dated 05-01: pre-fix this scored
    # against the wrong entry bar; now it must be marked skipped.
    broker = _FakeBroker({"NVDA": _bars(100.0, 10.0, DATES[5:])})
    repo = _FakeRepo(
        [{"id": 1, "symbol": "NVDA", "date": "2026-05-01", "outcome_status": "pending"}]
    )
    res = await backfill_outcomes(broker, repo)
    assert res == {"updated": 0, "scored": 0, "skipped": 1}
    row = repo.rows[1]
    assert row["outcome_status"] == "skipped"
    assert row.get("entry_close") is None  # never guessed
    # And skipped rows are not revisited by the next backfill.
    res2 = await backfill_outcomes(broker, repo)
    assert res2 == {"updated": 0, "scored": 0, "skipped": 0}


@pytest.mark.asyncio
async def test_audit_repairs_mislabeled_scored_row():
    # Row was scored against a wrong entry (the old bug); audit with a
    # covering window recomputes and repairs entry_close + adds path stats.
    broker = _FakeBroker({"NVDA": _bars(100.0, 10.0, DATES, spread=5.0)})
    repo = _FakeRepo(
        [
            {
                "id": 1,
                "symbol": "NVDA",
                "date": "2026-05-01",
                "outcome_status": "scored",
                "entry_close": 150.0,  # wrong: true entry close is 100.0
                "suggested_target": 130.0,
                "suggested_stop": 90.0,
            },
        ]
    )
    res = await audit_scored_outcomes(broker, repo)
    assert res["audited"] == 1
    assert res["repaired"] == 1
    row = repo.rows[1]
    assert row["entry_close"] == 100.0
    assert row["outcome_status"] == "scored"  # 20d recomputes fine
    assert row["first_hit"] == "target"


@pytest.mark.asyncio
async def test_audit_marks_uncoverable_row_skipped():
    # Bars start after the rec date even with the widened window → skipped.
    broker = _FakeBroker({"NVDA": _bars(100.0, 10.0, DATES[5:])})
    repo = _FakeRepo(
        [
            {
                "id": 1,
                "symbol": "NVDA",
                "date": "2026-05-01",
                "outcome_status": "scored",
                "entry_close": 150.0,
            },
        ]
    )
    res = await audit_scored_outcomes(broker, repo)
    assert res["skipped"] == 1
    assert repo.rows[1]["outcome_status"] == "skipped"


@pytest.mark.asyncio
async def test_audit_widens_window_to_cover_old_picks():
    broker = _FakeBroker({"NVDA": _bars(100.0, 10.0, DATES)})
    repo = _FakeRepo(
        [
            {
                "id": 1,
                "symbol": "NVDA",
                "date": "2026-05-01",
                "outcome_status": "scored",
                "entry_close": 100.0,
            },
        ]
    )
    await audit_scored_outcomes(broker, repo)
    # The fetch must cover the rec date: far more than the default 90 days
    # since 2026-05-01 is long past (relative to the frozen 'today' in CI this
    # is at least the default window).
    assert broker.requested_days and broker.requested_days[0] >= 90


@pytest.mark.asyncio
async def test_compute_scorecard_aggregates():
    repo = _FakeRepo(
        [
            {
                "id": 1,
                "symbol": "NVDA",
                "date": "2026-05-01",
                "fwd_return_1d": 1.0,
                "fwd_return_5d": 5.0,
                "fwd_return_20d": 20.0,
                "benchmark_return_5d": 2.0,
                "target_hit": True,
                "stop_hit": False,
                "first_hit": "target",
                "mfe_pct": 6.0,
                "mae_pct": -1.0,
            },
            {
                "id": 2,
                "symbol": "AAPL",
                "date": "2026-05-02",
                "fwd_return_1d": -1.0,
                "fwd_return_5d": -3.0,
                "fwd_return_20d": None,
                "benchmark_return_5d": 1.0,
                "target_hit": False,
                "stop_hit": True,
                "first_hit": "stop",
                "mfe_pct": 1.0,
                "mae_pct": -4.0,
            },
            {"id": 3, "symbol": "MSFT", "date": "2026-05-03", "fwd_return_5d": None},  # unlabeled
        ]
    )
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
    # Plan-quality block (the LLM-levels baseline)
    assert sc["n_with_levels"] == 2
    assert sc["levels_sufficient"] is False
    assert sc["target_hit_rate"] == pytest.approx(0.5)
    assert sc["stop_hit_rate"] == pytest.approx(0.5)
    assert sc["first_hit_counts"] == {"stop": 1, "target": 1}
    assert sc["avg_mfe_5d"] == pytest.approx(3.5)
    assert sc["avg_mae_5d"] == pytest.approx(-2.5)
    assert sc["avg_plan_return_5d"] is None  # no plan labels in this fixture
    assert sc["plan_exit_counts"] == {}


@pytest.mark.asyncio
async def test_compute_scorecard_band_coverage_from_stored_bands():
    def _row(rid, sym, lo5, hi5, band_lo, band_hi):
        return {
            "id": rid,
            "symbol": sym,
            "date": f"2026-05-0{rid}",
            "fwd_return_5d": 1.0,
            "realized_low_5d": lo5,
            "realized_high_5d": hi5,
            "candidates": {sym: {"quant_bands": {"5": {"low": band_lo, "high": band_hi}}}},
        }

    repo = _FakeRepo(
        [
            _row(1, "NVDA", 95.0, 105.0, 90.0, 110.0),  # covered
            _row(2, "AAPL", 95.0, 115.0, 90.0, 110.0),  # high breached band
            # No stored band → excluded from the coverage sample.
            {
                "id": 3,
                "symbol": "MSFT",
                "date": "2026-05-03",
                "fwd_return_5d": 1.0,
                "realized_low_5d": 95.0,
                "realized_high_5d": 105.0,
                "candidates": {},
            },
        ]
    )
    sc = await compute_scorecard(repo)
    assert sc["band_n"] == 2
    assert sc["band_coverage_5d"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_compute_scorecard_empty():
    repo = _FakeRepo([{"id": 1, "symbol": "NVDA", "date": "2026-05-01", "fwd_return_5d": None}])
    sc = await compute_scorecard(repo)
    assert sc["available"] is False
    assert sc["n_scored"] == 0


@pytest.mark.asyncio
async def test_whatif_equity_curve_compounds_in_date_order():
    repo = _FakeRepo(
        [
            {
                "id": 1,
                "symbol": "AAPL",
                "date": "2026-05-01",
                "fwd_return_5d": 10.0,
                "benchmark_return_5d": 2.0,
            },
            {
                "id": 2,
                "symbol": "MSFT",
                "date": "2026-05-02",
                "fwd_return_5d": -5.0,
                "benchmark_return_5d": 1.0,
            },
            {"id": 3, "symbol": "NVDA", "date": "2026-05-03", "fwd_return_5d": None},  # unscored
        ]
    )
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
async def test_whatif_plan_curve_compounds_bracket_outcomes():
    repo = _FakeRepo(
        [
            {
                "id": 1,
                "symbol": "AAPL",
                "date": "2026-05-01",
                "fwd_return_5d": 10.0,
                "plan_return_5d": 8.0,
            },
            {
                "id": 2,
                "symbol": "MSFT",
                "date": "2026-05-02",
                "fwd_return_5d": -5.0,
                "plan_return_5d": -6.0,
            },
            {"id": 3, "symbol": "NVDA", "date": "2026-05-03", "fwd_return_5d": 2.0},
        ]
    )
    wc = await whatif_equity_curve(repo, start=100.0)
    assert wc["plan_n"] == 2  # the third pick predates plan labeling
    # 100 * 1.08 * 0.94 = 101.52
    assert wc["plan_return_pct"] == pytest.approx(1.52)


@pytest.mark.asyncio
async def test_whatif_equity_curve_empty():
    repo = _FakeRepo([{"id": 1, "symbol": "NVDA", "date": "2026-05-01", "fwd_return_5d": None}])
    wc = await whatif_equity_curve(repo)
    assert wc["available"] is False
    assert wc["points"] == []
