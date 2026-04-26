"""Tests for embedding-based regime memory."""

from __future__ import annotations

from pathlib import Path

from halal_trader.ml.regime_memory import (
    RegimeFeatures,
    RegimeMemory,
    RegimeSnapshot,
    format_for_prompt,
)


def _calm() -> RegimeFeatures:
    return RegimeFeatures(
        volatility=0.005,
        trend=0.1,
        breadth=0.1,
        sentiment=0.0,
        drawdown=0.01,
        volume_ratio=1.0,
        correlation=0.2,
        realized_return_1d=0.001,
        rsi=50.0,
        spread_bps=2.0,
    )


def _crashing() -> RegimeFeatures:
    return RegimeFeatures(
        volatility=0.06,
        trend=-0.8,
        breadth=-0.7,
        sentiment=-0.6,
        drawdown=0.15,
        volume_ratio=3.0,
        correlation=0.85,
        realized_return_1d=-0.04,
        rsi=22.0,
        spread_bps=15.0,
    )


def _euphoric() -> RegimeFeatures:
    return RegimeFeatures(
        volatility=0.04,
        trend=0.85,
        breadth=0.7,
        sentiment=0.7,
        drawdown=0.0,
        volume_ratio=2.5,
        correlation=0.7,
        realized_return_1d=0.04,
        rsi=78.0,
        spread_bps=5.0,
    )


# ── Vector + cosine ───────────────────────────────────────────────


def test_features_to_vector_stable_length() -> None:
    v = _calm().to_vector()
    assert len(v) == len(_crashing().to_vector())


# ── Storage ───────────────────────────────────────────────────────


def test_add_and_size() -> None:
    mem = RegimeMemory()
    snap = RegimeSnapshot(date="2026-01-01", features=_calm())
    mem.add(snap)
    assert mem.size == 1


def test_add_today_dedupes_by_date() -> None:
    mem = RegimeMemory()
    mem.add_today(_calm(), today="2026-04-01", outcome_pnl_pct=0.01)
    mem.add_today(_crashing(), today="2026-04-01", outcome_pnl_pct=-0.05)
    assert mem.size == 1
    assert mem.snapshots[0].outcome_pnl_pct == -0.05


def test_capacity_fifo_trim() -> None:
    mem = RegimeMemory(capacity=3)
    for i in range(5):
        mem.add_today(_calm(), today=f"2026-04-0{i + 1}")
    assert mem.size == 3
    assert [s.date for s in mem.snapshots] == [
        "2026-04-03",
        "2026-04-04",
        "2026-04-05",
    ]


# ── Query ─────────────────────────────────────────────────────────


def test_query_finds_most_similar_first() -> None:
    mem = RegimeMemory()
    mem.add_today(_calm(), today="2026-01-01", outcome_pnl_pct=0.01)
    mem.add_today(_crashing(), today="2026-01-02", outcome_pnl_pct=-0.05)
    mem.add_today(_euphoric(), today="2026-01-03", outcome_pnl_pct=0.03)

    hits = mem.query(_crashing(), k=2)
    assert hits
    assert hits[0][0].date == "2026-01-02"
    assert hits[0][1] >= hits[1][1]


def test_query_respects_min_similarity() -> None:
    mem = RegimeMemory()
    mem.add_today(_calm(), today="2026-01-01")
    mem.add_today(_crashing(), today="2026-01-02")

    hits = mem.query(_euphoric(), k=5, min_similarity=0.95)
    assert hits == []


def test_query_empty_memory() -> None:
    mem = RegimeMemory()
    assert mem.query(_calm()) == []


def test_aggregate_outcome_weighted() -> None:
    mem = RegimeMemory()
    mem.add_today(_crashing(), today="2026-01-02", outcome_pnl_pct=-0.05, outcome_win_rate=0.2)
    mem.add_today(_calm(), today="2026-01-01", outcome_pnl_pct=0.01, outcome_win_rate=0.6)
    hits = mem.query(_crashing(), k=2)
    agg = mem.aggregate_outcome(hits)
    # Crashing match dominates by similarity, so aggregate should be negative.
    assert agg["n"] == 2
    assert agg["weighted_pnl_pct"] < 0


def test_aggregate_outcome_empty() -> None:
    mem = RegimeMemory()
    agg = mem.aggregate_outcome([])
    assert agg == {"weighted_pnl_pct": 0.0, "weighted_win_rate": 0.0, "n": 0}


# ── Persistence ───────────────────────────────────────────────────


def test_save_load_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "regime_memory.json"
    mem = RegimeMemory(capacity=10)
    mem.add_today(_calm(), today="2026-04-01", outcome_pnl_pct=0.01, note="quiet day")
    mem.add_today(_crashing(), today="2026-04-02", outcome_pnl_pct=-0.07, note="cpi shock")
    mem.save(p)
    back = RegimeMemory.load(p)
    assert back.size == 2
    assert back.capacity == 10
    assert back.snapshots[1].features.volatility == _crashing().volatility
    assert back.snapshots[1].note == "cpi shock"


# ── Prompt formatting ─────────────────────────────────────────────


def test_format_for_prompt_includes_outcome_label() -> None:
    mem = RegimeMemory()
    mem.add_today(_crashing(), today="2026-01-02", outcome_pnl_pct=-0.05, outcome_n_trades=4)
    hits = mem.query(_crashing(), k=1)
    text = format_for_prompt(_crashing(), hits)
    assert "2026-01-02" in text
    assert "-5" in text or "-5.0" in text or "-5.00" in text


def test_format_for_prompt_empty_returns_placeholder() -> None:
    text = format_for_prompt(_calm(), [])
    assert "No analogous" in text
