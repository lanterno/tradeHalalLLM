"""Tests for embedding-based regime memory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

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


async def test_add_and_size(engine: AsyncEngine) -> None:
    mem = RegimeMemory(engine=engine)
    snap = RegimeSnapshot(date="2026-01-01", features=_calm())
    await mem.add(snap)
    assert await mem.size() == 1


async def test_add_today_dedupes_by_date(engine: AsyncEngine) -> None:
    mem = RegimeMemory(engine=engine)
    await mem.add_today(_calm(), today="2026-04-01", outcome_pnl_pct=0.01)
    await mem.add_today(_crashing(), today="2026-04-01", outcome_pnl_pct=-0.05)
    assert await mem.size() == 1
    recent = await mem.recent()
    assert recent[0].outcome_pnl_pct == -0.05


async def test_capacity_fifo_trim(engine: AsyncEngine) -> None:
    import asyncio

    mem = RegimeMemory(engine=engine, capacity=3)
    for i in range(5):
        await mem.add_today(_calm(), today=f"2026-04-0{i + 1}")
        # Tiny sleep so the created_at column distinguishes insert order
        # — capacity trim deletes by oldest created_at.
        await asyncio.sleep(0.01)
    assert await mem.size() == 3
    recent = await mem.recent(limit=10)
    dates = sorted(s.date for s in recent)
    assert dates == ["2026-04-03", "2026-04-04", "2026-04-05"]


# ── Query ─────────────────────────────────────────────────────────


async def test_query_finds_most_similar_first(engine: AsyncEngine) -> None:
    mem = RegimeMemory(engine=engine)
    await mem.add_today(_calm(), today="2026-01-01", outcome_pnl_pct=0.01)
    await mem.add_today(_crashing(), today="2026-01-02", outcome_pnl_pct=-0.05)
    await mem.add_today(_euphoric(), today="2026-01-03", outcome_pnl_pct=0.03)

    hits = await mem.query(_crashing(), k=2)
    assert hits
    assert hits[0][0].date == "2026-01-02"
    assert hits[0][1] >= hits[1][1]


async def test_query_respects_min_similarity(engine: AsyncEngine) -> None:
    mem = RegimeMemory(engine=engine)
    await mem.add_today(_calm(), today="2026-01-01")
    await mem.add_today(_crashing(), today="2026-01-02")

    hits = await mem.query(_euphoric(), k=5, min_similarity=0.95)
    assert hits == []


async def test_query_empty_memory(engine: AsyncEngine) -> None:
    mem = RegimeMemory(engine=engine)
    assert await mem.query(_calm()) == []


async def test_aggregate_outcome_weighted(engine: AsyncEngine) -> None:
    mem = RegimeMemory(engine=engine)
    await mem.add_today(
        _crashing(), today="2026-01-02", outcome_pnl_pct=-0.05, outcome_win_rate=0.2
    )
    await mem.add_today(_calm(), today="2026-01-01", outcome_pnl_pct=0.01, outcome_win_rate=0.6)
    hits = await mem.query(_crashing(), k=2)
    agg = RegimeMemory.aggregate_outcome(hits)
    # Crashing match dominates by similarity, so aggregate should be negative.
    assert agg["n"] == 2
    assert agg["weighted_pnl_pct"] < 0


def test_aggregate_outcome_empty() -> None:
    agg = RegimeMemory.aggregate_outcome([])
    assert agg == {"weighted_pnl_pct": 0.0, "weighted_win_rate": 0.0, "n": 0}


async def test_round_trip_preserves_features(engine: AsyncEngine) -> None:
    mem = RegimeMemory(engine=engine)
    await mem.add_today(_crashing(), today="2026-04-02", outcome_pnl_pct=-0.07, note="cpi shock")
    recent = await mem.recent()
    assert len(recent) == 1
    assert recent[0].features.volatility == _crashing().volatility
    assert recent[0].note == "cpi shock"


# ── Prompt formatting ─────────────────────────────────────────────


async def test_format_for_prompt_includes_outcome_label(engine: AsyncEngine) -> None:
    mem = RegimeMemory(engine=engine)
    await mem.add_today(_crashing(), today="2026-01-02", outcome_pnl_pct=-0.05, outcome_n_trades=4)
    hits = await mem.query(_crashing(), k=1)
    text = format_for_prompt(_crashing(), hits)
    assert "2026-01-02" in text
    assert "-5" in text or "-5.0" in text or "-5.00" in text


def test_format_for_prompt_empty_returns_placeholder() -> None:
    text = format_for_prompt(_calm(), [])
    assert "No analogous" in text
