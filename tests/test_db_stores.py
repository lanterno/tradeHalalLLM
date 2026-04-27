"""Tests for the DB-backed RAG / thesis / regret stores.

All three stores share the same shape: async DB-backed implementations
of the JSON-sidecar interfaces. Tests use SQLite via the existing
fixture pattern — production runs against Postgres, but the storage
logic is dialect-agnostic.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from halal_trader.core.llm.rag_db import DBRationaleStore
from halal_trader.core.regret_db import DBRegretRecorder
from halal_trader.core.thesis_db import DBThesisTagStore


@pytest.fixture
async def engine(tmp_path):
    """Per-test SQLite engine with the full schema applied via create_all."""
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'db_stores.db'}")
    # Touch every model so SQLModel.metadata is fully populated.
    import halal_trader.db.models  # noqa: F401

    async with eng.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield eng
    await eng.dispose()


# ── DBRationaleStore ─────────────────────────────────────────────


async def test_rag_store_add_and_size(engine) -> None:
    store = DBRationaleStore(engine=engine)
    assert await store.size() == 0
    row = await store.add(
        trade_id="t1",
        symbol="BTCUSDT",
        text="rsi 35 oversold",
        outcome_pnl_pct=0.02,
    )
    assert row.outcome_win is True
    assert await store.size() == 1


async def test_rag_store_add_idempotent(engine) -> None:
    store = DBRationaleStore(engine=engine)
    await store.add(trade_id="t1", symbol="X", text="aaa", outcome_pnl_pct=0.01)
    await store.add(trade_id="t1", symbol="X", text="zzz", outcome_pnl_pct=-0.05)
    assert await store.size() == 1


async def test_rag_store_query_returns_sorted_hits(engine) -> None:
    store = DBRationaleStore(engine=engine)
    await store.add(
        trade_id="match",
        symbol="BTCUSDT",
        text="rsi 35 oversold bb lower",
        outcome_pnl_pct=0.02,
    )
    await store.add(
        trade_id="other",
        symbol="BTCUSDT",
        text="vwap rejection volume spike",
        outcome_pnl_pct=-0.01,
    )
    hits = await store.query("rsi oversold lower band", k=2, min_similarity=0.0)
    assert len(hits) == 2
    assert hits[0][0].trade_id == "match"
    assert hits[0][1] > hits[1][1]


async def test_rag_store_query_filters_by_symbol(engine) -> None:
    store = DBRationaleStore(engine=engine)
    await store.add(trade_id="btc", symbol="BTCUSDT", text="rsi 35", outcome_pnl_pct=0.02)
    await store.add(trade_id="eth", symbol="ETHUSDT", text="rsi 35", outcome_pnl_pct=-0.02)
    hits = await store.query("rsi 35", k=5, symbol="BTCUSDT", min_similarity=0.0)
    assert all(r.symbol == "BTCUSDT" for r, _ in hits)
    assert len(hits) == 1


async def test_rag_store_aggregate(engine) -> None:
    store = DBRationaleStore(engine=engine)
    await store.add(trade_id="win", symbol="X", text="aaa bbb", outcome_pnl_pct=0.05)
    await store.add(trade_id="lose", symbol="X", text="zzz xyz", outcome_pnl_pct=-0.03)
    hits = await store.query("aaa bbb", k=2, min_similarity=0.0)
    agg = await store.aggregate(hits)
    assert agg["n"] == 2


# ── DBThesisTagStore ─────────────────────────────────────────────


async def test_thesis_store_set_and_get(engine) -> None:
    store = DBThesisTagStore(engine=engine)
    await store.set("t1", "breakout", confidence=0.8, reason="20d high")
    assert await store.get("t1") == "breakout"


async def test_thesis_store_set_overwrites(engine) -> None:
    store = DBThesisTagStore(engine=engine)
    await store.set("t1", "breakout")
    await store.set("t1", "scalp")
    assert await store.get("t1") == "scalp"


async def test_thesis_store_unknown_tag_coerced(engine) -> None:
    store = DBThesisTagStore(engine=engine)
    await store.set("t1", "made_up_tag")
    assert await store.get("t1") == "unknown"


async def test_thesis_store_all(engine) -> None:
    store = DBThesisTagStore(engine=engine)
    await store.set("a", "breakout")
    await store.set("b", "trend_follow")
    assert await store.all() == {"a": "breakout", "b": "trend_follow"}


# ── DBRegretRecorder ─────────────────────────────────────────────


async def test_regret_recorder_append(engine) -> None:
    rec = DBRegretRecorder(engine=engine)
    await rec.append(
        {
            "trade_id": "t1",
            "symbol": "BTCUSDT",
            "regret": 0.5,
            "optimal_size_pct": 1.0,
            "actual_size_pct": 0.5,
            "pnl_pct": 0.02,
            "note": "x",
            "ts": "2026-04-27T00:00:00+00:00",
        }
    )
    rows = await rec.all()
    assert len(rows) == 1
    assert rows[0]["trade_id"] == "t1"
    assert rows[0]["regret"] == 0.5


async def test_regret_recorder_idempotent(engine) -> None:
    rec = DBRegretRecorder(engine=engine)
    payload = {
        "trade_id": "t1",
        "symbol": "X",
        "regret": 0.5,
        "optimal_size_pct": 1.0,
        "actual_size_pct": 0.5,
        "pnl_pct": 0.01,
        "note": "",
        "ts": "2026-04-27T00:00:00+00:00",
    }
    await rec.append(payload)
    await rec.append(payload)  # second append must be no-op
    rows = await rec.all()
    assert len(rows) == 1
