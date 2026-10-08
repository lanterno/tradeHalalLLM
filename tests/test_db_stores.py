"""Tests for the DB-backed RAG store."""

from __future__ import annotations

from halal_trader.core.llm.rag_db import DBRationaleStore

# ── DBRationaleStore ─────────────────────────────────────────────


async def test_rag_store_add_and_size(engine) -> None:
    store = DBRationaleStore(engine=engine)
    assert await store.size() == 0
    row = await store.add(
        trade_id="t1",
        symbol="AAPL",
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
        symbol="AAPL",
        text="rsi 35 oversold bb lower",
        outcome_pnl_pct=0.02,
    )
    await store.add(
        trade_id="other",
        symbol="AAPL",
        text="vwap rejection volume spike",
        outcome_pnl_pct=-0.01,
    )
    hits = await store.query("rsi oversold lower band", k=2, min_similarity=0.0)
    assert len(hits) == 2
    assert hits[0][0].trade_id == "match"
    assert hits[0][1] > hits[1][1]


async def test_rag_store_query_filters_by_symbol(engine) -> None:
    store = DBRationaleStore(engine=engine)
    await store.add(trade_id="aapl", symbol="AAPL", text="rsi 35", outcome_pnl_pct=0.02)
    await store.add(trade_id="msft", symbol="MSFT", text="rsi 35", outcome_pnl_pct=-0.02)
    hits = await store.query("rsi 35", k=5, symbol="AAPL", min_similarity=0.0)
    assert all(r.symbol == "AAPL" for r, _ in hits)
    assert len(hits) == 1


async def test_rag_store_aggregate(engine) -> None:
    store = DBRationaleStore(engine=engine)
    await store.add(trade_id="win", symbol="X", text="aaa bbb", outcome_pnl_pct=0.05)
    await store.add(trade_id="lose", symbol="X", text="zzz xyz", outcome_pnl_pct=-0.03)
    hits = await store.query("aaa bbb", k=2, min_similarity=0.0)
    agg = await store.aggregate(hits)
    assert agg["n"] == 2


async def test_rag_store_uses_hnsw_index(engine) -> None:
    """The query plan should hit the HNSW index, not a seq scan."""
    import sqlalchemy as sa

    store = DBRationaleStore(engine=engine)
    await store.add(trade_id="t1", symbol="X", text="aaa bbb", outcome_pnl_pct=0.01)

    q = list(store.embedder.embed("aaa bbb"))
    qstr = "[" + ",".join(str(x) for x in q) + "]"
    async with engine.connect() as conn:
        rows = await conn.execute(
            sa.text(
                "EXPLAIN SELECT trade_id FROM rag_rationales "
                f"ORDER BY embedding <=> '{qstr}'::vector LIMIT 5"
            )
        )
        plan = "\n".join(r[0] for r in rows.all())
    assert "ix_rag_rationales_embedding_hnsw" in plan, plan
