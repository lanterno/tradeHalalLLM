"""Tests for RAG over reasoning traces."""

from __future__ import annotations

from pathlib import Path

from halal_trader.core.llm.rag import (
    HashingEmbedder,
    RationaleStore,
    cosine,
    format_rag_for_prompt,
)

# ── Embedder ────────────────────────────────────────────────────


def test_embedder_deterministic() -> None:
    e = HashingEmbedder(dim=128)
    assert e.embed("rsi 35 oversold") == e.embed("rsi 35 oversold")


def test_embedder_normalises_to_unit() -> None:
    e = HashingEmbedder(dim=128)
    v = e.embed("rsi 35 oversold")
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_embedder_empty_string_returns_zero_vector() -> None:
    e = HashingEmbedder(dim=128)
    v = e.embed("")
    assert v == [0.0] * 128


def test_embedder_similar_texts_have_higher_cosine() -> None:
    e = HashingEmbedder(dim=512)
    a = e.embed("rsi 35 with bb lower band touch")
    b = e.embed("rsi 35 oversold at bb lower band")
    c = e.embed("vwap rejection on volume spike")
    assert cosine(a, b) > cosine(a, c)


def test_cosine_zero_vectors() -> None:
    assert cosine([], []) == 0.0
    assert cosine([0, 0], [0, 0]) == 0.0


# ── Store ───────────────────────────────────────────────────────


def _store(tmp_path: Path) -> RationaleStore:
    return RationaleStore(path=tmp_path / "rag.json")


def test_add_persists_and_returns_row(tmp_path: Path) -> None:
    s = _store(tmp_path)
    row = s.add(
        trade_id="t1",
        symbol="BTCUSDT",
        text="rsi 35 oversold",
        outcome_pnl_pct=0.02,
    )
    assert row.outcome_win is True
    assert s.size == 1


def test_add_idempotent_on_trade_id(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.add(trade_id="t1", symbol="BTCUSDT", text="x", outcome_pnl_pct=0.01)
    s.add(trade_id="t1", symbol="BTCUSDT", text="y", outcome_pnl_pct=-0.05)
    assert s.size == 1
    assert s.rows[0].text == "x"


def test_capacity_fifo_trim(tmp_path: Path) -> None:
    s = RationaleStore(path=tmp_path / "rag.json", capacity=3)
    for i in range(5):
        s.add(trade_id=f"t{i}", symbol="X", text=f"text {i}", outcome_pnl_pct=0.01)
    assert s.size == 3
    assert [r.trade_id for r in s.rows] == ["t2", "t3", "t4"]


def test_persists_across_instances(tmp_path: Path) -> None:
    p = tmp_path / "rag.json"
    s1 = RationaleStore(path=p)
    s1.add(trade_id="t1", symbol="X", text="aaa", outcome_pnl_pct=0.02)
    s2 = RationaleStore(path=p)
    assert s2.size == 1
    assert s2.rows[0].trade_id == "t1"


def test_resilient_to_corrupt_file(tmp_path: Path) -> None:
    p = tmp_path / "rag.json"
    p.write_text("{not json")
    s = RationaleStore(path=p)
    assert s.size == 0
    s.add(trade_id="t1", symbol="X", text="aaa", outcome_pnl_pct=0.01)
    assert s.size == 1


# ── Query ───────────────────────────────────────────────────────


def test_query_empty_store(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.query("anything") == []


def test_query_returns_most_similar_first(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.add(trade_id="match", symbol="BTCUSDT", text="rsi 35 oversold bb lower", outcome_pnl_pct=0.02)
    s.add(
        trade_id="other",
        symbol="BTCUSDT",
        text="vwap rejection volume spike",
        outcome_pnl_pct=-0.01,
    )
    hits = s.query("rsi oversold lower band", k=2, min_similarity=0.0)
    assert len(hits) == 2
    assert hits[0][0].trade_id == "match"
    assert hits[0][1] > hits[1][1]


def test_query_min_similarity_filters(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.add(trade_id="t1", symbol="X", text="totally unrelated content", outcome_pnl_pct=0.01)
    hits = s.query("rsi macd bb", min_similarity=0.5)
    assert hits == []


def test_query_filters_by_symbol(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.add(trade_id="btc", symbol="BTCUSDT", text="rsi 35 oversold", outcome_pnl_pct=0.02)
    s.add(trade_id="eth", symbol="ETHUSDT", text="rsi 35 oversold", outcome_pnl_pct=-0.02)
    hits = s.query("rsi 35 oversold", k=5, symbol="BTCUSDT")
    assert all(r.symbol == "BTCUSDT" for r, _ in hits)
    assert len(hits) == 1


def test_aggregate_weighted_outcome(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.add(trade_id="win", symbol="X", text="aaa bbb", outcome_pnl_pct=0.05)
    s.add(trade_id="lose", symbol="X", text="zzz xyz", outcome_pnl_pct=-0.03)
    hits = s.query("aaa bbb", k=2, min_similarity=0.0)
    agg = s.aggregate(hits)
    assert agg["n"] == 2
    # Match weighted toward the close hit (positive)
    assert agg["weighted_pnl_pct"] > 0


def test_aggregate_empty() -> None:
    s = RationaleStore(path=Path("/tmp/_unused.json"))
    s.path = Path("/tmp/_unused2.json")
    agg = s.aggregate([])
    assert agg["n"] == 0


# ── Prompt format ───────────────────────────────────────────────


def test_format_empty_returns_empty(tmp_path: Path) -> None:
    assert format_rag_for_prompt([]) == ""


def test_format_renders_top_hits(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.add(trade_id="t1", symbol="BTCUSDT", text="rsi 35", outcome_pnl_pct=0.02)
    hits = s.query("rsi 35")
    text = format_rag_for_prompt(hits)
    assert "BTCUSDT" in text
    assert "WIN" in text
