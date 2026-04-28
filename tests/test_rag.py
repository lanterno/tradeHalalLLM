"""Tests for RAG embedder + prompt formatting (pure helpers)."""

from __future__ import annotations

from halal_trader.core.llm.rag import (
    HashingEmbedder,
    RationaleRow,
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


# ── Prompt format ───────────────────────────────────────────────


def test_format_empty_returns_empty() -> None:
    assert format_rag_for_prompt([]) == ""


def test_format_renders_top_hits() -> None:
    row = RationaleRow(
        trade_id="t1",
        symbol="BTCUSDT",
        text="rsi 35",
        vector=[],
        outcome_pnl_pct=0.02,
        outcome_win=True,
    )
    text = format_rag_for_prompt([(row, 0.85)])
    assert "BTCUSDT" in text
    assert "WIN" in text
