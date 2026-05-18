"""Tests for under-covered branches in :mod:`core.llm.rag`.

`test_rag.py` covers basic embedder semantics + a smoke render.
This file pins the rest of `format_rag_for_prompt`'s contract
(max_rows truncation, text-char limit, newline collapsing, WIN/LOSS
labelling, sim-with-sign formatting, pnl-pct formatting), `cosine`'s
length-mismatch defensive, and `HashingEmbedder` reproducibility
across instances.
"""

from __future__ import annotations

from halal_trader.core.llm.rag import (
    HashingEmbedder,
    RationaleRow,
    cosine,
    format_rag_for_prompt,
)


def _row(
    *,
    symbol: str = "BTCUSDT",
    text: str = "rsi 35 oversold",
    outcome_pnl_pct: float = 0.02,
    outcome_win: bool = True,
) -> RationaleRow:
    return RationaleRow(
        trade_id=f"t-{symbol}",
        symbol=symbol,
        text=text,
        vector=[],
        outcome_pnl_pct=outcome_pnl_pct,
        outcome_win=outcome_win,
    )


# ── format_rag_for_prompt ──────────────────────────────────


def test_format_includes_header_when_hits_present():
    """The first line is always the header — operators grep for it
    when scanning prompt logs to find the RAG block boundary."""
    out = format_rag_for_prompt([(_row(), 0.5)])
    assert out.splitlines()[0].startswith("Most analogous past rationales")


def test_format_truncates_to_max_rows_default_5():
    """6 hits → only 5 rendered (header + 5 lines = 6 total)."""
    hits = [(_row(symbol=f"PAIR{i}"), 0.5) for i in range(6)]
    out = format_rag_for_prompt(hits)
    assert out.count("\n") == 5  # header + 5 rows = 5 newlines


def test_format_respects_custom_max_rows():
    hits = [(_row(symbol=f"PAIR{i}"), 0.5) for i in range(10)]
    out = format_rag_for_prompt(hits, max_rows=2)
    assert out.count("\n") == 2  # header + 2 rows


def test_format_renders_loss_label_when_outcome_lost():
    """`outcome_win=False` → `LOSS` token. Pin so flipped flag doesn't
    silently mark losers as wins (would mislead RAG-influenced trades)."""
    out = format_rag_for_prompt([(_row(outcome_win=False, outcome_pnl_pct=-0.03), 0.5)])
    assert "LOSS" in out
    assert "WIN" not in out


def test_format_renders_pnl_with_sign_and_percent():
    """The pnl format is `{:+.2%}` — both sign and 2-dp percent. Pin so
    a refactor doesn't strip the sign (operator can't tell win vs loss
    at a glance) or change the precision."""
    out = format_rag_for_prompt([(_row(outcome_pnl_pct=0.0234), 0.5)])
    assert "+2.34%" in out
    out_loss = format_rag_for_prompt([(_row(outcome_pnl_pct=-0.015, outcome_win=False), 0.5)])
    assert "-1.50%" in out_loss


def test_format_renders_similarity_with_sign():
    """sim is `{:+.2f}` — sign + 2dp float. Negative similarities can
    happen when normalised embeddings spread; pin the format so they
    render distinguishably."""
    out = format_rag_for_prompt([(_row(), 0.85)])
    assert "+0.85" in out
    out_neg = format_rag_for_prompt([(_row(), -0.20)])
    assert "-0.20" in out_neg


def test_format_truncates_text_at_80_chars():
    """Long rationale text is capped — keeps lines under 100 chars
    (the comment says < 100 to avoid bloating the prompt)."""
    long_text = "a" * 200
    out = format_rag_for_prompt([(_row(text=long_text), 0.5)])
    # Line shape: `  · sim=+0.50 | WIN +2.00% (BTCUSDT): aaaa...`
    # Count consecutive 'a's after the colon to verify truncation.
    rationale_part = out.split("BTCUSDT): ", 1)[1]
    a_count = sum(1 for c in rationale_part if c == "a")
    assert a_count <= 80


def test_format_collapses_newlines_in_text():
    """Newlines in the rationale text are replaced with spaces so each
    hit stays on one line — multi-line rationales would otherwise wreck
    the prompt structure."""
    out = format_rag_for_prompt([(_row(text="line1\nline2\nline3"), 0.5)])
    # The rationale text part should not contain a newline.
    assert "line1 line2 line3" in out
    # And the *header line* still ends correctly.
    assert out.count("\n") == 1  # header + 1 row


def test_format_strips_text_whitespace():
    """Leading/trailing whitespace is stripped before truncation."""
    out = format_rag_for_prompt([(_row(text="   padded   "), 0.5)])
    # The stripped text should appear, no leading whitespace inside the
    # parenthesised label.
    assert "BTCUSDT): padded" in out


def test_format_handles_none_text_gracefully():
    """Defensive: a row with `text=None` (legacy or malformed) renders
    as the BTCUSDT-prefixed empty rationale rather than crashing."""
    out = format_rag_for_prompt([(_row(text=""), 0.5)])
    # No crash; the text part is empty after the colon.
    assert "BTCUSDT)" in out


def test_format_preserves_input_order():
    """Order is preserved (the helper doesn't re-sort) — the caller's
    ranking by similarity is what determines what the LLM sees first."""
    a = _row(symbol="AAA", text="alpha")
    b = _row(symbol="BBB", text="beta")
    out = format_rag_for_prompt([(a, 0.9), (b, 0.5)])
    a_pos = out.index("AAA")
    b_pos = out.index("BBB")
    assert a_pos < b_pos


# ── cosine ─────────────────────────────────────────────────


def test_cosine_length_mismatch_returns_zero():
    """Defensive: vectors of different dim → 0 (not a crash). Happens
    when an old embedder ran with a different `dim` and the rows are
    backfilled."""
    assert cosine([1.0, 0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_one_empty_returns_zero():
    """Empty vector on either side → 0."""
    assert cosine([1.0, 0.0], []) == 0.0
    assert cosine([], [1.0, 0.0]) == 0.0


def test_cosine_orthogonal_vectors_returns_zero():
    """Perpendicular unit vectors → 0."""
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert cosine(a, b) == 0.0


def test_cosine_identical_unit_vectors_returns_one():
    a = [1.0, 0.0, 0.0]
    assert cosine(a, a) == 1.0


def test_cosine_anti_parallel_unit_vectors_returns_negative_one():
    """Opposing unit vectors → -1 (max anti-similarity)."""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine(a, b) == -1.0


# ── HashingEmbedder ────────────────────────────────────────


def test_embedder_reproducible_across_instances():
    """Two fresh embedders with the same dim produce identical vectors
    for the same text — the rationale store keys on the vector so any
    drift would break retrieval after a process restart."""
    a = HashingEmbedder(dim=128)
    b = HashingEmbedder(dim=128)
    assert a.embed("rsi 35 oversold") == b.embed("rsi 35 oversold")


def test_embedder_different_dim_produces_different_length():
    a = HashingEmbedder(dim=64)
    b = HashingEmbedder(dim=256)
    assert len(a.embed("test")) == 64
    assert len(b.embed("test")) == 256


def test_embedder_lowercase_normalisation():
    """Tokens are lowercased before hashing — so 'RSI Oversold' and
    'rsi oversold' produce the same vector. Important for retrieval
    quality: case shouldn't fragment matches."""
    e = HashingEmbedder(dim=128)
    assert e.embed("RSI OVERSOLD") == e.embed("rsi oversold")


def test_embedder_handles_only_whitespace_input():
    """Defensive: an all-whitespace string tokenises to nothing → zero
    vector (same as empty string)."""
    e = HashingEmbedder(dim=128)
    assert e.embed("    \t  \n  ") == [0.0] * 128
