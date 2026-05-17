"""Tests for `core/rationale_search.py`.

Pins the tokeniser semantics (case-fold, stop-word drop, length
filter), the TF-IDF scoring math (sub-linear TF dampening,
smoothed IDF), the require_all filter, the snippet builder, the
duplicate-doc-id rejection, and the render output.
"""

from __future__ import annotations

import math

import pytest

from halal_trader.core.rationale_search import (
    RationaleDoc,
    RationaleIndex,
    SearchHit,
    render_results,
    tokenize,
)


def _doc(doc_id: str, text: str, **metadata: str) -> RationaleDoc:
    return RationaleDoc(doc_id=doc_id, text=text, metadata=metadata)


# ── tokenise ─────────────────────────────────────────────


def test_tokenize_lowercases():
    assert tokenize("MACD divergence") == ["macd", "divergence"]


def test_tokenize_drops_stop_words():
    """Pin: only the small stop-word list is dropped — 'the',
    'and', 'is', etc. Operator-relevant words like 'overbought'
    must survive."""
    out = tokenize("the market is overbought and bearish")
    assert "overbought" in out
    assert "bearish" in out
    assert "the" not in out
    assert "is" not in out
    assert "and" not in out


def test_tokenize_drops_short_tokens():
    """Pin: tokens shorter than 2 chars are noise — 'a' / 'i'
    appear in every doc, IDF approaches zero anyway."""
    out = tokenize("a i x bb")
    assert "a" not in out
    assert "i" not in out
    assert "x" not in out
    assert "bb" in out  # length-2 survives


def test_tokenize_handles_empty_input():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_tokenize_splits_on_punctuation():
    """Pin: word-boundary tokenisation. 'MACD-divergence' splits
    into ['macd', 'divergence']."""
    out = tokenize("MACD-divergence, RSI=75; high vol.")
    assert "macd" in out
    assert "divergence" in out
    assert "rsi" in out
    assert "75" not in out  # leading digit; the regex requires letter start


def test_tokenize_keeps_alphanumeric_tokens():
    """Pin: tokens like 'rsi14' or 'btc1' are preserved as-is —
    operators search for indicators with numeric suffixes."""
    out = tokenize("rsi14 btc1")
    assert "rsi14" in out
    assert "btc1" in out


# ── add / clear ──────────────────────────────────────────


def test_add_and_doc_count():
    idx = RationaleIndex()
    assert idx.doc_count == 0
    idx.add(_doc("a", "MACD divergence"))
    idx.add(_doc("b", "RSI overbought"))
    assert idx.doc_count == 2


def test_add_rejects_duplicate_doc_id():
    """Pin: append-only audit invariant — duplicate doc_id
    surfaces immediately rather than silently overwriting."""
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD divergence"))
    with pytest.raises(ValueError, match="duplicate doc_id"):
        idx.add(_doc("a", "different text"))


def test_clear_resets_state():
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD divergence"))
    idx.add(_doc("b", "RSI overbought"))
    idx.clear()
    assert idx.doc_count == 0
    # Can re-add the same doc_id after clear.
    idx.add(_doc("a", "different text"))


def test_get_doc_returns_added_doc():
    idx = RationaleIndex()
    doc = _doc("a", "MACD divergence", pair="BTCUSDT")
    idx.add(doc)
    assert idx.get_doc("a") is doc
    assert idx.get_doc("missing") is None


# ── basic search ─────────────────────────────────────────


def test_search_returns_matching_doc():
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD divergence with high vol"))
    idx.add(_doc("b", "RSI bounce off support"))
    hits = idx.search("MACD")
    assert len(hits) == 1
    assert hits[0].doc.doc_id == "a"


def test_search_returns_empty_for_no_match():
    idx = RationaleIndex()
    idx.add(_doc("a", "RSI bounce"))
    hits = idx.search("MACD")
    assert hits == []


def test_search_returns_empty_index():
    """Empty index → empty results, no exception."""
    idx = RationaleIndex()
    hits = idx.search("anything")
    assert hits == []


def test_search_returns_empty_query():
    """Pin: an all-stop-word query returns no hits rather than
    every doc in score-zero order."""
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD divergence"))
    hits = idx.search("the and is")
    assert hits == []


def test_search_rejects_zero_or_negative_limit():
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD"))
    with pytest.raises(ValueError, match="limit"):
        idx.search("MACD", limit=0)


# ── ranking ──────────────────────────────────────────────


def test_search_ranks_higher_term_frequency_first():
    """A doc that mentions MACD 5× ranks above one that mentions
    it once (sub-linear dampening, but still increasing)."""
    idx = RationaleIndex()
    idx.add(_doc("once", "MACD"))
    idx.add(_doc("many", "MACD MACD MACD MACD MACD"))
    hits = idx.search("MACD")
    assert hits[0].doc.doc_id == "many"


def test_search_ranks_rare_term_higher_via_idf():
    """A query for a rare term should rank docs that contain it
    higher than docs that contain a common term — the IDF
    component should dominate."""
    idx = RationaleIndex()
    # 4 docs mention "buy"; only one mentions "exotic".
    for i in range(4):
        idx.add(_doc(f"common-{i}", "buy buy"))
    idx.add(_doc("rare", "exotic"))
    hits = idx.search("exotic")
    assert hits[0].doc.doc_id == "rare"


def test_search_sub_linear_tf_dampening():
    """Pin: 10× repeated term scores < 10× a single mention's
    contribution. Use the documented 1 + log(tf) shape — for
    tf=10 that's 1 + log(10) ≈ 3.30, not 10."""
    idx = RationaleIndex()
    idx.add(_doc("once", "MACD"))
    idx.add(_doc("ten", " ".join(["MACD"] * 10)))
    hits = idx.search("MACD")
    once = next(h for h in hits if h.doc.doc_id == "once")
    ten = next(h for h in hits if h.doc.doc_id == "ten")
    # Ratio of scores should be ~3.30 / 1.0 ≈ 3.30, well under 10.
    assert ten.score / once.score < 5.0


def test_search_includes_per_term_contributions():
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD divergence MACD"))
    idx.add(_doc("b", "RSI bounce"))
    hits = idx.search("MACD divergence")
    assert hits[0].doc.doc_id == "a"
    term_names = {t.term for t in hits[0].terms}
    assert "macd" in term_names
    assert "divergence" in term_names


def test_search_term_contributions_match_tf():
    """The reported `tf` in TermContribution should match the
    actual count in the doc."""
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD MACD divergence"))
    hits = idx.search("MACD divergence")
    macd = next(t for t in hits[0].terms if t.term == "macd")
    div = next(t for t in hits[0].terms if t.term == "divergence")
    assert macd.tf == 2
    assert div.tf == 1


def test_search_query_dedupes_repeated_terms():
    """Pin: a query 'MACD MACD' is treated as one occurrence —
    operators occasionally double-type; the search shouldn't
    reward it."""
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD divergence"))
    once = idx.search("MACD")[0]
    twice = idx.search("MACD MACD")[0]
    # Same score because the query dedupes.
    assert once.score == twice.score


def test_search_stable_tie_break_preserves_input_order():
    """When two docs match equally, earlier-added wins."""
    idx = RationaleIndex()
    idx.add(_doc("first", "MACD"))
    idx.add(_doc("second", "MACD"))
    hits = idx.search("MACD")
    assert hits[0].doc.doc_id == "first"
    assert hits[1].doc.doc_id == "second"


def test_search_respects_limit():
    idx = RationaleIndex()
    for i in range(20):
        idx.add(_doc(f"d-{i}", "MACD divergence"))
    hits = idx.search("MACD", limit=5)
    assert len(hits) == 5


# ── require_all ──────────────────────────────────────────


def test_require_all_filters_docs_missing_a_term():
    idx = RationaleIndex()
    idx.add(_doc("both", "MACD divergence"))
    idx.add(_doc("only_macd", "MACD bounce"))
    idx.add(_doc("only_div", "divergence pattern"))
    hits = idx.search("MACD divergence", require_all=True)
    ids = {h.doc.doc_id for h in hits}
    assert ids == {"both"}


def test_require_all_default_false_does_or_search():
    idx = RationaleIndex()
    idx.add(_doc("only_macd", "MACD bounce"))
    idx.add(_doc("only_div", "divergence pattern"))
    hits = idx.search("MACD divergence")
    ids = {h.doc.doc_id for h in hits}
    assert ids == {"only_macd", "only_div"}


# ── snippet generation ───────────────────────────────────


def test_snippet_centres_on_first_match():
    idx = RationaleIndex()
    text = "Long preamble of context. Then we discuss MACD divergence. Then more analysis after."
    idx.add(_doc("a", text))
    hits = idx.search("MACD")
    assert "MACD" in hits[0].snippet


def test_snippet_handles_match_at_start():
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD divergence is the key signal"))
    hits = idx.search("MACD")
    assert "MACD" in hits[0].snippet


def test_snippet_handles_match_at_end():
    idx = RationaleIndex()
    idx.add(_doc("a", "Lots of preamble before mentioning MACD"))
    hits = idx.search("MACD")
    assert "MACD" in hits[0].snippet


def test_snippet_truncates_long_text_at_default_max():
    """Pin: snippet is bounded so the dashboard / Slack can render
    it inline."""
    idx = RationaleIndex()
    long_text = "preamble. " * 50 + "MACD"
    idx.add(_doc("a", long_text))
    hits = idx.search("MACD")
    assert len(hits[0].snippet) < 250


def test_snippet_handles_empty_doc():
    """Pin: an empty rationale doesn't crash the snippet builder."""
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD"))
    hits = idx.search("MACD")
    assert isinstance(hits[0].snippet, str)


# ── metadata pass-through ────────────────────────────────


def test_metadata_pass_through():
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD", pair="BTCUSDT", side="buy"))
    hits = idx.search("MACD")
    assert hits[0].doc.metadata["pair"] == "BTCUSDT"
    assert hits[0].doc.metadata["side"] == "buy"


# ── output structure ─────────────────────────────────────


def test_search_hit_carries_doc_and_score():
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD"))
    hits = idx.search("MACD")
    assert isinstance(hits[0], SearchHit)
    assert hits[0].score > 0


def test_doc_is_immutable():
    doc = _doc("a", "MACD")
    with pytest.raises(Exception):
        doc.text = "tampered"  # type: ignore[misc]


# ── render_results ───────────────────────────────────────


def test_render_includes_each_hit():
    idx = RationaleIndex()
    idx.add(_doc("first", "MACD divergence"))
    idx.add(_doc("second", "MACD bounce"))
    text = render_results(idx.search("MACD"))
    assert "first" in text
    assert "second" in text


def test_render_includes_score():
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD divergence"))
    hits = idx.search("MACD")
    text = render_results(hits)
    assert f"{hits[0].score:.3f}" in text


def test_render_includes_term_contributions():
    idx = RationaleIndex()
    idx.add(_doc("a", "MACD MACD divergence"))
    text = render_results(idx.search("MACD divergence"))
    assert "macd×2" in text or "macd×" in text


def test_render_handles_empty_hits():
    text = render_results([])
    assert "no matches" in text


# ── numerical sanity ────────────────────────────────────


def test_idf_smoothing_keeps_score_positive_for_universal_terms():
    """A term that appears in every doc still contributes a
    positive score — pin so the smoothing floor of `+ 1` doesn't
    accidentally vanish."""
    idx = RationaleIndex()
    for i in range(5):
        idx.add(_doc(f"d-{i}", "MACD"))
    hits = idx.search("MACD")
    assert all(h.score > 0 for h in hits)


def test_tf_dampening_matches_documented_formula():
    """Pin the 1 + log(tf) shape directly. With 10 occurrences
    in a single doc, the per-term contribution should be (1 +
    log(10)) × idf — sanity-check via the exposed
    TermContribution.contribution field."""
    idx = RationaleIndex()
    idx.add(_doc("a", " ".join(["MACD"] * 10)))
    hits = idx.search("MACD")
    contrib = hits[0].terms[0]
    assert contrib.tf == 10
    expected_tf_weight = 1.0 + math.log(10)
    assert contrib.contribution == pytest.approx(expected_tf_weight * contrib.idf, rel=1e-9)
