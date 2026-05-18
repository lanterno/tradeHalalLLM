"""Tests for :func:`build_rag_query`.

The hashing embedder + cosine + format_rag_for_prompt are covered in
`test_rag.py`. This file pins the query builder — what gets fed to the
embedder each cycle to retrieve similar past setups.
"""

from __future__ import annotations

from halal_trader.core.llm.rag import build_rag_query

# ── Empty / edge paths ──────────────────────────────────────


def test_empty_inputs_returns_empty_string():
    out = build_rag_query(indicators_cache={}, sentiment_text="", regime_text="")
    assert out == ""


def test_skips_pairs_with_indicator_error():
    """Symbols whose bars failed parse don't contribute to the query."""
    out = build_rag_query(
        indicators_cache={"BTCUSDT": {"error": "insufficient data"}},
        sentiment_text="",
        regime_text="",
    )
    assert out == ""


def test_skips_pairs_with_empty_indicators_dict():
    """Empty per-pair dict is falsy → skipped entirely (matches the
    same gate as `error in inds`); avoids polluting the embed query
    with a bare pair name carrying no signal labels."""
    out = build_rag_query(
        indicators_cache={"BTCUSDT": {}},
        sentiment_text="",
        regime_text="",
    )
    assert out == ""


# ── RSI labelling ───────────────────────────────────────────


def test_rsi_below_35_labels_oversold():
    out = build_rag_query(
        indicators_cache={"BTCUSDT": {"rsi_14": 25}},
        sentiment_text="",
        regime_text="",
    )
    assert "rsi oversold" in out


def test_rsi_above_65_labels_overbought():
    out = build_rag_query(
        indicators_cache={"BTCUSDT": {"rsi_14": 75}},
        sentiment_text="",
        regime_text="",
    )
    assert "rsi overbought" in out


def test_rsi_in_band_labels_neutral():
    """35..65 RSI is the "no thesis" zone — labelled neutral."""
    out = build_rag_query(
        indicators_cache={"BTCUSDT": {"rsi_14": 50}},
        sentiment_text="",
        regime_text="",
    )
    assert "rsi neutral" in out


# ── MACD labelling ──────────────────────────────────────────


def test_macd_positive_labels_bullish():
    out = build_rag_query(
        indicators_cache={"BTCUSDT": {"macd_histogram": 0.5}},
        sentiment_text="",
        regime_text="",
    )
    assert "macd bullish" in out


def test_macd_negative_labels_bearish():
    out = build_rag_query(
        indicators_cache={"BTCUSDT": {"macd_histogram": -0.3}},
        sentiment_text="",
        regime_text="",
    )
    assert "macd bearish" in out


# ── BB position labelling ───────────────────────────────────


def test_bb_below_02_labels_lower():
    out = build_rag_query(
        indicators_cache={"BTCUSDT": {"bb_position": 0.1}},
        sentiment_text="",
        regime_text="",
    )
    assert "bb lower" in out


def test_bb_above_08_labels_upper():
    out = build_rag_query(
        indicators_cache={"BTCUSDT": {"bb_position": 0.9}},
        sentiment_text="",
        regime_text="",
    )
    assert "bb upper" in out


def test_bb_in_band_emits_no_label():
    """Middle of the bands → no label (avoids polluting the embedding)."""
    out = build_rag_query(
        indicators_cache={"BTCUSDT": {"bb_position": 0.5}},
        sentiment_text="",
        regime_text="",
    )
    assert "bb lower" not in out
    assert "bb upper" not in out


# ── Regime + sentiment text ─────────────────────────────────


def test_regime_text_appended_capped_at_80_chars():
    long_regime = "x" * 200
    out = build_rag_query(
        indicators_cache={},
        sentiment_text="",
        regime_text=long_regime,
    )
    # Only the first 80 chars of the regime should appear.
    assert "x" * 80 in out
    assert "x" * 81 not in out


def test_sentiment_text_appended_capped_at_80_chars():
    long_sent = "y" * 200
    out = build_rag_query(
        indicators_cache={},
        sentiment_text=long_sent,
        regime_text="",
    )
    assert "y" * 80 in out
    assert "y" * 81 not in out


# ── Full-output cap ─────────────────────────────────────────


def test_total_output_capped_at_600_chars():
    """Bound retrieval-time cost — even with extreme inputs."""
    cache = {
        f"PAIR{i}USDT": {"rsi_14": 50, "macd_histogram": 0.1, "bb_position": 0.5} for i in range(50)
    }
    out = build_rag_query(
        indicators_cache=cache,
        sentiment_text="z" * 500,
        regime_text="w" * 500,
    )
    assert len(out) <= 600


# ── Pair separator ──────────────────────────────────────────


def test_multiple_pairs_separated_by_pipe():
    """The ` | ` separator keeps the embedder seeing distinct setups."""
    out = build_rag_query(
        indicators_cache={
            "BTCUSDT": {"rsi_14": 30},
            "ETHUSDT": {"rsi_14": 70},
        },
        sentiment_text="",
        regime_text="",
    )
    assert " | " in out
    assert "BTCUSDT" in out
    assert "ETHUSDT" in out
