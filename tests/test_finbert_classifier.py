"""Tests for the FinBERT headline classifier (mocked pipeline — no download)."""

from __future__ import annotations

from typing import Any

from halal_trader.sentiment.finbert_classifier import FinBERTHeadlineClassifier


def _pipe(label: str, score: float):
    """A fake transformers sentiment pipeline: callable → [{label, score}]."""

    def _call(text: str) -> list[dict[str, Any]]:
        return [{"label": label, "score": score}]

    return _call


async def test_positive_maps_to_bullish_score():
    clf = FinBERTHeadlineClassifier(pipe=_pipe("positive", 0.91))
    out = await clf.classify(symbol="NVDA", headline="NVDA lands record AI orders")
    assert out.score == 0.91
    assert out.tag == "sentiment"
    assert "positive" in out.rationale


async def test_negative_does_not_fire():
    clf = FinBERTHeadlineClassifier(pipe=_pipe("negative", 0.88))
    out = await clf.classify(symbol="AAPL", headline="AAPL faces antitrust suit")
    assert out.score == 0.0  # bearish news never fires the bullish reactor
    assert "negative" in out.rationale


async def test_neutral_does_not_fire():
    clf = FinBERTHeadlineClassifier(pipe=_pipe("neutral", 0.7))
    out = await clf.classify(symbol="MSFT", headline="MSFT to hold annual meeting")
    assert out.score == 0.0


async def test_degrades_when_model_unavailable():
    # loader raises (no transformers / offline / bad model) → neutral, no raise.
    def _bad_loader(_name: str):
        raise RuntimeError("model download blocked")

    clf = FinBERTHeadlineClassifier(loader=_bad_loader)
    out = await clf.classify(symbol="NVDA", headline="anything")
    assert out.score == 0.0
    assert "unavailable" in out.rationale


async def test_inference_error_returns_zero():
    def _boom(_text: str):
        raise ValueError("cuda oom")

    clf = FinBERTHeadlineClassifier(pipe=_boom)
    out = await clf.classify(symbol="NVDA", headline="x")
    assert out.score == 0.0


async def test_lazy_load_only_on_first_classify():
    calls = {"n": 0}

    def _loader(_name: str):
        calls["n"] += 1
        return _pipe("positive", 0.8)

    clf = FinBERTHeadlineClassifier(loader=_loader)
    assert calls["n"] == 0  # not loaded at construction
    await clf.classify(symbol="NVDA", headline="up")
    await clf.classify(symbol="NVDA", headline="up again")
    assert calls["n"] == 1  # loaded once, reused


async def test_summary_included_and_truncated():
    seen = {}

    def _capture(text: str):
        seen["text"] = text
        return [{"label": "positive", "score": 0.6}]

    clf = FinBERTHeadlineClassifier(pipe=_capture)
    await clf.classify(symbol="NVDA", headline="H" * 400, summary="S" * 400)
    assert len(seen["text"]) <= 512  # char-capped for the model
