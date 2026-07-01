"""FinBERT headline classifier — a local, free, LLM-outage-resilient
implementation of the reactor's :class:`HeadlineClassifier` Protocol.

The production classifier is GPT-4o-mini (:class:`GPTHeadlineClassifier`);
when the LLM chain is down or in backoff (as happened live 2026-06-25) the
reactor's "fast in" half goes dark. FinBERT (ProsusAI/finbert) runs locally on
the already-installed ``transformers`` stack, so it keeps scoring headlines for
free with no API dependency.

Sentiment → the reactor's bullish momentum-impact score: a positive headline
maps to its FinBERT confidence, negative/neutral map to 0.0 — the reactor only
acts on bullish momentum, so a bearish headline should not fire a buy. Degrades
gracefully: if ``transformers`` or the model can't load, ``classify`` returns
``score=0.0`` (no fire) and never raises.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from halal_trader.sentiment.stocks_events import HeadlineClassification

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "ProsusAI/finbert"
_MAX_CHARS = 512  # FinBERT truncates at 512 tokens; char-cap is a cheap guard


def _default_loader(model_name: str) -> Any:
    from transformers import pipeline

    return pipeline("sentiment-analysis", model=model_name, truncation=True)


def _parse(result: Any) -> tuple[str, float]:
    """Normalise a transformers sentiment pipeline result → (label, confidence)."""
    row = result[0] if isinstance(result, list) and result else result
    if not isinstance(row, dict):
        return "neutral", 0.0
    label = str(row.get("label", "neutral")).lower()
    try:
        conf = float(row.get("score", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return label, conf


class FinBERTHeadlineClassifier:
    """``HeadlineClassifier`` backed by a local FinBERT sentiment model."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        pipe: Any = None,
        loader: Callable[[str], Any] | None = None,
    ) -> None:
        self._model_name = model_name or _DEFAULT_MODEL
        self._pipeline = pipe  # injectable for tests / preloading
        self._loader = loader or _default_loader
        self._load_failed = False

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None or self._load_failed:
            return
        try:
            self._pipeline = self._loader(self._model_name)
        except Exception as exc:  # noqa: BLE001 — any load failure → degrade
            self._load_failed = True
            logger.warning(
                "FinBERT unavailable (%s) — headline classifier returns neutral", exc
            )

    async def classify(
        self, *, symbol: str, headline: str, summary: str = ""
    ) -> HeadlineClassification:
        self._ensure_pipeline()
        if self._pipeline is None:
            return HeadlineClassification(score=0.0, rationale="finbert unavailable")
        text = headline if not summary else f"{headline}. {summary}"
        try:
            # transformers pipelines are blocking → run off the event loop.
            result = await asyncio.to_thread(self._pipeline, text[:_MAX_CHARS])
        except Exception as exc:  # noqa: BLE001 — inference failure → no fire
            logger.debug("FinBERT classify failed for %s: %s", symbol, exc)
            return HeadlineClassification(score=0.0)
        label, conf = _parse(result)
        # Bullish momentum-impact only: positive → confidence, else 0.
        score = round(conf, 4) if label == "positive" else 0.0
        return HeadlineClassification(
            score=score, tag="sentiment", rationale=f"FinBERT {label} {conf:.2f}"
        )
