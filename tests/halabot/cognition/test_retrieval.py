"""SetupRetriever — retrieval-grounded thesis context (Task B slice 2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from halabot.belief.schema import BeliefState, Direction, EvidenceItem
from halabot.cognition.retrieval import SetupRetriever, _query_text
from halabot.cognition.thesis import LlmThesisWriter

T0 = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)


@dataclass
class _Row:
    symbol: str
    text: str
    outcome_win: bool
    outcome_pnl_pct: float


class _FakeStore:
    def __init__(self, hits: list[tuple[Any, float]] | None = None, *, explode: bool = False):
        self.hits = hits or []
        self.explode = explode
        self.queries: list[str] = []

    async def query(self, text: str, *, k: int = 5, min_similarity: float = 0.1,
                    symbol: str | None = None) -> list[tuple[Any, float]]:
        if self.explode:
            raise RuntimeError("pg down")
        self.queries.append(text)
        return self.hits

    async def aggregate(self, hits: Any) -> dict[str, Any]:
        hits = list(hits)
        return {"n": len(hits), "weighted_pnl_pct": 0.011, "weighted_win_rate": 0.66}


def _belief(asset: str = "NVDA") -> BeliefState:
    b = BeliefState.neutral(asset)
    b.direction = Direction.LONG_BIAS
    b.evidence = [
        EvidenceItem(source="rsi", direction=0.8, weight=1.0, ts=T0),
        EvidenceItem(source="news", direction=-0.2, weight=0.5, ts=T0),
    ]
    return b


def test_query_text_is_deterministic_and_vocabulary_matched():
    b = _belief()
    q1, q2 = _query_text(b), _query_text(b)
    assert q1 == q2
    assert "NVDA" in q1 and "bullish" in q1  # direction + top evidence sign
    assert "rsi" in q1 and "news" in q1  # sources in corpus vocabulary


@pytest.mark.asyncio
async def test_context_renders_hits_and_weighted_stats():
    hits = [
        (_Row("NVDA", "rsi oversold bounce with momentum", True, 0.021), 0.61),
        (_Row("AMD", "macd bullish cross faded", False, -0.008), 0.44),
    ]
    out = await SetupRetriever(_FakeStore(hits)).context_for(_belief())
    assert "Similar past setups" in out
    assert "weighted win-rate 66%" in out
    assert "WIN +2.10% (NVDA)" in out
    assert "LOSS -0.80% (AMD)" in out
    assert all(len(line) < 120 for line in out.split("\n"))


@pytest.mark.asyncio
async def test_empty_hits_yield_empty_context():
    assert await SetupRetriever(_FakeStore([])).context_for(_belief()) == ""


@pytest.mark.asyncio
async def test_store_failure_degrades_to_empty_never_raises():
    assert await SetupRetriever(_FakeStore(explode=True)).context_for(_belief()) == ""


# ── thesis writer integration ──────────────────────────────────


class _CaptureLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return "grounded thesis"


@pytest.mark.asyncio
async def test_thesis_prompt_carries_grounding_block():
    llm = _CaptureLLM()
    hits = [(_Row("NVDA", "trend continuation held", True, 0.015), 0.7)]
    writer = LlmThesisWriter(llm, retriever=SetupRetriever(_FakeStore(hits)))
    out = await writer.write(_belief())
    assert out == "grounded thesis"
    assert "Similar past setups" in llm.prompts[0]
    assert llm.prompts[0].startswith("Asset NVDA")  # belief prompt still leads


@pytest.mark.asyncio
async def test_thesis_without_retriever_unchanged():
    llm = _CaptureLLM()
    await LlmThesisWriter(llm).write(_belief())
    assert "Similar past setups" not in llm.prompts[0]


@pytest.mark.asyncio
async def test_retriever_failure_still_writes_thesis():
    llm = _CaptureLLM()
    writer = LlmThesisWriter(llm, retriever=SetupRetriever(_FakeStore(explode=True)))
    out = await writer.write(_belief())
    assert out == "grounded thesis"
    assert "Similar past setups" not in llm.prompts[0]


def test_retrieval_flag_defaults_off():
    from halabot.platform.config import CognitionSettings

    assert CognitionSettings().retrieval_enabled is False
