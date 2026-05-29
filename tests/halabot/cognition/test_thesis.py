"""Sparse LLM thesis writer, gate, and headline scorer."""

from __future__ import annotations

import pytest

from halabot.belief.schema import BeliefState, Direction, EvidenceItem, Levels, Regime
from halabot.cognition.thesis import (
    LlmGate,
    LlmHeadlineScorer,
    LlmThesisWriter,
    _parse_polarity,
)


class _FakeLLM:
    def __init__(self, reply="thesis", *, breaker=False):
        self.reply = reply
        self.prompts: list[str] = []
        self._breaker = breaker

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.reply

    def breaker_open(self) -> bool:
        return self._breaker


def _belief() -> BeliefState:
    return BeliefState(
        asset="NVDA", regime=Regime.TRENDING_UP, direction=Direction.LONG_BIAS,
        conviction=0.7, levels=Levels(invalidation=95.0),
        evidence=[EvidenceItem(source="indicator.momentum", direction=0.8, weight=1.0)],
    )


@pytest.mark.asyncio
async def test_thesis_writer_prompts_and_truncates():
    llm = _FakeLLM(reply="x" * 1000)
    w = LlmThesisWriter(llm, max_chars=50)
    out = await w.write(_belief())
    assert len(out) == 50
    assert "NVDA" in llm.prompts[0]  # belief context reached the LLM


def test_llm_gate_reports_breaker():
    assert LlmGate(_FakeLLM(breaker=False)).breaker_open() is False
    assert LlmGate(_FakeLLM(breaker=True)).breaker_open() is True
    assert LlmGate(None).available() is False


@pytest.mark.asyncio
async def test_headline_scorer_parses_number():
    assert await LlmHeadlineScorer(_FakeLLM(reply="0.8")).score("beat earnings") == 0.8
    assert await LlmHeadlineScorer(_FakeLLM(reply="-1.5")).score("fraud") == -1.0  # clamped
    assert await LlmHeadlineScorer(_FakeLLM(reply="no idea")).score("???") is None


@pytest.mark.asyncio
async def test_headline_scorer_parses_json_and_passes_ticker():
    # The OpenAI backend runs in json_object mode, so the scorer asks for JSON.
    llm = _FakeLLM(reply='{"polarity": 0.6}')
    assert await llm_score(llm, "merger talks", asset="NVDA", summary="rumored deal") == 0.6
    # The prompt must carry the ticker + summary (impact is ticker-specific).
    assert "NVDA" in llm.prompts[0] and "rumored deal" in llm.prompts[0]


async def llm_score(llm, headline, **kw):
    return await LlmHeadlineScorer(llm).score(headline, **kw)


def test_parse_polarity_edge_cases():
    assert _parse_polarity("the score is +0.5 today") == 0.5
    assert _parse_polarity("") is None
    assert _parse_polarity("2") == 1.0  # clamped to [-1, 1]
    # JSON forms (the backend's json_object mode) parse first.
    assert _parse_polarity('{"polarity": -0.7}') == -0.7
    assert _parse_polarity('{"score": 1.0}') == 1.0
    assert _parse_polarity('{"polarity": 5}') == 1.0  # clamped
