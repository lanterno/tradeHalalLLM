"""Batch LLM scoring: post-cutoff only, roundups skipped, malformed replies dropped."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.llm_score import parse_scores, score_all, scorer_id
from halal_trader.events.store import EventRecord, EventRecorder


def test_parse_scores_keeps_only_well_formed_in_range_entries() -> None:
    raw = {"s": {"0": 0.4, "1": "-0.7", "2": 3.0, "x": 0.1, "9": 0.2}}
    assert parse_scores(raw, 3) == {0: 0.4, 1: -0.7}
    assert parse_scores({"oops": 1}, 3) == {}


class _LLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_json(self, prompt: str, system: str | None = None):
        self.prompts.append(prompt)
        lines = prompt.splitlines()
        # Answer all but the last item, to prove a skipped pair is not re-asked forever.
        return {"s": {str(i): 0.5 for i in range(len(lines) - 1)}}


async def test_scoring_covers_post_cutoff_company_news_once(engine: AsyncEngine) -> None:
    new, old = datetime(2026, 3, 2, 15, tzinfo=UTC), datetime(2025, 6, 2, 15, tzinfo=UTC)

    def rec(i: int, when: datetime, symbols: list[str], headline: str) -> EventRecord:
        return EventRecord(
            "alpaca",
            f"alpaca:{i}",
            "news",
            symbols[0],
            when,
            when,
            {"headline": headline, "symbols": symbols},
        )

    await EventRecorder(engine).record(
        [
            rec(1, new, ["AAA"], "AAA wins contract"),
            rec(2, new, ["BBB"], "BBB recalls product"),
            rec(3, old, ["CCC"], "CCC beats"),  # before the cutoff
            rec(4, new, ["A", "B", "C", "D"], "Top movers"),  # a roundup
        ]
    )
    llm = _LLM()
    assert await score_all(llm, engine, model="m", max_pairs=100) == 1
    assert len(llm.prompts) == 1  # the skipped pair was not asked again
    assert "CCC" not in llm.prompts[0] and "Top movers" not in llm.prompts[0]
    async with engine.connect() as conn:
        n = (
            await conn.execute(
                text("SELECT count(*) FROM event_scores WHERE scorer = :s"), {"s": scorer_id("m")}
            )
        ).scalar()
    assert n == 1
