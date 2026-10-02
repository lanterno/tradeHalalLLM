"""Does the LLM add anything a free reader does not? (roadmap Phase D, step 4)

On post-cutoff news only, three readings of the same headlines are
compared as predictors of net abnormal returns (study.evaluate):

* **llm** -- the batch LLM score in [-1, 1];
* **lexicon** -- the stdlib keyword polarity already in the codebase
  (positive weight minus negative weight), free and instant;
* **llm | lexicon neutral** -- the LLM score where the lexicon sees
  nothing: the information only the LLM supplies.

Headlines are collapsed to one observation per (symbol, New York day):
the mean score, timed at the day's **last** headline, so the observation
is only acted on once everything it averages is public. A catalyst's
10-30 repackaged headlines then count once.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.study import Observation, StudyResult, evaluate, summarise
from halal_trader.sentiment.headline_polarity import score_headline

_ET = ZoneInfo("America/New_York")
HORIZONS = (1, 5, 20)


@dataclass(frozen=True, slots=True)
class DayReading:
    symbol: str
    day: date
    last_at: datetime
    llm: float
    lexicon: float


async def day_readings(engine: AsyncEngine, scorer_prefix: str = "llm-batch:") -> list[DayReading]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT e.symbol, e.published_at, e.payload->>'headline' AS h, s.score "
                "FROM event_scores s JOIN events e ON e.id = s.event_id "
                "WHERE s.scorer LIKE :p AND e.kind = 'news'"
            ),
            {"p": scorer_prefix + "%"},
        )
        grouped: dict[tuple[str, date], list[tuple[datetime, float, float]]] = defaultdict(list)
        for r in rows:
            pos, neg = score_headline(r.h or "")
            day = r.published_at.astimezone(_ET).date()
            grouped[(r.symbol, day)].append((r.published_at, float(r.score), pos - neg))
    out = []
    for (symbol, day), items in grouped.items():
        out.append(
            DayReading(
                symbol,
                day,
                max(t for t, _, _ in items),
                sum(s for _, s, _ in items) / len(items),
                sum(x for _, _, x in items) / len(items),
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class Comparison:
    llm: StudyResult
    lexicon: StudyResult
    llm_where_lexicon_neutral: StudyResult
    days: int


async def compare(engine: AsyncEngine) -> Comparison:
    readings = await day_readings(engine)

    def obs(signal: str, subset: list[DayReading]) -> list[Observation]:
        return [Observation(r.symbol, r.last_at, getattr(r, signal)) for r in subset]

    llm = await evaluate(engine, obs("llm", readings), HORIZONS)
    lex = await evaluate(engine, obs("lexicon", readings), HORIZONS)
    neutral = [r for r in readings if r.lexicon == 0]
    only = await evaluate(engine, obs("llm", neutral), HORIZONS)
    return Comparison(summarise(llm), summarise(lex), summarise(only), len(readings))
