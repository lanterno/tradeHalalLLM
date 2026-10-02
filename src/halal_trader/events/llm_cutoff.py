"""Where the LLM's knowledge stops, measured rather than taken on trust (Phase D).

The model is asked for reported quarterly EPS figures whose true values
(and filing dates) are in ``eps_facts``. Accuracy by filing month falls
from "knows it" to "guesses" at the training cutoff; only news after that
month is honest evidence for LLM-scored signals (roadmap §5).

A guess can land close by luck (EPS is persistent), so the comparison is
accuracy over time, not any single answer: the cutoff is the last month
whose accuracy stays clearly above the guess rate of the months after.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

SYSTEM = (
    "You answer questions about historical company financial results from memory. "
    'Reply with JSON only: {"eps": number or null}. Use null if you do not know; '
    "do not estimate."
)
TOLERANCE = 0.02  # within 2% (or $0.01) counts as known


@dataclass(frozen=True, slots=True)
class Probe:
    company: str
    quarter_end: date
    filed: date
    eps: float


@dataclass(frozen=True, slots=True)
class MonthAccuracy:
    month: date
    asked: int
    answered: int  # not null
    correct: int


async def probes(
    engine: AsyncEngine, *, start: date, end: date, per_month: int = 10, seed: int = 7
) -> list[Probe]:
    """Quarterly diluted EPS of large, well-known filers, sampled per filing month."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    """
                    WITH big AS (
                        SELECT DISTINCT ON (cik) cik, symbol FROM halal_screen_results
                        WHERE cik IS NOT NULL AND (metrics->>'market_cap')::float > 5e10
                        ORDER BY cik, as_of DESC
                    ), firsts AS (
                        SELECT DISTINCT ON (f.cik, f."end") f.cik, f."end", f.filed, f.val
                        FROM eps_facts f
                        WHERE f.concept = 'EarningsPerShareDiluted' AND f.form = '10-Q'
                          AND f."end" - f.start BETWEEN 80 AND 100
                          AND f.filed BETWEEN :start AND :end
                        ORDER BY f.cik, f."end", f.filed
                    )
                    SELECT b.symbol, x."end", x.filed, x.val FROM firsts x
                    JOIN big b ON b.cik = x.cik
                    """
                ),
                {"start": start, "end": end},
            )
        ).all()
    by_month: dict[date, list[Probe]] = defaultdict(list)
    for r in rows:
        by_month[r.filed.replace(day=1)].append(Probe(r.symbol, r.end, r.filed, float(r.val)))
    rng = random.Random(seed)
    out: list[Probe] = []
    for month in sorted(by_month):
        pool = sorted(by_month[month], key=lambda p: (p.company, p.quarter_end))
        out += rng.sample(pool, min(per_month, len(pool)))
    return out


def correct(answer: Any, truth: float) -> bool:
    try:
        value = float(answer)
    except TypeError, ValueError:
        return False
    return abs(value - truth) <= max(TOLERANCE * abs(truth), 0.01)


async def run(llm: Any, items: list[Probe]) -> list[MonthAccuracy]:
    stats: dict[date, list[int]] = defaultdict(lambda: [0, 0, 0])
    for p in items:
        prompt = (
            f"What diluted EPS (US$ per share, GAAP) did the company with ticker "
            f"{p.company} report for its fiscal quarter ended {p.quarter_end:%Y-%m-%d}?"
        )
        try:
            raw = await llm.generate_json(prompt, system=SYSTEM)
        except Exception as exc:  # noqa: BLE001 -- one failed probe is one unanswered probe
            logger.debug("cutoff probe failed: %r", exc)
            raw = {}
        answer = raw.get("eps") if isinstance(raw, dict) else None
        s = stats[p.filed.replace(day=1)]
        s[0] += 1
        s[1] += answer is not None
        s[2] += correct(answer, p.eps)
    return [MonthAccuracy(m, *v) for m, v in sorted(stats.items())]
