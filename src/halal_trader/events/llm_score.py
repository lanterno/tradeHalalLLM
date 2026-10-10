"""Batch LLM scoring of stored headlines, post-cutoff only (roadmap Phase D).

Each (headline, symbol) pair gets a signed score in [-1, 1]: the direction
and size of the move the headline alone implies for that stock over the
next few days. Fifty pairs go in one call, the reply is terse JSON, and
every call is metered in the research pool (stopped by its monthly cap).

Only news published on or after ``POST_CUTOFF`` is scored: the model's
recall of reported EPS stops in 2025-09 (events/llm_cutoff.py), and two
months of margin are added. Earlier headlines are the model's training
data; their scores would measure memory, not judgement.

Multi-ticker roundups (more than 3 symbols on the article) are skipped
before any token is spent: they are market-wide, not company news.

A :class:`Variant` is one scorer: its system prompt (whose hash is in the
scorer id), and optionally a symbol filter, a context line per headline, and
``paired_with`` (score only events another scorer already scored, so two
scorers are compared on the same events). ``GENERIC`` is the original.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

POST_CUTOFF = datetime(2025, 12, 1, tzinfo=UTC)
BATCH = 50
MAX_SYMBOLS = 3

SYSTEM = (
    "You score stock-news headlines for a systematic trader. For each item "
    "'id|SYMBOL|headline', estimate the direction and size of the move this "
    "headline alone implies for SYMBOL over the next few trading days: -1 = "
    "strongly negative, 0 = no information (old, routine, market-wide, or about "
    "another company), +1 = strongly positive. Judge only from the text; do not "
    "use any knowledge of what happened afterwards. Reply with JSON only: "
    '{"s": {"<id>": <score>, ...}} with one entry per id, scores rounded to 1 decimal.'
)


Pair = tuple[str, str]  # (symbol, headline)
# (symbol, the headline's first publication) -> one line of context for it.
ContextFn = Callable[
    [AsyncEngine, list[tuple[str, datetime]]], Awaitable[dict[tuple[str, datetime], str]]
]


@dataclass(frozen=True)
class Variant:
    name: str  # the scorer id's prefix
    system: str
    symbols: frozenset[str] | None = None  # only these symbols' headlines
    context: ContextFn | None = None  # a context line in each item
    paired_with: str | None = None  # only events this scorer prefix already scored


GENERIC = Variant("llm-batch", SYSTEM)


def scorer_id(model: str, variant: Variant = GENERIC) -> str:
    return f"{variant.name}:{model}:{hashlib.sha1(variant.system.encode()).hexdigest()[:8]}"


@dataclass
class Pending:
    ids: list[int]
    first_at: datetime


async def pending(
    engine: AsyncEngine, scorer: str, limit: int, variant: Variant = GENERIC
) -> dict[Pair, Pending]:
    """(symbol, headline) -> event ids still unscored by ``scorer``, newest first."""
    where = ""
    params: dict[str, Any] = {"cut": POST_CUTOFF, "maxs": MAX_SYMBOLS, "sc": scorer, "n": limit}
    if variant.symbols is not None:
        where += " AND e.symbol = ANY(:syms)"
        params["syms"] = sorted(variant.symbols)
    if variant.paired_with is not None:
        where += (
            " AND EXISTS (SELECT 1 FROM event_scores p WHERE p.event_id = e.id "
            "AND p.scorer LIKE :paired)"
        )
        params["paired"] = variant.paired_with + ":%"
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT e.id, e.symbol, e.published_at, e.payload->>'headline' AS h FROM events e "
                "WHERE e.kind = 'news' AND e.published_at >= :cut "
                "AND jsonb_array_length(coalesce(e.payload->'symbols', '[]'::jsonb)) "
                "BETWEEN 1 AND :maxs "
                "AND NOT EXISTS (SELECT 1 FROM event_scores s "
                "WHERE s.event_id = e.id AND s.scorer = :sc)" + where + " "
                "ORDER BY e.published_at DESC LIMIT :n"
            ),
            params,
        )
        out: dict[Pair, Pending] = {}
        for r in rows:
            if r.h:
                got = out.setdefault((r.symbol, r.h), Pending([], r.published_at))
                got.ids.append(r.id)
                got.first_at = min(got.first_at, r.published_at)
        return out


def parse_scores(raw: Any, n: int) -> dict[int, float]:
    """{"s": {"0": 0.4, ...}} -> {0: 0.4}; anything malformed is dropped, never guessed."""
    table = raw.get("s") if isinstance(raw, dict) else None
    out: dict[int, float] = {}
    if not isinstance(table, dict):
        return out
    for key, value in table.items():
        try:
            i, v = int(key), float(value)
        except TypeError, ValueError:
            continue
        if 0 <= i < n and -1.0 <= v <= 1.0:
            out[i] = v
    return out


async def score_batch(
    llm: Any,
    engine: AsyncEngine,
    scorer: str,
    items: list[tuple[Pair, Pending]],
    variant: Variant = GENERIC,
) -> int:
    contexts: dict[tuple[str, datetime], str] = {}
    if variant.context is not None:
        contexts = await variant.context(engine, [(sym, p.first_at) for (sym, _), p in items])

    def line(i: int, sym: str, headline: str, first_at: datetime) -> str:
        # One item per line: a headline's own line breaks would split it in two.
        head = " ".join(headline.split())[:200].replace("|", "/")
        if variant.context is None:
            return f"{i}|{sym}|{head}"
        ctx = " ".join(contexts.get((sym, first_at), "no context").split()).replace("|", "/")
        return f"{i}|{sym}|{ctx}|{head}"

    lines = "\n".join(line(i, sym, h, p.first_at) for i, ((sym, h), p) in enumerate(items))
    raw = await llm.generate_json(lines, system=variant.system)
    scores = parse_scores(raw, len(items))
    rows = [
        {"e": event_id, "s": scorer, "v": scores[i], "at": datetime.now(UTC)}
        for i, (_, p) in enumerate(items)
        if i in scores
        for event_id in p.ids
    ]
    if rows:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO event_scores (event_id, scorer, score, scored_at) "
                    "VALUES (:e, :s, :v, :at) ON CONFLICT ON CONSTRAINT "
                    "uq_event_scores_event_scorer DO NOTHING"
                ),
                rows,
            )
    return len(scores)


async def score_all(
    llm: Any,
    engine: AsyncEngine,
    *,
    model: str,
    max_pairs: int,
    concurrency: int = 4,
    variant: Variant = GENERIC,
) -> int:
    """Score up to ``max_pairs`` unscored post-cutoff pairs; stops early on the budget cap."""
    from halal_trader.core.llm.spend import BudgetExhausted

    scorer = scorer_id(model, variant)
    done = 0
    tried: set[tuple[str, str]] = set()  # a pair the model skipped is not asked again this run
    results: list[Any] = []
    while done < max_pairs:
        fresh = await pending(engine, scorer, BATCH * concurrency * 4 + len(tried), variant)
        todo = [(k, v) for k, v in fresh.items() if k not in tried][: BATCH * concurrency * 4]
        if not todo:
            break
        tried.update(k for k, _ in todo)
        batches = [todo[i : i + BATCH] for i in range(0, len(todo), BATCH)]
        for group in (batches[i : i + concurrency] for i in range(0, len(batches), concurrency)):
            results = await asyncio.gather(
                *(score_batch(llm, engine, scorer, b, variant) for b in group),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, BudgetExhausted):
                    logger.warning("research LLM budget reached: stopping")
                    return done
                if isinstance(r, Exception):
                    logger.warning("scoring batch failed: %r", r)
                else:
                    done += r
            if done >= max_pairs:
                break
        logger.info("llm scoring: %d pairs scored", done)
        if results and all(isinstance(r, Exception) for r in results):
            break  # every batch failing: stop rather than spin
    return done
