"""Does a tech-industry expert read tech headlines better? (pre-registered 2026-10-09)

The operator's strategy turns to technology (software, hardware and
semiconductors) with the LLM as a tech-industry expert. The generic batch
score showed no edge on post-cutoff news (events/llm_eval.py), so before
any of it trades, two expert scorers are measured against it on the **same
events** (post-cutoff tech headlines the generic scorer already scored):

* **expert** -- a tech-industry analyst reading the headline alone: does
  expertise by itself change anything?
* **context** -- the same analyst, given for each headline what an analyst
  would have open at that moment, all published before it: the company's
  industry, its latest earnings result and guidance against consensus, and
  its 20-session return against SPUS to the previous close.

Pre-registered rule (written before any score was seen): the context scorer
adds signal only if its 5-day IC is positive with t >= 2 and exceeds the
generic score's 5-day IC on the same readings by at least 0.02. Horizons,
day collapsing and returns are llm_eval's, unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.llm_score import GENERIC, Variant
from halal_trader.events.study import Observation, StudyResult, evaluate, summarise
from halal_trader.market_hours import MARKET_TZ

HORIZONS = (1, 5, 20)
MOMENTUM_SESSIONS = 20
BENCHMARK = "SPUS"

_REPLY = (
    "Reply with JSON only: "
    '{"s": {"<id>": <score>, ...}} with one entry per id, scores rounded to 1 decimal.'
)
EXPERT_SYSTEM = (
    "You are a senior technology-sector equity analyst: software, internet "
    "platforms, semiconductors and chip equipment, and computer and networking "
    "hardware. You know the industry's economics: product and platform cycles, "
    "the semiconductor and memory cycles, hyperscaler and enterprise capex, "
    "pricing power and competition, and how news at one company moves its "
    "suppliers, customers and rivals. For each item 'id|SYMBOL|headline', "
    "estimate the direction and size of the move this headline implies for "
    "SYMBOL over the next few trading days, as that analyst would: -1 = strongly "
    "negative, 0 = no information (old, routine, market-wide, already expected, "
    "or about another company with no read-through), +1 = strongly positive. "
    "Judge only from the text and your understanding of the industry; do not use "
    "any knowledge of what happened afterwards. " + _REPLY
)
CONTEXT_SYSTEM = (
    "You are a senior technology-sector equity analyst: software, internet "
    "platforms, semiconductors and chip equipment, and computer and networking "
    "hardware. You know the industry's economics: product and platform cycles, "
    "the semiconductor and memory cycles, hyperscaler and enterprise capex, "
    "pricing power and competition, and how news at one company moves its "
    "suppliers, customers and rivals. Each item is 'id|SYMBOL|context|headline'; "
    "the context is what you had open when the headline appeared: the company's "
    "industry, its latest earnings result and guidance against consensus, and "
    "its stock's 20-day return against the halal index fund SPUS. Weigh the "
    "headline against it: news the last report already implied, or a move the "
    "stock has already made, is worth less. Estimate the direction and size of "
    "the move this headline implies for SYMBOL over the next few trading days: "
    "-1 = strongly negative, 0 = no information, +1 = strongly positive. Judge "
    "only from the text, the context and your understanding of the industry; do "
    "not use any knowledge of what happened afterwards. " + _REPLY
)


async def tech_symbols(engine: AsyncEngine) -> frozenset[str]:
    """Every symbol the newest screen puts in Technology, whatever its verdict."""
    from halal_trader.halal import strict
    from halal_trader.halal.sector_limits import TECHNOLOGY, cap_sector

    as_of = await strict.newest_screen(engine)
    if as_of is None:
        return frozenset()
    return frozenset(
        r.symbol
        for r in await strict.screen_rows(engine, as_of)
        if cap_sector(r.symbol, r.sic_description) == TECHNOLOGY
    )


def _pct(v: Any) -> str:
    try:
        return f"{float(v):+.0%}"
    except TypeError, ValueError:
        return "n/a"


def _result_text(f: dict[str, Any], at: datetime) -> str:
    return (
        f"last report {at:%Y-%m-%d} ({f.get('period') or '?'}): EPS {f.get('eps_verdict') or '?'} "
        f"{_pct(f.get('eps_surprise'))}, sales {f.get('sales_verdict') or '?'} "
        f"{_pct(f.get('sales_surprise'))}"
    )


def _guidance_text(f: dict[str, Any]) -> str:
    return (
        f"guidance {f.get('action') or 'given'} {f.get('metric') or ''} "
        f"{f.get('period') or ''} {_pct(f.get('surprise'))} vs consensus"
    ).replace("  ", " ")


async def tech_context(
    engine: AsyncEngine, items: list[tuple[str, datetime]]
) -> dict[tuple[str, datetime], str]:
    """For each (symbol, moment): what was public before that moment."""
    from halal_trader.events.earnings_parse import EXTRACTOR
    from halal_trader.halal import strict

    if not items:
        return {}
    symbols = sorted({s for s, _ in items})
    lo = min(t for _, t in items)
    hi = max(t for _, t in items)
    as_of = await strict.newest_screen(engine)
    industry = {
        r.symbol: (r.sic_description or "unknown industry").lower()
        for r in (await strict.screen_rows(engine, as_of, symbols=symbols) if as_of else [])
    }
    facts: dict[str, list[tuple[datetime, str, dict[str, Any]]]] = defaultdict(list)
    closes: dict[str, dict[date, float]] = defaultdict(dict)
    async with engine.connect() as conn:
        for r in await conn.execute(
            text(
                "SELECT e.symbol, e.published_at, f.kind, f.fields FROM event_facts f "
                "JOIN events e ON e.id = f.event_id WHERE f.extractor = :x "
                "AND f.kind IN ('result', 'guidance') AND e.symbol = ANY(:s) "
                "AND e.published_at >= :lo AND e.published_at < :hi"
            ),
            {"x": EXTRACTOR, "s": symbols, "lo": lo - timedelta(days=200), "hi": hi},
        ):
            facts[r.symbol].append((r.published_at, r.kind, dict(r.fields)))
        for r in await conn.execute(
            text(
                "SELECT symbol, day, close FROM daily_bars WHERE adjustment = 'all' "
                "AND symbol = ANY(:s) AND day >= :lo AND day < :hi"
            ),
            {
                "s": [*symbols, BENCHMARK],
                "lo": lo.astimezone(MARKET_TZ).date() - timedelta(days=45),
                "hi": hi.astimezone(MARKET_TZ).date(),
            },
        ):
            closes[r.symbol][r.day] = float(r.close)
    for rows in facts.values():
        rows.sort(key=lambda x: x[0])
    sessions = sorted(closes.get(BENCHMARK, {}))

    def momentum(symbol: str, at: datetime) -> str:
        before = [d for d in sessions if d < at.astimezone(MARKET_TZ).date()]
        if len(before) <= MOMENTUM_SESSIONS:
            return "20d vs SPUS n/a"
        start, end = before[-MOMENTUM_SESSIONS - 1], before[-1]
        own, bench = closes.get(symbol, {}), closes[BENCHMARK]
        if start not in own or end not in own:
            return "20d vs SPUS n/a"
        rel = (own[end] / own[start]) - (bench[end] / bench[start])
        return f"20d vs SPUS {rel:+.1%}"

    out = {}
    for symbol, at in items:
        parts = [industry.get(symbol, "unknown industry")]
        prior = [x for x in facts.get(symbol, []) if x[0] < at]
        result = next((x for x in reversed(prior) if x[1] == "result"), None)
        guidance = next((x for x in reversed(prior) if x[1] == "guidance"), None)
        parts.append(_result_text(result[2], result[0]) if result else "no earnings result on file")
        if guidance is not None:
            parts.append(_guidance_text(guidance[2]))
        parts.append(momentum(symbol, at))
        out[(symbol, at)] = "; ".join(parts)
    return out


def _polarity(headline: str) -> float:
    from halal_trader.sentiment.headline_polarity import score_headline

    pos, neg = score_headline(headline)
    return pos - neg


def variants(symbols: frozenset[str]) -> dict[str, Variant]:
    return {
        "expert": Variant("llm-tech", EXPERT_SYSTEM, symbols=symbols, paired_with=GENERIC.name),
        "context": Variant(
            "llm-techctx",
            CONTEXT_SYSTEM,
            symbols=symbols,
            context=tech_context,
            paired_with=GENERIC.name,
        ),
    }


@dataclass(frozen=True, slots=True)
class TechComparison:
    days: int  # (symbol, day) readings every scorer has
    results: dict[str, StudyResult]  # "generic", "expert", "context", "lexicon"


async def compare(engine: AsyncEngine, model: str) -> TechComparison:
    """Each scorer on the (symbol, day) readings all of them scored, as llm_eval reads them."""
    from halal_trader.events.llm_score import scorer_id

    symbols = await tech_symbols(engine)
    ids = {"generic": scorer_id(model)} | {
        k: scorer_id(model, v) for k, v in variants(symbols).items()
    }
    name = {v: k for k, v in ids.items()}
    scores: dict[int, dict[str, float]] = defaultdict(dict)
    meta: dict[int, tuple[str, datetime, str]] = {}
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT s.scorer, e.id, e.symbol, e.published_at, e.payload->>'headline' AS h, "
                "s.score FROM event_scores s JOIN events e ON e.id = s.event_id "
                "WHERE s.scorer = ANY(:ids) AND e.kind = 'news' AND e.symbol = ANY(:syms)"
            ),
            {"ids": list(ids.values()), "syms": sorted(symbols)},
        )
        for r in rows:
            scores[r.id][name[r.scorer]] = float(r.score)
            meta[r.id] = (r.symbol, r.published_at, r.h or "")
    # Only events every scorer read; then one reading per (symbol, day), timed at
    # its last headline, as llm_eval does.
    days: dict[tuple[str, date], list[int]] = defaultdict(list)
    for event_id, by in scores.items():
        if len(by) == len(ids):
            symbol, at, _ = meta[event_id]
            days[(symbol, at.astimezone(MARKET_TZ).date())].append(event_id)

    def signal(label: str, event_ids: list[int]) -> float:
        if label == "lexicon":
            vals = [_polarity(meta[i][2]) for i in event_ids]
        else:
            vals = [scores[i][label] for i in event_ids]
        return sum(vals) / len(vals)

    results = {}
    for label in [*ids, "lexicon"]:
        obs = [
            Observation(symbol, max(meta[i][1] for i in event_ids), signal(label, event_ids))
            for (symbol, _), event_ids in sorted(days.items())
        ]
        results[label] = summarise(await evaluate(engine, obs, HORIZONS))
    return TechComparison(len(days), results)
