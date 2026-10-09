"""The tech-expert scoring test: point-in-time context, same events as the generic scorer."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.earnings_parse import EXTRACTOR
from halal_trader.events.llm_score import GENERIC, score_all, scorer_id
from halal_trader.events.store import EventRecord, EventRecorder
from halal_trader.events.tech_expert import tech_context, tech_symbols, variants

MODEL = "z-ai/glm-5.2"
HEADLINE_AT = datetime(2026, 3, 2, 15, tzinfo=UTC)


async def _screen(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES "
                "('2026-03-01', 'NVDA', 1, 'SEMICONDUCTORS & RELATED DEVICES', 'halal', '[]', "
                "'{}', 'v12', now()), "
                "('2026-03-01', 'XOM', 2, 'PETROLEUM REFINING', 'halal', '[]', '{}', 'v12', now())"
            )
        )


async def _fact(engine: AsyncEngine, sid: str, at: datetime, kind: str, fields: str) -> None:
    await EventRecorder(engine).record(
        [EventRecord("alpaca", sid, "news", "NVDA", at, at, {"headline": sid})]
    )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO event_facts (event_id, extractor, kind, fields) "
                "SELECT id, :x, :k, CAST(:f AS JSONB) FROM events WHERE source_id = :s"
            ),
            {"x": EXTRACTOR, "k": kind, "f": fields, "s": sid},
        )


async def test_the_context_holds_only_what_was_public_before_the_headline(
    engine: AsyncEngine,
) -> None:
    await _screen(engine)
    await _fact(
        engine,
        "r1",
        HEADLINE_AT - timedelta(days=30),
        "result",
        '{"period": "Q4", "eps_verdict": "beats", "eps_surprise": 0.06, '
        '"sales_verdict": "beat", "sales_surprise": 0.04}',
    )
    await _fact(  # published after the headline: must not leak
        engine,
        "r2",
        HEADLINE_AT + timedelta(hours=1),
        "result",
        '{"period": "Q1", "eps_verdict": "misses", "eps_surprise": -0.5}',
    )
    days = [date(2026, 1, 26) + timedelta(days=i) for i in range(40)]
    sessions = [d for d in days if d.weekday() < 5 and d < date(2026, 3, 2)]
    async with engine.begin() as conn:
        for symbol, step in (("NVDA", 1.0), ("SPUS", 0.0)):
            await conn.execute(
                text(
                    "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                    "volume, fetched_at) VALUES (:s, :d, 'all', :c, :c, :c, :c, 1, now())"
                ),
                [{"s": symbol, "d": d, "c": 100.0 + step * i} for i, d in enumerate(sessions)],
            )
    ctx = (await tech_context(engine, [("NVDA", HEADLINE_AT)]))[("NVDA", HEADLINE_AT)]
    assert ctx.startswith("semiconductors & related devices; last report 2026-01-31 (Q4)")
    assert "EPS beats +6%" in ctx and "misses" not in ctx
    n = len(sessions)
    assert f"20d vs SPUS {(100 + n - 1) / (100 + n - 21) - 1:+.1%}" in ctx


class _LLM:
    def __init__(self) -> None:
        self.prompts: list[tuple[str, str | None]] = []

    async def generate_json(self, prompt: str, system: str | None = None):
        self.prompts.append((prompt, system))
        return {"s": {str(i): 0.5 for i in range(len(prompt.splitlines()))}}


async def test_experts_score_only_tech_headlines_the_generic_scorer_read(
    engine: AsyncEngine,
) -> None:
    await _screen(engine)
    records = [
        EventRecord("alpaca", f"n{i}", "news", sym, HEADLINE_AT, HEADLINE_AT, p)
        for i, (sym, p) in enumerate(
            [
                ("NVDA", {"headline": "Nvidia wins deal", "symbols": ["NVDA"]}),
                ("NVDA", {"headline": "Nvidia unscored", "symbols": ["NVDA"]}),
                ("XOM", {"headline": "Exxon output", "symbols": ["XOM"]}),
            ]
        )
    ]
    await EventRecorder(engine).record(records)
    async with engine.begin() as conn:  # the generic scorer read the first and the third
        await conn.execute(
            text(
                "INSERT INTO event_scores (event_id, scorer, score, scored_at) "
                "SELECT id, :sc, 0.1, now() FROM events WHERE source_id IN ('n0', 'n2')"
            ),
            {"sc": scorer_id(MODEL, GENERIC)},
        )
    symbols = await tech_symbols(engine)
    assert symbols == {"NVDA"}
    llm = _LLM()
    v = variants(symbols)["context"]
    assert await score_all(llm, engine, model=MODEL, max_pairs=10, variant=v) == 1
    prompt, system = llm.prompts[0]
    assert system == v.system and prompt.startswith("0|NVDA|semiconductors")
    assert prompt.endswith("|Nvidia wins deal")
    # Its own scorer id: never averaged into the generic scorer's readings.
    assert scorer_id(MODEL, v).startswith("llm-techctx:")
    assert scorer_id(MODEL).startswith("llm-batch:")
