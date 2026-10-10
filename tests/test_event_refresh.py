"""The evening event refresh: every step runs, a failing one is reported, not fatal."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

import halal_trader.events.earnings_parse as ep
from halal_trader.events import daily, history


async def three(*a, **k):
    return 3


def _stub_steps(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Every step but news and facts returns 3; news fails. Returns the settings."""

    async def companies(_engine):
        return {1: "AAA"}

    async def symbols(_engine):
        return {"AAA"}

    async def boom(*a, **k):
        raise RuntimeError("alpaca down")

    monkeypatch.setattr(history, "covered_companies", companies)
    monkeypatch.setattr(history, "covered_symbols", symbols)
    monkeypatch.setattr(history, "backfill_news", boom)
    monkeypatch.setattr(daily, "refresh_filings", three)
    monkeypatch.setattr(history, "backfill_eps", three)
    monkeypatch.setattr(history, "backfill_insiders", three)
    import halal_trader.events.llm_score as ls

    monkeypatch.setattr(ls, "score_all", three)
    monkeypatch.setattr("halal_trader.core.llm.create_classifier_llm", lambda s: object())
    return SimpleNamespace(
        edgar=SimpleNamespace(user_agent="t t@example.invalid"),
        alpaca=SimpleNamespace(api_key="k", secret_key="s", paper_trade=True),
        llm=SimpleNamespace(model="m", monthly_live_usd=25.0, monthly_research_usd=15.0),
        glm=None,
    )


async def test_each_step_reports_and_the_rest_still_run(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _stub_steps(monkeypatch)
    monkeypatch.setattr(ep, "extract_all", three)
    run = await daily.refresh_events(engine, settings, today=date(2026, 10, 2))

    assert run.counts == {"filings": 3, "eps": 3, "insiders": 3, "facts": 3, "llm_scores": 3}
    assert len(run.errors) == 1 and "alpaca down" in run.errors[0]
    # Filings refresh is weekly: a second run the same day skips it.
    run2 = await daily.refresh_events(engine, settings, today=date(2026, 10, 2))
    assert run2.counts["filings"] == 0


async def _seed_superseded(engine: AsyncEngine) -> None:
    """One earnings headline, read by v3 and by a pin-less v4 parser."""
    from halal_trader.events.store import EventRecord, EventRecorder

    t = datetime(2026, 10, 1, 20, tzinfo=UTC)
    headline = "Micron Technology Q4 Adj. EPS $33.42 Beats $31.45 Estimate"
    await EventRecorder(engine).record(
        [EventRecord("alpaca", "1", "news", "MU", t, t, {"headline": headline})]
    )
    async with engine.begin() as conn:
        for label in (ep.EXTRACTOR_V3, ep.EXTRACTOR_V4):
            await conn.execute(
                text(
                    "INSERT INTO event_facts (event_id, extractor, kind, fields) "
                    "SELECT id, :x, 'none', '{}'::jsonb FROM events"
                ),
                {"x": label},
            )


async def _labels(engine: AsyncEngine) -> dict[str, int]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT extractor, count(*) AS n FROM event_facts GROUP BY 1")
        )
        return {r.extractor: r.n for r in rows}


async def test_the_facts_step_reads_every_event_then_drops_the_superseded_facts(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _stub_steps(monkeypatch)
    await _seed_superseded(engine)
    run = await daily.refresh_events(engine, settings, today=date(2026, 10, 2))

    assert run.counts["facts"] == 1
    assert await _labels(engine) == {ep.EXTRACTOR_V3: 1, ep.EXTRACTOR: 1}


async def test_a_failed_extraction_deletes_no_facts(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _stub_steps(monkeypatch)
    await _seed_superseded(engine)

    async def interrupted(*a, **k):
        raise RuntimeError("connection lost")

    monkeypatch.setattr(ep, "extract_all", interrupted)
    run = await daily.refresh_events(engine, settings, today=date(2026, 10, 2))

    assert "facts" not in run.counts
    assert any("events facts" in e and "connection lost" in e for e in run.errors)
    # Not every event is read under the current label: the old facts stay.
    assert await _labels(engine) == {ep.EXTRACTOR_V3: 1, ep.EXTRACTOR_V4: 1}
