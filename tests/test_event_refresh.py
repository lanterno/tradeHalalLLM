"""The evening event refresh: every step runs, a failing one is reported, not fatal."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events import daily, history


async def test_each_step_reports_and_the_rest_still_run(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def companies(_engine):
        return {1: "AAA"}

    async def symbols(_engine):
        return {"AAA"}

    async def boom(*a, **k):
        raise RuntimeError("alpaca down")

    async def three(*a, **k):
        return 3

    monkeypatch.setattr(history, "covered_companies", companies)
    monkeypatch.setattr(history, "covered_symbols", symbols)
    monkeypatch.setattr(history, "backfill_news", boom)
    monkeypatch.setattr(daily, "refresh_filings", three)
    monkeypatch.setattr(history, "backfill_eps", three)
    monkeypatch.setattr(history, "backfill_insiders", three)
    import halal_trader.events.earnings_parse as ep
    import halal_trader.events.llm_score as ls

    monkeypatch.setattr(ep, "extract_all", three)
    monkeypatch.setattr(ls, "score_all", three)

    settings = SimpleNamespace(
        edgar=SimpleNamespace(user_agent="t t@example.invalid"),
        alpaca=SimpleNamespace(api_key="k", secret_key="s"),
        llm=SimpleNamespace(model="m", monthly_live_usd=25.0, monthly_research_usd=15.0),
        glm=None,
    )
    monkeypatch.setattr("halal_trader.core.llm.create_classifier_llm", lambda s: object())
    run = await daily.refresh_events(engine, settings, today=date(2026, 10, 2))

    assert run.counts == {"filings": 3, "eps": 3, "insiders": 3, "facts": 3, "llm_scores": 3}
    assert len(run.errors) == 1 and "alpaca down" in run.errors[0]
    # Filings refresh is weekly: a second run the same day skips it.
    run2 = await daily.refresh_events(engine, settings, today=date(2026, 10, 2))
    assert run2.counts["filings"] == 0
