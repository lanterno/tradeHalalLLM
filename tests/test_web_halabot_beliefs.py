"""/api/halabot/* — belief-board bridge from the dashboard to the shadow engine."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine

from halal_trader.web import app as web_app


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WEB_API_TOKEN", "secret")
    monkeypatch.setenv("WEB_REQUIRE_CONFIRMATION", "false")
    app = web_app.create_app()
    with TestClient(app) as c:
        yield c


def _seed_belief(database_url: str, asset: str, conviction: float) -> None:
    """Bootstrap the hb_ schema and persist one belief via the real store."""

    async def _go() -> None:
        from halabot.belief.schema import BeliefState, Catalyst, Direction
        from halabot.belief.store import PgBeliefStore
        from halabot.platform.db import bootstrap_schema

        engine = create_async_engine(database_url)
        try:
            await bootstrap_schema(engine)
            b = BeliefState.neutral(asset)
            b.direction = Direction.LONG_BIAS
            b.conviction = conviction
            b.thesis = "uptrend holding above support"
            b.catalysts_pending = [
                Catalyst(
                    kind="CPI",
                    scheduled_for=datetime(2026, 7, 14, 12, 30, tzinfo=UTC),
                    expected_impact=0.9,
                    detail="CPI release",
                )
            ]
            b.last_updated = datetime.now(UTC)
            await PgBeliefStore(engine).put(b)
        finally:
            await engine.dispose()

    asyncio.run(_go())


def test_board_empty_is_available_false(client):
    r = client.get("/api/halabot/beliefs")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["beliefs"] == []


def test_board_lists_seeded_belief_with_catalysts(client, database_url):
    _seed_belief(database_url, "NVDA", 0.62)
    r = client.get("/api/halabot/beliefs")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    row = next(b for b in body["beliefs"] if b["asset"] == "NVDA")
    assert row["conviction"] == pytest.approx(0.62)
    assert row["thesis"] == "uptrend holding above support"
    assert row["catalysts_pending"] == [
        {
            "kind": "CPI",
            "scheduled_for": "2026-07-14T12:30:00+00:00",
            "expected_impact": 0.9,
            "detail": "CPI release",
        }
    ]
    assert "support" in row and "horizon" in row  # new payload keys


def test_single_asset_detail_and_404(client, database_url):
    _seed_belief(database_url, "AAPL", 0.4)
    assert client.get("/api/halabot/beliefs/aapl").json()["asset"] == "AAPL"
    assert client.get("/api/halabot/beliefs/ZZZZ").status_code == 404


def test_decisions_endpoint_empty_list(client, database_url):
    _seed_belief(database_url, "AAPL", 0.4)  # ensures hb_ tables exist
    r = client.get("/api/halabot/decisions?limit=5")
    assert r.status_code == 200
    assert r.json() == []


def test_bad_correlation_id_is_400(client):
    assert client.get("/api/halabot/decisions/not-a-uuid").status_code == 400
