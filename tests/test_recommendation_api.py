"""Daily recommendation web API — GET latest / history (read path)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine

from halal_trader.db.repository import Repository
from halal_trader.web import app as web_app


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WEB_API_TOKEN", "secret")
    monkeypatch.setenv("WEB_REQUIRE_CONFIRMATION", "false")
    app = web_app.create_app()
    with TestClient(app) as c:
        c.headers["X-Trader-Token"] = "secret"
        yield c


def _seed(database_url: str, rec: dict[str, Any]) -> int:
    async def _go() -> int:
        engine = create_async_engine(database_url)
        try:
            return await Repository(engine).save_recommendation(rec)
        finally:
            await engine.dispose()

    return asyncio.run(_go())


def _rec(symbol: str = "NVDA", date: str = "2026-06-20") -> dict[str, Any]:
    return {
        "date": date,
        "symbol": symbol,
        "conviction": 0.8,
        "thesis": "uptrend",
        "halal_note": "AAOIFI compliant",
        "suggested_entry": 130.0,
        "suggested_target": 145.0,
        "suggested_stop": 124.0,
        "catalysts": "AI demand",
        "risks": "valuation",
        "universe_size": 20,
        "model": "test",
        "prompt_version": "recommendation.daily.system@abc",
        "candidates": {"NVDA": {"price": 130.0}},
    }


def test_latest_empty_returns_available_false(client):
    r = client.get("/api/recommendation")
    assert r.status_code == 200
    assert r.json() == {"available": False}


def test_latest_returns_saved_pick(client, database_url):
    _seed(database_url, _rec("NVDA"))
    r = client.get("/api/recommendation")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["symbol"] == "NVDA"
    assert body["conviction"] == 0.8
    assert body["candidates"]["NVDA"]["price"] == 130.0
    assert "created_at" in body  # serialized ISO datetime


def test_latest_is_most_recent(client, database_url):
    _seed(database_url, _rec("AAPL"))
    _seed(database_url, _rec("MSFT"))  # newer
    r = client.get("/api/recommendation")
    assert r.json()["symbol"] == "MSFT"


def test_history_returns_list_newest_first(client, database_url):
    _seed(database_url, _rec("AAPL"))
    _seed(database_url, _rec("MSFT"))
    r = client.get("/api/recommendation/history?limit=5")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    assert [row["symbol"] for row in rows[:2]] == ["MSFT", "AAPL"]
