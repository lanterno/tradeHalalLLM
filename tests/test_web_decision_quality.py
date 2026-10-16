"""GET /api/decisions/quality serves the evening run's last score."""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy.ext.asyncio import create_async_engine

from halal_trader.trading import decision_quality as dq


def test_the_route_serves_the_last_score(
    database_url: str, tmp_path: Any, monkeypatch: Any
) -> None:
    from fastapi.testclient import TestClient

    from halal_trader import config
    from halal_trader.core.heartbeat import beat
    from halal_trader.web import app as web_app

    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(config, "_settings", None)

    async def seed() -> None:
        engine = create_async_engine(database_url)
        try:
            await beat(engine, dq.BEAT, {"trips": 3, "entry_edge": {"mean": -0.002}})
        finally:
            await engine.dispose()

    with TestClient(web_app.create_app()) as client:
        assert client.get("/api/decisions/quality").json() is None
        asyncio.run(seed())
        body = client.get("/api/decisions/quality").json()
    config._settings = None
    assert body["trips"] == 3 and body["entry_edge"]["mean"] == -0.002
    assert body["computed_at"]
