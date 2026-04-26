"""Mobile summary endpoint + state-push WebSocket tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "mobile.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WEB_API_TOKEN", "secret")
    monkeypatch.setenv("WEB_REQUIRE_CONFIRMATION", "false")

    import halal_trader.config as _config

    _config._settings = None

    from halal_trader.db import admin

    admin.upgrade("head")

    web_app.app_state.clear()
    app = web_app.create_app()

    with TestClient(app) as c:
        c.headers["X-Trader-Token"] = "secret"
        yield c

    _config._settings = None


# ── Summary endpoint ─────────────────────────────────────────


def test_summary_returns_baseline_payload(client):
    r = client.get("/api/mobile/summary")
    assert r.status_code == 200
    body = r.json()
    # Required keys present even when nothing has been initialised.
    assert "halt" in body
    assert "bot_running" in body
    assert "open_positions_by_asset" in body
    assert body["halt"]["enabled"] is False


def test_summary_reflects_app_state(client):
    web_app.app_state["bot_running"] = True
    web_app.app_state["open_positions_by_asset"] = {"crypto": 3, "stock": 1}
    web_app.app_state["llm_cost_today_usd"] = 0.42

    r = client.get("/api/mobile/summary")
    body = r.json()
    assert body["bot_running"] is True
    assert body["open_positions_by_asset"] == {"crypto": 3, "stock": 1}
    assert body["llm_cost_today_usd"] == 0.42


def test_summary_reflects_engaged_halt(client):
    """After /api/admin/halt, the mobile summary should show enabled=True."""
    client.post("/api/admin/halt", json={"reason": "test halt drill"})
    body = client.get("/api/mobile/summary").json()
    assert body["halt"]["enabled"] is True
    assert body["halt"]["reason"] == "test halt drill"


# ── WebSocket push ──────────────────────────────────────────


def test_ws_state_pushes_first_payload(client):
    """The WS handshake should immediately send a summary payload."""
    with client.websocket_connect("/ws/state") as ws:
        payload = ws.receive_json(mode="text")
        assert "halt" in payload
        assert "ts" in payload
