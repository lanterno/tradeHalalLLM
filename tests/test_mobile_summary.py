"""Mobile summary endpoint + state-push WebSocket tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

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
    rt = client.app.state.ctx.runtime
    rt.bot_running = True
    rt.open_positions_by_asset = {"crypto": [{}, {}, {}], "stock": [{}]}
    rt.llm_cost_today_usd = 0.42

    r = client.get("/api/mobile/summary")
    body = r.json()
    assert body["bot_running"] is True
    # Mobile summary echoes the dict; route serialises lists per asset class.
    assert body["llm_cost_today_usd"] == 0.42


def test_summary_exposes_risk_market_discriminator(client):
    """The cycle pushes ``risk_state["market"]``; the summary must
    surface it as ``drawdown_market`` so the phone shows whose risk
    snapshot the drawdown belongs to."""
    rt = client.app.state.ctx.runtime
    rt.risk_state = {
        "market": "stocks",
        "drawdown_pct": 0.018,
        "portfolio_heat_pct": 0.04,
    }
    body = client.get("/api/mobile/summary").json()
    assert body["drawdown_pct"] == 0.018
    assert body["drawdown_market"] == "stocks"


def test_summary_drawdown_market_none_when_no_risk_state(client):
    """No cycle has run yet → no risk_state → ``drawdown_market`` None."""
    body = client.get("/api/mobile/summary").json()
    assert body["drawdown_pct"] is None
    assert body["drawdown_market"] is None


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
