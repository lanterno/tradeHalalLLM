"""Mobile summary endpoint + state-push WebSocket tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from halal_trader.config import get_settings
from halal_trader.web import app as web_app


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WEB_API_TOKEN", "secret")
    monkeypatch.setattr(get_settings().web, "require_confirmation", False)
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


def test_summary_reflects_the_bot_from_the_database(client, database_url):
    """Liveness and today's LLM spend come from the DB (the bot is another process)."""
    import asyncio
    from decimal import Decimal

    from sqlalchemy.ext.asyncio import create_async_engine

    from halal_trader.core.heartbeat import STOCK_CYCLE, STOCK_PROCESS
    from halal_trader.core.llm.spend import SpendMeter
    from tests._beats import write_beat

    write_beat(database_url, STOCK_PROCESS)
    write_beat(database_url, STOCK_CYCLE)  # alive also means cycling during market hours

    async def spend() -> None:
        engine = create_async_engine(database_url)
        try:
            await SpendMeter(engine, consumer="stock", cap_usd=0).record(Decimal("0.42"))
        finally:
            await engine.dispose()

    asyncio.run(spend())

    body = client.get("/api/mobile/summary").json()
    assert body["bot_running"] is True
    assert body["llm_cost_today_usd"] == 0.42


def test_summary_exposes_risk_market_discriminator(client, database_url):
    """The cycle pushes ``risk_state["market"]``; the summary must
    surface it as ``drawdown_market`` so the phone shows whose risk
    snapshot the drawdown belongs to."""
    from halal_trader.core.heartbeat import STOCK_CYCLE
    from tests._beats import write_beat

    write_beat(
        database_url, STOCK_CYCLE, {"risk": {"drawdown_pct": 0.018, "portfolio_heat_pct": 0.04}}
    )
    body = client.get("/api/mobile/summary").json()
    assert body["drawdown_pct"] == 0.018
    assert body["drawdown_market"] == "stocks"


def test_summary_drawdown_market_none_when_no_risk_state(client):
    """No cycle has run yet → no risk_state → ``drawdown_market`` None."""
    body = client.get("/api/mobile/summary").json()
    assert body["drawdown_pct"] is None
    assert body["drawdown_market"] is None


def test_summary_reflects_engaged_halt(client):
    """After a halt, the mobile summary should show enabled=True."""
    r = client.post(
        "/api/system/halt", json={"reason": "test halt drill"}, headers={"X-Trader-Confirm": "true"}
    )
    assert r.status_code == 200, r.text
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
