"""Operator lifecycle endpoint tests — halt, resume, pause, cancel, close."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WEB_API_TOKEN", "secret")
    # Disable confirmation in tests so we don't have to forge two headers per call.
    monkeypatch.setenv("WEB_REQUIRE_CONFIRMATION", "false")
    web_app.app_state.clear()
    app = web_app.create_app()

    # Mock crypto broker for cancel/close tests.
    crypto = MagicMock()
    crypto.get_open_orders = AsyncMock(return_value=[])
    crypto.cancel_order = AsyncMock(return_value={"orderId": "x"})
    crypto.get_balances = AsyncMock(return_value=[])
    crypto.place_order = AsyncMock(return_value={"orderId": "y"})
    web_app.app_state["crypto_broker"] = crypto

    with TestClient(app) as c:
        c.headers["X-Trader-Token"] = "secret"
        yield c


# ── Halt ──────────────────────────────────────────────────────


def test_halt_status_initially_off(client):
    r = client.get("/api/admin/halt")
    assert r.status_code == 200
    assert r.json()["enabled"] is False


def test_halt_engages_with_reason(client):
    r = client.post("/api/admin/halt", json={"reason": "drill from dashboard"})
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["reason"] == "drill from dashboard"
    assert body["set_by"] == "dashboard"


def test_halt_rejects_short_reason(client):
    r = client.post("/api/admin/halt", json={"reason": "x"})
    assert r.status_code == 422


def test_resume_clears_halt(client):
    client.post("/api/admin/halt", json={"reason": "drill from dashboard"})
    r = client.post("/api/admin/resume")
    assert r.status_code == 200
    assert r.json()["enabled"] is False
    # GET reflects the cleared state.
    assert client.get("/api/admin/halt").json()["enabled"] is False


# ── Per-pair pause ────────────────────────────────────────────


def test_pause_pair_round_trip(client):
    r = client.post("/api/admin/pairs/BTCUSDT/pause", json={"reason": "bad fills"})
    assert r.status_code == 200
    assert r.json() == {"pair": "BTCUSDT", "paused": True}

    r2 = client.get("/api/admin/pairs/paused")
    rows = r2.json()
    assert len(rows) == 1
    assert rows[0]["pair"] == "BTCUSDT"
    assert rows[0]["reason"] == "bad fills"


def test_resume_pair_404_when_not_paused(client):
    r = client.delete("/api/admin/pairs/BTCUSDT/pause")
    assert r.status_code == 404


def test_resume_pair_clears(client):
    client.post("/api/admin/pairs/BTCUSDT/pause", json={"reason": "bad fills"})
    r = client.delete("/api/admin/pairs/BTCUSDT/pause")
    assert r.status_code == 200
    assert client.get("/api/admin/pairs/paused").json() == []


# ── Cancel orders ─────────────────────────────────────────────


def test_cancel_all_orders_no_open_orders(client):
    r = client.delete("/api/admin/orders?asset_class=crypto")
    assert r.status_code == 200
    body = r.json()
    assert body["cancelled"] == []
    assert body["failed"] == []


def test_cancel_one_order_calls_broker(client):
    r = client.delete("/api/admin/orders/abc123?symbol=BTCUSDT&asset_class=crypto")
    assert r.status_code == 200
    web_app.app_state["crypto_broker"].cancel_order.assert_awaited_once_with(
        symbol="BTCUSDT", order_id="abc123"
    )


def test_cancel_invalid_asset_class(client):
    r = client.delete("/api/admin/orders?asset_class=options")
    assert r.status_code == 400


def test_cancel_503_when_broker_not_bound(client):
    web_app.app_state.pop("crypto_broker", None)
    r = client.delete("/api/admin/orders?asset_class=crypto")
    assert r.status_code == 503


# ── Force close ──────────────────────────────────────────────


def test_force_close_crypto_calls_broker(client):
    crypto = web_app.app_state["crypto_broker"]
    bal = MagicMock()
    bal.asset = "BTC"
    bal.free = 0.5
    crypto.get_balances = AsyncMock(return_value=[bal])
    r = client.post(
        "/api/admin/positions/BTCUSDT/close",
        json={"asset_class": "crypto", "reason": "operator_intervention"},
    )
    assert r.status_code == 200
    crypto.place_order.assert_awaited_once()


def test_force_close_404_when_no_balance(client):
    crypto = web_app.app_state["crypto_broker"]
    crypto.get_balances = AsyncMock(return_value=[])
    r = client.post(
        "/api/admin/positions/BTCUSDT/close",
        json={"asset_class": "crypto", "reason": "x"},
    )
    assert r.status_code == 404


# ── Auth + confirmation gating ───────────────────────────────


def test_no_auth_token_rejected(client):
    """Even a benign POST is 401 without the header."""
    c = TestClient(client.app)  # fresh client with no default header
    r = c.post("/api/admin/halt", json={"reason": "drill from dashboard"})
    assert r.status_code == 401
