"""Operator lifecycle endpoint tests — halt, resume, cancel, close."""

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
    app = web_app.create_app()

    # Mock stock broker for cancel/close tests.
    stock = MagicMock()
    stock.get_open_orders = AsyncMock(return_value=[])
    stock.cancel_order = AsyncMock(return_value={"orderId": "x"})
    stock.close_position = AsyncMock(return_value={"ok": True})

    with TestClient(app) as c:
        c.app.state.ctx.runtime.stock_broker = stock
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


# ── Cancel orders ─────────────────────────────────────────────


def test_cancel_all_orders_no_open_orders(client):
    r = client.delete("/api/admin/orders")
    assert r.status_code == 200
    body = r.json()
    assert body["cancelled"] == []
    assert body["failed"] == []


def test_cancel_one_order_calls_broker(client):
    r = client.delete("/api/admin/orders/abc123?symbol=AAPL")
    assert r.status_code == 200
    client.app.state.ctx.runtime.stock_broker.cancel_order.assert_awaited_once_with(
        symbol="AAPL", order_id="abc123"
    )


def test_cancel_invalid_asset_class(client):
    r = client.delete("/api/admin/orders?asset_class=crypto")
    assert r.status_code == 400


def test_cancel_503_when_broker_not_bound(client):
    client.app.state.ctx.runtime.stock_broker = None
    r = client.delete("/api/admin/orders")
    assert r.status_code == 503


# ── Force close ──────────────────────────────────────────────


def test_force_close_stock_calls_broker(client):
    stock = client.app.state.ctx.runtime.stock_broker
    r = client.post(
        "/api/admin/positions/AAPL/close",
        json={"asset_class": "stock", "reason": "operator_intervention"},
    )
    assert r.status_code == 200
    stock.close_position.assert_awaited_once_with("AAPL")


def test_force_close_rejects_crypto_asset_class(client):
    r = client.post(
        "/api/admin/positions/BTCUSDT/close",
        json={"asset_class": "crypto", "reason": "x"},
    )
    assert r.status_code == 422


# ── Auth + confirmation gating ───────────────────────────────


def test_no_auth_token_rejected(client):
    """Even a benign POST is 401 without the header."""
    c = TestClient(client.app)  # fresh client with no default header
    r = c.post("/api/admin/halt", json={"reason": "drill from dashboard"})
    assert r.status_code == 401
