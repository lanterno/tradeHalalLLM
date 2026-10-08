"""Halal & compliance admin endpoint tests."""

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


# ── Purification ─────────────────────────────────────────────


def test_purification_initially_empty(client):
    r = client.get("/api/admin/purification")
    assert r.status_code == 200
    body = r.json()
    assert body["outstanding"] == []
    assert body["totals"]["outstanding_usd"] == 0.0


def test_record_purification_then_lists(client):
    r = client.post(
        "/api/admin/purification",
        json={
            "symbol": "AAPL",
            "dividend_usd": 100.0,
            "haram_pct": 0.05,
            "notes": "Q1 dividend",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["purification_usd"] == 5.0

    listing = client.get("/api/admin/purification").json()
    assert len(listing["outstanding"]) == 1
    assert listing["totals"]["outstanding_usd"] == 5.0


def test_mark_paid_round_trip(client):
    r = client.post(
        "/api/admin/purification",
        json={"symbol": "AAPL", "dividend_usd": 100.0, "haram_pct": 0.05},
    )
    eid = r.json()["id"]
    paid = client.post(f"/api/admin/purification/{eid}/mark_paid")
    assert paid.status_code == 200
    listing = client.get("/api/admin/purification").json()
    assert listing["outstanding"] == []
    assert listing["totals"]["paid_usd"] == 5.0


def test_mark_paid_404(client):
    r = client.post("/api/admin/purification/9999/mark_paid")
    assert r.status_code == 404


def test_record_rejects_negative_dividend(client):
    r = client.post(
        "/api/admin/purification",
        json={"symbol": "X", "dividend_usd": -10, "haram_pct": 0.05},
    )
    assert r.status_code == 422


def test_record_rejects_haram_pct_above_one(client):
    r = client.post(
        "/api/admin/purification",
        json={"symbol": "X", "dividend_usd": 10, "haram_pct": 1.5},
    )
    assert r.status_code == 422


# ── Halal cache refresh ──────────────────────────────────────


def test_the_screener_bypassing_refresh_route_is_gone(client):
    """POST /api/admin/halal/refresh wrote the default 20 symbols as halal
    without screening, into the cache the live bot trades from. Deleted."""
    assert client.post("/api/admin/halal/refresh").status_code in (404, 405)


# ── Sector allocation ────────────────────────────────────────
