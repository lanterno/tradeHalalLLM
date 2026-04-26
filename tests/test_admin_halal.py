"""Halal & compliance admin endpoint tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "halal.db"
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


def test_halal_refresh_returns_count(client):
    """No Zoya configured → falls back to default list."""
    r = client.post("/api/admin/halal/refresh")
    assert r.status_code == 200
    body = r.json()
    assert body["refreshed"] is True
    # Default fallback list ships ~20 large-caps.
    assert body["halal_symbol_count"] >= 5


# ── Sector allocation ────────────────────────────────────────


def test_sector_allocation_empty_when_no_state(client):
    r = client.get("/api/admin/halal/sector-allocation")
    assert r.status_code == 200
    body = r.json()
    assert body["allocations"] == []
    assert body["total_equity_usd"] == 0.0


def test_sector_allocation_buckets_by_sector(client):
    p1 = MagicMock()
    p1.symbol = "AAPL"
    p1.qty = 50
    p1.current_price = 200.0
    p1.avg_entry_price = 200.0
    p2 = MagicMock()
    p2.symbol = "JNJ"
    p2.qty = 10
    p2.current_price = 150.0
    p2.avg_entry_price = 150.0
    web_app.app_state["stock_positions"] = [p1, p2]
    web_app.app_state["stock_equity"] = 20_000

    r = client.get("/api/admin/halal/sector-allocation").json()
    by_sector = {row["sector"]: row["value_usd"] for row in r["allocations"]}
    assert by_sector["Technology"] == 10_000  # 50 * 200
    assert by_sector["Healthcare"] == 1_500  # 10 * 150
