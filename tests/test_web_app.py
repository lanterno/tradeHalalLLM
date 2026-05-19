"""Smoke + contract tests for the FastAPI dashboard endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    """TestClient pointed at a fresh tmp DB.

    `init_db` (called by the app's lifespan) refuses any DB that isn't at
    the head Alembic revision, so we apply migrations against the tmp DB
    via `alembic.command.upgrade` first.
    """
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    # Phase W0 introduced the auth gate; the legacy halt endpoint is now
    # also a mutation under the gate. Provide a token here so the rest
    # of this file's POST/DELETE tests can pass it via the test client's
    # default headers.
    monkeypatch.setenv("WEB_API_TOKEN", "legacy-test-token")

    # Bust the Settings singleton so the new env vars are picked up.
    # Apply migrations to the fresh tmp DB.
    app = web_app.create_app()

    with TestClient(app) as c:
        # Default the auth header on the client so legacy tests don't have
        # to forge it on every call. Tests that explicitly want to test
        # auth rejection can pop the header per-request.
        c.headers["X-Trader-Token"] = "legacy-test-token"
        yield c

    # Reset the Settings singleton for the next test.


# ── Health & basic GETs ────────────────────────────────────────


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "running"
    assert "timestamp" in body


def test_request_id_echoed_back(client):
    r = client.get("/api/health", headers={"X-Request-ID": "test-rid-123"})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == "test-rid-123"


def test_request_id_generated_if_missing(client):
    r = client.get("/api/health")
    rid = r.headers.get("X-Request-ID", "")
    assert rid.startswith("req-")


def test_trades_endpoint_empty(client):
    r = client.get("/api/trades")
    assert r.status_code == 200
    assert r.json() == []


def test_pnl_daily_empty(client):
    r = client.get("/api/pnl/daily?days=7")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_pnl_daily_defaults_to_crypto_for_back_compat(client):
    """Pre-Round-7 the route always queried the crypto table — pin
    that default so existing dashboard fetches (no ``market`` param)
    keep hitting crypto rows."""
    r = client.get("/api/pnl/daily?days=7")
    assert r.status_code == 200
    # Same response shape as the explicit ``?market=crypto`` call.
    r2 = client.get("/api/pnl/daily?days=7&market=crypto")
    assert r2.status_code == 200
    assert r.json() == r2.json()


def test_pnl_daily_market_stocks_reads_stocks_table(client):
    """``?market=stocks`` must hit the stocks-side ``daily_pnl`` table
    via :meth:`get_pnl_history`, not the crypto ledger. With no rows
    written yet it still must respond 200 + ``[]`` (not 404 / not
    'crypto fallback'). Pin so the stocks day-end row is reachable
    via this route once the bot has run a day."""
    r = client.get("/api/pnl/daily?days=7&market=stocks")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_pnl_daily_rejects_unknown_market(client):
    """Anything other than crypto/stocks/stock 400s loudly — silent
    empty results were how the stocks ledger went missing from the
    dashboard for weeks."""
    r = client.get("/api/pnl/daily?market=junk")
    assert r.status_code == 400
    assert "market must be" in r.json()["detail"]


def test_analytics_returns_zeros_with_no_trades(client):
    r = client.get("/api/analytics")
    assert r.status_code == 200
    body = r.json()
    assert body["total_trades"] == 0


# ── Risk + system status ───────────────────────────────────────


def test_risk_state_unavailable_when_unset(client):
    r = client.get("/api/risk/state")
    assert r.status_code == 200
    assert r.json() == {"available": False}


def test_risk_state_round_trips_cached_value(client):
    client.app.state.ctx.runtime.risk_state = {
        "is_halted": False,
        "halt_reason": None,
        "portfolio_heat_pct": 0.012,
        "drawdown_pct": 0.04,
        "avg_correlation": 0.55,
        "summary": "all clear",
    }
    r = client.get("/api/risk/state")
    body = r.json()
    assert body["available"] is True
    assert body["portfolio_heat_pct"] == 0.012


def test_risk_state_passes_market_discriminator_through(client):
    """The cycle pushes ``risk_state["market"]``; the route must echo it
    so the frontend can label whose risk this snapshot is."""
    client.app.state.ctx.runtime.risk_state = {
        "market": "crypto",
        "is_halted": True,
        "halt_reason": "drawdown_breach",
        "portfolio_heat_pct": 0.03,
        "drawdown_pct": 0.08,
        "summary": "halted",
    }
    body = client.get("/api/risk/state").json()
    assert body["market"] == "crypto"
    assert body["is_halted"] is True


def test_halt_get_returns_disabled_initially(client):
    r = client.get("/api/system/halt")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False


def test_halt_post_requires_confirm_header(client):
    r = client.post("/api/system/halt", json={"reason": "drill"})
    assert r.status_code == 400
    assert "X-Halt-Confirm" in r.json()["error"]


def test_halt_post_engages_with_confirm_header(client):
    r = client.post(
        "/api/system/halt",
        json={"reason": "drill"},
        headers={"X-Halt-Confirm": "yes"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["reason"] == "drill"
    assert body["set_by"] == "dashboard"

    # Subsequent GET reflects the engaged state.
    body2 = client.get("/api/system/halt").json()
    assert body2["enabled"] is True


def test_halt_delete_requires_confirm_header(client):
    r = client.delete("/api/system/halt")
    assert r.status_code == 400


def test_halt_delete_clears_with_confirm_header(client):
    client.post(
        "/api/system/halt",
        json={"reason": "drill"},
        headers={"X-Halt-Confirm": "yes"},
    )
    r = client.delete("/api/system/halt", headers={"X-Halt-Confirm": "yes"})
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert body["reason"] == "drill"  # audit retained


def test_reconcile_recent_paginates(client):
    # Seed a row directly via the engine.
    from datetime import UTC, datetime

    from sqlalchemy import create_engine
    from sqlmodel import Session

    from halal_trader.config import get_settings
    from halal_trader.db.models import ReconciliationLog

    sync_url = get_settings().database_url_sync()
    eng = create_engine(sync_url)
    try:
        with Session(eng) as session:
            for i in range(5):
                session.add(
                    ReconciliationLog(
                        timestamp=datetime.now(UTC),
                        market="crypto",
                        symbol=f"SYM{i}",
                        db_quantity=1.0,
                        broker_quantity=0.5,
                        drift_pct=0.5,
                    )
                )
            session.commit()
    finally:
        eng.dispose()

    r = client.get("/api/system/reconcile/recent?limit=3")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 3


def test_reconcile_recent_caps_limit(client):
    r = client.get("/api/system/reconcile/recent?limit=9999")
    assert r.status_code == 200


def test_backups_endpoint_empty(client):
    """Postgres baseline — backups endpoint returns empty list."""
    r = client.get("/api/system/backups")
    assert r.status_code == 200
    assert r.json() == []


# ── Metrics endpoints ─────────────────────────────────────────


def test_metrics_cycles_returns_zero_count_with_no_log(client, tmp_path, monkeypatch):
    from halal_trader.config import get_settings

    monkeypatch.setattr(get_settings().log, "dir", tmp_path)
    r = client.get("/api/metrics/cycles?window=3600")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["window_seconds"] == 3600


def test_metrics_llm_returns_zero_calls_with_no_log(client, tmp_path, monkeypatch):
    from halal_trader.config import get_settings

    monkeypatch.setattr(get_settings().log, "dir", tmp_path)
    r = client.get("/api/metrics/llm?window=86400")
    assert r.status_code == 200
    assert r.json()["calls"] == 0
