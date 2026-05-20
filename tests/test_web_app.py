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


def test_request_id_present_on_auth_rejected_request(client):
    """Auth-rejected 401s must STILL carry an X-Request-ID header.
    Pre-Round-7 the correlate middleware was registered before auth,
    so 401s went out without the header and the operator had no
    correlatable trace ID for failed-auth attempts.

    Pin so a future middleware reshuffle doesn't silently regress
    this — the auth-rejected request must still hit the correlate
    middleware's response phase to pick up the header.
    """
    # Strip the default token so auth_middleware rejects us.
    client.headers.pop("X-Trader-Token", None)
    r = client.post(
        "/api/system/halt",
        json={"reason": "auth-rejection trace test"},
        headers={"X-Request-ID": "auth-rejected-correlator"},
    )
    assert r.status_code == 401
    # The echo header must still appear on the rejected response.
    assert r.headers.get("X-Request-ID") == "auth-rejected-correlator"


def test_audit_actor_reflects_request_id(client):
    """``web_actions.actor`` must equal the request-id of the call,
    not the default ``"anon"``. Pre-fix the correlate middleware ran
    AFTER audit on the inbound path, so the audit row's
    ``request_id_var.get()`` always read an empty value and every
    row recorded ``"anon"``.

    This test exercises the path end-to-end: send a mutation with an
    explicit X-Request-ID, then read it back via the activity-log
    endpoint and confirm the actor matches.
    """
    client.post(
        "/api/system/halt",
        json={"reason": "actor-correlation test"},
        headers={"X-Halt-Confirm": "yes", "X-Request-ID": "actor-cor-rid-42"},
    )
    # Clean up: resume the halt so the test DB ends in a known state.
    client.delete("/api/system/halt", headers={"X-Halt-Confirm": "yes"})

    r = client.get("/api/activity?limit=5")
    assert r.status_code == 200
    rows = r.json()
    # The most-recent POST row should carry our explicit request-id
    # as actor — not "anon".
    post_rows = [
        row for row in rows if row.get("method") == "POST" and row.get("path") == "/api/system/halt"
    ]
    assert post_rows, f"no POST audit row found in {rows[:3]}"
    assert post_rows[0]["actor"] == "actor-cor-rid-42", (
        f"expected actor='actor-cor-rid-42', got {post_rows[0]['actor']!r}. "
        f"Likely correlate_request is running AFTER audit on inbound path."
    )


def test_trades_endpoint_empty(client):
    r = client.get("/api/trades")
    assert r.status_code == 200
    assert r.json() == []


def test_trades_endpoint_market_stocks_reads_stocks_table(client):
    """``?market=stocks`` reads the stocks ``trades`` table (returns
    list of stock trades), not the crypto ledger. Empty pre-trade is
    correct; the route just must not 404 or fall back to crypto."""
    r = client.get("/api/trades?market=stocks")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_trades_endpoint_default_is_crypto_for_back_compat(client):
    """No ``market`` param → crypto, matching pre-Round-7 behavior."""
    a = client.get("/api/trades")
    b = client.get("/api/trades?market=crypto")
    assert a.status_code == 200
    assert b.status_code == 200
    assert a.json() == b.json()


def test_trades_endpoint_rejects_unknown_market(client):
    r = client.get("/api/trades?market=options")
    assert r.status_code == 400
    assert "market must be" in r.json()["detail"]


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


def test_analytics_market_stocks_routes_to_cross_asset_analytics(client):
    """``?market=stocks`` builds a fresh CrossAssetAnalytics(asset_class=
    'stock') and reads ``get_completed_stock_round_trips`` instead of
    ``get_completed_round_trips``. Same response shape, sourced from
    the stocks ``trades`` table."""
    r = client.get("/api/analytics?market=stocks")
    assert r.status_code == 200
    body = r.json()
    # Same field set as the crypto path — frontend renders identically.
    for key in ("total_trades", "wins", "losses", "win_rate", "profit_factor"):
        assert key in body


def test_analytics_rejects_unknown_market(client):
    r = client.get("/api/analytics?market=options")
    assert r.status_code == 400
    assert "market must be" in r.json()["detail"]


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


# ── /api/positions market dispatch ───────────────────────────


def test_positions_default_is_crypto_for_back_compat(client):
    """No ``market`` param → crypto path, matching pre-Round-7."""
    a = client.get("/api/positions")
    b = client.get("/api/positions?market=crypto")
    assert a.status_code == 200
    assert b.status_code == 200
    assert a.json() == b.json()


def test_positions_market_stocks_reads_stocks_open_trades(client):
    """``?market=stocks`` calls ``get_open_trades`` (stocks repo)
    instead of ``get_open_crypto_trades``. Empty pre-trade is fine
    — the route just must return 200 and a list, not 404 or
    silently fall back to crypto."""
    r = client.get("/api/positions?market=stocks")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_positions_rejects_unknown_market(client):
    r = client.get("/api/positions?market=options")
    assert r.status_code == 400
    assert "market must be" in r.json()["detail"]


# ── /api/system/status both-market cadence ────────────────────


def test_system_status_exposes_both_market_cadences(client):
    """Pre-Round-7 the ``cycle_interval_seconds`` field only carried
    the crypto value (60s default). A stocks operator saw "60" even
    though the stocks cycle runs every 15 min. Pin both new fields
    and the back-compat key."""
    r = client.get("/api/system/status")
    assert r.status_code == 200
    body = r.json()
    # Back-compat: ``cycle_interval_seconds`` mirrors crypto.
    assert "cycle_interval_seconds" in body
    # New: explicit crypto + stocks fields.
    assert "crypto_cycle_interval_seconds" in body
    assert "stocks_cycle_interval_seconds" in body
    # Defaults: crypto = 60s, stocks = 15min * 60 = 900s.
    assert body["crypto_cycle_interval_seconds"] == 60
    assert body["stocks_cycle_interval_seconds"] == 900
    # Back-compat field equals crypto value.
    assert body["cycle_interval_seconds"] == body["crypto_cycle_interval_seconds"]
