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


# ── SPA static serving / path traversal ────────────────────────


def test_resolve_static_serves_and_blocks_traversal(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=root>")
    (dist / "app.js").write_text("ok")
    (dist / "assets" / "x.css").write_text("x")
    (tmp_path / "secret.txt").write_text("top secret")  # OUTSIDE dist

    # Legit files inside dist resolve.
    assert web_app._resolve_static(dist, "app.js") == (dist / "app.js").resolve()
    assert web_app._resolve_static(dist, "assets/x.css") == (dist / "assets" / "x.css").resolve()
    # Traversal escaping dist is refused (caller falls back to index.html).
    assert web_app._resolve_static(dist, "../secret.txt") is None
    assert web_app._resolve_static(dist, "../../etc/passwd") is None
    # A missing file inside dist is also None (SPA client-route fallback).
    assert web_app._resolve_static(dist, "does-not-exist.js") is None


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


def test_pnl_daily_empty(client):
    r = client.get("/api/pnl/daily?days=7")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


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
        "market": "stocks",
        "is_halted": True,
        "halt_reason": "drawdown_breach",
        "portfolio_heat_pct": 0.03,
        "drawdown_pct": 0.08,
        "summary": "halted",
    }
    body = client.get("/api/risk/state").json()
    assert body["market"] == "stocks"
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
                        market="stocks",
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


def test_positions_reads_stocks_open_trades(client):
    """``/api/positions`` reads ``get_open_trades`` (stocks repo).
    Empty pre-trade is fine — the route just must return 200 and a list."""
    r = client.get("/api/positions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ── /api/system/status cadence ────────────────────


def test_system_status_exposes_stocks_cadence(client):
    """The stocks cycle runs every 15 min; the status route reports it."""
    r = client.get("/api/system/status")
    assert r.status_code == 200
    body = r.json()
    # Default: 15min * 60 = 900s.
    assert body["stocks_cycle_interval_seconds"] == 900
    assert "crypto_cycle_interval_seconds" not in body


def test_system_status_classifier_health_null_without_reactor(client):
    """Dashboard-only process (no co-hosted bot) → no reactor
    available → classifier_health is None, not missing."""
    r = client.get("/api/system/status")
    body = r.json()
    assert "classifier_health" in body
    assert body["classifier_health"] is None


def test_system_status_classifier_health_surfaces_telemetry(client):
    """When the bot's wired up the reactor, /api/system/status surfaces
    its classifier telemetry so the operator can answer "is the brain
    healthy" with one HTTP call. Pin the contract."""

    class _FakeClassifier:
        def get_telemetry(self):
            return {
                "total_calls": 42,
                "total_successes": 40,
                "total_failures": 2,
                "total_short_circuits": 0,
                "calls_today": 42,
                "daily_cap": 250,
                "quota_exhausted": False,
                "calls_by_provider": {"openai": 40},
                "cost_usd_total": 0.021,
            }

    class _FakeReactor:
        classifier = _FakeClassifier()

    client.app.state.ctx.runtime.stocks_news_reactor = _FakeReactor()
    body = client.get("/api/system/status").json()
    health = body["classifier_health"]
    assert health is not None
    assert health["total_calls"] == 42
    assert health["calls_by_provider"] == {"openai": 40}
    assert health["cost_usd_total"] == 0.021
    assert health["quota_exhausted"] is False


def test_system_status_classifier_health_handles_telemetry_exception(client):
    """A misbehaving classifier (e.g. attribute renamed by mistake)
    must not 500 the entire status endpoint — degrade gracefully to
    None so /api/system/status stays the safe always-on probe."""

    class _BoomClassifier:
        def get_telemetry(self):
            raise RuntimeError("boom")

    class _FakeReactor:
        classifier = _BoomClassifier()

    client.app.state.ctx.runtime.stocks_news_reactor = _FakeReactor()
    r = client.get("/api/system/status")
    assert r.status_code == 200
    assert r.json()["classifier_health"] is None
