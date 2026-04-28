"""Confirmation envelope + activity feed tests."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app
from halal_trader.web.middleware.confirm import require_confirmation

# ── require_confirmation as a pure dependency ─────────────────


def _confirm_app(monkeypatch, *, require: bool):
    monkeypatch.setenv("WEB_API_TOKEN", "")
    monkeypatch.setenv("WEB_REQUIRE_CONFIRMATION", "true" if require else "false")
    import halal_trader.config as _cfg

    monkeypatch.setattr(_cfg, "_settings", None)

    app = FastAPI()

    @app.post("/api/admin/destroy", dependencies=[Depends(require_confirmation)])
    async def destroy():
        return {"destroyed": True}

    return app


def test_confirm_required_412_without_header(monkeypatch):
    app = _confirm_app(monkeypatch, require=True)
    c = TestClient(app)
    r = c.post("/api/admin/destroy")
    assert r.status_code == 412


def test_confirm_required_passes_with_header(monkeypatch):
    app = _confirm_app(monkeypatch, require=True)
    c = TestClient(app)
    r = c.post("/api/admin/destroy", headers={"X-Trader-Confirm": "true"})
    assert r.status_code == 200


def test_confirm_disabled_lets_request_through(monkeypatch):
    """Tests can opt out via WEB_REQUIRE_CONFIRMATION=false."""
    app = _confirm_app(monkeypatch, require=False)
    c = TestClient(app)
    r = c.post("/api/admin/destroy")
    assert r.status_code == 200


def test_confirm_header_case_insensitive(monkeypatch):
    """``True`` and ``TRUE`` should both work — easier on operators."""
    app = _confirm_app(monkeypatch, require=True)
    c = TestClient(app)
    r = c.post("/api/admin/destroy", headers={"X-Trader-Confirm": "TRUE"})
    assert r.status_code == 200


# ── /api/activity endpoint ────────────────────────────────────


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WEB_API_TOKEN", "secret")
    web_app.app_state.clear()
    app = web_app.create_app()

    @app.post("/api/admin/echo")
    async def echo():
        return {"ok": True}

    with TestClient(app) as c:
        yield c


def test_activity_empty_initially(client):
    r = client.get("/api/activity")
    assert r.status_code == 200
    assert r.json() == []


def test_activity_lists_recent_mutations_newest_first(client):
    for _ in range(3):
        client.post("/api/admin/echo", headers={"X-Trader-Token": "secret"})
    r = client.get("/api/activity?limit=10")
    rows = r.json()
    assert len(rows) == 3
    # All three should be ok-outcome echo posts.
    for row in rows:
        assert row["path"] == "/api/admin/echo"
        assert row["outcome"] == "ok"
        assert row["status_code"] == 200
    # Newest first by id ordering.
    assert rows[0]["id"] > rows[-1]["id"]


def test_activity_limit_respected(client):
    for _ in range(5):
        client.post("/api/admin/echo", headers={"X-Trader-Token": "secret"})
    r = client.get("/api/activity?limit=2")
    assert len(r.json()) == 2
