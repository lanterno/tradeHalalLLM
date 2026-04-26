"""Auth middleware contract tests."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from halal_trader.web.middleware.auth import auth_middleware


def _app(token: str = "") -> FastAPI:
    """A minimal app that wires the auth middleware + a couple of endpoints."""
    app = FastAPI()
    app.middleware("http")(auth_middleware)

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.post("/api/admin/halt")
    async def halt():
        return {"halted": True}

    @app.post("/api/research/backtest/run")
    async def run_bt():
        return {"job_id": "x"}

    return app


@pytest.fixture
def with_token(monkeypatch):
    monkeypatch.setenv("WEB_API_TOKEN", "secret-123")
    import halal_trader.config as _cfg

    monkeypatch.setattr(_cfg, "_settings", None)
    return _app()


@pytest.fixture
def without_token(monkeypatch):
    monkeypatch.setenv("WEB_API_TOKEN", "")
    import halal_trader.config as _cfg

    monkeypatch.setattr(_cfg, "_settings", None)
    return _app()


# ── GET requests are always allowed ───────────────────────────


def test_get_passes_without_token(without_token):
    c = TestClient(without_token)
    assert c.get("/api/health").status_code == 200


def test_get_passes_even_with_token_set(with_token):
    """Read endpoints don't need the token; only mutations do."""
    c = TestClient(with_token)
    assert c.get("/api/health").status_code == 200


# ── No token configured → mutations 503 ───────────────────────


def test_mutation_503_when_token_unset(without_token):
    c = TestClient(without_token)
    r = c.post("/api/admin/halt")
    assert r.status_code == 503
    assert r.json()["error"] == "mutations_disabled"


def test_non_admin_mutation_also_gated(without_token):
    """Any POST under /api/ must be gated, not just /api/admin."""
    c = TestClient(without_token)
    assert c.post("/api/research/backtest/run").status_code == 503


# ── Token configured → mutations require matching header ──────


def test_mutation_401_when_header_missing(with_token):
    c = TestClient(with_token)
    r = c.post("/api/admin/halt")
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"


def test_mutation_401_when_header_wrong(with_token):
    c = TestClient(with_token)
    r = c.post("/api/admin/halt", headers={"X-Trader-Token": "wrong"})
    assert r.status_code == 401


def test_mutation_passes_with_correct_header(with_token):
    c = TestClient(with_token)
    r = c.post("/api/admin/halt", headers={"X-Trader-Token": "secret-123"})
    assert r.status_code == 200
    assert r.json() == {"halted": True}


# ── Length-mismatch comparison still constant-time ────────────


def test_mismatched_token_lengths_still_rejected(with_token):
    """A short prefix of the real token must not match."""
    c = TestClient(with_token)
    r = c.post("/api/admin/halt", headers={"X-Trader-Token": "secret-12"})
    assert r.status_code == 401
