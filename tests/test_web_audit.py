"""Mutation audit middleware tests — round-trip via repo."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WEB_API_TOKEN", "secret")
    web_app.app_state.clear()
    app = web_app.create_app()

    # Add a fake admin endpoint so we can trigger an audit row.
    @app.post("/api/admin/echo")
    async def echo(payload: dict):
        return {"got": payload.get("x")}

    @app.post("/api/admin/boom")
    async def boom():
        raise RuntimeError("intentional")

    with TestClient(app) as c:
        yield c


def _sync_engine():
    from sqlalchemy import create_engine

    from halal_trader.config import get_settings

    return create_engine(get_settings().database_url_sync())


def _audited_rows(client):
    eng = _sync_engine()
    try:
        with eng.begin() as conn:
            rows = conn.execute(
                sa.text("SELECT method, path, outcome, status_code, error FROM web_actions")
            )
            return list(rows)
    finally:
        eng.dispose()


def test_successful_mutation_writes_ok_row(client):
    r = client.post(
        "/api/admin/echo",
        json={"x": 1},
        headers={"X-Trader-Token": "secret"},
    )
    assert r.status_code == 200
    rows = _audited_rows(client)
    assert len(rows) == 1
    method, path, outcome, status_code, error = rows[0]
    assert method == "POST"
    assert path == "/api/admin/echo"
    assert outcome == "ok"
    assert status_code == 200
    assert error is None


def test_rejected_mutation_does_not_write_row(client):
    """Auth rejection (401) must NOT produce an audit row — would otherwise
    spam the table whenever a misconfigured client retries."""
    r = client.post("/api/admin/echo", json={"x": 1})
    assert r.status_code == 401
    rows = _audited_rows(client)
    assert rows == []


def test_handler_exception_marks_row_error(client):
    with pytest.raises(Exception):
        client.post("/api/admin/boom", headers={"X-Trader-Token": "secret"})
    rows = _audited_rows(client)
    assert len(rows) == 1
    _, _, outcome, _, error = rows[0]
    assert outcome == "error"
    assert "intentional" in (error or "")


def test_get_request_writes_no_audit(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    rows = _audited_rows(client)
    assert rows == []


def test_payload_truncated_for_huge_bodies(client):
    big = {"x": "y" * 10_000}
    r = client.post("/api/admin/echo", json=big, headers={"X-Trader-Token": "secret"})
    assert r.status_code == 200
    eng = _sync_engine()
    try:
        with eng.begin() as conn:
            payload = conn.execute(sa.text("SELECT payload FROM web_actions")).scalar_one()
    finally:
        eng.dispose()
    assert payload.endswith("…[truncated]")
    assert len(payload) < 6_000
