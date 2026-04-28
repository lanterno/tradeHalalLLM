"""Backtest job queue tests."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WEB_API_TOKEN", "secret")
    monkeypatch.setenv("WEB_REQUIRE_CONFIRMATION", "false")
    web_app.app_state.clear()
    app = web_app.create_app()

    with TestClient(app) as c:
        c.headers["X-Trader-Token"] = "secret"
        yield c


def _wait_for_status(client, job_id: int, *, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/research/jobs/{job_id}").json()
        if body["status"] in ("ok", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


# ── Empty params → error row, no crash ───────────────────────


def test_backtest_with_empty_klines_records_error(client):
    r = client.post(
        "/api/research/backtest/run",
        json={"kind": "backtest", "params": {"klines": []}},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    body = _wait_for_status(client, job_id)
    assert body["status"] == "error"
    assert "klines" in (body["error"] or "")


# ── Monte Carlo job runs end-to-end ──────────────────────────


def test_monte_carlo_job_completes(client):
    """Monte Carlo doesn't need klines — just a trade list."""
    r = client.post(
        "/api/research/backtest/run",
        json={
            "kind": "monte_carlo",
            "params": {
                "trades": [
                    {"pnl": 10},
                    {"pnl": -5},
                    {"pnl": 15},
                    {"pnl": -2},
                ],
                "runs": 20,
                "seed": 42,
            },
        },
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    body = _wait_for_status(client, job_id)
    assert body["status"] == "ok"
    assert body["result"]["runs"] == 20
    assert "max_drawdown_pct_p95" in body["result"]


# ── List + pin/unpin ─────────────────────────────────────────


def test_list_jobs_includes_recently_enqueued(client):
    client.post(
        "/api/research/backtest/run",
        json={"kind": "monte_carlo", "params": {"trades": [{"pnl": 1}]}},
    )
    rows = client.get("/api/research/jobs").json()
    assert len(rows) == 1
    assert rows[0]["kind"] == "monte_carlo"


def test_pin_unpin_round_trips(client):
    r = client.post(
        "/api/research/backtest/run",
        json={"kind": "monte_carlo", "params": {"trades": [{"pnl": 1}]}},
    )
    job_id = r.json()["job_id"]
    pin = client.post(f"/api/research/jobs/{job_id}/pin")
    assert pin.status_code == 200
    assert pin.json()["pinned"] is True

    unpin = client.delete(f"/api/research/jobs/{job_id}/pin")
    assert unpin.status_code == 200
    assert unpin.json()["pinned"] is False


def test_pin_404_for_unknown_job(client):
    r = client.post("/api/research/jobs/9999/pin")
    assert r.status_code == 404


# ── Validation ───────────────────────────────────────────────


def test_invalid_kind_rejected(client):
    r = client.post(
        "/api/research/backtest/run",
        json={"kind": "options_chain", "params": {}},
    )
    assert r.status_code == 422


def test_get_unknown_job_404(client):
    r = client.get("/api/research/jobs/9999")
    assert r.status_code == 404
