"""Runtime config overlay + prompt picker + A/B weights tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "cfg.db"
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


# ── Runtime config CRUD ──────────────────────────────────────


def test_runtime_config_initially_empty(client):
    r = client.get("/api/admin/config/runtime")
    assert r.status_code == 200
    assert r.json() == {}


def test_patch_runtime_config_round_trips(client):
    r = client.patch(
        "/api/admin/config/runtime/CRYPTO_MAX_POSITION_PCT",
        json={"value": 0.15},
    )
    assert r.status_code == 200
    assert r.json()["value"] == 0.15

    listing = client.get("/api/admin/config/runtime").json()
    assert listing["CRYPTO_MAX_POSITION_PCT"] == 0.15


def test_patch_unknown_key_404(client):
    r = client.patch(
        "/api/admin/config/runtime/NOT_A_REAL_KEY",
        json={"value": 1},
    )
    assert r.status_code == 404


def test_patch_out_of_range_pct_rejected(client):
    """0.99 might be acceptable but 1.5 should be — verify the >1 guard."""
    r = client.patch(
        "/api/admin/config/runtime/CRYPTO_MAX_POSITION_PCT",
        json={"value": 1.5},
    )
    assert r.status_code == 422


def test_patch_negative_pct_rejected(client):
    r = client.patch(
        "/api/admin/config/runtime/CRYPTO_MAX_POSITION_PCT",
        json={"value": -0.1},
    )
    assert r.status_code == 422


def test_patch_string_value_for_int_field_rejected(client):
    r = client.patch(
        "/api/admin/config/runtime/CRYPTO_TRADING_INTERVAL_SECONDS",
        json={"value": "not-a-number"},
    )
    assert r.status_code == 422


def test_delete_runtime_config_reverts(client):
    client.patch(
        "/api/admin/config/runtime/CRYPTO_MAX_POSITION_PCT",
        json={"value": 0.15},
    )
    r = client.delete("/api/admin/config/runtime/CRYPTO_MAX_POSITION_PCT")
    assert r.status_code == 200
    assert client.get("/api/admin/config/runtime").json() == {}


def test_delete_unknown_key_404(client):
    r = client.delete("/api/admin/config/runtime/CRYPTO_MAX_POSITION_PCT")
    assert r.status_code == 404


# ── Prompt registry ─────────────────────────────────────────


def test_list_prompts_includes_strategy_modules(client):
    r = client.get("/api/admin/prompts")
    assert r.status_code == 200
    rows = r.json()
    names = {row["name"] for row in rows}
    assert "crypto.strategy.system" in names
    assert "trading.strategy.system" in names


def test_set_active_prompt_round_trips(client):
    rows = client.get("/api/admin/prompts").json()
    target = next(r for r in rows if r["name"] == "crypto.strategy.system")
    r = client.post("/api/admin/prompts/active", json={"version": target["short"]})
    assert r.status_code == 200
    # After setting, the listing should mark this version active.
    refreshed = client.get("/api/admin/prompts").json()
    active = [row for row in refreshed if row["active"]]
    assert len(active) == 1
    assert active[0]["short"] == target["short"]


def test_set_active_prompt_rejects_malformed(client):
    r = client.post("/api/admin/prompts/active", json={"version": "no-at-sign"})
    assert r.status_code == 422


# ── A/B router weights ──────────────────────────────────────


def test_ab_weights_initially_empty(client):
    r = client.get("/api/admin/ab/weights")
    assert r.status_code == 200
    assert r.json() == {}


def test_patch_ab_weights_round_trips(client):
    body = {"v1": 0.7, "v2": 0.3}
    r = client.patch("/api/admin/ab/weights", json=body)
    assert r.status_code == 200
    assert r.json() == body
    assert client.get("/api/admin/ab/weights").json() == body


def test_patch_ab_weights_rejects_negative(client):
    r = client.patch("/api/admin/ab/weights", json={"v1": -0.1})
    assert r.status_code == 422


def test_patch_ab_weights_rejects_empty(client):
    r = client.patch("/api/admin/ab/weights", json={})
    assert r.status_code == 422
