"""GET /api/config/schema — exposes every Settings field for the dashboard."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "schema.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))

    import halal_trader.config as _config

    _config._settings = None

    from halal_trader.db import admin

    admin.upgrade("head")

    web_app.app_state.clear()
    app = web_app.create_app()

    with TestClient(app) as c:
        yield c

    _config._settings = None


def test_schema_returns_list_of_fields(client):
    r = client.get("/api/config/schema")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    # Every row has the four required keys.
    for row in rows[:5]:
        assert {"env_name", "type", "default", "secret"} <= set(row.keys())


def test_schema_includes_well_known_envvars(client):
    rows = client.get("/api/config/schema").json()
    env_names = {row["env_name"] for row in rows}
    # Smoke-check a few representative keys from every settings group.
    assert "LLM_PROVIDER" in env_names
    assert "LLM_MODEL" in env_names
    assert "BINANCE_API_KEY" in env_names
    assert "ALPACA_API_KEY" in env_names
    assert "LLM_DAILY_USD_CAP" in env_names


def test_schema_flags_secrets(client):
    rows = client.get("/api/config/schema").json()
    by_name = {r["env_name"]: r for r in rows}
    # Anything with api_key / secret in the name is a secret; defaults aren't.
    assert by_name["BINANCE_API_KEY"]["secret"] is True
    assert by_name["BINANCE_SECRET_KEY"]["secret"] is True
    assert by_name["LLM_MODEL"]["secret"] is False
    assert by_name["LLM_PROVIDER"]["secret"] is False


def test_schema_default_for_simple_int_field(client):
    rows = client.get("/api/config/schema").json()
    by_name = {r["env_name"]: r for r in rows}
    # Crypto trading interval ships at 60s by default.
    assert by_name["CRYPTO_TRADING_INTERVAL_SECONDS"]["default"] == 60
