"""GET /api/config/schema — exposes every Settings field for the dashboard."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    # database_url points the app's lifespan at a disposable test database;
    # without it the app fell back to the default URL, the operator's own.
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    app = web_app.create_app()

    with TestClient(app) as c:
        yield c


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
    assert "GLM_API_KEY" in env_names
    assert "GLM_BASE_URL" in env_names
    assert "LLM_MODEL" in env_names
    assert "ALPACA_API_KEY" in env_names
    assert "ZOYA_API_KEY" in env_names
    assert "LLM_DAILY_USD_CAP" in env_names


def test_schema_flags_secrets(client):
    rows = client.get("/api/config/schema").json()
    by_name = {r["env_name"]: r for r in rows}
    # Anything with api_key / secret in the name is a secret; defaults aren't.
    assert by_name["ALPACA_API_KEY"]["secret"] is True
    assert by_name["ALPACA_SECRET_KEY"]["secret"] is True
    assert by_name["GLM_API_KEY"]["secret"] is True
    assert by_name["LLM_MODEL"]["secret"] is False
    assert by_name["GLM_BASE_URL"]["secret"] is False


def test_schema_default_for_simple_int_field(client):
    rows = client.get("/api/config/schema").json()
    by_name = {r["env_name"]: r for r in rows}
    # The stock cycle ships at 15 minutes by default.
    assert by_name["TRADING_INTERVAL_MINUTES"]["default"] == 15


def test_schema_never_sends_a_secrets_default(client):
    rows = client.get("/api/config/schema").json()
    by_name = {r["env_name"]: r for r in rows}

    # Its default holds the repo-default Postgres password.
    assert by_name["DATABASE_URL"]["secret"] is True
    assert by_name["DATABASE_URL"]["default"] is None
    for name in ("GLM_API_KEY", "LIVE_MODE_CONFIRMATION", "TELEGRAM_BOT_TOKEN"):
        assert by_name[name]["secret"] is True
    assert all(r["default"] is None for r in rows if r["secret"])
    assert "trader-dev-only" not in str(rows)
