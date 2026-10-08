"""CORS stays closed: the built SPA is same-origin and the Vite dev server proxies."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import halal_trader.config as config
from halal_trader.web import app as web_app

DEV = "http://localhost:5173"


def _allowed_origin(monkeypatch: pytest.MonkeyPatch) -> str | None:
    monkeypatch.setattr(config, "_settings", None)
    with TestClient(web_app.create_app()) as client:
        r = client.get("/api/health", headers={"Origin": DEV})
    return r.headers.get("access-control-allow-origin")


def test_cors_is_closed_by_default(
    database_url: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    assert _allowed_origin(monkeypatch) is None
