"""CORS stays closed unless WEB_CORS_DEV_ORIGINS opens it for the Vite dev server."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import halal_trader.config as config
from halal_trader.web import app as web_app

DEV = "http://localhost:5173"


def _allowed_origin(monkeypatch: pytest.MonkeyPatch, flag: str | None) -> str | None:
    if flag is not None:
        monkeypatch.setenv("WEB_CORS_DEV_ORIGINS", flag)
    monkeypatch.setattr(config, "_settings", None)
    with TestClient(web_app.create_app()) as client:
        r = client.get("/api/health", headers={"Origin": DEV})
    return r.headers.get("access-control-allow-origin")


def test_cors_is_closed_by_default(
    database_url: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    assert _allowed_origin(monkeypatch, None) is None


def test_the_dev_flag_opens_it_for_vite_only(
    database_url: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    assert _allowed_origin(monkeypatch, "true") == DEV
