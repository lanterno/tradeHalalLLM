"""Routes that were removed stay removed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    from halal_trader.config import get_settings

    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    # A valid token, so a mutation reaches routing instead of the auth gate.
    monkeypatch.setattr(get_settings().web, "api_token", "t")
    with TestClient(web_app.create_app()) as c:
        c.headers["X-Trader-Token"] = "t"
        yield c


@pytest.mark.parametrize(
    ("method", "path"),
    [
        # Wrote the default 20 symbols as halal, unscreened, into the cache the
        # live bot trades from.
        ("POST", "/api/admin/halal/refresh"),
        # The legacy hand-typed purification ledger (purification_accruals is
        # the dividend ledger).
        ("GET", "/api/admin/purification"),
        ("POST", "/api/admin/purification"),
        # Read in-process bot state the web process never has.
        ("GET", "/api/admin/halal/sector-allocation"),
        ("GET", "/api/mobile/summary"),
        ("GET", "/api/sse"),
    ],
)
def test_a_removed_route_is_gone(client: TestClient, method: str, path: str) -> None:
    assert client.request(method, path).status_code in (404, 405)
