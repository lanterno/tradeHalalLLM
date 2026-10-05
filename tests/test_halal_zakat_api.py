"""GET /api/halal/zakat and /api/halal/purification for the Halal page."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ZAKAT_HAWL_HIJRI", "09-01")
    from halal_trader import config

    monkeypatch.setattr(config, "_settings", None)  # re-read the environment
    app = web_app.create_app()
    with TestClient(app) as c:
        yield c
    config._settings = None


def test_zakat_shows_the_next_hawl_and_both_methods_if_due_today(client) -> None:
    body = client.get("/api/halal/zakat").json()
    assert body["configured"] is True and body["hawl_hijri"] == "09-01"
    assert body["next_hawl_hijri"].endswith("-09-01 AH")
    assert 0 < body["days_to_next"] <= 355
    assert [a["account"] for a in body["accounts"]] == ["core", "paper"]
    for account in body["accounts"]:
        today = account["if_due_today"]
        assert {"trade_goods_zakat", "income_zakat", "chosen", "amount"} <= set(today)
        assert today["amount"] == max(today["trade_goods_zakat"], today["income_zakat"])
        assert account["last_recorded"] is None


def test_purification_lists_the_year_per_holding(client) -> None:
    body = client.get("/api/halal/purification?year=2026").json()
    assert body == {"year": 2026, "lines": [], "dividends": 0, "amount": 0}
