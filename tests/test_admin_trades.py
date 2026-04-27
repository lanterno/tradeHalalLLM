"""Per-trade intervention endpoint tests."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "trades.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
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


def _seed_crypto_trade(client, *, side="buy", entry=70_000.0, sl=None, tp=None):
    """Insert one crypto trade row directly via the engine."""
    eng = web_app.app_state["engine"]
    import asyncio

    async def _seed():
        async with eng.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO crypto_trades "
                    "(timestamp, pair, side, quantity, price, filled_price, "
                    " entry_price, status, exchange, stop_loss, target_price) "
                    "VALUES (:ts, 'BTCUSDT', :side, 0.01, :p, :p, :p, "
                    "        'open', 'binance', :sl, :tp)"
                ),
                {"ts": "2026-04-26T12:00:00", "side": side, "p": entry, "sl": sl, "tp": tp},
            )
            row = await conn.execute(sa.text("SELECT max(id) FROM crypto_trades"))
            return row.scalar_one()

    return asyncio.run(_seed())


# ── Edit SL/TP ───────────────────────────────────────────────


def test_edit_sl_tp_round_trips(client):
    tid = _seed_crypto_trade(client, entry=70_000.0)
    r = client.patch(
        f"/api/admin/trades/{tid}/sl_tp",
        json={"asset_class": "crypto", "stop_loss": 68_000.0, "target_price": 72_000.0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stop_loss"] == 68_000.0
    assert body["target_price"] == 72_000.0


def test_edit_sl_tp_404_for_unknown(client):
    r = client.patch(
        "/api/admin/trades/9999/sl_tp",
        json={"asset_class": "crypto", "stop_loss": 1.0},
    )
    assert r.status_code == 404


def test_edit_rejects_sl_at_or_above_entry(client):
    tid = _seed_crypto_trade(client, entry=100.0)
    r = client.patch(
        f"/api/admin/trades/{tid}/sl_tp",
        json={"asset_class": "crypto", "stop_loss": 105.0},
    )
    assert r.status_code == 422


def test_edit_rejects_tp_at_or_below_entry(client):
    tid = _seed_crypto_trade(client, entry=100.0)
    r = client.patch(
        f"/api/admin/trades/{tid}/sl_tp",
        json={"asset_class": "crypto", "target_price": 95.0},
    )
    assert r.status_code == 422


def test_edit_requires_at_least_one_field(client):
    tid = _seed_crypto_trade(client)
    r = client.patch(
        f"/api/admin/trades/{tid}/sl_tp",
        json={"asset_class": "crypto"},
    )
    assert r.status_code == 422


def test_edit_rejects_sell_trade(client):
    tid = _seed_crypto_trade(client, side="sell")
    r = client.patch(
        f"/api/admin/trades/{tid}/sl_tp",
        json={"asset_class": "crypto", "stop_loss": 1.0},
    )
    assert r.status_code == 409


# ── Manual close ─────────────────────────────────────────────


def test_manual_close_marks_trade_closed(client):
    tid = _seed_crypto_trade(client, entry=70_000.0)
    r = client.post(
        f"/api/admin/trades/{tid}/close",
        json={"asset_class": "crypto", "exit_price": 71_000.0, "reason": "news_event"},
    )
    assert r.status_code == 200
    eng = web_app.app_state["engine"]
    import asyncio

    async def _read():
        async with eng.begin() as conn:
            row = await conn.execute(
                sa.text("SELECT status, exit_price, exit_reason FROM crypto_trades WHERE id = :i"),
                {"i": tid},
            )
            return row.first()

    status, exit_price, reason = asyncio.run(_read())
    assert status == "closed"
    assert exit_price == 71_000.0
    assert reason == "news_event"


# ── Audit drawer ─────────────────────────────────────────────


def test_audit_drawer_returns_full_payload(client):
    tid = _seed_crypto_trade(client, entry=70_000.0)
    r = client.get(f"/api/trades/crypto/{tid}/audit")
    assert r.status_code == 200
    body = r.json()
    assert body["trade"]["pair"] == "BTCUSDT"
    # No screening or snapshot was seeded — both should be None.
    assert body["receipt"]["compliance_status"] == "unattested"
    assert body["indicator_snapshot"] is None


def test_audit_drawer_404_for_unknown(client):
    r = client.get("/api/trades/crypto/9999/audit")
    assert r.status_code == 404


def test_audit_drawer_invalid_asset_class(client):
    r = client.get("/api/trades/options/1/audit")
    assert r.status_code == 400
