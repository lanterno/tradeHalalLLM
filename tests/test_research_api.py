"""Research API endpoint tests — replay, prompt-version diff, halal audit."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    web_app.app_state.clear()
    app = web_app.create_app()

    with TestClient(app) as c:
        yield c


def _sync_engine():
    from sqlalchemy import create_engine

    from halal_trader.config import get_settings

    return create_engine(get_settings().database_url_sync())


def _seed_decisions(client: TestClient) -> None:
    """Insert a couple of decisions covering two prompt versions."""
    from datetime import UTC, datetime

    eng = _sync_engine()
    try:
        with eng.begin() as conn:
            for minute, version, in_tok, out_tok, cache, cost in [
                (0, "v1", 1000, 100, 800, 0.01),
                (1, "v1", 1100, 110, 900, 0.011),
                (2, "v2", 950, 90, 700, 0.009),
            ]:
                conn.execute(
                    sa.text(
                        "INSERT INTO llm_decisions "
                        "(timestamp, provider, model, prompt_summary, raw_response, "
                        " prompt_version, input_tokens, output_tokens, "
                        " cache_read_tokens, cost_usd) "
                        "VALUES (:ts, 'anthropic', 'claude-opus-4-7', :sum, '{}', "
                        " :ver, :it, :ot, :crt, :cost)"
                    ),
                    {
                        "ts": datetime(2026, 4, 26, 12, minute, 0, tzinfo=UTC),
                        "sum": f"cycle {minute}",
                        "ver": f"crypto.strategy.system@{version}",
                        "it": in_tok,
                        "ot": out_tok,
                        "crt": cache,
                        "cost": cost,
                    },
                )
    finally:
        eng.dispose()


def test_replay_unknown_decision_returns_404(client):
    r = client.get("/api/research/replay/9999")
    assert r.status_code == 404


def test_replay_returns_decision_payload(client):
    _seed_decisions(client)
    r = client.get("/api/research/replay/1")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "anthropic"
    assert body["prompt_version"] == "crypto.strategy.system@v1"
    assert body["cost_usd"] == 0.01


def test_prompt_versions_groups_and_aggregates(client):
    _seed_decisions(client)
    r = client.get("/api/research/prompt-versions")
    assert r.status_code == 200
    rows = r.json()
    by_version = {row["version"]: row for row in rows}
    v1 = by_version["crypto.strategy.system@v1"]
    v2 = by_version["crypto.strategy.system@v2"]
    assert v1["count"] == 2
    assert v2["count"] == 1
    # v1 cost = 0.01 + 0.011 = 0.021
    assert abs(v1["total_cost_usd"] - 0.021) < 1e-9
    # cache_read_ratio = 1700 / 2100 ≈ 0.81
    assert 0.7 < v1["cache_read_ratio"] < 0.9


def test_halal_audit_one_trade_404(client):
    r = client.get("/api/research/halal-audit/crypto/9999")
    assert r.status_code == 404


def test_halal_audit_invalid_asset_class(client):
    r = client.get("/api/research/halal-audit/options/1")
    assert r.status_code == 400


def test_halal_audit_for_symbol_returns_empty_list(client):
    r = client.get("/api/research/halal-audit/crypto/symbol/BTCUSDT")
    assert r.status_code == 200
    assert r.json() == []


def test_halal_audit_for_symbol_returns_receipts(client):
    """After seeding a trade with a screening FK, the audit returns receipts."""
    from datetime import UTC, datetime

    eng = _sync_engine()
    ts = datetime(2026, 4, 26, 12, 0, 0, tzinfo=UTC)
    try:
        with eng.begin() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO halal_screenings "
                    "(timestamp, symbol, asset_class, source, decision, cache_hit) "
                    "VALUES (:ts, 'BTC', 'crypto', 'coingecko_rules', 'halal', false) "
                    "RETURNING id"
                ),
                {"ts": ts},
            )
            sid = conn.execute(sa.text("SELECT max(id) FROM halal_screenings")).scalar_one()
            conn.execute(
                sa.text(
                    "INSERT INTO crypto_trades "
                    "(timestamp, pair, side, quantity, price, halal_screening_id, "
                    " status, exchange) "
                    "VALUES (:ts, 'BTCUSDT', 'buy', 0.01, 70000.0, :sid, 'open', "
                    " 'binance')"
                ),
                {"ts": ts, "sid": sid},
            )
    finally:
        eng.dispose()

    r = client.get("/api/research/halal-audit/crypto/symbol/BTCUSDT")
    assert r.status_code == 200
    receipts = r.json()
    assert len(receipts) == 1
    assert receipts[0]["compliance_status"] == "halal"
