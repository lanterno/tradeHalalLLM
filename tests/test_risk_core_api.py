"""GET /api/risk/core: the core portfolio's concentration and drawdown."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from halal_trader.web import app as web_app
from halal_trader.web.routes.risk import drawdown


def test_drawdown_is_measured_from_the_accounts_own_peak() -> None:
    points = [(date(2026, 10, 1), 100.0), (date(2026, 10, 2), 110.0), (date(2026, 10, 5), 105.0)]
    d = drawdown(points, 99.0)
    assert d["drawdown_pct"] == pytest.approx(99 / 110 - 1, abs=1e-5)
    assert d["peak_equity"] == 110.0 and d["peak_day"] == "2026-10-02"
    assert d["history_from"] == "2026-10-01" and d["history_days"] == 3


def test_a_new_high_is_no_drawdown() -> None:
    d = drawdown([(date(2026, 10, 1), 100.0)], 101.0)
    assert d["drawdown_pct"] == 0.0 and d["peak_day"] is None and d["peak_equity"] == 101.0
    assert drawdown([], 50.0)["drawdown_pct"] == 0.0


async def _seed(url: str) -> None:
    engine = create_async_engine(url)
    positions = [
        {
            "symbol": f"S{i:02d}",
            "qty": 1.0,
            "price": 10.0 * (12 - i),
            "market_value": 10.0 * (12 - i),
            "unrealized_pl": 0.0,
            "change_today": 0.0,
        }
        for i in range(12)
    ]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO account_snapshots (account, taken_at, equity, cash, last_equity, "
                "positions) VALUES ('core', :t, 1000, 220, 1010, CAST(:p AS JSONB))"
            ),
            {"t": datetime.now(UTC) - timedelta(minutes=2), "p": json.dumps(positions)},
        )
        await conn.execute(
            text(
                "INSERT INTO broker_equity (account, day, equity, profit_loss, profit_loss_pct, "
                "synced_at) VALUES ('core', '2026-09-01', 1100, 0, 0, now())"
            )
        )
        for i in range(12):
            await conn.execute(
                text(
                    "INSERT INTO halal_screen_results (as_of, symbol, sic_description, verdict, "
                    "reasons, metrics, method, screened_at) VALUES ('2026-10-01', :s, :sic, :v, "
                    "'[]', '{}', 't', now())"
                ),
                {
                    "s": f"S{i:02d}",
                    "sic": "Services-Prepackaged Software"
                    if i < 3
                    else "Pharmaceutical Preparations",
                    "v": "not_halal" if i == 11 else "halal",
                },
            )
    await engine.dispose()


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    asyncio.run(_seed(database_url))
    with TestClient(web_app.create_app()) as c:
        yield c


def test_core_risk_reads_the_snapshot(client) -> None:
    body = client.get("/api/risk/core").json()
    assert body["available"] is True and body["source"] == "snapshot"
    assert body["positions"] == 12
    # Values 120, 110, ..., 10: the top ten hold 120 + ... + 30 = 750 of 1000.
    assert body["top10_weight"] == pytest.approx(0.75)
    assert body["largest"] == {"symbol": "S00", "weight": 0.12}
    assert body["sectors"][0] == {"sector": "Healthcare", "weight": pytest.approx(0.45)}
    assert body["cash_pct"] == pytest.approx(0.22)
    assert body["failing_screen"] == ["S11"]
    # The peak is the ledger's 1,100 on 09-01, not the current 1,000.
    assert body["drawdown_pct"] == pytest.approx(1000 / 1100 - 1, abs=1e-5)
    assert body["peak_day"] == "2026-09-01"


def test_core_risk_without_an_account_is_unavailable(database_url, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    with TestClient(web_app.create_app()) as c:
        assert c.get("/api/risk/core").json() == {"available": False}
