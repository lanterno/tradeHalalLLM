"""GET /api/insights/purification: what is owed, from the dividend ledger."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from halal_trader.web import app as web_app


async def _seed(url: str) -> None:
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        for account, did, symbol, amount, paid in (
            ("paper", "d1", "AAPL", 0.07, False),
            ("paper", "d2", "MSFT", 0.30, False),
            ("core", "d3", "MSFT", 1.25, False),
            ("core", "d4", "JNJ", 2.00, True),
            ("book:core", "d5", "MSFT", 9.99, False),  # a forward book: notional, left out
        ):
            await conn.execute(
                text(
                    "INSERT INTO purification_accruals (account, dividend_id, symbol, ex_date, "
                    "shares, dividend, impure_ratio, amount, method, accrued_at, paid_at) VALUES "
                    "(:a, :d, :s, '2026-08-20', 1, 1, 0.02, :m, 't', now(), "
                    "CASE WHEN :p THEN now() END)"
                ),
                {"a": account, "d": did, "s": symbol, "m": amount, "p": paid},
            )
    await engine.dispose()


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    with TestClient(web_app.create_app()) as c:
        yield c


def test_no_ledger_rows_is_unavailable(client) -> None:
    assert client.get("/api/insights/purification").json() == {"available": False}


def test_outstanding_purification_comes_from_the_dividend_ledger(client, database_url) -> None:
    asyncio.run(_seed(database_url))
    body = client.get("/api/insights/purification").json()
    assert body["available"] is True
    assert body["total_usd"] == pytest.approx(1.62)
    assert body["by_account"] == {"paper": 0.37, "core": 1.25}
    assert body["by_symbol"] == {"AAPL": 0.07, "MSFT": 1.55}
    assert body["disbursed_total_usd"] == 2.0 and body["n_entries"] == 3
