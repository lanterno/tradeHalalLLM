"""GET /api/core for the dashboard's Core page."""

from __future__ import annotations

import asyncio
from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from halal_trader.web import app as web_app


async def _seed(url: str) -> None:
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO forward_books (name, strategy, params, created_at) "
                "VALUES ('core', 'core-strict-cap', '{}', now())"
            )
        )
        for day, nav, equity in (
            (date(2026, 10, 5), 1.0, 1000.0),
            (date(2026, 10, 6), 1.02, 1010.0),
        ):
            await conn.execute(
                text(
                    "INSERT INTO forward_book_days (book, day, nav, day_return, turnover, weights, "
                    "rebalance_next, recorded_at) VALUES ('core', :d, :n, 0, 0, "
                    '\'{"MSFT": 0.6, "NVDA": 0.4}\', false, now())'
                ),
                {"d": day, "n": nav},
            )
            await conn.execute(
                text(
                    "INSERT INTO broker_equity (account, day, equity, profit_loss, "
                    "profit_loss_pct, synced_at) VALUES ('core', :d, :e, 0, 0, now())"
                ),
                {"d": day, "e": equity},
            )
        await conn.execute(
            text(
                "INSERT INTO broker_activities (id, account, activity_type, transaction_time, "
                "symbol, side, qty, price, raw) VALUES ('c1', 'core', 'FILL', :t, 'MSFT', 'buy', "
                "1.5, 400, '{}')"
            ),
            {"t": datetime.fromisoformat("2026-10-05 15:41:00-04:00")},
        )
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES ('MSFT', '2026-10-06', 'raw', 404, 404, 404, 404, 1, now())"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO core_orders (submitted_at, symbol, side, qty, est_price, notional, "
                "reason, status) VALUES (:t, 'MSFT', 'buy', 1.5, 400, 600, 'rebalance', "
                "'submitted')"
            ),
            {"t": datetime.fromisoformat("2026-10-05 15:40:30-04:00")},
        )
    await engine.dispose()


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    from halal_trader import config

    monkeypatch.setattr(config, "_settings", None)
    asyncio.run(_seed(database_url))
    app = web_app.create_app()
    with TestClient(app) as c:
        yield c
    config._settings = None


def test_core_page_shows_holdings_against_targets_and_the_gate(client) -> None:
    body = client.get("/api/core").json()

    assert body["equity"] == 1010.0
    by_symbol = {h["symbol"]: h for h in body["holdings"]}
    assert by_symbol["MSFT"]["value"] == 606.0  # 1.5 shares at the last close
    assert by_symbol["MSFT"]["weight"] == pytest.approx(0.6, abs=1e-4)
    assert by_symbol["NVDA"]["shares"] is None and by_symbol["NVDA"]["target"] == 0.4
    assert body["series"][-1] == {"date": "2026-10-06", "account": 101.0, "book": 102.0}
    assert body["readiness"]["ready"] is False and body["readiness"]["min_days"] == 20
    assert [o["symbol"] for o in body["orders"]] == ["MSFT"]
    assert "alpaca_api_key" not in str(body)
