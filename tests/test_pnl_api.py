"""GET /api/pnl/daily: a calendar window, and the equity change named as such."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from halal_trader.market_hours import today_eastern
from halal_trader.web import app as web_app
from halal_trader.web.routes.pnl import with_equity_change


async def _seed(url: str) -> None:
    today = today_eastern()
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        for ago, start, end, realized in (
            (0, 100.0, None, 0.0),  # today, still open
            (3, 100.0, 101.5, 1.5),
            (29, 100.0, 99.0, 5429.33),  # an old row whose "realized" is not the change
            (31, 100.0, 102.0, 2.0),  # outside a 30-day window
            (120, 100.0, 103.0, 3.0),
        ):
            await conn.execute(
                text(
                    "INSERT INTO daily_pnl (date, starting_equity, ending_equity, realized_pnl, "
                    "trades_count) VALUES (:d, :s, :e, :r, 0)"
                ),
                {
                    "d": (today - timedelta(days=ago)).isoformat(),
                    "s": start,
                    "e": end,
                    "r": realized,
                },
            )
    await engine.dispose()


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    asyncio.run(_seed(database_url))
    with TestClient(web_app.create_app()) as c:
        yield c


def test_days_is_a_calendar_window_not_a_row_count(client) -> None:
    rows = client.get("/api/pnl/daily?days=30").json()
    today = today_eastern()
    assert [r["date"] for r in rows] == [
        (today - timedelta(days=ago)).isoformat() for ago in (0, 3, 29)
    ]
    assert len(client.get("/api/pnl/daily?days=365").json()) == 5


def test_each_day_carries_its_equity_change(client) -> None:
    rows = {r["date"]: r for r in client.get("/api/pnl/daily?days=30").json()}
    today = today_eastern()
    assert rows[today.isoformat()]["equity_change"] is None  # the day is still open
    old = rows[(today - timedelta(days=29)).isoformat()]
    assert old["equity_change"] == -1.0 and old["realized_pnl"] == 5429.33


def test_equity_change_is_end_minus_start() -> None:
    assert with_equity_change({"starting_equity": 98681.66, "ending_equity": 98955.44})[
        "equity_change"
    ] == pytest.approx(273.78)
