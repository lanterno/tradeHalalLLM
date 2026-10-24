"""GET /api/trades: the filters pick the rows before the page is cut."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import Trade
from halal_trader.web import app as web_app


async def _store(url: str, trades: list[Trade]) -> None:
    engine = create_async_engine(url)
    async with AsyncSession(engine) as session:
        session.add_all(trades)
        await session.commit()
    await engine.dispose()


def _trade(symbol: str, at: datetime, *, side: str = "buy") -> Trade:
    return Trade(symbol=symbol, side=side, quantity=1, price=100, status="filled", timestamp=at)


@pytest.fixture
def make_client(database_url, tmp_path, monkeypatch) -> Iterator[Callable[..., TestClient]]:
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    clients: list[TestClient] = []

    def make(*trades: Trade) -> TestClient:
        asyncio.run(_store(database_url, list(trades)))
        c = TestClient(web_app.create_app())
        c.__enter__()
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def test_a_symbol_older_than_the_newest_page_is_still_found(make_client) -> None:
    now = datetime(2026, 10, 22, 15, tzinfo=UTC)
    newest = [_trade("AAPL", now - timedelta(minutes=i)) for i in range(30)]
    client = make_client(*newest, _trade("INTU", now - timedelta(days=3)))

    rows = client.get("/api/trades", params={"symbol": "INTU", "limit": 25}).json()

    assert [r["symbol"] for r in rows] == ["INTU"]


def test_dates_are_eastern_trading_days_both_inclusive(make_client) -> None:
    client = make_client(
        _trade("A", datetime(2026, 10, 20, 15, tzinfo=UTC)),  # 11:00 ET on the 20th
        _trade("B", datetime(2026, 10, 21, 14, tzinfo=UTC)),  # 10:00 ET on the 21st
        _trade("C", datetime(2026, 10, 22, 3, 30, tzinfo=UTC)),  # 23:30 ET, still the 21st
        _trade("D", datetime(2026, 10, 22, 14, tzinfo=UTC)),  # 10:00 ET on the 22nd
    )

    r = client.get("/api/trades", params={"from_date": "2026-10-21", "to_date": "2026-10-21"})

    assert r.status_code == 200
    assert [t["symbol"] for t in r.json()] == ["C", "B"]


def test_offset_pages_through_the_filtered_rows(make_client) -> None:
    now = datetime(2026, 10, 22, 15, tzinfo=UTC)
    sells = [_trade(f"S{i}", now - timedelta(hours=i), side="sell") for i in range(4)]
    buys = [_trade(f"B{i}", now - timedelta(minutes=30 + 60 * i)) for i in range(4)]
    client = make_client(*sells, *buys)

    rows = client.get("/api/trades", params={"side": "sell", "limit": 2, "offset": 2}).json()

    assert [r["symbol"] for r in rows] == ["S2", "S3"]


def test_a_date_that_is_not_one_is_refused(make_client) -> None:
    client = make_client()

    assert client.get("/api/trades", params={"from_date": "yesterday"}).status_code == 422
