"""GET /api/positions: both accounts, the core first, marked by the broker."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from halal_trader.web import app as web_app
from halal_trader.web.routes.positions import broker_position


async def _run(url: str, *statements: tuple[str, dict]) -> None:
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        for sql, params in statements:
            await conn.execute(text(sql), params)
    await engine.dispose()


SNAPSHOT = (
    "INSERT INTO account_snapshots (account, taken_at, equity, cash, last_equity, positions) "
    "VALUES (:a, :t, :e, :c, :e, CAST(:p AS JSONB))"
)
OPEN_MSFT = (
    "INSERT INTO trades (timestamp, symbol, side, quantity, price, status, filled_price, "
    "filled_quantity, stop_loss, entry_type) VALUES (:t, 'MSFT', 'buy', 24, 401.84, 'filled', "
    "401.84, 24, 489.67, 'reactor_momentum')",
    {"t": datetime(2026, 7, 20, 18, 16, tzinfo=UTC)},
)


@pytest.fixture
def make_client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("CORE_ENABLED", "true")
    monkeypatch.setenv("DAY_TRADER_ENABLED", "false")
    from halal_trader import config

    monkeypatch.setattr(config, "_settings", None)
    clients = []

    def make(*statements: tuple[str, dict]) -> TestClient:
        asyncio.run(_run(database_url, *statements))
        c = TestClient(web_app.create_app())
        c.__enter__()
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)
    config._settings = None


def test_a_broker_position_carries_its_cost() -> None:
    p = broker_position(
        {
            "symbol": "MSFT",
            "qty": 24,
            "price": 529.7,
            "market_value": 12712.8,
            "unrealized_pl": 3068.64,
            "change_today": 0.00861,
        }
    )
    assert p["cost_basis"] == pytest.approx(9644.16)
    assert p["avg_entry"] == pytest.approx(401.84)
    assert p["unrealized_pl_pct"] == pytest.approx(3068.64 / 9644.16, abs=1e-5)


def test_positions_are_marked_by_the_snapshot_core_first(make_client) -> None:
    taken = datetime.now(UTC) - timedelta(minutes=3)
    msft = {
        "symbol": "MSFT",
        "qty": 24.0,
        "price": 529.7,
        "prev_close": 525.18,
        "change_today": 0.00861,
        "market_value": 12712.8,
        "unrealized_pl": 3068.64,
    }
    core = [
        {
            "symbol": "AAPL",
            "qty": 2.0,
            "price": 300.0,
            "market_value": 600.0,
            "unrealized_pl": -10.0,
            "change_today": -0.002,
        },
        {
            "symbol": "A",
            "qty": 1.0,
            "price": 170.0,
            "market_value": 170.0,
            "unrealized_pl": 1.0,
            "change_today": 0.0,
        },
    ]
    client = make_client(
        (
            SNAPSHOT,
            {"a": "paper", "t": taken, "e": 105381.85, "c": 92669.05, "p": json.dumps([msft])},
        ),
        (SNAPSHOT, {"a": "core", "t": taken, "e": 1000.0, "c": 230.0, "p": json.dumps(core)}),
        OPEN_MSFT,
    )
    body = client.get("/api/positions").json()

    assert [a["account"] for a in body["accounts"]] == ["core", "paper"]
    core_acct, paper = body["accounts"]
    assert core_acct["status"] == "active" and core_acct["source"] == "snapshot"
    assert [p["symbol"] for p in core_acct["positions"]] == ["AAPL", "A"]  # largest first
    assert core_acct["positions"][0]["weight"] == pytest.approx(0.6)
    assert core_acct["invested"] == 770.0 and core_acct["unrealized_pl"] == -9.0
    assert 150 <= core_acct["age_seconds"] <= 400

    assert paper["status"] == "retired"
    (pos,) = paper["positions"]
    # Marked at the broker's price, not at entry: the gain is the broker's.
    assert pos["price"] == 529.7 and pos["unrealized_pl"] == 3068.64
    assert pos["avg_entry"] == pytest.approx(401.84)
    assert pos["stop_loss"] == 489.67 and pos["opened_at"].startswith("2026-07-20")


def test_without_a_snapshot_the_ledger_is_marked_at_the_last_close(make_client) -> None:
    client = make_client(
        OPEN_MSFT,
        (
            "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
            "fetched_at) VALUES ('MSFT', '2026-10-05', 'raw', 520, 530, 518, 525.18, 1, now())",
            {},
        ),
    )
    (paper,) = client.get("/api/positions").json()["accounts"]
    assert paper["source"] == "ledger" and paper["as_of"] == "2026-10-05"
    (pos,) = paper["positions"]
    assert pos["price"] == 525.18
    assert pos["unrealized_pl"] == pytest.approx(24 * (525.18 - 401.84), abs=0.01)
