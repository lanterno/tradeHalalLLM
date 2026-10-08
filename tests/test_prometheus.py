"""Prometheus exposition + endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from halal_trader.web import app as web_app
from halal_trader.web.prometheus import MetricSnapshot, collect, render_metrics

# ── render_metrics ────────────────────────────────────────────


def test_render_emits_help_and_type_headers():
    snaps = [MetricSnapshot(name="x", help_text="a counter", value=42)]
    text = render_metrics(snaps)
    assert "# HELP x a counter" in text
    assert "# TYPE x gauge" in text
    assert "x 42" in text


def test_render_groups_headers_per_metric_name():
    """Multiple snapshots of the same metric only emit headers once."""
    snaps = [
        MetricSnapshot(name="open", help_text="positions", value=1, labels={"a": "X"}),
        MetricSnapshot(name="open", help_text="positions", value=2, labels={"a": "Y"}),
    ]
    text = render_metrics(snaps)
    assert text.count("# HELP open") == 1
    assert text.count("# TYPE open") == 1
    assert 'open{a="X"} 1' in text
    assert 'open{a="Y"} 2' in text


def test_render_escapes_label_values():
    snap = MetricSnapshot(name="x", help_text="x", value=1, labels={"k": 'v"\\n'})
    text = render_metrics([snap])
    # Quote, backslash, newline all escaped.
    assert 'k="v\\"\\\\n"' in text


def test_render_empty_returns_empty_string():
    assert render_metrics([]) == ""


# ── collect: every gauge from the database ────────────────────


async def test_the_gauges_are_read_from_the_database(engine) -> None:  # type: ignore[no-untyped-def]
    import json
    from datetime import UTC, datetime

    from sqlalchemy import text

    from halal_trader.core.heartbeat import STOCK_CYCLE, STOCK_PROCESS, beat

    await beat(engine, STOCK_PROCESS)
    await beat(engine, STOCK_CYCLE, {"risk": {"drawdown_pct": -0.02, "portfolio_heat_pct": 0.01}})
    async with engine.begin() as conn:
        for consumer, usd in (("stock", 0.3), ("shadow", 0.1), ("research", 0.5)):
            await conn.execute(
                text(
                    "INSERT INTO llm_spend (day, consumer, spent_usd, calls) VALUES (:d, :c, :u, 1)"
                ),
                {"d": datetime.now(UTC).date(), "c": consumer, "u": usd},
            )
        await conn.execute(
            text(
                "INSERT INTO account_snapshots (account, taken_at, equity, cash, positions) "
                "VALUES ('core', now(), 1000, 10, CAST(:p AS JSONB))"
            ),
            {"p": json.dumps([{"symbol": "MSFT"}, {"symbol": "AAPL"}])},
        )

    got = {(s.name, tuple(sorted(s.labels.items()))): s.value for s in await collect(engine)}

    assert got[("halal_trader_bot_running", ())] == 1.0
    assert ("halal_trader_heartbeat_age_seconds", (("component", STOCK_PROCESS),)) in got
    assert got[("halal_trader_drawdown_pct", ())] == -0.02
    assert got[("halal_trader_llm_spend_today_usd", (("pool", "live"),))] == 0.4
    assert got[("halal_trader_llm_spend_today_usd", (("pool", "research"),))] == 0.5
    assert got[("halal_trader_account_equity_usd", (("account", "core"),))] == 1000.0
    assert got[("halal_trader_open_positions", (("account", "core"),))] == 2.0


@pytest.fixture
def client(tmp_path, monkeypatch, database_url):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    with TestClient(web_app.create_app()) as c:
        yield c


def test_metrics_endpoint_returns_text(client) -> None:  # type: ignore[no-untyped-def]
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "halal_trader_bot_running 0" in r.text  # no bot has beaten


async def test_bot_running_comes_from_the_heartbeat(engine) -> None:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime, timedelta

    from halal_trader.core.heartbeat import STOCK_PROCESS, beat
    from halal_trader.web.prometheus import bot_alive

    assert await bot_alive(engine) is False
    await beat(engine, STOCK_PROCESS)
    assert await bot_alive(engine) is True
    await beat(engine, STOCK_PROCESS, now=datetime.now(UTC) - timedelta(minutes=10))
    assert await bot_alive(engine) is False
