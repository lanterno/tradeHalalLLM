"""Tests for /api/insights/* routes."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from halal_trader.core.context import DashboardContext, RuntimeView
from halal_trader.core.event_bus import EventBus
from halal_trader.core.insights_hub import InsightsHub
from halal_trader.core.shadow import ShadowLedger
from halal_trader.web.routes.insights import register


def _client(
    *,
    shadow: ShadowLedger | None = None,
    runtime: RuntimeView | None = None,
) -> TestClient:
    app = FastAPI()
    hub = InsightsHub(shadow=shadow if shadow is not None else ShadowLedger())
    ctx = DashboardContext(
        engine=MagicMock(),
        repo=MagicMock(),
        hub=hub,
        analytics=MagicMock(),
        settings=MagicMock(),
        bus=EventBus(),
        runtime=runtime if runtime is not None else RuntimeView(),
    )
    app.state.ctx = ctx
    register(app)
    return TestClient(app)


# ── shadow ───────────────────────────────────────────────────────


def test_shadow_unavailable_without_ledger() -> None:
    client = _client()
    r = client.get("/api/insights/shadow")
    assert r.json() == {"available": False}


def test_shadow_with_ledger() -> None:
    led = ShadowLedger()
    for i in range(40):
        led.record(cycle_id=f"c{i}", live_equity=100, shadow_equity=100 + i * 0.05)
    client = _client(shadow=led)
    body = client.get("/api/insights/shadow").json()
    assert body["available"] is True
    assert body["n"] == 40
    assert body["level"] in ("ok", "watch", "diverged")


# ── new surfaces ─────────────────────────────────────────────────


def test_treasury_unavailable_without_account_snapshot() -> None:
    client = _client()
    assert client.get("/api/insights/treasury").json() == {"available": False}
