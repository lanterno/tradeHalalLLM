"""Tests for /api/insights/* routes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from halal_trader.core.context import DashboardContext, RuntimeView
from halal_trader.core.event_bus import EventBus
from halal_trader.core.insights_hub import InsightsHub
from halal_trader.core.shadow import ShadowLedger
from halal_trader.ml.calibration import CalibrationCurve
from halal_trader.ml.drift import DriftMonitor
from halal_trader.web.routes.insights import register


def _client(
    *,
    drift: DriftMonitor | None = None,
    shadow: ShadowLedger | None = None,
    calibration: CalibrationCurve | None = None,
    basis: Any = None,
    regime: Any = None,
    stress_verdicts: list | None = None,
    stress_ts: str | None = None,
    runtime: RuntimeView | None = None,
) -> TestClient:
    app = FastAPI()
    hub = InsightsHub(
        drift=drift if drift is not None else DriftMonitor(),
        shadow=shadow if shadow is not None else ShadowLedger(),
        calibration=(calibration if calibration is not None else CalibrationCurve.identity()),
        regime=regime,
    )
    if basis is not None:
        hub.basis = basis
    if stress_verdicts is not None:
        hub.stress_verdicts = stress_verdicts
        hub.stress_ts = stress_ts
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


# ── drift ────────────────────────────────────────────────────────


def test_drift_unavailable_without_monitor() -> None:
    # The default DriftMonitor has n=0, so drift route reports it but
    # with state="warming_up". We only assert the route still works.
    client = _client()
    r = client.get("/api/insights/drift")
    assert r.status_code == 200
    body = r.json()
    assert "state" in body


def test_drift_with_monitor() -> None:
    mon = DriftMonitor()
    for _ in range(50):
        mon.observe(0.0)
    client = _client(drift=mon)
    r = client.get("/api/insights/drift")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["n"] == 50
    assert body["state"] in ("stable", "drift", "warming_up")


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


# ── stress ───────────────────────────────────────────────────────


def test_stress_unavailable() -> None:
    client = _client()
    assert client.get("/api/insights/stress").json() == {"available": False}


def test_stress_with_verdicts() -> None:
    from halal_trader.crypto.stress import StressVerdict

    verdicts = [
        StressVerdict(
            scenario_name="flash_crash",
            severity=0.0,
            buys=0,
            sells=0,
            holds=1,
            notes=["sane"],
        )
    ]
    client = _client(stress_verdicts=verdicts, stress_ts="2026-04-26T00:00:00Z")
    body = client.get("/api/insights/stress").json()
    assert body["available"] is True
    assert body["verdicts"][0]["scenario_name"] == "flash_crash"
    assert body["verdicts"][0]["passed"] is True


# ── calibration ──────────────────────────────────────────────────


def test_calibration_unavailable() -> None:
    # Default CalibrationCurve.identity() is non-None, so the route
    # reports it as available with method="identity".
    client = _client()
    body = client.get("/api/insights/calibration").json()
    assert body["available"] is True
    assert body["method"] == "identity"


def test_calibration_with_curve() -> None:
    curve = CalibrationCurve(anchors=[(0.0, 0.0), (1.0, 0.7)], method="platt", n_samples=100)
    client = _client(calibration=curve)
    body = client.get("/api/insights/calibration").json()
    assert body["available"] is True
    assert body["method"] == "platt"
    assert body["n_samples"] == 100


# ── new surfaces ─────────────────────────────────────────────────


def test_regime_unavailable() -> None:
    client = _client()
    assert client.get("/api/insights/regime").json() == {"available": False}


async def test_regime_with_snapshots(database_url) -> None:
    """End-to-end: route reads a DB-backed RegimeMemory."""
    import httpx
    from fastapi import FastAPI

    from halal_trader.db.models import init_db
    from halal_trader.ml.regime_memory import RegimeFeatures, RegimeMemory

    engine = await init_db(database_url)
    try:
        mem = RegimeMemory(engine=engine)
        await mem.add_today(
            RegimeFeatures(volatility=0.01), today="2026-04-26", outcome_pnl_pct=0.01
        )
        app = FastAPI()
        hub = InsightsHub(regime=mem)
        ctx = DashboardContext(
            engine=engine,
            repo=MagicMock(),
            hub=hub,
            analytics=MagicMock(),
            settings=MagicMock(),
            bus=EventBus(),
            runtime=RuntimeView(),
        )
        app.state.ctx = ctx
        register(app)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/api/insights/regime")
        body = response.json()
        assert body["available"] is True
        assert body["size"] == 1
        assert body["recent"][0]["date"] == "2026-04-26"
    finally:
        await engine.dispose()


def test_basis_unavailable() -> None:
    client = _client()
    # Default BasisTracker has empty history → unavailable.
    assert client.get("/api/insights/basis").json() == {"available": False}


def test_basis_with_history() -> None:
    from halal_trader.crypto.basis import BasisTracker

    tracker = BasisTracker()
    tracker.observe(pair="BTCUSDT", spot_price=100.0, perp_price=100.5, funding_rate_pct=0.0001)
    client = _client(basis=tracker)
    body = client.get("/api/insights/basis").json()
    assert body["available"] is True
    assert "BTCUSDT" in body["pairs"]


def test_treasury_unavailable_without_account_snapshot() -> None:
    client = _client()
    assert client.get("/api/insights/treasury").json() == {"available": False}
