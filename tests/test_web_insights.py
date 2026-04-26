"""Tests for /api/insights/* routes."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from halal_trader.core.shadow import ShadowLedger
from halal_trader.ml.calibration import CalibrationCurve
from halal_trader.ml.drift import DriftMonitor
from halal_trader.web.routes.insights import register


def _client(insights_state: dict | None = None) -> TestClient:
    app = FastAPI()
    state: dict = {"insights": insights_state or {}}
    register(app, state)
    return TestClient(app)


# ── drift ────────────────────────────────────────────────────────


def test_drift_unavailable_without_monitor() -> None:
    client = _client()
    r = client.get("/api/insights/drift")
    assert r.status_code == 200
    assert r.json() == {"available": False}


def test_drift_with_monitor() -> None:
    mon = DriftMonitor()
    for _ in range(50):
        mon.observe(0.0)
    client = _client({"drift_monitor": mon})
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
    client = _client({"shadow_ledger": led})
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
    client = _client({"stress_verdicts": verdicts, "stress_ts": "2026-04-26T00:00:00Z"})
    body = client.get("/api/insights/stress").json()
    assert body["available"] is True
    assert body["verdicts"][0]["scenario_name"] == "flash_crash"
    assert body["verdicts"][0]["passed"] is True


# ── calibration ──────────────────────────────────────────────────


def test_calibration_unavailable() -> None:
    client = _client()
    assert client.get("/api/insights/calibration").json() == {"available": False}


def test_calibration_with_curve() -> None:
    curve = CalibrationCurve(anchors=[(0.0, 0.0), (1.0, 0.7)], method="platt", n_samples=100)
    client = _client({"calibration_curve": curve})
    body = client.get("/api/insights/calibration").json()
    assert body["available"] is True
    assert body["method"] == "platt"
    assert body["n_samples"] == 100


# ── new surfaces ─────────────────────────────────────────────────


def test_regime_unavailable() -> None:
    client = _client()
    assert client.get("/api/insights/regime").json() == {"available": False}


def test_regime_with_snapshots() -> None:
    from halal_trader.ml.regime_memory import RegimeFeatures, RegimeMemory

    mem = RegimeMemory()
    mem.add_today(RegimeFeatures(volatility=0.01), today="2026-04-26", outcome_pnl_pct=0.01)
    client = _client({"regime_memory": mem})
    body = client.get("/api/insights/regime").json()
    assert body["available"] is True
    assert body["size"] == 1
    assert body["recent"][0]["date"] == "2026-04-26"


def test_basis_unavailable() -> None:
    client = _client()
    assert client.get("/api/insights/basis").json() == {"available": False}


def test_basis_with_history() -> None:
    from halal_trader.crypto.basis import BasisTracker

    tracker = BasisTracker()
    tracker.observe(pair="BTCUSDT", spot_price=100.0, perp_price=100.5, funding_rate_pct=0.0001)
    client = _client({"basis_tracker": tracker})
    body = client.get("/api/insights/basis").json()
    assert body["available"] is True
    assert "BTCUSDT" in body["pairs"]


def test_treasury_unavailable_without_account_snapshot() -> None:
    client = _client()
    assert client.get("/api/insights/treasury").json() == {"available": False}
