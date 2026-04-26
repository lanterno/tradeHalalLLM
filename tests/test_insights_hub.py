"""Tests for the insights hub."""

from __future__ import annotations

from halal_trader.core.insights_hub import hub, reset_hub
from halal_trader.core.shadow import ShadowLedger
from halal_trader.ml.calibration import CalibrationCurve
from halal_trader.ml.drift import DriftMonitor
from halal_trader.ml.regime_memory import RegimeMemory


def test_hub_default_attributes_present() -> None:
    reset_hub()
    from halal_trader.core.insights_hub import hub as h

    assert isinstance(h.drift, DriftMonitor)
    assert isinstance(h.regime, RegimeMemory)
    assert isinstance(h.shadow, ShadowLedger)
    assert isinstance(h.calibration, CalibrationCurve)


def test_hub_state_snapshot_has_expected_keys() -> None:
    reset_hub()
    from halal_trader.core.insights_hub import hub as h

    snap = h.to_app_state()
    assert {
        "drift_monitor",
        "regime_memory",
        "shadow_ledger",
        "calibration_curve",
    } <= set(snap.keys())


def test_hub_modifications_persist_until_reset() -> None:
    reset_hub()
    from halal_trader.core.insights_hub import hub as h

    h.shadow.record(cycle_id="c1", live_equity=100, shadow_equity=99)
    assert h.shadow.size == 1
    reset_hub()
    from halal_trader.core.insights_hub import hub as h2

    assert h2.shadow.size == 0


def test_app_state_keys_match_web_route_lookup() -> None:
    """Sanity-check: hub keys align with what the insights routes read."""
    reset_hub()
    snap = hub.to_app_state()
    # The web routes look these up under app_state["insights"]
    for key in ("drift_monitor", "shadow_ledger", "calibration_curve"):
        assert key in snap
