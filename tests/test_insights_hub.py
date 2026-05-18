"""Tests for the InsightsHub dataclass."""

from __future__ import annotations

from halal_trader.core.insights_hub import InsightsHub
from halal_trader.core.shadow import ShadowLedger
from halal_trader.ml.calibration import CalibrationCurve
from halal_trader.ml.drift import DriftMonitor


def test_hub_default_attributes_present() -> None:
    h = InsightsHub()
    assert isinstance(h.drift, DriftMonitor)
    # regime is None until the bot composes a DB engine and wires it.
    assert h.regime is None
    assert isinstance(h.shadow, ShadowLedger)
    assert isinstance(h.calibration, CalibrationCurve)


def test_hub_state_snapshot_has_expected_keys() -> None:
    h = InsightsHub()
    snap = h.snapshot()
    assert {
        "drift_monitor",
        "regime_memory",
        "shadow_ledger",
        "calibration_curve",
    } <= set(snap.keys())


def test_hub_modifications_persist_within_an_instance() -> None:
    """Two ``InsightsHub`` instances are independent — no module-level state."""
    h1 = InsightsHub()
    h1.shadow.record(cycle_id="c1", live_equity=100, shadow_equity=99)
    assert h1.shadow.size == 1

    h2 = InsightsHub()
    assert h2.shadow.size == 0


def test_app_state_keys_match_web_route_lookup() -> None:
    """Sanity-check: hub keys align with what the insights routes read."""
    h = InsightsHub()
    snap = h.snapshot()
    for key in ("drift_monitor", "shadow_ledger", "calibration_curve"):
        assert key in snap
