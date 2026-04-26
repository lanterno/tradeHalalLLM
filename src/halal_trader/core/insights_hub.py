"""Process-wide hub for the new analytics modules.

The cycle, monitor, dashboard, CLI, and tests all want to read/write
the same drift monitor, regime memory, shadow ledger, calibration
curve, etc. Threading those through every constructor is noise — and
the alternative (singletons sprinkled across modules) is hard to test
and impossible to swap.

This module gives one explicit place to look:

    from halal_trader.core.insights_hub import hub
    hub.drift.observe(pnl_pct)
    hub.regime.add_today(features, ...)
    hub.shadow.record(cycle_id=..., live_equity=..., shadow_equity=...)

Every attribute is opt-in: cycles that don't care simply don't write,
and dashboards / CLIs that don't see writes show "not available".

Tests reset the hub via :func:`reset_hub` (called from a fixture).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from halal_trader.core.shadow import ShadowLedger
from halal_trader.ml.calibration import CalibrationCurve
from halal_trader.ml.drift import DriftMonitor
from halal_trader.ml.regime_memory import RegimeMemory


@dataclass
class InsightsHub:
    """Container for in-process analytics state."""

    drift: DriftMonitor = field(default_factory=DriftMonitor)
    regime: RegimeMemory = field(default_factory=RegimeMemory)
    shadow: ShadowLedger = field(default_factory=ShadowLedger)
    calibration: CalibrationCurve = field(default_factory=CalibrationCurve.identity)

    def to_app_state(self) -> dict:
        """Snapshot suitable for ``app_state["insights"]`` (web routes)."""
        return {
            "drift_monitor": self.drift,
            "regime_memory": self.regime,
            "shadow_ledger": self.shadow,
            "calibration_curve": self.calibration,
        }


hub = InsightsHub()


def reset_hub() -> None:
    """Replace every analytic with a fresh default. Intended for tests."""
    global hub
    hub = InsightsHub()
