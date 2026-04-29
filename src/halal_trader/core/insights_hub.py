"""Process-wide hub for the new analytics modules.

The cycle, monitor, dashboard, CLI, and tests all want to read/write
the same drift monitor, regime memory, shadow ledger, calibration
curve, etc. Threading those through every constructor would be noise;
*one* explicit instance per process is cleaner — the bot constructs it
in ``crypto/components.py`` and passes it to the cycle / monitor /
post-close fan-out, the web app instantiates its own at startup.

There is intentionally **no module-level singleton**. If you need a
hub, take it as a constructor argument or build a fresh one in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from halal_trader.core.shadow import ShadowLedger
from halal_trader.crypto.basis import BasisTracker
from halal_trader.ml.calibration import CalibrationCurve
from halal_trader.ml.drift import DriftMonitor
from halal_trader.ml.regime_memory import RegimeMemory


@dataclass
class InsightsHub:
    """Container for in-process analytics state."""

    drift: DriftMonitor = field(default_factory=DriftMonitor)
    # DB-backed once the bot is composed (CryptoComponents wires it
    # from the engine). Stays None for pure-CLI / dashboard contexts
    # that don't run the cycle.
    regime: RegimeMemory | None = None
    shadow: ShadowLedger = field(default_factory=ShadowLedger)
    calibration: CalibrationCurve = field(default_factory=CalibrationCurve.identity)
    basis: BasisTracker = field(default_factory=BasisTracker)
    # Optional reference to the dashboard's mutable RuntimeView so the
    # cycle can push live state (risk_state, last_cycle, …) into the
    # surface the dashboard reads. Threaded by the composition root.
    runtime: Any = None
    # Latest computed velocity result per symbol; populated by the
    # sentiment manager once it exposes raw mention timestamps.
    velocity: dict[str, Any] = field(default_factory=dict)
    # RAG store over closed-trade rationales — populated by the post-
    # close fan-out, queried by the cycle to surface analogous
    # past setups.
    rag: object | None = None
    # Latest on-chain whale-flow signals per ERC-20 symbol; populated
    # by the cycle from EtherscanWhaleFlow when ETHERSCAN_API_KEY is
    # set. Empty when the source is disabled.
    whale_flows: dict[str, Any] = field(default_factory=dict)

    def to_app_state(self) -> dict[str, Any]:
        """Snapshot suitable for ``app_state["insights"]`` (web routes)."""
        return {
            "drift_monitor": self.drift,
            "regime_memory": self.regime,
            "shadow_ledger": self.shadow,
            "calibration_curve": self.calibration,
            "basis_tracker": self.basis,
            "velocity": self.velocity,
            "rag": self.rag,
            "whale_flows": self.whale_flows,
        }
