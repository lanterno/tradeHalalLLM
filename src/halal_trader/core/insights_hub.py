"""Process-wide hub for the new analytics modules.

The cycle, monitor, dashboard, CLI, and tests all want to read/write
the same drift monitor, regime memory, shadow ledger, calibration
curve, etc. Threading those through every constructor would be noise;
*one* explicit instance per process is cleaner — the web app
instantiates its own at startup and tests build fresh ones.

There is intentionally **no module-level singleton**. If you need a
hub, take it as a constructor argument or build a fresh one in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from halal_trader.core.shadow import ShadowLedger


@dataclass
class InsightsHub:
    """Container for in-process analytics state."""

    shadow: ShadowLedger = field(default_factory=ShadowLedger)
    # Optional reference to the dashboard's mutable RuntimeView so the
    # cycle can push live state (risk_state, last_cycle, …) into the
    # surface the dashboard reads. Threaded by the composition root.
    runtime: Any = None
    # RAG store over closed-trade rationales — populated by the post-
    # close fan-out, queried by the cycle to surface analogous
    # past setups.
    rag: object | None = None

    def snapshot(self) -> dict[str, Any]:
        """Dict snapshot of every insights component the dashboard exposes.

        Used by the ``/api/insights`` route (which now reads via
        ``Depends(get_ctx)`` and serialises this dict back out) and by
        the test suite. The shape is intentionally tolerant — None
        values for optional components are fine.
        """
        return {
            "shadow_ledger": self.shadow,
            "rag": self.rag,
        }
