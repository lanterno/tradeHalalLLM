"""Post-close analytics fan-out — one call hooks every recorder.

The monitor / executor close path needs to fire several recorders
when a trade closes (drift observation, thesis tag, regret record,
round-trip purification, ML retraining label). Calling each of them
inline at the close-site bloats the monitor and makes it harder to
test.

This module exposes a single :func:`record_close` that takes a small
:class:`CloseEvent` describing the closed trade plus an optional
context (indicators at entry, reasoning) and dispatches to:

* :class:`DriftMonitor` (process-wide, via ``insights_hub.drift``)
* :class:`ShadowLedger` — *not* wired here (driven by cycle, not close)
* :class:`ThesisTagStore` (persistent sidecar) + heuristic tagger
* Regret sidecar (JSON) via :class:`RegretSidecar`
* :class:`RoundTripLedger` purification accrual

Each step is best-effort: a failure in one recorder does not prevent
the others from running. The monitor logs a debug-level trace for
each failure but the call to :func:`record_close` never raises.

The ``insights_hub`` is the default sink; tests can pass an explicit
hub to assert behaviour in isolation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Close event ──────────────────────────────────────────────────


@dataclass
class CloseEvent:
    """Minimum data needed by every post-close recorder."""

    trade_id: str
    symbol: str  # 'BTCUSDT' for crypto, 'AAPL' for stocks
    side: str  # 'buy' or 'sell' — almost always 'buy' for closing entries
    entry_price: float
    exit_price: float
    exit_reason: str
    realized_pnl_usd: float
    return_pct: float
    quantity: float = 0.0
    indicators: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    setup_type: str | None = None
    hold_seconds: int = 0
    confidence: float | None = None
    closed_at: datetime | None = None


# ── Sidecar for regret records ───────────────────────────────────


@dataclass
class RegretSidecar:
    """JSON-on-disk append-only store of RegretRecord rows."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("regret sidecar unreadable: %s — starting fresh", exc)
            return []

    def append(self, record: dict) -> None:
        records = self._load()
        records.append(record)
        self.path.write_text(json.dumps(records, indent=2))

    def all(self) -> list[dict]:
        return self._load()


# ── Recorder bundle ──────────────────────────────────────────────


@dataclass
class CloseRecorders:
    """Holds the writers used by :func:`record_close`.

    Each field is optional; ``None`` means that recorder is skipped.
    """

    hub: Any | None = None  # InsightsHub
    thesis_store: Any | None = None  # ThesisTagStore
    regret_sidecar: RegretSidecar | None = None
    purification_ledger: Any | None = None  # RoundTripLedger
    purification_rules: Mapping[str, Any] | None = None


# ── Main entry point ─────────────────────────────────────────────


def record_close(event: CloseEvent, recorders: CloseRecorders) -> dict[str, Any]:
    """Dispatch a close event to every configured recorder.

    Returns a dict summarising what fired, suitable for INFO logging.
    Errors are caught per-recorder so partial failure is observable
    but never blocks the close-path.
    """
    summary: dict[str, Any] = {
        "trade_id": event.trade_id,
        "symbol": event.symbol,
        "return_pct": event.return_pct,
    }

    # Drift monitor — feed the residual / return as the signal.
    if recorders.hub is not None and getattr(recorders.hub, "drift", None) is not None:
        try:
            recorders.hub.drift.observe(event.return_pct)
            summary["drift_state"] = recorders.hub.drift.state
        except Exception as exc:  # noqa: BLE001
            logger.debug("drift observe failed: %s", exc)

    # Thesis tagger — heuristic only at close-time; LLM refinement is async.
    if recorders.thesis_store is not None:
        try:
            from halal_trader.core.thesis import (
                TaggedTradeContext,
                heuristic_tag,
            )

            ctx = TaggedTradeContext(
                trade_id=event.trade_id,
                symbol=event.symbol,
                side=event.side,
                entry_price=event.entry_price,
                exit_price=event.exit_price,
                exit_reason=event.exit_reason,
                pnl_pct=event.return_pct,
                hold_seconds=event.hold_seconds,
                setup_type=event.setup_type,
                indicators=event.indicators,
                reasoning=event.reasoning,
            )
            tag = heuristic_tag(ctx)
            recorders.thesis_store.set(
                event.trade_id,
                tag,
                method="heuristic",
                reason="close-time tag",
            )
            summary["thesis_tag"] = tag
        except Exception as exc:  # noqa: BLE001
            logger.debug("thesis tagger failed: %s", exc)

    # Regret sidecar.
    if recorders.regret_sidecar is not None:
        try:
            from halal_trader.core.regret import (
                ClosedTradeView,
                hindsight_regret,
            )

            view = ClosedTradeView(
                trade_id=event.trade_id,
                symbol=event.symbol,
                action_size_pct=1.0,
                pnl_pct=event.return_pct,
                confidence=event.confidence or 0.5,
                setup_type=event.setup_type,
            )
            rec = hindsight_regret(view)
            recorders.regret_sidecar.append(
                {
                    "trade_id": rec.trade_id,
                    "symbol": rec.symbol,
                    "regret": rec.regret,
                    "optimal_size_pct": rec.optimal_size_pct,
                    "actual_size_pct": rec.actual_size_pct,
                    "pnl_pct": rec.pnl_pct,
                    "note": rec.note,
                    "ts": (event.closed_at or datetime.now(UTC)).isoformat(),
                }
            )
            summary["regret"] = rec.regret
        except Exception as exc:  # noqa: BLE001
            logger.debug("regret recorder failed: %s", exc)

    # Round-trip purification.
    if (
        recorders.purification_ledger is not None
        and recorders.purification_rules is not None
        and event.realized_pnl_usd > 0
    ):
        try:
            from halal_trader.halal.round_trip_purification import (
                record_round_trip,
            )

            entry = record_round_trip(
                recorders.purification_ledger,
                recorders.purification_rules,
                trade_id=event.trade_id,
                symbol=event.symbol,
                gain_usd=event.realized_pnl_usd,
            )
            if entry is not None:
                summary["purification_due_usd"] = entry.purification_due_usd
        except Exception as exc:  # noqa: BLE001
            logger.debug("purification recorder failed: %s", exc)

    return summary
