"""Ops-event → alert triage.

Centralises the "what counts as alert-worthy" logic so individual call
sites don't each grow ad-hoc thresholds. Each event type has:

* a stable ``error_type`` string (the AlertSink dedup key)
* a threshold function over the relevant input state
* a message builder

Call sites pass their state object into the corresponding ``maybe_*``
helper; if the threshold is breached, the helper fires the alert via
the supplied :class:`AlertSink` and returns ``True``. Below threshold
returns ``False`` and is silent.

Why a separate module rather than scattering thresholds across cycle /
monitor / strategy: any ops policy change (e.g. raise the SL-slippage
threshold from 50bps to 100bps) lives in one file rather than three,
and the test suite can exhaustively cover the threshold matrix.
"""

from __future__ import annotations

import logging
from typing import Any

from halal_trader.core.divergence import DivergenceReport

logger = logging.getLogger(__name__)


# Suggested defaults — operator-overridable.
DEFAULT_FILL_RATIO_FLOOR = 0.5
DEFAULT_FILL_RATIO_WINDOW = 10
DEFAULT_SCREENER_LATENCY_S = 10.0
DEFAULT_DIVERGENCE_BPS = 25.0


async def maybe_alert_low_fill_ratio(
    *,
    alerts: Any,
    attempted: int,
    filled: int,
    floor: float = DEFAULT_FILL_RATIO_FLOOR,
    window_size: int = DEFAULT_FILL_RATIO_WINDOW,
) -> bool:
    """Fire if the recent fill ratio (filled/attempted) drops below ``floor``.

    Requires at least ``window_size`` attempts so a single rejected
    order doesn't trigger an alert on cold start.
    """
    if attempted < window_size:
        return False
    ratio = filled / attempted if attempted > 0 else 0.0
    if ratio >= floor:
        return False
    msg = (
        f"Fill ratio {ratio:.0%} over the last {attempted} attempts "
        f"is below the {floor:.0%} floor — broker may be rejecting orders. "
        f"Inspect recent trade rows for status='rejected'."
    )
    await _fire(alerts, "ops.low_fill_ratio", msg)
    return True


async def maybe_alert_screener_timeout(
    *,
    alerts: Any,
    last_refresh_seconds: float,
    threshold_seconds: float = DEFAULT_SCREENER_LATENCY_S,
) -> bool:
    """Fire if the halal screener took longer than expected to refresh.

    A slow-but-eventually-completing screener is fine; a hang means our
    halal cache stales out and we either trade old data or skip cycles.
    """
    if last_refresh_seconds < threshold_seconds:
        return False
    msg = (
        f"Halal screener refresh took {last_refresh_seconds:.1f}s "
        f"(threshold {threshold_seconds:.1f}s). Cache freshness at risk; "
        f"check Zoya / CoinGecko availability."
    )
    await _fire(alerts, "ops.screener_timeout", msg)
    return True


async def maybe_alert_divergence(
    *,
    alerts: Any,
    report: DivergenceReport,
    threshold_bps: float = DEFAULT_DIVERGENCE_BPS,
) -> bool:
    """Fire when paper-vs-live slippage diverges beyond ``threshold_bps``.

    Distinct from the threshold inside ``DivergenceReport`` — that's the
    *display* threshold (which gets shown as "EXCEEDS"); this is the
    *alert* threshold for paging the operator.
    """
    if report.sample_size == 0:
        return False
    if report.mean_divergence_bps < threshold_bps:
        return False
    msg = (
        f"Paper-vs-live slippage divergence {report.mean_divergence_bps:+.1f}bps "
        f"over {report.sample_size} trades exceeds alert threshold "
        f"{threshold_bps:.1f}bps. Backtester is over-promising — adjust "
        f"slippage baseline or halt live trading until model updated."
    )
    await _fire(alerts, "ops.divergence", msg)
    return True


async def maybe_alert_ws_dropped(
    *,
    alerts: Any,
    consecutive_failures: int,
    fail_floor: int = 5,
) -> bool:
    """Fire when a WS reconnect loop has failed ``fail_floor`` times in a row."""
    if consecutive_failures < fail_floor:
        return False
    msg = (
        f"WebSocket connection has failed to reconnect {consecutive_failures} "
        f"times in a row. Real-time price feed offline; SL/TP enforcement "
        f"is degraded. Check exchange WS status."
    )
    await _fire(alerts, "ops.ws_dropped", msg)
    return True


async def _fire(alerts: Any, error_type: str, details: str) -> None:
    """Send via AlertSink if available; tolerate missing/None sinks gracefully."""
    if alerts is None:
        return
    try:
        await alerts.notify(error_type, details)
    except Exception as e:
        logger.debug("AlertSink.notify failed for %s: %s", error_type, e)
