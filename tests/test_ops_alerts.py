"""Ops-event alert triage tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from halal_trader.core.divergence import DivergenceReport
from halal_trader.core.ops_alerts import (
    maybe_alert_divergence,
    maybe_alert_low_fill_ratio,
    maybe_alert_screener_timeout,
    maybe_alert_ws_dropped,
)


def _alerts():
    a = MagicMock()
    a.notify = AsyncMock()
    return a


# ── low fill ratio ────────────────────────────────────────────


async def test_low_fill_ratio_silent_when_under_window():
    alerts = _alerts()
    fired = await maybe_alert_low_fill_ratio(alerts=alerts, attempted=3, filled=0, window_size=10)
    assert fired is False
    alerts.notify.assert_not_called()


async def test_low_fill_ratio_fires_when_below_floor():
    alerts = _alerts()
    fired = await maybe_alert_low_fill_ratio(alerts=alerts, attempted=20, filled=5, floor=0.5)
    assert fired is True
    args, _ = alerts.notify.await_args
    assert args[0] == "ops.low_fill_ratio"


async def test_low_fill_ratio_silent_when_at_or_above_floor():
    alerts = _alerts()
    fired = await maybe_alert_low_fill_ratio(alerts=alerts, attempted=20, filled=15, floor=0.5)
    assert fired is False


# ── screener timeout ──────────────────────────────────────────


async def test_screener_timeout_silent_when_fast():
    alerts = _alerts()
    fired = await maybe_alert_screener_timeout(
        alerts=alerts, last_refresh_seconds=2.0, threshold_seconds=10.0
    )
    assert fired is False


async def test_screener_timeout_fires_when_slow():
    alerts = _alerts()
    fired = await maybe_alert_screener_timeout(
        alerts=alerts, last_refresh_seconds=15.0, threshold_seconds=10.0
    )
    assert fired is True
    args, _ = alerts.notify.await_args
    assert args[0] == "ops.screener_timeout"


# ── divergence ────────────────────────────────────────────────


def _report(bps: float, n: int = 100) -> DivergenceReport:
    return DivergenceReport(
        sample_size=n,
        mean_divergence_bps=bps,
        p95_divergence_bps=bps + 5,
        mean_paper_bps=5.0,
        mean_live_bps=5.0 + bps,
        exceeds_threshold=False,
        threshold_bps=10.0,
    )


async def test_divergence_silent_when_no_samples():
    alerts = _alerts()
    fired = await maybe_alert_divergence(alerts=alerts, report=_report(bps=999, n=0))
    assert fired is False


async def test_divergence_silent_when_under_alert_threshold():
    alerts = _alerts()
    fired = await maybe_alert_divergence(alerts=alerts, report=_report(bps=10), threshold_bps=25)
    assert fired is False


async def test_divergence_fires_when_above_alert_threshold():
    alerts = _alerts()
    fired = await maybe_alert_divergence(alerts=alerts, report=_report(bps=50), threshold_bps=25)
    assert fired is True


# ── WS dropped ────────────────────────────────────────────────


async def test_ws_dropped_silent_when_few_failures():
    alerts = _alerts()
    fired = await maybe_alert_ws_dropped(alerts=alerts, consecutive_failures=2)
    assert fired is False


async def test_ws_dropped_fires_after_floor():
    alerts = _alerts()
    fired = await maybe_alert_ws_dropped(alerts=alerts, consecutive_failures=10, fail_floor=5)
    assert fired is True
    args, _ = alerts.notify.await_args
    assert args[0] == "ops.ws_dropped"


# ── _fire safety net ──────────────────────────────────────────


async def test_alert_helpers_tolerate_none_alert_sink():
    """All maybe_* helpers must accept alerts=None without crashing."""
    await maybe_alert_low_fill_ratio(alerts=None, attempted=20, filled=5)
    await maybe_alert_screener_timeout(alerts=None, last_refresh_seconds=999)
    await maybe_alert_divergence(alerts=None, report=_report(bps=999))
    await maybe_alert_ws_dropped(alerts=None, consecutive_failures=999)


async def test_alert_helpers_swallow_sink_failure():
    """A blowing-up AlertSink shouldn't propagate up to the cycle."""
    bad = MagicMock()
    bad.notify = AsyncMock(side_effect=RuntimeError("network gone"))
    fired = await maybe_alert_ws_dropped(alerts=bad, consecutive_failures=10)
    assert fired is True  # we still claim the threshold was crossed
