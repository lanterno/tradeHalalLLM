"""Per-asset circuit breaker — opens on N unexpected errors, rejections don't count."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from halabot.execution.breaker import PerAssetBreaker

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


def test_opens_after_threshold_unexpected_errors():
    b = PerAssetBreaker(threshold=3, cooldown_s=900)
    assert not b.record_error("NVDA", T0)
    assert not b.record_error("NVDA", T0)
    assert b.record_error("NVDA", T0)  # 3rd → opens
    assert b.is_open("NVDA", T0)


def test_rejections_do_not_trip_breaker():
    b = PerAssetBreaker(threshold=2)
    b.record_error("NVDA", T0, rejection=True)
    b.record_error("NVDA", T0, rejection=True)
    assert not b.is_open("NVDA", T0)  # -1013/-2010 style rejections never trip it


def test_success_resets_consecutive_count():
    b = PerAssetBreaker(threshold=3)
    b.record_error("NVDA", T0)
    b.record_error("NVDA", T0)
    b.record_success("NVDA")
    assert not b.record_error("NVDA", T0)  # count restarted
    assert not b.is_open("NVDA", T0)


def test_cooldown_elapses_and_closes():
    b = PerAssetBreaker(threshold=1, cooldown_s=600)
    b.record_error("NVDA", T0)  # opens immediately
    assert b.is_open("NVDA", T0)
    assert b.is_open("NVDA", T0 + timedelta(seconds=300))  # still cooling
    assert not b.is_open("NVDA", T0 + timedelta(seconds=601))  # elapsed → closed


def test_breaker_is_per_asset():
    b = PerAssetBreaker(threshold=1)
    b.record_error("NVDA", T0)
    assert b.is_open("NVDA", T0)
    assert not b.is_open("AAPL", T0)  # isolated
