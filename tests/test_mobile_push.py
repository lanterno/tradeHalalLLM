"""Tests for `halal_trader.web.mobile_push` (Wave 3.J).

Covers: notification kinds + priority ladder, quiet hours including
wraparound, rate-limit gate, delivery state machine, no-secret render.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, time, timedelta, timezone

import pytest

from halal_trader.web.mobile_push import (
    DEFAULT_POLICY,
    DeliveryOrderError,
    DeliveryRecord,
    DeliveryStatus,
    DeviceRegistration,
    GateOutcome,
    NotificationKind,
    NotificationPolicy,
    NotificationRequest,
    Platform,
    Priority,
    count_recent_sends,
    evaluate_gate,
    mark_delivered,
    mark_failed,
    mark_sent,
    priority_for,
    render_delivery,
    render_request,
    start_delivery,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_notification_kind_string_values_pinned() -> None:
    assert NotificationKind.TRADE_FILL.value == "trade_fill"
    assert NotificationKind.RISK_HALT.value == "risk_halt"
    assert NotificationKind.DAILY_SUMMARY.value == "daily_summary"


def test_priority_string_values_pinned() -> None:
    assert Priority.LOW.value == "low"
    assert Priority.NORMAL.value == "normal"
    assert Priority.CRITICAL.value == "critical"


def test_platform_string_values_pinned() -> None:
    assert Platform.IOS.value == "ios"
    assert Platform.ANDROID.value == "android"


def test_delivery_status_string_values_pinned() -> None:
    assert DeliveryStatus.PENDING.value == "pending"
    assert DeliveryStatus.SENT.value == "sent"
    assert DeliveryStatus.DELIVERED.value == "delivered"
    assert DeliveryStatus.FAILED.value == "failed"


def test_gate_outcome_string_values_pinned() -> None:
    assert GateOutcome.SEND.value == "send"
    assert GateOutcome.HOLD_QUIET_HOURS.value == "hold_quiet_hours"
    assert GateOutcome.HOLD_RATE_LIMIT.value == "hold_rate_limit"
    assert GateOutcome.HOLD_NO_DEVICE.value == "hold_no_device"


# --------------------------- priority_for ------------------------------------


def test_risk_halt_is_critical() -> None:
    """Pin: risk_halt is the bypass-all-gates priority."""

    assert priority_for(NotificationKind.RISK_HALT) is Priority.CRITICAL


def test_trade_fill_is_normal() -> None:
    assert priority_for(NotificationKind.TRADE_FILL) is Priority.NORMAL


def test_daily_summary_is_low() -> None:
    assert priority_for(NotificationKind.DAILY_SUMMARY) is Priority.LOW


# --------------------------- NotificationPolicy ------------------------------


def test_default_policy_quiet_hours_22_07() -> None:
    """Pin: default 22:00-07:00 wraparound."""

    assert DEFAULT_POLICY.quiet_hours_start == time(22, 0)
    assert DEFAULT_POLICY.quiet_hours_end == time(7, 0)
    assert DEFAULT_POLICY.rate_limit_per_hour == 30


def test_policy_rejects_zero_rate_limit() -> None:
    with pytest.raises(ValueError, match="rate_limit"):
        NotificationPolicy(rate_limit_per_hour=0)


def test_policy_rejects_negative_rate_limit() -> None:
    with pytest.raises(ValueError, match="rate_limit"):
        NotificationPolicy(rate_limit_per_hour=-1)


def test_policy_rejects_quiet_start_equals_end() -> None:
    with pytest.raises(ValueError, match="quiet_hours"):
        NotificationPolicy(quiet_hours_start=time(10, 0), quiet_hours_end=time(10, 0))


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.rate_limit_per_hour = 100  # type: ignore[misc]


# --------------------------- DeviceRegistration ------------------------------


def _device(**overrides: object) -> DeviceRegistration:
    base: dict[str, object] = {
        "user_id": "user_1",
        "device_id": "dev_1",
        "platform": Platform.ANDROID,
        "device_token": "fcm_token_string",
        "timezone_offset_minutes": 0,
        "registered_at": T0,
    }
    base.update(overrides)
    return DeviceRegistration(**base)  # type: ignore[arg-type]


def test_device_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        _device(user_id="")


def test_device_rejects_empty_device_id() -> None:
    with pytest.raises(ValueError, match="device_id"):
        _device(device_id="")


def test_device_rejects_empty_token() -> None:
    with pytest.raises(ValueError, match="device_token"):
        _device(device_token="")


def test_device_rejects_naive_registered_at() -> None:
    with pytest.raises(ValueError, match="registered_at"):
        _device(registered_at=datetime(2026, 5, 1))


def test_device_rejects_extreme_tz_offset() -> None:
    with pytest.raises(ValueError, match="timezone_offset"):
        _device(timezone_offset_minutes=900)


def test_device_ios_token_must_be_hex() -> None:
    """Pin: APNs tokens are hex strings."""

    with pytest.raises(ValueError, match="hex"):
        _device(
            platform=Platform.IOS,
            device_token="not_hex_!!@@",
        )


def test_device_ios_token_hex_accepted() -> None:
    d = _device(
        platform=Platform.IOS,
        device_token="abc123DEADBEEF",
    )
    assert d.platform is Platform.IOS


def test_device_is_frozen() -> None:
    d = _device()
    with pytest.raises(FrozenInstanceError):
        d.revoked = True  # type: ignore[misc]


# --------------------------- NotificationRequest -----------------------------


def _request(**overrides: object) -> NotificationRequest:
    base: dict[str, object] = {
        "notification_id": "ntf_1",
        "user_id": "user_1",
        "kind": NotificationKind.TRADE_FILL,
        "title": "Trade filled",
        "body": "BTCUSDT buy filled at $50000",
        "requested_at": T0,
    }
    base.update(overrides)
    return NotificationRequest(**base)  # type: ignore[arg-type]


def test_request_rejects_empty_notification_id() -> None:
    with pytest.raises(ValueError, match="notification_id"):
        _request(notification_id="")


def test_request_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        _request(user_id="")


def test_request_rejects_empty_title() -> None:
    with pytest.raises(ValueError, match="title"):
        _request(title="")


def test_request_rejects_empty_body() -> None:
    with pytest.raises(ValueError, match="body"):
        _request(body="")


def test_request_rejects_naive_requested_at() -> None:
    with pytest.raises(ValueError, match="requested_at"):
        _request(requested_at=datetime(2026, 5, 1))


def test_request_rejects_oversized_title() -> None:
    """Pin: titles capped at 100 chars (APNs limit)."""

    with pytest.raises(ValueError, match="title too long"):
        _request(title="x" * 101)


def test_request_rejects_oversized_body() -> None:
    with pytest.raises(ValueError, match="body too long"):
        _request(body="x" * 1001)


def test_request_is_frozen() -> None:
    r = _request()
    with pytest.raises(FrozenInstanceError):
        r.title = "other"  # type: ignore[misc]


# --------------------------- evaluate_gate -----------------------------------


def test_gate_no_device_holds() -> None:
    request = _request()
    decision = evaluate_gate(request, registration=None, recent_send_count=0)
    assert decision.outcome is GateOutcome.HOLD_NO_DEVICE


def test_gate_revoked_device_holds() -> None:
    request = _request()
    revoked = _device(revoked=True)
    decision = evaluate_gate(request, registration=revoked, recent_send_count=0)
    assert decision.outcome is GateOutcome.HOLD_NO_DEVICE


def test_gate_normal_during_business_hours_sends() -> None:
    """Pin: trade fill at 14:00 UTC (operator at UTC) → SEND."""

    request = _request(
        kind=NotificationKind.TRADE_FILL,
        requested_at=datetime(2026, 5, 1, 14, 0, tzinfo=UTC),
    )
    device = _device(timezone_offset_minutes=0)
    decision = evaluate_gate(request, registration=device, recent_send_count=0)
    assert decision.outcome is GateOutcome.SEND


def test_gate_normal_during_quiet_hours_holds() -> None:
    """Pin: trade fill at 23:00 user-local → HOLD_QUIET_HOURS."""

    # 23:00 UTC, user at UTC offset 0 → user local 23:00 (in quiet hours)
    request = _request(
        kind=NotificationKind.TRADE_FILL,
        requested_at=datetime(2026, 5, 1, 23, 0, tzinfo=UTC),
    )
    device = _device(timezone_offset_minutes=0)
    decision = evaluate_gate(request, registration=device, recent_send_count=0)
    assert decision.outcome is GateOutcome.HOLD_QUIET_HOURS


def test_gate_critical_bypasses_quiet_hours() -> None:
    """Pin: RISK_HALT at 03:00 user-local still SENDs."""

    request = _request(
        kind=NotificationKind.RISK_HALT,
        requested_at=datetime(2026, 5, 1, 3, 0, tzinfo=UTC),
    )
    device = _device(timezone_offset_minutes=0)
    decision = evaluate_gate(request, registration=device, recent_send_count=0)
    assert decision.outcome is GateOutcome.SEND


def test_gate_critical_bypasses_rate_limit() -> None:
    """Pin: RISK_HALT bypasses rate limit even with 1000 recent sends."""

    request = _request(kind=NotificationKind.RISK_HALT)
    device = _device()
    decision = evaluate_gate(request, registration=device, recent_send_count=1000)
    assert decision.outcome is GateOutcome.SEND


def test_gate_normal_at_rate_limit_holds() -> None:
    """Pin: 30 recent sends → next non-critical HOLDS."""

    request = _request(kind=NotificationKind.TRADE_FILL)
    device = _device()
    decision = evaluate_gate(request, registration=device, recent_send_count=30)
    assert decision.outcome is GateOutcome.HOLD_RATE_LIMIT


def test_gate_normal_just_below_rate_limit_sends() -> None:
    request = _request(kind=NotificationKind.TRADE_FILL)
    device = _device()
    decision = evaluate_gate(request, registration=device, recent_send_count=29)
    assert decision.outcome is GateOutcome.SEND


def test_gate_user_in_other_timezone_quiet_hours() -> None:
    """Pin: NY user (UTC-5) at 14:00 UTC = 09:00 NY → SEND.

    Same UTC time at 04:00 UTC = 23:00 NY → HOLD."""

    ny_tz_offset = -300  # -5 hours

    # 09:00 NY = 14:00 UTC → SEND
    request = _request(
        kind=NotificationKind.TRADE_FILL,
        requested_at=datetime(2026, 5, 1, 14, 0, tzinfo=UTC),
    )
    device = _device(timezone_offset_minutes=ny_tz_offset)
    assert (
        evaluate_gate(request, registration=device, recent_send_count=0).outcome is GateOutcome.SEND
    )

    # 23:00 NY = 04:00 UTC next day → HOLD
    request2 = _request(
        kind=NotificationKind.TRADE_FILL,
        requested_at=datetime(2026, 5, 2, 4, 0, tzinfo=UTC),
    )
    assert (
        evaluate_gate(request2, registration=device, recent_send_count=0).outcome
        is GateOutcome.HOLD_QUIET_HOURS
    )


def test_gate_decision_carries_message() -> None:
    request = _request()
    decision = evaluate_gate(request, registration=None, recent_send_count=0)
    assert "user_1" in decision.message


def test_quiet_hours_wraparound_handles_morning() -> None:
    """Pin: 06:00 user-local is in quiet hours (22:00-07:00 wrap)."""

    request = _request(
        kind=NotificationKind.TRADE_FILL,
        requested_at=datetime(2026, 5, 1, 6, 0, tzinfo=UTC),
    )
    device = _device(timezone_offset_minutes=0)
    decision = evaluate_gate(request, registration=device, recent_send_count=0)
    assert decision.outcome is GateOutcome.HOLD_QUIET_HOURS


def test_quiet_hours_wraparound_07_boundary_exclusive() -> None:
    """Pin: 07:00 user-local is OUTSIDE quiet hours (end exclusive)."""

    request = _request(
        kind=NotificationKind.TRADE_FILL,
        requested_at=datetime(2026, 5, 1, 7, 0, tzinfo=UTC),
    )
    device = _device(timezone_offset_minutes=0)
    decision = evaluate_gate(request, registration=device, recent_send_count=0)
    assert decision.outcome is GateOutcome.SEND


def test_quiet_hours_22_boundary_inclusive() -> None:
    """Pin: 22:00 user-local IS in quiet hours (start inclusive)."""

    request = _request(
        kind=NotificationKind.TRADE_FILL,
        requested_at=datetime(2026, 5, 1, 22, 0, tzinfo=UTC),
    )
    device = _device(timezone_offset_minutes=0)
    decision = evaluate_gate(request, registration=device, recent_send_count=0)
    assert decision.outcome is GateOutcome.HOLD_QUIET_HOURS


# --------------------------- Delivery state machine --------------------------


def test_start_delivery_pending() -> None:
    record = start_delivery(notification_id="n1", now=T0)
    assert record.status is DeliveryStatus.PENDING


def test_start_delivery_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="notification_id"):
        start_delivery(notification_id="", now=T0)


def test_start_delivery_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        start_delivery(notification_id="n1", now=datetime(2026, 5, 1))


def test_mark_sent_advances_state() -> None:
    record = start_delivery(notification_id="n1", now=T0)
    record = mark_sent(record, now=T0 + timedelta(seconds=1))
    assert record.status is DeliveryStatus.SENT


def test_mark_delivered_advances_state() -> None:
    record = start_delivery(notification_id="n1", now=T0)
    record = mark_sent(record, now=T0)
    record = mark_delivered(record, now=T0 + timedelta(seconds=2))
    assert record.status is DeliveryStatus.DELIVERED


def test_cant_skip_to_delivered() -> None:
    """Pin: PENDING → DELIVERED skip is rejected."""

    record = start_delivery(notification_id="n1", now=T0)
    with pytest.raises(DeliveryOrderError):
        mark_delivered(record, now=T0)


def test_mark_failed_from_pending() -> None:
    record = start_delivery(notification_id="n1", now=T0)
    record = mark_failed(record, now=T0, failure_reason="apns rejected")
    assert record.status is DeliveryStatus.FAILED
    assert record.failure_reason == "apns rejected"


def test_mark_failed_from_sent() -> None:
    """Pin: SENT can transition to FAILED (delivery confirmation
    timed out)."""

    record = start_delivery(notification_id="n1", now=T0)
    record = mark_sent(record, now=T0)
    record = mark_failed(record, now=T0, failure_reason="confirmation timeout")
    assert record.status is DeliveryStatus.FAILED


def test_mark_failed_already_failed_rejected() -> None:
    record = start_delivery(notification_id="n1", now=T0)
    record = mark_failed(record, now=T0, failure_reason="x")
    with pytest.raises(DeliveryOrderError):
        mark_failed(record, now=T0, failure_reason="y")


def test_mark_failed_requires_reason() -> None:
    record = start_delivery(notification_id="n1", now=T0)
    with pytest.raises(ValueError, match="failure_reason"):
        mark_failed(record, now=T0, failure_reason="")


def test_record_validates_failed_requires_reason() -> None:
    with pytest.raises(ValueError, match="failure_reason"):
        DeliveryRecord(
            notification_id="n1",
            status=DeliveryStatus.FAILED,
            last_updated_at=T0,
            failure_reason="",
        )


def test_record_validates_non_failed_no_reason() -> None:
    with pytest.raises(ValueError, match="failure_reason"):
        DeliveryRecord(
            notification_id="n1",
            status=DeliveryStatus.SENT,
            last_updated_at=T0,
            failure_reason="ghost reason",
        )


def test_record_is_frozen() -> None:
    record = start_delivery(notification_id="n1", now=T0)
    with pytest.raises(FrozenInstanceError):
        record.status = DeliveryStatus.SENT  # type: ignore[misc]


def test_delivery_returns_new_state() -> None:
    """Pin: state operations are immutable."""

    original = start_delivery(notification_id="n1", now=T0)
    new_record = mark_sent(original, now=T0)
    assert original.status is DeliveryStatus.PENDING
    assert new_record.status is DeliveryStatus.SENT


# --------------------------- count_recent_sends ------------------------------


def test_count_recent_sends_basic() -> None:
    records = [
        DeliveryRecord(
            notification_id=f"n{i}",
            status=DeliveryStatus.SENT,
            last_updated_at=T0 - timedelta(minutes=i * 5),
        )
        for i in range(5)
    ]
    user_ids = [f"n{i}" for i in range(5)]
    count = count_recent_sends(records, user_notification_ids=user_ids, now=T0)
    # All 5 were within last 1 hour
    assert count == 5


def test_count_recent_sends_window_boundary() -> None:
    """Pin: notification at exactly 60 min old is counted (>= cutoff)."""

    records = [
        DeliveryRecord(
            notification_id="n1",
            status=DeliveryStatus.SENT,
            last_updated_at=T0 - timedelta(hours=1),
        ),
    ]
    count = count_recent_sends(records, user_notification_ids=["n1"], now=T0)
    assert count == 1


def test_count_recent_sends_excludes_old() -> None:
    records = [
        DeliveryRecord(
            notification_id="n1",
            status=DeliveryStatus.SENT,
            last_updated_at=T0 - timedelta(hours=2),
        ),
    ]
    count = count_recent_sends(records, user_notification_ids=["n1"], now=T0)
    assert count == 0


def test_count_recent_sends_excludes_pending_and_failed() -> None:
    """Pin: only SENT + DELIVERED count toward rate limit."""

    records = [
        DeliveryRecord(
            notification_id="n1",
            status=DeliveryStatus.PENDING,
            last_updated_at=T0,
        ),
        DeliveryRecord(
            notification_id="n2",
            status=DeliveryStatus.FAILED,
            last_updated_at=T0,
            failure_reason="x",
        ),
        DeliveryRecord(
            notification_id="n3",
            status=DeliveryStatus.SENT,
            last_updated_at=T0,
        ),
        DeliveryRecord(
            notification_id="n4",
            status=DeliveryStatus.DELIVERED,
            last_updated_at=T0,
        ),
    ]
    count = count_recent_sends(
        records,
        user_notification_ids=["n1", "n2", "n3", "n4"],
        now=T0,
    )
    assert count == 2


def test_count_recent_sends_filters_by_user_ids() -> None:
    records = [
        DeliveryRecord(
            notification_id="n_my",
            status=DeliveryStatus.SENT,
            last_updated_at=T0,
        ),
        DeliveryRecord(
            notification_id="n_other",
            status=DeliveryStatus.SENT,
            last_updated_at=T0,
        ),
    ]
    count = count_recent_sends(records, user_notification_ids=["n_my"], now=T0)
    assert count == 1


def test_count_recent_sends_naive_now_rejected() -> None:
    with pytest.raises(ValueError, match="now"):
        count_recent_sends(
            [],
            user_notification_ids=[],
            now=datetime(2026, 5, 1),
        )


# --------------------------- render ------------------------------------------


def test_render_request_includes_emoji_per_kind() -> None:
    fill = render_request(_request(kind=NotificationKind.TRADE_FILL))
    halt = render_request(_request(kind=NotificationKind.RISK_HALT))
    summary = render_request(_request(kind=NotificationKind.DAILY_SUMMARY))
    assert "💸" in fill
    assert "🚨" in halt
    assert "📊" in summary


def test_render_request_includes_priority_emoji() -> None:
    halt = render_request(_request(kind=NotificationKind.RISK_HALT))
    fill = render_request(_request(kind=NotificationKind.TRADE_FILL))
    summary = render_request(_request(kind=NotificationKind.DAILY_SUMMARY))
    assert "🔴" in halt  # critical
    assert "🟡" in fill  # normal
    assert "🔵" in summary  # low


def test_render_request_no_secret_leak() -> None:
    """Pin: render never includes device_token / APNs key / FCM key."""

    request = _request()
    out = render_request(request)
    assert "device_token" not in out.lower()
    assert "fcm" not in out.lower()
    assert "apns" not in out.lower()
    assert "key" not in out.lower()


def test_render_delivery_status_emoji() -> None:
    record = start_delivery(notification_id="n1", now=T0)
    out = render_delivery(record)
    assert "⏳" in out
    record_sent = mark_sent(record, now=T0)
    assert "📤" in render_delivery(record_sent)


def test_render_delivery_failed_includes_reason() -> None:
    record = start_delivery(notification_id="n1", now=T0)
    record = mark_failed(record, now=T0, failure_reason="apns 410 gone")
    out = render_delivery(record)
    assert "❌" in out
    assert "apns 410 gone" in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_risk_halt_at_3am_delivers() -> None:
    """Real-world: bot halts at 3am operator-local; the operator's
    phone buzzes (CRITICAL bypasses quiet hours)."""

    # Operator at UTC; risk halt at 03:00 UTC
    request = _request(
        kind=NotificationKind.RISK_HALT,
        title="Bot halted",
        body="Drawdown exceeded 10%",
        requested_at=datetime(2026, 5, 1, 3, 0, tzinfo=UTC),
    )
    device = _device(timezone_offset_minutes=0)
    decision = evaluate_gate(request, registration=device, recent_send_count=0)
    assert decision.outcome is GateOutcome.SEND

    # Delivery state machine
    record = start_delivery(notification_id=request.notification_id, now=T0)
    record = mark_sent(record, now=T0 + timedelta(seconds=1))
    record = mark_delivered(record, now=T0 + timedelta(seconds=3))
    assert record.status is DeliveryStatus.DELIVERED


def test_e2e_daily_summary_at_3am_held() -> None:
    """Real-world: daily summary at 3am operator-local → HELD."""

    request = _request(
        kind=NotificationKind.DAILY_SUMMARY,
        title="Daily summary",
        body="Today: +2.3%",
        requested_at=datetime(2026, 5, 1, 3, 0, tzinfo=UTC),
    )
    device = _device(timezone_offset_minutes=0)
    decision = evaluate_gate(request, registration=device, recent_send_count=0)
    assert decision.outcome is GateOutcome.HOLD_QUIET_HOURS


def test_e2e_noisy_strategy_rate_limited() -> None:
    """Real-world: 30 trade-fill notifications in last hour → 31st HOLDS.

    But concurrent risk_halt still SENDS."""

    fill_request = _request(kind=NotificationKind.TRADE_FILL)
    halt_request = _request(
        notification_id="ntf_halt",
        kind=NotificationKind.RISK_HALT,
    )
    device = _device()

    # Trade fill held
    decision = evaluate_gate(fill_request, registration=device, recent_send_count=30)
    assert decision.outcome is GateOutcome.HOLD_RATE_LIMIT

    # Risk halt still sends
    decision = evaluate_gate(halt_request, registration=device, recent_send_count=30)
    assert decision.outcome is GateOutcome.SEND


def test_e2e_replay_consistency() -> None:
    """Same operations produce equal delivery records."""

    def build() -> DeliveryRecord:
        record = start_delivery(notification_id="n1", now=T0)
        record = mark_sent(record, now=T0 + timedelta(seconds=1))
        return record

    a = build()
    b = build()
    assert a == b
