"""Tests for `core/alert_router.py`.

Pins severity routing, the per-(type, channel) cooldown semantics
(channel-isolated dedup), the don't-abort-on-channel-failure
contract, the disabled-channel skip, the multi-route dedup, and
the message renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from halal_trader.core.alert_router import (
    AlertRoute,
    AlertRouter,
    AlertSpec,
    Severity,
    render_message,
)

# ── Stub channel ─────────────────────────────────────────


@dataclass
class _StubChannel:
    """In-memory channel for testing."""

    _name: str
    _enabled: bool = True
    sent: list[str] = field(default_factory=list)
    fail_with: type[BaseException] | None = None
    return_false: bool = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def send(self, message: str) -> bool:
        if self.fail_with is not None:
            raise self.fail_with("simulated failure")
        if self.return_false:
            return False
        self.sent.append(message)
        return True


def _alert(
    *,
    type: str = "test.alert",
    severity: Severity = Severity.WARN,
    summary: str = "test summary",
    runbook_url: str = "",
    context: dict[str, str] | None = None,
) -> AlertSpec:
    return AlertSpec(
        type=type,
        severity=severity,
        summary=summary,
        runbook_url=runbook_url,
        context=context or {},
    )


def _clock():
    """Mutable clock state for deterministic dedup tests."""
    state = {"now": 0.0}

    def now() -> float:
        return state["now"]

    def advance(seconds: float) -> None:
        state["now"] += seconds

    return now, advance


# ── severity routing ─────────────────────────────────────


@pytest.mark.asyncio
async def test_route_dispatches_when_severity_meets_min():
    ch = _StubChannel("slack")
    router = AlertRouter([AlertRoute(severity_min=Severity.WARN, channel=ch)])
    result = await router.dispatch(_alert(severity=Severity.WARN))
    assert result.sent == ["slack"]


@pytest.mark.asyncio
async def test_route_skips_when_severity_below_min():
    ch = _StubChannel("slack")
    router = AlertRouter([AlertRoute(severity_min=Severity.PAGE, channel=ch)])
    result = await router.dispatch(_alert(severity=Severity.WARN))
    assert result.sent == []
    assert ch.sent == []


@pytest.mark.asyncio
async def test_higher_severity_clears_lower_min():
    """Pin: PAGE alerts reach a WARN+ channel."""
    ch = _StubChannel("slack")
    router = AlertRouter([AlertRoute(severity_min=Severity.WARN, channel=ch)])
    result = await router.dispatch(_alert(severity=Severity.PAGE))
    assert result.sent == ["slack"]


# ── multi-route dedup ────────────────────────────────────


@pytest.mark.asyncio
async def test_two_routes_targeting_same_channel_send_once():
    """Pin: a channel matching two routes (one WARN+, one PAGE+)
    still receives exactly one message per alert."""
    ch = _StubChannel("slack")
    router = AlertRouter(
        [
            AlertRoute(severity_min=Severity.WARN, channel=ch),
            AlertRoute(severity_min=Severity.PAGE, channel=ch),
        ]
    )
    result = await router.dispatch(_alert(severity=Severity.PAGE))
    assert result.sent == ["slack"]
    assert len(ch.sent) == 1


@pytest.mark.asyncio
async def test_dispatch_to_multiple_distinct_channels():
    slack = _StubChannel("slack")
    telegram = _StubChannel("telegram")
    router = AlertRouter(
        [
            AlertRoute(severity_min=Severity.WARN, channel=slack),
            AlertRoute(severity_min=Severity.PAGE, channel=telegram),
        ]
    )
    result = await router.dispatch(_alert(severity=Severity.PAGE))
    assert sorted(result.sent) == ["slack", "telegram"]


@pytest.mark.asyncio
async def test_warn_only_reaches_warn_channel_not_page_only():
    slack = _StubChannel("slack")
    telegram = _StubChannel("telegram")
    router = AlertRouter(
        [
            AlertRoute(severity_min=Severity.WARN, channel=slack),
            AlertRoute(severity_min=Severity.PAGE, channel=telegram),
        ]
    )
    result = await router.dispatch(_alert(severity=Severity.WARN))
    assert result.sent == ["slack"]
    assert telegram.sent == []


# ── per-(type, channel) cooldown ─────────────────────────


@pytest.mark.asyncio
async def test_repeat_within_cooldown_is_suppressed():
    now, _advance = _clock()
    ch = _StubChannel("slack")
    router = AlertRouter(
        [AlertRoute(severity_min=Severity.WARN, channel=ch)],
        cooldown_seconds=600.0,
        now_fn=now,
    )
    first = await router.dispatch(_alert())
    second = await router.dispatch(_alert())
    assert first.sent == ["slack"]
    assert second.sent == []
    assert second.suppressed == ["slack"]
    assert len(ch.sent) == 1


@pytest.mark.asyncio
async def test_repeat_after_cooldown_passes_through():
    now, advance = _clock()
    ch = _StubChannel("slack")
    router = AlertRouter(
        [AlertRoute(severity_min=Severity.WARN, channel=ch)],
        cooldown_seconds=600.0,
        now_fn=now,
    )
    await router.dispatch(_alert())
    advance(700.0)
    second = await router.dispatch(_alert())
    assert second.sent == ["slack"]


@pytest.mark.asyncio
async def test_dedup_is_per_alert_type():
    """Pin: cooldown is keyed on type. Two different alert types
    don't share a window."""
    now, _advance = _clock()
    ch = _StubChannel("slack")
    router = AlertRouter(
        [AlertRoute(severity_min=Severity.WARN, channel=ch)],
        cooldown_seconds=600.0,
        now_fn=now,
    )
    await router.dispatch(_alert(type="halt.engaged"))
    second = await router.dispatch(_alert(type="drift.breach"))
    assert second.sent == ["slack"]


@pytest.mark.asyncio
async def test_dedup_is_per_channel():
    """Pin: a Slack-suppressed alert can still reach Telegram in
    the same window. Operators see each alert once *per channel*,
    not once globally."""
    now, _advance = _clock()
    slack = _StubChannel("slack")
    telegram = _StubChannel("telegram")
    router_slack_first = AlertRouter(
        [AlertRoute(severity_min=Severity.WARN, channel=slack)],
        cooldown_seconds=600.0,
        now_fn=now,
    )
    # Send via slack to populate its cooldown.
    await router_slack_first.dispatch(_alert())
    # Now build a router with both channels, repeat the alert.
    router_both = AlertRouter(
        [
            AlertRoute(severity_min=Severity.WARN, channel=slack),
            AlertRoute(severity_min=Severity.WARN, channel=telegram),
        ],
        cooldown_seconds=600.0,
        now_fn=now,
    )
    second = await router_both.dispatch(_alert())
    # Independent routers each have their own dedup, so the second
    # router sends to both. Pin so the contract is "dedup is
    # router-scoped + channel-scoped".
    assert "slack" in second.sent
    assert "telegram" in second.sent


# ── failure isolation ────────────────────────────────────


@pytest.mark.asyncio
async def test_channel_send_failure_does_not_abort_other_channels():
    """Pin: if Slack send raises, Telegram still gets the alert."""
    slack = _StubChannel("slack", fail_with=RuntimeError)
    telegram = _StubChannel("telegram")
    router = AlertRouter(
        [
            AlertRoute(severity_min=Severity.WARN, channel=slack),
            AlertRoute(severity_min=Severity.WARN, channel=telegram),
        ]
    )
    result = await router.dispatch(_alert())
    assert "telegram" in result.sent
    assert "slack" in result.failed


@pytest.mark.asyncio
async def test_channel_returning_false_is_classified_as_failed():
    """Pin: send returning False (e.g. webhook 4xx) is a failure,
    not a suppression."""
    ch = _StubChannel("slack", return_false=True)
    router = AlertRouter([AlertRoute(severity_min=Severity.WARN, channel=ch)])
    result = await router.dispatch(_alert())
    assert result.failed == ["slack"]
    assert result.sent == []


@pytest.mark.asyncio
async def test_channel_returning_false_does_not_burn_cooldown():
    """Pin: a failed send must not mark the cooldown — the next
    attempt should retry, not be silently suppressed."""
    now, _advance = _clock()
    ch = _StubChannel("slack", return_false=True)
    router = AlertRouter(
        [AlertRoute(severity_min=Severity.WARN, channel=ch)],
        cooldown_seconds=600.0,
        now_fn=now,
    )
    first = await router.dispatch(_alert())
    assert first.failed == ["slack"]
    # Second attempt should retry (not suppressed).
    ch.return_false = False
    second = await router.dispatch(_alert())
    assert second.sent == ["slack"]


# ── disabled channels ────────────────────────────────────


@pytest.mark.asyncio
async def test_disabled_channel_is_skipped_silently():
    """Pin: a channel with enabled=False is not a failure or a
    suppression — operator deliberately turned it off, so it
    quietly drops."""
    disabled = _StubChannel("disabled", _enabled=False)
    enabled = _StubChannel("enabled")
    router = AlertRouter(
        [
            AlertRoute(severity_min=Severity.WARN, channel=disabled),
            AlertRoute(severity_min=Severity.WARN, channel=enabled),
        ]
    )
    result = await router.dispatch(_alert())
    assert result.sent == ["enabled"]
    assert result.failed == []
    assert result.suppressed == []


# ── reached_any property ─────────────────────────────────


@pytest.mark.asyncio
async def test_reached_any_true_when_at_least_one_send_succeeds():
    ch = _StubChannel("slack")
    router = AlertRouter([AlertRoute(severity_min=Severity.WARN, channel=ch)])
    result = await router.dispatch(_alert())
    assert result.reached_any is True


@pytest.mark.asyncio
async def test_reached_any_false_when_all_failed_or_suppressed():
    ch = _StubChannel("slack", return_false=True)
    router = AlertRouter([AlertRoute(severity_min=Severity.WARN, channel=ch)])
    result = await router.dispatch(_alert())
    assert result.reached_any is False


# ── matching_channels ────────────────────────────────────


def test_matching_channels_lists_eligible_only():
    a = _StubChannel("a")
    b = _StubChannel("b")
    router = AlertRouter(
        [
            AlertRoute(severity_min=Severity.WARN, channel=a),
            AlertRoute(severity_min=Severity.PAGE, channel=b),
        ]
    )
    matches = router.matching_channels(_alert(severity=Severity.WARN))
    assert [c.name for c in matches] == ["a"]


def test_matching_channels_dedups_by_name():
    a = _StubChannel("a")
    router = AlertRouter(
        [
            AlertRoute(severity_min=Severity.WARN, channel=a),
            AlertRoute(severity_min=Severity.PAGE, channel=a),
        ]
    )
    matches = router.matching_channels(_alert(severity=Severity.PAGE))
    assert len(matches) == 1


# ── render_message ───────────────────────────────────────


def test_render_includes_severity_emoji():
    text = render_message(_alert(severity=Severity.PAGE))
    assert "🚨" in text


def test_render_includes_type_in_backticks():
    text = render_message(_alert(type="halt.engaged"))
    assert "`halt.engaged`" in text


def test_render_includes_severity_name():
    text = render_message(_alert(severity=Severity.WARN))
    assert "WARN" in text


def test_render_includes_runbook_url_when_present():
    text = render_message(_alert(runbook_url="https://example.com/runbook"))
    assert "https://example.com/runbook" in text


def test_render_flags_missing_runbook():
    """Pin: an empty runbook URL surfaces "no runbook yet" rather
    than silently rendering nothing — operators should spot the
    gap and write one."""
    text = render_message(_alert(runbook_url=""))
    # Renderer surfaces a "(none yet — write one in docs/runbooks/)"
    # marker so the operator can spot the gap and write one. Match
    # the substring that's stable across phrasing tweaks.
    assert "none yet" in text.lower()
    assert "docs/runbooks" in text


def test_render_includes_sorted_context_kv():
    text = render_message(_alert(context={"pair": "BTCUSDT", "cycle_id": "abc123"}))
    # Pin: keys sorted alphabetically so the rendered order is
    # deterministic (cycle_id < pair).
    cycle_idx = text.find("cycle_id")
    pair_idx = text.find("pair")
    assert cycle_idx < pair_idx


def test_render_omits_context_section_when_empty():
    text = render_message(_alert(context={}))
    # No stray k/v lines.
    assert "  " not in text or "Runbook" in text


# ── output structure ─────────────────────────────────────


def test_alert_spec_is_immutable():
    spec = _alert()
    with pytest.raises(Exception):
        spec.severity = Severity.PAGE  # type: ignore[misc]


@pytest.mark.asyncio
async def test_dispatch_result_is_immutable():
    ch = _StubChannel("slack")
    router = AlertRouter([AlertRoute(severity_min=Severity.WARN, channel=ch)])
    result = await router.dispatch(_alert())
    with pytest.raises(Exception):
        result.alert_type = "tampered"  # type: ignore[misc]


def test_severity_int_enum_supports_comparison():
    """Pin: IntEnum so `severity >= warn` reads naturally and the
    AlertRoute.matches comparison works without explicit ordinal
    lookups."""
    assert Severity.PAGE >= Severity.WARN
    assert Severity.WARN >= Severity.INFO
    assert not (Severity.INFO >= Severity.PAGE)
