"""Tests for marketplace/signal_subscription.py — Round-5 Wave 21.B."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.marketplace.signal_subscription import (
    BillingEvent,
    BillingPeriod,
    FeeRollup,
    FeeTier,
    Subscription,
    SubscriptionStatus,
    TierPricing,
    accrue_due_fees,
    can_resubscribe,
    cancel_subscription,
    default_pricing,
    render_rollup,
    render_subscription,
    rollup_events,
)


def _sub(
    subscription_id: str = "S1",
    subscriber_id: str = "alice",
    author_id: str = "bob",
    tier: FeeTier = FeeTier.PRO,
    billing_period: BillingPeriod = BillingPeriod.WEEKLY,
    started_on: date = date(2026, 5, 1),
    platform_fee_pct: float = 0.20,
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE,
    cancelled_on: date | None = None,
) -> Subscription:
    return Subscription(
        subscription_id=subscription_id,
        subscriber_id=subscriber_id,
        author_id=author_id,
        tier=tier,
        billing_period=billing_period,
        started_on=started_on,
        platform_fee_pct=platform_fee_pct,
        status=status,
        cancelled_on=cancelled_on,
    )


# --- TierPricing validation ----------------------------------


def test_tier_pricing_valid():
    p = TierPricing(period_fee_usd=10.0, max_signals_per_period=5)
    assert p.max_signals_per_period == 5


def test_tier_pricing_negative_fee_rejected():
    with pytest.raises(ValueError):
        TierPricing(period_fee_usd=-1.0, max_signals_per_period=5)


def test_tier_pricing_excessive_fee_rejected():
    with pytest.raises(ValueError):
        TierPricing(period_fee_usd=10_000.0, max_signals_per_period=5)


def test_tier_pricing_zero_max_signals_rejected():
    with pytest.raises(ValueError):
        TierPricing(period_fee_usd=10.0, max_signals_per_period=0)


def test_default_pricing_keys():
    p = default_pricing()
    for tier in FeeTier:
        assert tier in p


# --- Subscription validation -----------------------------


def test_subscription_valid():
    s = _sub()
    assert s.tier is FeeTier.PRO


def test_subscription_self_dealing_rejected():
    with pytest.raises(ValueError):
        _sub(subscriber_id="x", author_id="x")


def test_subscription_empty_id_rejected():
    with pytest.raises(ValueError):
        _sub(subscription_id="")


def test_subscription_invalid_platform_fee_rejected():
    with pytest.raises(ValueError):
        _sub(platform_fee_pct=0.50)


def test_subscription_cancelled_without_date_rejected():
    with pytest.raises(ValueError):
        _sub(status=SubscriptionStatus.CANCELLED, cancelled_on=None)


def test_subscription_active_with_cancel_date_rejected():
    with pytest.raises(ValueError):
        _sub(
            status=SubscriptionStatus.ACTIVE,
            cancelled_on=date(2026, 5, 15),
        )


def test_subscription_cancel_before_start_rejected():
    with pytest.raises(ValueError):
        _sub(
            status=SubscriptionStatus.CANCELLED,
            cancelled_on=date(2026, 4, 1),
        )


def test_subscription_immutable():
    s = _sub()
    with pytest.raises(AttributeError):
        s.tier = FeeTier.STARTER  # type: ignore[misc]


# --- accrue_due_fees ----------------------------------


def test_accrue_no_events_before_first_period_end():
    s = _sub(billing_period=BillingPeriod.WEEKLY)
    events = accrue_due_fees(s, until=date(2026, 5, 3))
    assert events == ()


def test_accrue_one_period_after_first_end():
    s = _sub(billing_period=BillingPeriod.WEEKLY, started_on=date(2026, 5, 1))
    events = accrue_due_fees(s, until=date(2026, 5, 8))
    assert len(events) == 1
    assert events[0].period_index == 0
    assert events[0].period_end_on == date(2026, 5, 8)


def test_accrue_multiple_periods():
    s = _sub(billing_period=BillingPeriod.WEEKLY, started_on=date(2026, 5, 1))
    events = accrue_due_fees(s, until=date(2026, 5, 29))
    # 5/8, 5/15, 5/22, 5/29 → 4 periods.
    assert len(events) == 4


def test_accrue_truncated_by_cancellation():
    s = _sub(
        billing_period=BillingPeriod.WEEKLY,
        started_on=date(2026, 5, 1),
        status=SubscriptionStatus.CANCELLED,
        cancelled_on=date(2026, 5, 15),
    )
    events = accrue_due_fees(s, until=date(2026, 6, 1))
    # Only periods that end on or before 2026-05-15: 5/8, 5/15.
    assert len(events) == 2


def test_accrue_daily_period():
    s = _sub(
        billing_period=BillingPeriod.DAILY,
        started_on=date(2026, 5, 1),
        tier=FeeTier.STARTER,
    )
    events = accrue_due_fees(s, until=date(2026, 5, 5))
    assert len(events) == 4


def test_accrue_monthly_period():
    s = _sub(
        billing_period=BillingPeriod.MONTHLY,
        started_on=date(2026, 5, 1),
    )
    events = accrue_due_fees(s, until=date(2026, 9, 1))
    # 30d cadence: 5/31, 6/30, 7/30, 8/29 → 4 periods.
    assert len(events) == 4


def test_accrue_fee_split_pinned():
    """Pin: platform = gross × pct; author = gross − platform."""
    s = _sub(billing_period=BillingPeriod.WEEKLY, platform_fee_pct=0.25)
    events = accrue_due_fees(s, until=date(2026, 5, 8))
    assert len(events) == 1
    # PRO tier default = $50/period.
    assert events[0].gross_usd == 50.0
    assert events[0].platform_share_usd == pytest.approx(12.50)
    assert events[0].author_share_usd == pytest.approx(37.50)


def test_accrue_unknown_tier_rejected():
    s = _sub()
    bad_pricing = {FeeTier.STARTER: TierPricing(period_fee_usd=10.0, max_signals_per_period=5)}
    with pytest.raises(ValueError):
        accrue_due_fees(s, until=date(2026, 5, 30), pricing=bad_pricing)


# --- cancel_subscription -----------------------------


def test_cancel_active():
    s = _sub()
    s2 = cancel_subscription(s, on=date(2026, 5, 15))
    assert s2.status is SubscriptionStatus.CANCELLED


def test_cancel_already_cancelled_rejected():
    s = cancel_subscription(_sub(), on=date(2026, 5, 15))
    with pytest.raises(ValueError):
        cancel_subscription(s, on=date(2026, 5, 20))


def test_cancel_before_started_rejected():
    s = _sub()
    with pytest.raises(ValueError):
        cancel_subscription(s, on=date(2026, 4, 1))


# --- can_resubscribe --------------------------------


def test_can_resubscribe_no_prior():
    ok, _ = can_resubscribe("alice", "bob", [], on=date(2026, 6, 1))
    assert ok


def test_can_resubscribe_cooldown_active():
    prior = cancel_subscription(_sub(), on=date(2026, 5, 30))
    ok, reason = can_resubscribe("alice", "bob", [prior], on=date(2026, 6, 2), cooldown_days=7)
    assert not ok
    assert "cooldown" in reason


def test_can_resubscribe_cooldown_expired():
    prior = cancel_subscription(_sub(), on=date(2026, 5, 1))
    ok, _ = can_resubscribe("alice", "bob", [prior], on=date(2026, 6, 1), cooldown_days=7)
    assert ok


def test_can_resubscribe_active_blocks():
    prior = _sub()  # active
    ok, reason = can_resubscribe("alice", "bob", [prior], on=date(2026, 6, 1))
    assert not ok
    assert "existing active" in reason


def test_can_resubscribe_other_pairs_dont_count():
    prior = cancel_subscription(
        _sub(subscriber_id="alice", author_id="charlie"), on=date(2026, 5, 30)
    )
    ok, _ = can_resubscribe("alice", "bob", [prior], on=date(2026, 6, 1), cooldown_days=7)
    assert ok


def test_can_resubscribe_invalid_cooldown_rejected():
    with pytest.raises(ValueError):
        can_resubscribe("alice", "bob", [], on=date(2026, 6, 1), cooldown_days=0)


# --- rollup_events ----------------------------------


def test_rollup_empty_returns_none():
    assert rollup_events([]) is None


def test_rollup_sums_correctly():
    events = [
        BillingEvent(
            subscription_id="S1",
            period_index=i,
            period_end_on=date(2026, 5, 8 + 7 * i),
            gross_usd=50.0,
            platform_share_usd=10.0,
            author_share_usd=40.0,
        )
        for i in range(3)
    ]
    rollup = rollup_events(events)
    assert rollup is not None
    assert rollup.n_periods == 3
    assert rollup.total_gross_usd == 150.0
    assert rollup.total_platform_usd == 30.0
    assert rollup.total_author_usd == 120.0


def test_rollup_mixed_subscription_rejected():
    events = [
        BillingEvent(
            subscription_id="S1",
            period_index=0,
            period_end_on=date(2026, 5, 8),
            gross_usd=50.0,
            platform_share_usd=10.0,
            author_share_usd=40.0,
        ),
        BillingEvent(
            subscription_id="S2",
            period_index=0,
            period_end_on=date(2026, 5, 8),
            gross_usd=50.0,
            platform_share_usd=10.0,
            author_share_usd=40.0,
        ),
    ]
    with pytest.raises(ValueError):
        rollup_events(events)


# --- Render ---------------------------------------


def test_render_subscription_no_secret_leak():
    s = _sub(
        subscriber_id="alice@example.com",
        author_id="bob@example.com",
    )
    out = render_subscription(s)
    assert "alice@example.com" not in out
    assert "bob@example.com" not in out


def test_render_subscription_status_emoji():
    s = _sub()
    out = render_subscription(s)
    assert "✅" in out
    cancelled = cancel_subscription(s, on=date(2026, 5, 15))
    out2 = render_subscription(cancelled)
    assert "🚫" in out2


def test_render_rollup_format():
    rollup = FeeRollup(
        subscription_id="S1",
        n_periods=3,
        total_gross_usd=150.0,
        total_platform_usd=30.0,
        total_author_usd=120.0,
    )
    out = render_rollup(rollup)
    assert "S1" in out
    assert "platform=$30.00" in out
    assert "author=$120.00" in out
