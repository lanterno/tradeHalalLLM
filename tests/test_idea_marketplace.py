"""Tests for community/idea_marketplace.py — Round-5 Wave 17.B."""

from __future__ import annotations

from datetime import datetime

import pytest

from halal_trader.community.idea_marketplace import (
    AuthorAttribution,
    FeeStructure,
    IdeaSide,
    IdeaStatus,
    RiskBand,
    Subscription,
    TradeIdea,
    assert_idea_compliant,
    assert_no_performance_carry,
    author_attribution,
    render_attribution,
    render_idea,
    subscribe,
    transition_idea,
    unsubscribe,
)


def _idea(
    idea_id: str = "I1",
    author_id: str = "alice",
    ticker: str = "AAPL",
    side: IdeaSide = IdeaSide.LONG,
    entry: float = 100.0,
    target: float = 120.0,
    stop: float = 95.0,
    risk: RiskBand = RiskBand.MEDIUM,
    rationale: str = "Strong fundamentals + technical breakout",
    published: datetime = datetime(2026, 5, 1, 9, 0),
    horizon_days: int = 30,
    fee_structure: FeeStructure = FeeStructure.FLAT_PER_SUBSCRIPTION,
    fee_amount: float = 5.0,
    status: IdeaStatus = IdeaStatus.PUBLISHED,
    closed_at: datetime | None = None,
    realised_return: float | None = None,
) -> TradeIdea:
    return TradeIdea(
        idea_id=idea_id,
        author_id=author_id,
        ticker=ticker,
        side=side,
        entry_price=entry,
        target_price=target,
        stop_price=stop,
        risk_band=risk,
        rationale_summary=rationale,
        published_at=published,
        horizon_days=horizon_days,
        fee_structure=fee_structure,
        fee_amount_usd=fee_amount,
        status=status,
        closed_at=closed_at,
        realised_return_pct=realised_return,
    )


# --- TradeIdea validation -----------------------------------------------


def test_idea_valid_long():
    i = _idea()
    assert i.side is IdeaSide.LONG


def test_idea_valid_short():
    i = _idea(side=IdeaSide.SHORT, entry=100.0, target=80.0, stop=110.0)
    assert i.side is IdeaSide.SHORT


def test_idea_long_invalid_geometry_rejected():
    with pytest.raises(ValueError):
        _idea(entry=100.0, target=80.0, stop=95.0)
    with pytest.raises(ValueError):
        _idea(entry=100.0, target=120.0, stop=110.0)


def test_idea_short_invalid_geometry_rejected():
    with pytest.raises(ValueError):
        _idea(side=IdeaSide.SHORT, entry=100.0, target=120.0, stop=110.0)


def test_idea_zero_entry_rejected():
    with pytest.raises(ValueError):
        _idea(entry=0)


def test_idea_empty_rationale_rejected():
    with pytest.raises(ValueError):
        _idea(rationale=" ")


def test_idea_long_rationale_rejected():
    with pytest.raises(ValueError):
        _idea(rationale="x" * 600)


def test_idea_negative_fee_rejected():
    with pytest.raises(ValueError):
        _idea(fee_amount=-1.0)


def test_idea_excessive_fee_rejected():
    """Pin: > $100 is suspicious for a per-subscription Wakalah."""
    with pytest.raises(ValueError):
        _idea(fee_amount=200.0)


def test_idea_immutable():
    i = _idea()
    with pytest.raises(AttributeError):
        i.entry_price = 0  # type: ignore[misc]


def test_idea_reward_to_risk_long():
    """LONG: (target - entry) / (entry - stop) = (120-100)/(100-95) = 4."""
    i = _idea()
    assert i.reward_to_risk() == pytest.approx(4.0)


def test_idea_reward_to_risk_short():
    """SHORT: (entry - target) / (stop - entry) = (100-80)/(110-100) = 2."""
    i = _idea(side=IdeaSide.SHORT, entry=100.0, target=80.0, stop=110.0)
    assert i.reward_to_risk() == pytest.approx(2.0)


# --- Subscription validation --------------------------------------------


def _sub(
    sid: str = "S1",
    iid: str = "I1",
    follower: str = "bob",
    subscribed_at: datetime = datetime(2026, 5, 2, 9, 0),
    fee: float = 5.0,
) -> Subscription:
    return Subscription(
        subscription_id=sid,
        idea_id=iid,
        follower_id=follower,
        subscribed_at=subscribed_at,
        fee_paid_usd=fee,
    )


def test_subscription_valid():
    s = _sub()
    assert s.is_active()


def test_subscription_empty_id_rejected():
    with pytest.raises(ValueError):
        _sub(sid="")


def test_subscription_negative_fee_rejected():
    with pytest.raises(ValueError):
        _sub(fee=-1.0)


def test_subscription_unsub_before_sub_rejected():
    with pytest.raises(ValueError):
        Subscription(
            subscription_id="S1",
            idea_id="I1",
            follower_id="bob",
            subscribed_at=datetime(2026, 5, 5),
            fee_paid_usd=5.0,
            unsubscribed_at=datetime(2026, 5, 1),
        )


def test_subscription_active_then_inactive():
    s = _sub()
    s2 = unsubscribe(s, unsubscribed_at=datetime(2026, 6, 1))
    # is_active: True if no unsub OR now < unsub.
    assert s2.is_active(now=datetime(2026, 5, 15))
    assert not s2.is_active(now=datetime(2026, 7, 1))


# --- subscribe -----------------------------------------------------------


def test_subscribe_records_fee():
    i = _idea(fee_amount=10.0)
    s = subscribe(
        i,
        subscription_id="S1",
        follower_id="bob",
        subscribed_at=datetime(2026, 5, 2),
    )
    assert s.fee_paid_usd == 10.0


def test_subscribe_self_rejected():
    """Author cannot subscribe to their own idea."""
    i = _idea(author_id="alice")
    with pytest.raises(ValueError):
        subscribe(
            i,
            subscription_id="S1",
            follower_id="alice",
            subscribed_at=datetime(2026, 5, 2),
        )


def test_subscribe_to_revoked_rejected():
    i = _idea(status=IdeaStatus.REVOKED)
    with pytest.raises(ValueError):
        subscribe(
            i,
            subscription_id="S1",
            follower_id="bob",
            subscribed_at=datetime(2026, 5, 2),
        )


def test_subscribe_to_closed_rejected():
    i = _idea(
        status=IdeaStatus.CLOSED,
        closed_at=datetime(2026, 5, 5),
        realised_return=0.05,
    )
    with pytest.raises(ValueError):
        subscribe(
            i,
            subscription_id="S1",
            follower_id="bob",
            subscribed_at=datetime(2026, 5, 6),
        )


def test_subscribe_to_triggered_allowed():
    i = _idea(status=IdeaStatus.TRIGGERED)
    s = subscribe(
        i,
        subscription_id="S1",
        follower_id="bob",
        subscribed_at=datetime(2026, 5, 2),
    )
    assert s.subscription_id == "S1"


# --- unsubscribe ---------------------------------------------------------


def test_unsubscribe_double_rejected():
    s = _sub()
    s2 = unsubscribe(s, unsubscribed_at=datetime(2026, 6, 1))
    with pytest.raises(ValueError):
        unsubscribe(s2, unsubscribed_at=datetime(2026, 6, 5))


# --- transition_idea — legal moves --------------------------------------


def test_transition_published_to_triggered():
    i = _idea(status=IdeaStatus.PUBLISHED)
    i2 = transition_idea(i, new_status=IdeaStatus.TRIGGERED, at=datetime(2026, 5, 5))
    assert i2.status is IdeaStatus.TRIGGERED


def test_transition_triggered_to_closed_requires_return():
    i = _idea(status=IdeaStatus.TRIGGERED)
    with pytest.raises(ValueError):
        transition_idea(i, new_status=IdeaStatus.CLOSED, at=datetime(2026, 5, 5))


def test_transition_to_closed_records_return():
    i = _idea(status=IdeaStatus.TRIGGERED)
    i2 = transition_idea(
        i,
        new_status=IdeaStatus.CLOSED,
        at=datetime(2026, 5, 30),
        realised_return_pct=0.10,
    )
    assert i2.status is IdeaStatus.CLOSED
    assert i2.realised_return_pct == 0.10
    assert i2.closed_at == datetime(2026, 5, 30)


def test_transition_published_to_revoked():
    i = _idea(status=IdeaStatus.PUBLISHED)
    i2 = transition_idea(i, new_status=IdeaStatus.REVOKED, at=datetime(2026, 5, 5))
    assert i2.status is IdeaStatus.REVOKED


def test_transition_closed_is_terminal():
    i = _idea(
        status=IdeaStatus.CLOSED,
        closed_at=datetime(2026, 5, 5),
        realised_return=0.05,
    )
    with pytest.raises(ValueError):
        transition_idea(
            i,
            new_status=IdeaStatus.PUBLISHED,
            at=datetime(2026, 5, 6),
        )


def test_transition_revoked_is_terminal():
    i = _idea(status=IdeaStatus.REVOKED)
    with pytest.raises(ValueError):
        transition_idea(
            i,
            new_status=IdeaStatus.PUBLISHED,
            at=datetime(2026, 5, 6),
        )


def test_transition_triggered_to_revoked_rejected():
    i = _idea(status=IdeaStatus.TRIGGERED)
    with pytest.raises(ValueError):
        transition_idea(i, new_status=IdeaStatus.REVOKED, at=datetime(2026, 5, 6))


# --- assert_no_performance_carry ----------------------------------------


def test_assert_no_performance_carry_passes_for_flat():
    i = _idea(fee_structure=FeeStructure.FLAT_PER_SUBSCRIPTION)
    assert_no_performance_carry(i)
    i2 = _idea(fee_structure=FeeStructure.FLAT_PER_FOLLOW_DAY)
    assert_no_performance_carry(i2)


# --- assert_idea_compliant ----------------------------------------------


def test_assert_compliant_passes():
    i = _idea(ticker="AAPL")
    assert_idea_compliant(i, is_ticker_halal=lambda t: True)


def test_assert_compliant_haram_rejected():
    i = _idea(ticker="MO")  # tobacco proxy
    with pytest.raises(ValueError):
        assert_idea_compliant(i, is_ticker_halal=lambda t: t != "MO")


# --- author_attribution -------------------------------------------------


def test_attribution_per_author_rollup():
    i1 = _idea(idea_id="I1", author_id="alice")
    i2 = _idea(idea_id="I2", author_id="alice")
    i3 = _idea(idea_id="I3", author_id="charlie")
    s1 = _sub(sid="S1", iid="I1", follower="bob", fee=5.0)
    s2 = _sub(sid="S2", iid="I1", follower="dave", fee=5.0)
    s3 = _sub(sid="S3", iid="I3", follower="bob", fee=5.0)
    out = author_attribution([i1, i2, i3], [s1, s2, s3])
    by_author = {a.author_id: a for a in out}
    assert by_author["alice"].n_ideas_published == 2
    assert by_author["alice"].n_subscriptions == 2
    assert by_author["alice"].total_wakalah_fees_usd == 10.0
    assert by_author["charlie"].n_ideas_published == 1
    assert by_author["charlie"].n_subscriptions == 1


def test_attribution_win_rate_only_closed():
    i_open = _idea(idea_id="I1", author_id="alice", status=IdeaStatus.PUBLISHED)
    i_win = _idea(
        idea_id="I2",
        author_id="alice",
        status=IdeaStatus.CLOSED,
        closed_at=datetime(2026, 5, 30),
        realised_return=0.10,
    )
    i_loss = _idea(
        idea_id="I3",
        author_id="alice",
        status=IdeaStatus.CLOSED,
        closed_at=datetime(2026, 5, 30),
        realised_return=-0.05,
    )
    out = author_attribution([i_open, i_win, i_loss], [])
    a = out[0]
    assert a.win_rate == pytest.approx(0.50)
    assert a.closed_ideas_avg_return_pct == pytest.approx(0.025)


def test_attribution_no_closed_ideas_zero_metrics():
    i = _idea(idea_id="I1", author_id="alice", status=IdeaStatus.PUBLISHED)
    out = author_attribution([i], [])
    assert out[0].win_rate == 0.0
    assert out[0].closed_ideas_avg_return_pct == 0.0


def test_attribution_sorted_by_author():
    i_a = _idea(idea_id="I1", author_id="alice")
    i_z = _idea(idea_id="I2", author_id="zoe")
    out = author_attribution([i_z, i_a], [])
    assert [a.author_id for a in out] == ["alice", "zoe"]


# --- Render --------------------------------------------------------------


def test_render_idea_no_secret_leak():
    i = _idea(author_id="alice@example.com")
    out = render_idea(i)
    assert "alice@example.com" not in out


def test_render_idea_includes_status_emoji():
    i = _idea(status=IdeaStatus.PUBLISHED)
    out = render_idea(i)
    assert "📢" in out


def test_render_attribution_no_secret_leak():
    a = AuthorAttribution(
        author_id="alice@example.com",
        n_ideas_published=2,
        n_subscriptions=4,
        total_wakalah_fees_usd=20.0,
        closed_ideas_avg_return_pct=0.05,
        win_rate=0.5,
    )
    out = render_attribution(a)
    assert "alice@example.com" not in out


def test_render_idea_reward_to_risk_visible():
    i = _idea()
    out = render_idea(i)
    assert "R:R=" in out
