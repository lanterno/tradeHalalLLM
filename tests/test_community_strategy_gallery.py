"""Tests for community/strategy_gallery.py — Round-5 Wave 17.C.

Note: distinct from `tests/test_strategy_gallery.py` which covers the
public web-side curation engine. This file covers the community-side
subscription + Wakalah-fee + performance ledger primitives.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from halal_trader.community.strategy_gallery import (
    BillingPeriod,
    GallerySubscription,
    ListingStatus,
    PerformanceEntry,
    RiskBand,
    StrategyListing,
    Visibility,
    append_performance,
    cancel,
    compute_fee_split,
    render_fee_split,
    render_ledger,
    render_listing,
    subscribe,
    transition_listing,
    verify_ledger,
)


def _listing(
    listing_id: str = "L1",
    author: str = "alice",
    name: str = "Halal Momentum",
    description: str = "Buy halal stocks with high RSI + volume.",
    visibility: Visibility = Visibility.PAID,
    risk: RiskBand = RiskBand.MEDIUM,
    billing: BillingPeriod = BillingPeriod.MONTHLY,
    fee: float = 50.0,
    platform_fee: float = 0.20,
    status: ListingStatus = ListingStatus.PUBLISHED,
    published_at: datetime | None = datetime(2026, 5, 1, 9, 0),
) -> StrategyListing:
    return StrategyListing(
        listing_id=listing_id,
        author_id=author,
        name=name,
        description=description,
        visibility=visibility,
        risk_band=risk,
        billing_period=billing,
        wakalah_fee_per_period_usd=fee,
        platform_fee_pct=platform_fee,
        published_at=published_at,
        status=status,
    )


# --- StrategyListing validation -----------------------------------------


def test_listing_valid():
    L = _listing()
    assert L.author_take_per_period() == pytest.approx(40.0)
    assert L.platform_take_per_period() == pytest.approx(10.0)


def test_listing_empty_id_rejected():
    with pytest.raises(ValueError):
        _listing(listing_id="")


def test_listing_long_name_rejected():
    with pytest.raises(ValueError):
        _listing(name="x" * 200)


def test_listing_empty_description_rejected():
    with pytest.raises(ValueError):
        _listing(description=" ")


def test_listing_long_description_rejected():
    with pytest.raises(ValueError):
        _listing(description="x" * 1500)


def test_listing_negative_fee_rejected():
    with pytest.raises(ValueError):
        _listing(fee=-1.0)


def test_listing_excessive_fee_rejected():
    with pytest.raises(ValueError):
        _listing(fee=2000.0)


def test_listing_free_with_fee_rejected():
    with pytest.raises(ValueError):
        _listing(visibility=Visibility.FREE, fee=10.0)


def test_listing_free_zero_fee_ok():
    L = _listing(visibility=Visibility.FREE, fee=0.0)
    assert L.visibility is Visibility.FREE


def test_listing_platform_fee_above_30pct_rejected():
    with pytest.raises(ValueError):
        _listing(platform_fee=0.40)


def test_listing_published_requires_timestamp():
    with pytest.raises(ValueError):
        _listing(status=ListingStatus.PUBLISHED, published_at=None)


def test_listing_immutable():
    L = _listing()
    with pytest.raises(AttributeError):
        L.wakalah_fee_per_period_usd = 0.0  # type: ignore[misc]


def test_author_take_pinned():
    L = _listing(fee=100.0, platform_fee=0.20)
    assert L.author_take_per_period() == pytest.approx(80.0)
    assert L.platform_take_per_period() == pytest.approx(20.0)


def test_zero_platform_fee_full_to_author():
    L = _listing(fee=100.0, platform_fee=0.0)
    assert L.author_take_per_period() == pytest.approx(100.0)
    assert L.platform_take_per_period() == 0.0


# --- Subscription validation --------------------------------------------


def _sub(
    sid: str = "S1",
    listing_id: str = "L1",
    subscriber: str = "bob",
    started: datetime = datetime(2026, 5, 5, 9, 0),
    next_billing: datetime = datetime(2026, 6, 4, 9, 0),
) -> GallerySubscription:
    return GallerySubscription(
        subscription_id=sid,
        listing_id=listing_id,
        subscriber_id=subscriber,
        started_at=started,
        next_billing_at=next_billing,
    )


def test_subscription_valid():
    s = _sub()
    assert s.is_active(datetime(2026, 5, 10))


def test_subscription_next_billing_before_start_rejected():
    with pytest.raises(ValueError):
        GallerySubscription(
            subscription_id="S1",
            listing_id="L1",
            subscriber_id="bob",
            started_at=datetime(2026, 5, 5),
            next_billing_at=datetime(2026, 5, 1),
        )


def test_subscription_active_after_cancel_until_cancel_date():
    s = _sub()
    s2 = cancel(s, cancelled_at=datetime(2026, 6, 10))
    assert s2.is_active(datetime(2026, 6, 1))
    assert not s2.is_active(datetime(2026, 7, 1))


# --- subscribe -----------------------------------------------------------


def test_subscribe_basic():
    L = _listing()
    s = subscribe(
        L,
        subscription_id="S1",
        subscriber_id="bob",
        started_at=datetime(2026, 5, 5),
    )
    assert s.next_billing_at == datetime(2026, 5, 5) + timedelta(days=30)


def test_subscribe_quarterly_uses_90_days():
    L = _listing(billing=BillingPeriod.QUARTERLY)
    s = subscribe(
        L,
        subscription_id="S1",
        subscriber_id="bob",
        started_at=datetime(2026, 5, 5),
    )
    assert (s.next_billing_at - s.started_at) == timedelta(days=90)


def test_subscribe_annual_uses_365_days():
    L = _listing(billing=BillingPeriod.ANNUAL)
    s = subscribe(
        L,
        subscription_id="S1",
        subscriber_id="bob",
        started_at=datetime(2026, 5, 5),
    )
    assert (s.next_billing_at - s.started_at) == timedelta(days=365)


def test_subscribe_self_rejected():
    L = _listing(author="alice")
    with pytest.raises(ValueError):
        subscribe(
            L,
            subscription_id="S1",
            subscriber_id="alice",
            started_at=datetime(2026, 5, 5),
        )


def test_subscribe_to_draft_rejected():
    L = _listing(status=ListingStatus.DRAFT, published_at=None)
    with pytest.raises(ValueError):
        subscribe(
            L,
            subscription_id="S1",
            subscriber_id="bob",
            started_at=datetime(2026, 5, 5),
        )


def test_subscribe_to_archived_rejected():
    L = _listing(status=ListingStatus.ARCHIVED)
    with pytest.raises(ValueError):
        subscribe(
            L,
            subscription_id="S1",
            subscriber_id="bob",
            started_at=datetime(2026, 5, 5),
        )


def test_subscribe_to_deprecated_rejected_for_new():
    L = _listing(status=ListingStatus.DEPRECATED)
    with pytest.raises(ValueError):
        subscribe(
            L,
            subscription_id="S1",
            subscriber_id="bob",
            started_at=datetime(2026, 5, 5),
        )


def test_cancel_double_rejected():
    s = _sub()
    s2 = cancel(s, cancelled_at=datetime(2026, 6, 10))
    with pytest.raises(ValueError):
        cancel(s2, cancelled_at=datetime(2026, 6, 15))


# --- compute_fee_split --------------------------------------------------


def test_compute_fee_split_arithmetic():
    L = _listing(fee=100.0, platform_fee=0.25)
    s = _sub(listing_id="L1")
    split = compute_fee_split(L, s)
    assert split.gross_fee_usd == 100.0
    assert split.platform_take_usd == pytest.approx(25.0)
    assert split.author_take_usd == pytest.approx(75.0)


def test_compute_fee_split_listing_mismatch_rejected():
    L = _listing(listing_id="L1")
    s = _sub(listing_id="L2")
    with pytest.raises(ValueError):
        compute_fee_split(L, s)


# --- PerformanceEntry validation + ledger -------------------------------


def _entry(
    listing_id: str = "L1",
    period_end: date = date(2026, 5, 31),
    return_pct: float = 0.05,
    drawdown_pct: float = 0.03,
    bench: float = 0.02,
    n_subs: int = 10,
    prev_hash: str = "",
) -> PerformanceEntry:
    return PerformanceEntry(
        listing_id=listing_id,
        period_end=period_end,
        return_pct=return_pct,
        drawdown_pct=drawdown_pct,
        benchmark_return_pct=bench,
        n_subscribers=n_subs,
        prev_hash=prev_hash,
    )


def test_entry_valid():
    e = _entry()
    assert e.return_pct == 0.05


def test_entry_negative_drawdown_rejected():
    with pytest.raises(ValueError):
        _entry(drawdown_pct=-0.01)


def test_entry_unreasonable_return_rejected():
    with pytest.raises(ValueError):
        _entry(return_pct=10.0)


def test_entry_hash_stable():
    e1 = _entry()
    e2 = _entry()
    assert e1.entry_hash() == e2.entry_hash()


def test_entry_hash_changes_with_fields():
    e1 = _entry(return_pct=0.05)
    e2 = _entry(return_pct=0.10)
    assert e1.entry_hash() != e2.entry_hash()


def test_append_first_requires_empty_prev_hash():
    e_bad = _entry(prev_hash="abc")
    with pytest.raises(ValueError):
        append_performance((), e_bad)


def test_append_chains_correctly():
    e1 = _entry(period_end=date(2026, 5, 31))
    ledger = append_performance((), e1)
    e2 = _entry(period_end=date(2026, 6, 30), prev_hash=e1.entry_hash())
    ledger = append_performance(ledger, e2)
    assert len(ledger) == 2
    assert verify_ledger(ledger)


def test_append_wrong_prev_hash_rejected():
    e1 = _entry(period_end=date(2026, 5, 31))
    ledger = append_performance((), e1)
    e2 = _entry(period_end=date(2026, 6, 30), prev_hash="wrong")
    with pytest.raises(ValueError):
        append_performance(ledger, e2)


def test_append_period_must_strictly_increase():
    e1 = _entry(period_end=date(2026, 5, 31))
    ledger = append_performance((), e1)
    e2 = _entry(period_end=date(2026, 5, 31), prev_hash=e1.entry_hash())
    with pytest.raises(ValueError):
        append_performance(ledger, e2)


def test_append_listing_id_mismatch_rejected():
    e1 = _entry(listing_id="L1", period_end=date(2026, 5, 31))
    ledger = append_performance((), e1)
    e2 = _entry(
        listing_id="L2",
        period_end=date(2026, 6, 30),
        prev_hash=e1.entry_hash(),
    )
    with pytest.raises(ValueError):
        append_performance(ledger, e2)


def test_verify_ledger_empty():
    assert verify_ledger([])


def test_verify_ledger_detects_tamper():
    e1 = _entry(period_end=date(2026, 5, 31))
    e2 = _entry(period_end=date(2026, 6, 30), prev_hash="wrong")
    assert not verify_ledger((e1, e2))


# --- transition_listing -------------------------------------------------


def test_transition_draft_to_published():
    L = _listing(status=ListingStatus.DRAFT, published_at=None)
    L2 = transition_listing(L, new_status=ListingStatus.PUBLISHED, at=datetime(2026, 5, 1))
    assert L2.status is ListingStatus.PUBLISHED
    assert L2.published_at == datetime(2026, 5, 1)


def test_transition_published_to_deprecated():
    L = _listing(status=ListingStatus.PUBLISHED)
    L2 = transition_listing(L, new_status=ListingStatus.DEPRECATED, at=datetime(2026, 7, 1))
    assert L2.status is ListingStatus.DEPRECATED


def test_transition_deprecated_to_archived():
    L = _listing(status=ListingStatus.DEPRECATED)
    L2 = transition_listing(L, new_status=ListingStatus.ARCHIVED, at=datetime(2026, 7, 1))
    assert L2.status is ListingStatus.ARCHIVED


def test_transition_archived_terminal():
    L = _listing(status=ListingStatus.ARCHIVED)
    with pytest.raises(ValueError):
        transition_listing(L, new_status=ListingStatus.PUBLISHED, at=datetime(2026, 7, 1))


def test_transition_skips_intermediate_rejected():
    L = _listing(status=ListingStatus.DRAFT, published_at=None)
    with pytest.raises(ValueError):
        transition_listing(L, new_status=ListingStatus.DEPRECATED, at=datetime(2026, 7, 1))


# --- Render --------------------------------------------------------------


def test_render_listing_no_secret_leak():
    L = _listing(author="alice@example.com")
    out = render_listing(L)
    assert "alice@example.com" not in out


def test_render_listing_visibility_emoji():
    L_paid = _listing(visibility=Visibility.PAID)
    L_free = _listing(visibility=Visibility.FREE, fee=0.0)
    assert "💸" in render_listing(L_paid)
    assert "🆓" in render_listing(L_free)


def test_render_fee_split_format():
    L = _listing(fee=100.0)
    s = _sub()
    split = compute_fee_split(L, s)
    out = render_fee_split(split)
    assert "Billed" in out
    assert "platform=" in out
    assert "author=" in out


def test_render_ledger_empty():
    assert "No performance" in render_ledger([])


def test_render_ledger_includes_periods():
    e1 = _entry(period_end=date(2026, 5, 31))
    out = render_ledger([e1])
    assert "2026-05-31" in out
