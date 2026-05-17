"""Tests for `halal_trader.web.marketplace` (Wave 7.E).

Covers: listing validation gate (PII, pricing, status), publish
flow, subscription lifecycle (trial→active, pause/resume/cancel),
revenue-share math, render no-secret contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.web.marketplace import (
    DEFAULT_POLICY,
    HalalCertLevel,
    LicenseTerm,
    ListingStatus,
    ListingViolationError,
    MarketplaceListing,
    MarketplacePolicy,
    RevenueSplit,
    Subscription,
    SubscriptionStatus,
    cancel_subscription,
    compute_split,
    convert_to_active,
    pause_subscription,
    publish_listing,
    render_listing,
    render_split,
    render_subscription,
    resume_subscription,
    start_subscription,
    take_down_listing,
    validate_listing,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_license_term_string_values_pinned() -> None:
    assert LicenseTerm.PERSONAL_USE.value == "personal_use"
    assert LicenseTerm.COMMERCIAL_USE.value == "commercial_use"
    assert LicenseTerm.NON_COMMERCIAL_USE.value == "non_commercial_use"
    assert LicenseTerm.RESEARCH_ONLY.value == "research_only"


def test_listing_status_string_values_pinned() -> None:
    assert ListingStatus.DRAFT.value == "draft"
    assert ListingStatus.PUBLISHED.value == "published"
    assert ListingStatus.UNLISTED.value == "unlisted"
    assert ListingStatus.TAKEN_DOWN.value == "taken_down"


def test_subscription_status_string_values_pinned() -> None:
    assert SubscriptionStatus.TRIAL.value == "trial"
    assert SubscriptionStatus.ACTIVE.value == "active"
    assert SubscriptionStatus.PAUSED.value == "paused"
    assert SubscriptionStatus.CANCELLED.value == "cancelled"


def test_halal_cert_level_string_values_pinned() -> None:
    assert HalalCertLevel.BASIC.value == "basic"
    assert HalalCertLevel.MODERATE.value == "moderate"
    assert HalalCertLevel.STRICT.value == "strict"
    assert HalalCertLevel.SCHOLAR_REVIEWED.value == "scholar_reviewed"


# --------------------------- MarketplacePolicy -------------------------------


def test_default_policy_values() -> None:
    assert DEFAULT_POLICY.default_trial_days == 7
    assert DEFAULT_POLICY.author_share == 0.90
    assert DEFAULT_POLICY.platform_share == pytest.approx(0.10)
    assert DEFAULT_POLICY.price_floor_usd == 1.0
    assert DEFAULT_POLICY.price_ceiling_usd == 999.0


def test_policy_rejects_trial_below_min() -> None:
    with pytest.raises(ValueError, match="default_trial_days"):
        MarketplacePolicy(default_trial_days=0)


def test_policy_rejects_trial_above_max() -> None:
    with pytest.raises(ValueError, match="default_trial_days"):
        MarketplacePolicy(default_trial_days=31)


def test_policy_accepts_trial_at_lower_boundary() -> None:
    p = MarketplacePolicy(default_trial_days=1)
    assert p.default_trial_days == 1


def test_policy_accepts_trial_at_upper_boundary() -> None:
    p = MarketplacePolicy(default_trial_days=30)
    assert p.default_trial_days == 30


def test_policy_rejects_author_share_at_zero() -> None:
    with pytest.raises(ValueError, match="author_share"):
        MarketplacePolicy(author_share=0.0)


def test_policy_rejects_author_share_at_one() -> None:
    """Pin: author_share must be < 1.0 — platform must take some cut."""

    with pytest.raises(ValueError, match="author_share"):
        MarketplacePolicy(author_share=1.0)


def test_policy_rejects_zero_price_floor() -> None:
    with pytest.raises(ValueError, match="price_floor_usd"):
        MarketplacePolicy(price_floor_usd=0.0)


def test_policy_rejects_ceiling_below_floor() -> None:
    with pytest.raises(ValueError, match="price_ceiling_usd"):
        MarketplacePolicy(price_floor_usd=10.0, price_ceiling_usd=5.0)


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.author_share = 0.5  # type: ignore[misc]


# --------------------------- MarketplaceListing ------------------------------


def _listing(**overrides: object) -> MarketplaceListing:
    base: dict[str, object] = {
        "listing_id": "lst_1",
        "author_anonymous_handle": "strategist_abc12345",
        "name": "Halal Momentum Strategy",
        "description": "A momentum-following strategy with 5% NPI screening",
        "strategy_kind": "momentum",
        "halal_cert_level": HalalCertLevel.MODERATE,
        "license_term": LicenseTerm.COMMERCIAL_USE,
        "monthly_price_usd": 19.99,
        "status": ListingStatus.DRAFT,
    }
    base.update(overrides)
    return MarketplaceListing(**base)  # type: ignore[arg-type]


def test_listing_rejects_empty_listing_id() -> None:
    with pytest.raises(ValueError, match="listing_id"):
        _listing(listing_id="")


def test_listing_rejects_empty_author() -> None:
    with pytest.raises(ValueError, match="author"):
        _listing(author_anonymous_handle="")


def test_listing_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        _listing(name="")


def test_listing_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="description"):
        _listing(description="")


def test_listing_rejects_zero_price() -> None:
    with pytest.raises(ValueError, match="monthly_price_usd"):
        _listing(monthly_price_usd=0)


def test_listing_rejects_negative_price() -> None:
    with pytest.raises(ValueError, match="monthly_price_usd"):
        _listing(monthly_price_usd=-1.0)


def test_listing_rejects_naive_published_at() -> None:
    with pytest.raises(ValueError, match="published_at"):
        _listing(published_at=datetime(2026, 5, 1))


def test_listing_is_frozen() -> None:
    listing = _listing()
    with pytest.raises(FrozenInstanceError):
        listing.monthly_price_usd = 99.99  # type: ignore[misc]


# --------------------------- validate_listing --------------------------------


def test_validate_listing_clean_passes() -> None:
    validate_listing(_listing())


def test_validate_rejects_price_below_floor() -> None:
    with pytest.raises(ListingViolationError, match="below"):
        validate_listing(_listing(monthly_price_usd=0.50))


def test_validate_accepts_price_at_floor() -> None:
    """Pin: $1.00 inclusive."""

    validate_listing(_listing(monthly_price_usd=1.00))


def test_validate_rejects_price_above_ceiling() -> None:
    with pytest.raises(ListingViolationError, match="above"):
        validate_listing(_listing(monthly_price_usd=1500.0))


def test_validate_accepts_price_at_ceiling() -> None:
    """Pin: $999 inclusive."""

    validate_listing(_listing(monthly_price_usd=999.0))


def test_validate_rejects_email_in_description() -> None:
    with pytest.raises(ListingViolationError, match="PII"):
        validate_listing(_listing(description="Contact me at trader@example.com"))


def test_validate_rejects_phone_in_description() -> None:
    with pytest.raises(ListingViolationError, match="PII"):
        validate_listing(_listing(description="Call me at +1-555-123-4567 for help"))


def test_validate_rejects_ssn_in_description() -> None:
    with pytest.raises(ListingViolationError, match="PII"):
        validate_listing(_listing(description="My SSN is 123-45-6789"))


def test_validate_rejects_ip_in_description() -> None:
    with pytest.raises(ListingViolationError, match="PII"):
        validate_listing(_listing(description="Server at 192.168.1.100 hosts the model"))


def test_validate_rejects_email_in_name() -> None:
    """Pin: PII check covers name as well."""

    with pytest.raises(ListingViolationError, match="PII"):
        validate_listing(_listing(name="alice@x.com momentum"))


def test_validate_rejects_published_listing() -> None:
    """Pin: only DRAFT or UNLISTED can be re-validated."""

    with pytest.raises(ListingViolationError, match="DRAFT or UNLISTED"):
        validate_listing(_listing(status=ListingStatus.PUBLISHED))


def test_validate_rejects_taken_down_listing() -> None:
    with pytest.raises(ListingViolationError, match="DRAFT or UNLISTED"):
        validate_listing(_listing(status=ListingStatus.TAKEN_DOWN))


def test_validate_accepts_unlisted_for_republishing() -> None:
    """Pin: UNLISTED is a valid input for re-validation + republish."""

    validate_listing(_listing(status=ListingStatus.UNLISTED))


def test_violation_carries_listing_id_and_reason() -> None:
    try:
        validate_listing(_listing(monthly_price_usd=2000.0))
    except ListingViolationError as e:
        assert e.listing_id == "lst_1"
        assert "above" in e.reason


# --------------------------- publish_listing ---------------------------------


def test_publish_flips_status_and_records_time() -> None:
    listing = _listing()
    published = publish_listing(listing, now=T0)
    assert published.status is ListingStatus.PUBLISHED
    assert published.published_at == T0


def test_publish_returns_new_state() -> None:
    """Pin: state is immutable."""

    listing = _listing()
    published = publish_listing(listing, now=T0)
    assert listing.status is ListingStatus.DRAFT
    assert published.status is ListingStatus.PUBLISHED


def test_publish_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        publish_listing(_listing(), now=datetime(2026, 5, 1))


def test_publish_runs_validation() -> None:
    """Pin: publish_listing applies the validation gate."""

    with pytest.raises(ListingViolationError):
        publish_listing(_listing(monthly_price_usd=2000.0), now=T0)


# --------------------------- take_down_listing -------------------------------


def test_take_down_listing() -> None:
    listing = publish_listing(_listing(), now=T0)
    taken_down = take_down_listing(listing)
    assert taken_down.status is ListingStatus.TAKEN_DOWN
    assert taken_down.published_at == T0


# --------------------------- start_subscription ------------------------------


def _published_listing() -> MarketplaceListing:
    return publish_listing(_listing(), now=T0)


def test_start_subscription_basic() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="subscriber_xyz",
        now=T0,
    )
    assert sub.status is SubscriptionStatus.TRIAL
    assert sub.trial_end_at == T0 + timedelta(days=7)


def test_start_subscription_uses_custom_trial_days() -> None:
    listing = _published_listing()
    policy = MarketplacePolicy(default_trial_days=14)
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="subscriber",
        now=T0,
        policy=policy,
    )
    assert sub.trial_end_at == T0 + timedelta(days=14)


def test_start_subscription_rejects_unpublished_listing() -> None:
    listing = _listing()  # DRAFT
    with pytest.raises(ValueError, match="status"):
        start_subscription(
            subscription_id="sub_1",
            listing=listing,
            subscriber_anonymous_handle="s",
            now=T0,
        )


def test_start_subscription_rejects_taken_down_listing() -> None:
    listing = take_down_listing(_published_listing())
    with pytest.raises(ValueError, match="status"):
        start_subscription(
            subscription_id="sub_1",
            listing=listing,
            subscriber_anonymous_handle="s",
            now=T0,
        )


def test_start_subscription_rejects_naive_now() -> None:
    listing = _published_listing()
    with pytest.raises(ValueError, match="now"):
        start_subscription(
            subscription_id="sub_1",
            listing=listing,
            subscriber_anonymous_handle="s",
            now=datetime(2026, 5, 1),
        )


# --------------------------- Subscription validation -------------------------


def test_subscription_rejects_empty_subscription_id() -> None:
    with pytest.raises(ValueError, match="subscription_id"):
        Subscription(
            subscription_id="",
            listing_id="lst",
            subscriber_anonymous_handle="s",
            status=SubscriptionStatus.TRIAL,
            started_at=T0,
            trial_end_at=T0 + timedelta(days=7),
        )


def test_subscription_rejects_naive_started_at() -> None:
    with pytest.raises(ValueError, match="started_at"):
        Subscription(
            subscription_id="s1",
            listing_id="lst",
            subscriber_anonymous_handle="s",
            status=SubscriptionStatus.TRIAL,
            started_at=datetime(2026, 5, 1),
            trial_end_at=T0 + timedelta(days=7),
        )


def test_subscription_rejects_trial_end_before_start() -> None:
    with pytest.raises(ValueError, match="trial_end_at"):
        Subscription(
            subscription_id="s1",
            listing_id="lst",
            subscriber_anonymous_handle="s",
            status=SubscriptionStatus.TRIAL,
            started_at=T0,
            trial_end_at=T0 - timedelta(days=1),
        )


def test_subscription_is_frozen() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="s",
        now=T0,
    )
    with pytest.raises(FrozenInstanceError):
        sub.status = SubscriptionStatus.CANCELLED  # type: ignore[misc]


# --------------------------- convert_to_active -------------------------------


def test_convert_to_active_after_trial() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="s",
        now=T0,
    )
    after_trial = T0 + timedelta(days=7)
    converted = convert_to_active(sub, now=after_trial)
    assert converted.status is SubscriptionStatus.ACTIVE


def test_convert_to_active_before_trial_end_rejected() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="s",
        now=T0,
    )
    with pytest.raises(ValueError, match="trial_end_at"):
        convert_to_active(sub, now=T0 + timedelta(days=3))


def test_convert_only_from_trial() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="s",
        now=T0,
    )
    sub = convert_to_active(sub, now=T0 + timedelta(days=7))
    with pytest.raises(ValueError, match="TRIAL"):
        convert_to_active(sub, now=T0 + timedelta(days=14))


# --------------------------- pause / resume ----------------------------------


def test_pause_active() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="s",
        now=T0,
    )
    sub = convert_to_active(sub, now=T0 + timedelta(days=7))
    sub = pause_subscription(sub)
    assert sub.status is SubscriptionStatus.PAUSED


def test_pause_trial_rejected() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="s",
        now=T0,
    )
    with pytest.raises(ValueError, match="ACTIVE"):
        pause_subscription(sub)


def test_resume_paused() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="s",
        now=T0,
    )
    sub = convert_to_active(sub, now=T0 + timedelta(days=7))
    sub = pause_subscription(sub)
    sub = resume_subscription(sub)
    assert sub.status is SubscriptionStatus.ACTIVE


def test_resume_active_rejected() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="s",
        now=T0,
    )
    sub = convert_to_active(sub, now=T0 + timedelta(days=7))
    with pytest.raises(ValueError, match="PAUSED"):
        resume_subscription(sub)


# --------------------------- cancel_subscription -----------------------------


def test_cancel_from_trial() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="s",
        now=T0,
    )
    sub = cancel_subscription(sub, now=T0 + timedelta(days=2))
    assert sub.status is SubscriptionStatus.CANCELLED
    assert sub.cancelled_at == T0 + timedelta(days=2)


def test_cancel_from_active() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="s",
        now=T0,
    )
    sub = convert_to_active(sub, now=T0 + timedelta(days=7))
    sub = cancel_subscription(sub, now=T0 + timedelta(days=30))
    assert sub.status is SubscriptionStatus.CANCELLED


def test_cancel_already_cancelled_rejected() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="s",
        now=T0,
    )
    sub = cancel_subscription(sub, now=T0)
    with pytest.raises(ValueError, match="already cancelled"):
        cancel_subscription(sub, now=T0)


def test_cancel_naive_now_rejected() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="s",
        now=T0,
    )
    with pytest.raises(ValueError, match="now"):
        cancel_subscription(sub, now=datetime(2026, 5, 1))


# --------------------------- compute_split -----------------------------------


def test_compute_split_default_90_10() -> None:
    listing = _listing()
    split = compute_split(listing=listing, cycle_revenue_usd=100.0)
    assert split.author_amount_usd == 90.0
    assert split.platform_amount_usd == 10.0
    assert split.cycle_revenue_usd == 100.0


def test_compute_split_zero_revenue() -> None:
    listing = _listing()
    split = compute_split(listing=listing, cycle_revenue_usd=0.0)
    assert split.author_amount_usd == 0.0
    assert split.platform_amount_usd == 0.0


def test_compute_split_custom_share() -> None:
    listing = _listing()
    policy = MarketplacePolicy(author_share=0.70)
    split = compute_split(listing=listing, cycle_revenue_usd=100.0, policy=policy)
    assert split.author_amount_usd == 70.0
    assert split.platform_amount_usd == 30.0


def test_compute_split_rejects_negative_revenue() -> None:
    listing = _listing()
    with pytest.raises(ValueError, match="cycle_revenue_usd"):
        compute_split(listing=listing, cycle_revenue_usd=-1.0)


def test_compute_split_rounding_consistency() -> None:
    """Pin: small rounding tolerance — author + platform always sums to revenue."""

    listing = _listing()
    # $19.99 * 0.90 = 17.991 → rounds to 17.99
    split = compute_split(listing=listing, cycle_revenue_usd=19.99)
    total = split.author_amount_usd + split.platform_amount_usd
    assert abs(total - 19.99) < 0.01


def test_revenue_split_rejects_negative_amounts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RevenueSplit(
            listing_id="x",
            cycle_revenue_usd=10.0,
            author_amount_usd=-1.0,
            platform_amount_usd=11.0,
        )


def test_revenue_split_rejects_inconsistent_sum() -> None:
    """Pin: amounts must sum to revenue (within $0.01 rounding)."""

    with pytest.raises(ValueError, match="!="):
        RevenueSplit(
            listing_id="x",
            cycle_revenue_usd=100.0,
            author_amount_usd=50.0,
            platform_amount_usd=30.0,
        )


# --------------------------- render ------------------------------------------


def test_render_listing_includes_name_and_price() -> None:
    listing = _listing(name="My Strategy", monthly_price_usd=29.99)
    out = render_listing(listing)
    assert "My Strategy" in out
    assert "$29.99" in out


def test_render_listing_no_secret_leak() -> None:
    """Pin: render never includes Stripe IDs / card data / payout accounts."""

    listing = _listing()
    out = render_listing(listing)
    assert "cus_" not in out.lower()
    assert "in_" not in out.lower().replace("includes", "").replace("license", "").replace(
        "listing", ""
    )
    assert "payout" not in out.lower()
    assert "card" not in out.lower()
    assert "ach" not in out.lower()


def test_render_subscription_includes_id() -> None:
    listing = _published_listing()
    sub = start_subscription(
        subscription_id="sub_alice",
        listing=listing,
        subscriber_anonymous_handle="anon123",
        now=T0,
    )
    out = render_subscription(sub)
    assert "sub_alice" in out
    assert "anon123" in out
    assert "trial" in out.lower()


def test_render_split_includes_amounts() -> None:
    listing = _listing()
    split = compute_split(listing=listing, cycle_revenue_usd=100.0)
    out = render_split(split)
    assert "$100.00" in out
    assert "$90.00" in out
    assert "$10.00" in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_publish_subscribe_convert_cancel() -> None:
    """Real-world: author publishes, subscriber starts trial, converts,
    cancels three months in."""

    listing = _listing()
    listing = publish_listing(listing, now=T0)
    sub = start_subscription(
        subscription_id="sub_1",
        listing=listing,
        subscriber_anonymous_handle="anon_buyer",
        now=T0,
    )
    sub = convert_to_active(sub, now=T0 + timedelta(days=7))
    sub = cancel_subscription(sub, now=T0 + timedelta(days=90))
    assert sub.status is SubscriptionStatus.CANCELLED
    # Compute revenue split after 3 months at $19.99/mo
    revenue = 19.99 * 3
    split = compute_split(listing=listing, cycle_revenue_usd=revenue)
    # 90/10 split
    assert abs(split.author_amount_usd - 19.99 * 3 * 0.9) < 0.05


def test_e2e_replay_consistency() -> None:
    """Pin: applying same operations produces equal states."""

    def build() -> Subscription:
        listing = publish_listing(_listing(), now=T0)
        sub = start_subscription(
            subscription_id="s1",
            listing=listing,
            subscriber_anonymous_handle="a",
            now=T0,
        )
        sub = convert_to_active(sub, now=T0 + timedelta(days=7))
        return sub

    a = build()
    b = build()
    assert a == b
