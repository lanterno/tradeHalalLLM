"""Tests for `halal_trader.web.newsletter` (Wave 10.D).

Covers: section validation (PII + handle denylist), digest assembly,
canonical section ordering, subscription state machine, no-secret
render.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.web.newsletter import (
    AlreadyUnsubscribedError,
    Digest,
    DigestViolationError,
    Section,
    SectionKind,
    Subscription,
    SubscriptionStatus,
    active_subscribers,
    render_digest,
    render_section,
    sections_by_kind,
    subscribe,
    unsubscribe,
    validate_digest,
    validate_section,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_section_kind_string_values_pinned() -> None:
    assert SectionKind.TOP_PERFORMERS.value == "top_performers"
    assert SectionKind.REGULATORY.value == "regulatory"
    assert SectionKind.SCHOLAR_UPDATES.value == "scholar_updates"
    assert SectionKind.WHAT_DIDNT_WORK.value == "what_didnt_work"


def test_subscription_status_string_values_pinned() -> None:
    assert SubscriptionStatus.SUBSCRIBED.value == "subscribed"
    assert SubscriptionStatus.UNSUBSCRIBED.value == "unsubscribed"


# --------------------------- Section validation ------------------------------


def _section(**overrides: object) -> Section:
    base: dict[str, object] = {
        "section_id": "s1",
        "kind": SectionKind.TOP_PERFORMERS,
        "title": "Top strategies this quarter",
        "body": "The momentum + halal-screen cohort hit 1.5 Sharpe on average.",
    }
    base.update(overrides)
    return Section(**base)  # type: ignore[arg-type]


def test_section_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="section_id"):
        _section(section_id="")


def test_section_rejects_empty_title() -> None:
    with pytest.raises(ValueError, match="title"):
        _section(title="")


def test_section_rejects_empty_body() -> None:
    with pytest.raises(ValueError, match="body"):
        _section(body="")


def test_section_rejects_oversized_title() -> None:
    with pytest.raises(ValueError, match="title too long"):
        _section(title="x" * 121)


def test_section_rejects_oversized_body() -> None:
    with pytest.raises(ValueError, match="body too long"):
        _section(body="x" * 4001)


def test_section_is_frozen() -> None:
    s = _section()
    with pytest.raises(FrozenInstanceError):
        s.title = "other"  # type: ignore[misc]


# --------------------------- validate_section --------------------------------


def test_validate_clean_section_passes() -> None:
    validate_section(_section())


def test_validate_rejects_email_in_body() -> None:
    with pytest.raises(DigestViolationError, match="PII"):
        validate_section(_section(body="Contact us at hello@example.com"))


def test_validate_rejects_email_in_title() -> None:
    with pytest.raises(DigestViolationError, match="PII"):
        validate_section(_section(title="Mail trader@x.com"))


def test_validate_rejects_ssn() -> None:
    with pytest.raises(DigestViolationError, match="PII"):
        validate_section(_section(body="A user accidentally pasted 123-45-6789"))


def test_validate_rejects_phone() -> None:
    with pytest.raises(DigestViolationError, match="PII"):
        validate_section(_section(body="Call +1-555-123-4567"))


def test_validate_rejects_ip() -> None:
    with pytest.raises(DigestViolationError, match="PII"):
        validate_section(_section(body="Server at 192.168.1.1"))


def test_validate_rejects_api_key() -> None:
    """Pin: long alphanumeric blob caught."""

    fake = "BinanceA" + "Bc" * 25
    with pytest.raises(DigestViolationError, match="PII"):
        validate_section(_section(body=f"key: {fake}"))


def test_validate_rejects_handle_in_body() -> None:
    """Pin: @username (Twitter / Discord handle) rejected."""

    with pytest.raises(DigestViolationError, match="handle"):
        validate_section(_section(body="Shoutout to @alice for the strategy"))


def test_validate_rejects_handle_in_title() -> None:
    with pytest.raises(DigestViolationError, match="handle"):
        validate_section(_section(title="Strategy by @alice"))


def test_violation_carries_section_id() -> None:
    try:
        validate_section(_section(body="email: a@b.com"))
    except DigestViolationError as e:
        assert e.section_id == "s1"
        assert "PII" in e.reason


def test_validate_clean_strategy_discussion() -> None:
    """Pin: a clean section that talks about strategies + Sharpe ratios
    without naming individuals or PII passes."""

    section = _section(
        body=(
            "The momentum + halal-screen cohort hit 1.5 Sharpe on average; "
            "the mean-reversion cohort lagged at 0.8 Sharpe. Drawdown was "
            "well-contained across the universe."
        ),
    )
    validate_section(section)  # no raise


# --------------------------- Digest validation -------------------------------


def _digest(**overrides: object) -> Digest:
    base: dict[str, object] = {
        "digest_id": "q1_2026",
        "quarter_label": "2026-Q1",
        "published_at": T0,
        "sections": (
            _section(section_id="s_top", kind=SectionKind.TOP_PERFORMERS),
            _section(
                section_id="s_reg",
                kind=SectionKind.REGULATORY,
                title="SEC RIA registration timeline",
                body="The platform's SEC RIA registration is on track for Q3.",
            ),
        ),
    }
    base.update(overrides)
    return Digest(**base)  # type: ignore[arg-type]


def test_digest_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="digest_id"):
        _digest(digest_id="")


def test_digest_rejects_empty_quarter_label() -> None:
    with pytest.raises(ValueError, match="quarter_label"):
        _digest(quarter_label="")


def test_digest_rejects_naive_published_at() -> None:
    with pytest.raises(ValueError, match="published_at"):
        _digest(published_at=datetime(2026, 5, 1))


def test_digest_rejects_empty_sections() -> None:
    with pytest.raises(ValueError, match="sections"):
        _digest(sections=())


def test_digest_rejects_duplicate_section_ids() -> None:
    sections = (
        _section(section_id="s1"),
        _section(section_id="s1"),  # duplicate
    )
    with pytest.raises(ValueError, match="duplicate"):
        _digest(sections=sections)


def test_digest_is_frozen() -> None:
    d = _digest()
    with pytest.raises(FrozenInstanceError):
        d.quarter_label = "2026-Q2"  # type: ignore[misc]


# --------------------------- validate_digest ---------------------------------


def test_validate_clean_digest_passes() -> None:
    validate_digest(_digest())


def test_validate_digest_rejects_pii_in_any_section() -> None:
    sections = (
        _section(section_id="s_clean"),
        _section(
            section_id="s_pii",
            body="Contact: alice@example.com",
        ),
    )
    digest = _digest(sections=sections)
    with pytest.raises(DigestViolationError):
        validate_digest(digest)


# --------------------------- sections_by_kind --------------------------------


def test_sections_by_kind_filters() -> None:
    sections = (
        _section(section_id="s1", kind=SectionKind.TOP_PERFORMERS),
        _section(section_id="s2", kind=SectionKind.REGULATORY),
        _section(section_id="s3", kind=SectionKind.TOP_PERFORMERS),
    )
    digest = _digest(sections=sections)
    top = sections_by_kind(digest, SectionKind.TOP_PERFORMERS)
    assert len(top) == 2
    assert all(s.kind is SectionKind.TOP_PERFORMERS for s in top)


def test_sections_by_kind_empty_when_kind_absent() -> None:
    digest = _digest()
    scholar_sections = sections_by_kind(digest, SectionKind.SCHOLAR_UPDATES)
    assert scholar_sections == ()


# --------------------------- Subscription validation -------------------------


def test_subscription_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="subscription_id"):
        Subscription(
            subscription_id="",
            subscriber_anonymous_handle="anon",
            status=SubscriptionStatus.SUBSCRIBED,
            subscribed_at=T0,
        )


def test_subscription_rejects_empty_handle() -> None:
    with pytest.raises(ValueError, match="anonymous_handle"):
        Subscription(
            subscription_id="s1",
            subscriber_anonymous_handle="",
            status=SubscriptionStatus.SUBSCRIBED,
            subscribed_at=T0,
        )


def test_subscription_rejects_naive_subscribed_at() -> None:
    with pytest.raises(ValueError, match="subscribed_at"):
        Subscription(
            subscription_id="s1",
            subscriber_anonymous_handle="anon",
            status=SubscriptionStatus.SUBSCRIBED,
            subscribed_at=datetime(2026, 5, 1),
        )


def test_subscription_unsubscribed_requires_timestamp() -> None:
    """Pin: UNSUBSCRIBED status requires unsubscribed_at."""

    with pytest.raises(ValueError, match="unsubscribed_at"):
        Subscription(
            subscription_id="s1",
            subscriber_anonymous_handle="anon",
            status=SubscriptionStatus.UNSUBSCRIBED,
            subscribed_at=T0,
            unsubscribed_at=None,
        )


def test_subscription_subscribed_must_not_have_timestamp() -> None:
    """Pin: SUBSCRIBED can't carry an unsubscribed_at."""

    with pytest.raises(ValueError, match="unsubscribed_at"):
        Subscription(
            subscription_id="s1",
            subscriber_anonymous_handle="anon",
            status=SubscriptionStatus.SUBSCRIBED,
            subscribed_at=T0,
            unsubscribed_at=T0,
        )


def test_subscription_rejects_unsubscribed_before_subscribed() -> None:
    with pytest.raises(ValueError, match="unsubscribed_at"):
        Subscription(
            subscription_id="s1",
            subscriber_anonymous_handle="anon",
            status=SubscriptionStatus.UNSUBSCRIBED,
            subscribed_at=T0,
            unsubscribed_at=T0 - timedelta(days=1),
        )


def test_subscription_is_frozen() -> None:
    s = subscribe(
        subscription_id="s1",
        subscriber_anonymous_handle="anon",
        now=T0,
    )
    with pytest.raises(FrozenInstanceError):
        s.status = SubscriptionStatus.UNSUBSCRIBED  # type: ignore[misc]


# --------------------------- subscribe + unsubscribe -------------------------


def test_subscribe_basic() -> None:
    s = subscribe(
        subscription_id="s1",
        subscriber_anonymous_handle="anon123",
        now=T0,
    )
    assert s.status is SubscriptionStatus.SUBSCRIBED
    assert s.unsubscribed_at is None


def test_subscribe_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="subscription_id"):
        subscribe(subscription_id="", subscriber_anonymous_handle="x", now=T0)


def test_subscribe_rejects_empty_handle() -> None:
    with pytest.raises(ValueError, match="anonymous_handle"):
        subscribe(subscription_id="s1", subscriber_anonymous_handle="", now=T0)


def test_subscribe_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        subscribe(
            subscription_id="s1",
            subscriber_anonymous_handle="x",
            now=datetime(2026, 5, 1),
        )


def test_unsubscribe_records_timestamp() -> None:
    s = subscribe(
        subscription_id="s1",
        subscriber_anonymous_handle="anon",
        now=T0,
    )
    later = T0 + timedelta(days=30)
    u = unsubscribe(s, now=later)
    assert u.status is SubscriptionStatus.UNSUBSCRIBED
    assert u.unsubscribed_at == later


def test_unsubscribe_already_unsubscribed_raises() -> None:
    """Pin: cannot double-unsubscribe."""

    s = subscribe(
        subscription_id="s1",
        subscriber_anonymous_handle="anon",
        now=T0,
    )
    u = unsubscribe(s, now=T0 + timedelta(days=1))
    with pytest.raises(AlreadyUnsubscribedError) as exc_info:
        unsubscribe(u, now=T0 + timedelta(days=2))
    assert exc_info.value.subscription_id == "s1"


def test_unsubscribe_returns_new_state() -> None:
    """Pin: subscriptions are immutable; unsubscribe returns new record."""

    s = subscribe(
        subscription_id="s1",
        subscriber_anonymous_handle="anon",
        now=T0,
    )
    u = unsubscribe(s, now=T0 + timedelta(days=1))
    assert s.status is SubscriptionStatus.SUBSCRIBED
    assert u.status is SubscriptionStatus.UNSUBSCRIBED


def test_unsubscribe_rejects_naive_now() -> None:
    s = subscribe(
        subscription_id="s1",
        subscriber_anonymous_handle="anon",
        now=T0,
    )
    with pytest.raises(ValueError, match="now"):
        unsubscribe(s, now=datetime(2026, 5, 1))


# --------------------------- active_subscribers ------------------------------


def test_active_subscribers_filters_unsubscribed() -> None:
    s1 = subscribe(
        subscription_id="s1",
        subscriber_anonymous_handle="a1",
        now=T0,
    )
    s2 = subscribe(
        subscription_id="s2",
        subscriber_anonymous_handle="a2",
        now=T0,
    )
    s2 = unsubscribe(s2, now=T0 + timedelta(days=10))
    active = active_subscribers([s1, s2])
    ids = {s.subscription_id for s in active}
    assert ids == {"s1"}


def test_active_subscribers_empty_when_all_unsubscribed() -> None:
    s = subscribe(
        subscription_id="s1",
        subscriber_anonymous_handle="anon",
        now=T0,
    )
    s = unsubscribe(s, now=T0 + timedelta(days=1))
    assert active_subscribers([s]) == ()


# --------------------------- render_section ----------------------------------


def test_render_section_includes_kind_heading() -> None:
    s = _section(kind=SectionKind.TOP_PERFORMERS)
    out = render_section(s)
    assert "🏆" in out
    assert "Top Performers" in out


def test_render_section_includes_title_and_body() -> None:
    s = _section(title="Strong quarter", body="Sharpe averaged 1.5 across all halal cohorts")
    out = render_section(s)
    assert "Strong quarter" in out
    assert "1.5" in out


def test_render_section_emoji_per_kind() -> None:
    """Pin: each kind has a distinct emoji."""

    top = render_section(_section(kind=SectionKind.TOP_PERFORMERS))
    reg = render_section(
        _section(
            kind=SectionKind.REGULATORY,
            title="x",
            body="y",
            section_id="s_reg",
        )
    )
    sch = render_section(
        _section(
            kind=SectionKind.SCHOLAR_UPDATES,
            title="x",
            body="y",
            section_id="s_sch",
        )
    )
    nope = render_section(
        _section(
            kind=SectionKind.WHAT_DIDNT_WORK,
            title="x",
            body="y",
            section_id="s_no",
        )
    )
    assert "🏆" in top
    assert "⚖️" in reg
    assert "📚" in sch
    assert "🔍" in nope


# --------------------------- render_digest -----------------------------------


def test_render_digest_includes_quarter_label() -> None:
    d = _digest(quarter_label="2026-Q1")
    out = render_digest(d)
    assert "2026-Q1" in out


def test_render_digest_includes_published_date() -> None:
    d = _digest()
    out = render_digest(d)
    assert "2026-05-01" in out


def test_render_digest_canonical_section_order() -> None:
    """Pin: sections render in TOP → REG → SCHOLAR → WHAT_DIDNT order
    regardless of input order."""

    sections = (
        _section(section_id="s_no", kind=SectionKind.WHAT_DIDNT_WORK),
        _section(
            section_id="s_top",
            kind=SectionKind.TOP_PERFORMERS,
            title="Top",
            body="Top body",
        ),
        _section(
            section_id="s_reg",
            kind=SectionKind.REGULATORY,
            title="Reg",
            body="Reg body",
        ),
        _section(
            section_id="s_sch",
            kind=SectionKind.SCHOLAR_UPDATES,
            title="Sch",
            body="Sch body",
        ),
    )
    digest = _digest(sections=sections)
    out = render_digest(digest)

    # Find the index of each kind heading in the output
    top_idx = out.index("🏆")
    reg_idx = out.index("⚖️")
    sch_idx = out.index("📚")
    nope_idx = out.index("🔍")

    assert top_idx < reg_idx < sch_idx < nope_idx


def test_render_digest_no_secret_leak_structural() -> None:
    """Pin: render output never includes email / API key / phone.

    Validation rejects PII at construction so a contributor that
    tried to slip an email into a section would have failed; this
    test asserts the rendered output is clean even for the canonical
    digest.
    """

    out = render_digest(_digest())
    import re

    assert not re.search(r"\w+@\w+\.\w+", out)
    assert "@" not in out  # no handles either
    # Don't fire on long-but-clean strings
    assert "subscriber_email" not in out.lower()
    assert "send_api_key" not in out.lower()


# --------------------------- e2e flows ---------------------------------------


def test_e2e_full_quarterly_digest() -> None:
    """Real-world: build a Q1 digest with all four section kinds."""

    sections = (
        _section(
            section_id="top_q1",
            kind=SectionKind.TOP_PERFORMERS,
            title="Q1 top performers",
            body=(
                "The momentum-screen cohort averaged 1.8 Sharpe; "
                "mean-reversion lagged at 0.6 Sharpe."
            ),
        ),
        _section(
            section_id="reg_q1",
            kind=SectionKind.REGULATORY,
            title="SEC RIA progress",
            body=("The platform's SEC RIA registration cleared Phase 1 review in February."),
        ),
        _section(
            section_id="sch_q1",
            kind=SectionKind.SCHOLAR_UPDATES,
            title="AAOIFI Standard 21 reaffirmed",
            body=(
                "The Shariah Supervisory Board reaffirmed AAOIFI Standard "
                "21 on REIT screening. The 5% NPI threshold remains in effect."
            ),
        ),
        _section(
            section_id="no_q1",
            kind=SectionKind.WHAT_DIDNT_WORK,
            title="High-frequency crypto strategies",
            body=(
                "HFT-style crypto strategies underperformed the simpler "
                "DCA + halal-screen approach by 3.2% on average."
            ),
        ),
    )
    digest = Digest(
        digest_id="q1_2026",
        quarter_label="2026-Q1",
        published_at=T0,
        sections=sections,
    )
    validate_digest(digest)
    out = render_digest(digest)
    assert "2026-Q1" in out
    assert "1.8 Sharpe" in out
    assert "AAOIFI" in out


def test_e2e_subscription_lifecycle() -> None:
    """Real-world: user subscribes Q1, unsubscribes Q3."""

    s = subscribe(
        subscription_id="sub_alice",
        subscriber_anonymous_handle="anon_alice",
        now=T0,
    )
    assert s.status is SubscriptionStatus.SUBSCRIBED

    # Q3: user unsubscribes
    s = unsubscribe(s, now=T0 + timedelta(days=180))
    assert s.status is SubscriptionStatus.UNSUBSCRIBED

    # Cannot unsubscribe twice
    with pytest.raises(AlreadyUnsubscribedError):
        unsubscribe(s, now=T0 + timedelta(days=181))


def test_e2e_pii_caught_before_send() -> None:
    """Pin: the load-bearing pre-send pin — a contributor that
    accidentally pasted a real email into a section gets caught
    before the digest reaches 1000 subscribers."""

    sections = (
        _section(
            section_id="bad",
            body="Reach out to alice.smith@halal-trader.dev for questions",
        ),
    )
    digest = Digest(
        digest_id="bad_q1",
        quarter_label="2026-Q1",
        published_at=T0,
        sections=sections,
    )
    with pytest.raises(DigestViolationError, match="PII"):
        validate_digest(digest)


def test_e2e_replay_consistency() -> None:
    def build() -> Subscription:
        return subscribe(
            subscription_id="s1",
            subscriber_anonymous_handle="anon",
            now=T0,
        )

    a = build()
    b = build()
    assert a == b
