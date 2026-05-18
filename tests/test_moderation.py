"""Tests for `halal_trader.web.moderation` (Wave 10.C).

Covers: content classification priority order, PII auto-removal,
moderator state transitions, no-secret render contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.web.moderation import (
    DEFAULT_POLICY,
    ClassificationResult,
    ContentClassification,
    MessageReview,
    ModerationPolicy,
    ReviewStatus,
    ReviewTransitionError,
    auto_decide,
    classify,
    initial_review,
    is_visible_to_channel,
    moderator_approve,
    moderator_remove,
    render_classification,
    render_review,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_content_classification_string_values_pinned() -> None:
    assert ContentClassification.CLEAN.value == "clean"
    assert ContentClassification.SPAM.value == "spam"
    assert ContentClassification.HARASSMENT.value == "harassment"
    assert ContentClassification.FINANCIAL_ADVICE.value == "financial_advice"
    assert ContentClassification.PII_LEAK.value == "pii_leak"


def test_review_status_string_values_pinned() -> None:
    assert ReviewStatus.PENDING.value == "pending"
    assert ReviewStatus.AUTO_APPROVED.value == "auto_approved"
    assert ReviewStatus.FLAGGED.value == "flagged"
    assert ReviewStatus.ESCALATED.value == "escalated"
    assert ReviewStatus.REMOVED.value == "removed"


# --------------------------- ModerationPolicy --------------------------------


def test_default_policy() -> None:
    assert DEFAULT_POLICY.spam_threshold == 3
    assert DEFAULT_POLICY.spam_window == timedelta(seconds=60)
    assert DEFAULT_POLICY.detect_pii is True


def test_policy_rejects_spam_threshold_below_2() -> None:
    with pytest.raises(ValueError, match="spam_threshold"):
        ModerationPolicy(spam_threshold=1)


def test_policy_rejects_zero_spam_window() -> None:
    with pytest.raises(ValueError, match="spam_window"):
        ModerationPolicy(spam_window=timedelta(0))


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.spam_threshold = 10  # type: ignore[misc]


# --------------------------- ClassificationResult ----------------------------


def test_classification_result_rejects_severity_above_one() -> None:
    with pytest.raises(ValueError, match="severity_score"):
        ClassificationResult(
            classification=ContentClassification.CLEAN,
            flagged_phrases=(),
            severity_score=1.01,
        )


def test_classification_result_rejects_negative_severity() -> None:
    with pytest.raises(ValueError, match="severity_score"):
        ClassificationResult(
            classification=ContentClassification.CLEAN,
            flagged_phrases=(),
            severity_score=-0.01,
        )


def test_classification_result_is_frozen() -> None:
    r = ClassificationResult(
        classification=ContentClassification.CLEAN,
        flagged_phrases=(),
        severity_score=0.0,
    )
    with pytest.raises(FrozenInstanceError):
        r.severity_score = 0.5  # type: ignore[misc]


# --------------------------- classify: CLEAN ---------------------------------


def test_classify_empty_string_is_clean() -> None:
    result = classify("")
    assert result.classification is ContentClassification.CLEAN
    assert result.severity_score == 0.0


def test_classify_normal_message_is_clean() -> None:
    result = classify("Hey everyone, how's the BTCUSDT setup looking?")
    assert result.classification is ContentClassification.CLEAN


def test_classify_strategy_discussion_is_clean() -> None:
    result = classify("My RSI strategy hit 1.5 Sharpe last quarter")
    assert result.classification is ContentClassification.CLEAN


# --------------------------- classify: PII (highest priority) ----------------


def test_classify_email_is_pii() -> None:
    result = classify("Contact me at trader@example.com")
    assert result.classification is ContentClassification.PII_LEAK
    assert "email" in result.flagged_phrases
    assert result.severity_score == 1.0


def test_classify_ssn_is_pii() -> None:
    result = classify("My SSN is 123-45-6789 — please help")
    assert result.classification is ContentClassification.PII_LEAK
    assert "ssn" in result.flagged_phrases


def test_classify_ip_is_pii() -> None:
    result = classify("My server is at 192.168.1.100 fyi")
    assert result.classification is ContentClassification.PII_LEAK
    assert "ip" in result.flagged_phrases


def test_classify_phone_is_pii() -> None:
    result = classify("Call me at +1-555-123-4567 if anyone is stuck")
    assert result.classification is ContentClassification.PII_LEAK
    assert "phone" in result.flagged_phrases


def test_classify_api_key_is_pii() -> None:
    """Pin: long alphanumeric string treated as API key shape."""

    fake_key = "BinanceA" + "Bc" * 25  # 58 chars
    result = classify(f"Here's my key: {fake_key}")
    assert result.classification is ContentClassification.PII_LEAK
    assert "api_key" in result.flagged_phrases


def test_classify_pii_takes_priority_over_harassment() -> None:
    """Pin: PII detection short-circuits — it's the load-bearing block."""

    result = classify("you are an idiot, my email is trader@example.com")
    # PII wins because it's the maximally destructive outcome
    assert result.classification is ContentClassification.PII_LEAK


def test_classify_pii_takes_priority_over_financial_advice() -> None:
    result = classify("you should buy AAPL — DM me at me@example.com")
    assert result.classification is ContentClassification.PII_LEAK


def test_classify_pii_disabled_via_policy() -> None:
    """Pin: detect_pii=False allows other classifications to run."""

    policy = ModerationPolicy(detect_pii=False)
    result = classify(
        "you should buy AAPL — DM me at me@example.com",
        policy=policy,
    )
    # PII detection off → falls through to financial advice
    assert result.classification is ContentClassification.FINANCIAL_ADVICE


# --------------------------- classify: HARASSMENT ----------------------------


def test_classify_harassment_basic() -> None:
    result = classify("you are an idiot")
    assert result.classification is ContentClassification.HARASSMENT
    assert result.severity_score == 0.9


def test_classify_harassment_kill_yourself() -> None:
    result = classify("just kill yourself already")
    assert result.classification is ContentClassification.HARASSMENT


def test_classify_harassment_case_insensitive() -> None:
    result = classify("YOU'RE STUPID")
    assert result.classification is ContentClassification.HARASSMENT


def test_classify_harassment_takes_priority_over_financial_advice() -> None:
    """Pin: harassment > financial advice (more severe)."""

    result = classify("you should buy AAPL but you are an idiot for not")
    assert result.classification is ContentClassification.HARASSMENT


# --------------------------- classify: FINANCIAL_ADVICE ----------------------


def test_classify_you_should_buy() -> None:
    result = classify("you should buy AAPL right now")
    assert result.classification is ContentClassification.FINANCIAL_ADVICE
    assert "you should buy" in result.flagged_phrases
    assert result.severity_score == 0.6


def test_classify_you_should_sell() -> None:
    result = classify("you should sell BTCUSDT before it dumps")
    assert result.classification is ContentClassification.FINANCIAL_ADVICE


def test_classify_guaranteed_profit() -> None:
    result = classify("This setup is guaranteed profit")
    assert result.classification is ContentClassification.FINANCIAL_ADVICE


def test_classify_risk_free() -> None:
    result = classify("Try this risk free strategy")
    assert result.classification is ContentClassification.FINANCIAL_ADVICE


def test_classify_i_recommend_buying() -> None:
    result = classify("I recommend buying TSLA tomorrow")
    assert result.classification is ContentClassification.FINANCIAL_ADVICE


def test_classify_definitely_buy() -> None:
    result = classify("definitely buy NVDA before earnings")
    assert result.classification is ContentClassification.FINANCIAL_ADVICE


# --------------------------- classify: SPAM ----------------------------------


def test_classify_recent_count_below_threshold_is_clean() -> None:
    """Pin: 2 recent identical (below default 3) is still CLEAN."""

    result = classify("hello", recent_identical_count=2)
    assert result.classification is ContentClassification.CLEAN


def test_classify_recent_count_at_threshold_is_spam() -> None:
    """Pin: 3 recent identical hits the inclusive boundary → SPAM."""

    result = classify("hello", recent_identical_count=3)
    assert result.classification is ContentClassification.SPAM


def test_classify_high_recent_count_is_spam() -> None:
    result = classify("free crypto giveaway click now", recent_identical_count=10)
    assert result.classification is ContentClassification.SPAM


def test_classify_spam_priority_below_financial_advice() -> None:
    """Pin: financial advice + high spam count → FINANCIAL_ADVICE wins.

    A spammed financial-advice message is more important to surface
    than the spam aspect.
    """

    result = classify("you should buy SHIB", recent_identical_count=5)
    assert result.classification is ContentClassification.FINANCIAL_ADVICE


def test_classify_custom_spam_threshold() -> None:
    """Strict policy: 2-message threshold."""

    strict = ModerationPolicy(spam_threshold=2)
    result = classify("hello", recent_identical_count=2, policy=strict)
    assert result.classification is ContentClassification.SPAM


# --------------------------- MessageReview validation ------------------------


def test_review_rejects_empty_message_id() -> None:
    with pytest.raises(ValueError, match="message_id"):
        MessageReview(
            message_id="",
            classification=ContentClassification.CLEAN,
            status=ReviewStatus.PENDING,
            decided_at=T0,
        )


def test_review_rejects_naive_decided_at() -> None:
    with pytest.raises(ValueError, match="decided_at"):
        MessageReview(
            message_id="m1",
            classification=ContentClassification.CLEAN,
            status=ReviewStatus.PENDING,
            decided_at=datetime(2026, 5, 1),
        )


def test_review_escalated_requires_moderator() -> None:
    """Pin: escalated/removed status need moderator attribution."""

    with pytest.raises(ValueError, match="moderator"):
        MessageReview(
            message_id="m1",
            classification=ContentClassification.HARASSMENT,
            status=ReviewStatus.ESCALATED,
            decided_at=T0,
            moderator="",
        )


def test_review_removed_requires_moderator() -> None:
    with pytest.raises(ValueError, match="moderator"):
        MessageReview(
            message_id="m1",
            classification=ContentClassification.PII_LEAK,
            status=ReviewStatus.REMOVED,
            decided_at=T0,
            moderator="",
        )


def test_review_auto_approved_no_moderator_required() -> None:
    review = MessageReview(
        message_id="m1",
        classification=ContentClassification.CLEAN,
        status=ReviewStatus.AUTO_APPROVED,
        decided_at=T0,
    )
    assert review.moderator == ""


def test_review_is_frozen() -> None:
    review = initial_review(
        message_id="m1",
        classification=ContentClassification.CLEAN,
        now=T0,
    )
    with pytest.raises(FrozenInstanceError):
        review.status = ReviewStatus.AUTO_APPROVED  # type: ignore[misc]


# --------------------------- initial_review + auto_decide --------------------


def test_initial_review_pending() -> None:
    review = initial_review(
        message_id="m1",
        classification=ContentClassification.CLEAN,
        now=T0,
    )
    assert review.status is ReviewStatus.PENDING


def test_auto_decide_clean_auto_approved() -> None:
    review = initial_review(message_id="m1", classification=ContentClassification.CLEAN, now=T0)
    after = auto_decide(review, now=T0)
    assert after.status is ReviewStatus.AUTO_APPROVED


def test_auto_decide_pii_removed() -> None:
    """Pin: PII auto-removed without moderator review.

    The leak is the maximally destructive outcome; auto-blocking
    means the channel never sees the API key / SSN / etc.
    """

    review = initial_review(
        message_id="m1",
        classification=ContentClassification.PII_LEAK,
        now=T0,
    )
    after = auto_decide(review, now=T0)
    assert after.status is ReviewStatus.REMOVED
    assert after.moderator == "auto"


def test_auto_decide_harassment_escalated() -> None:
    """Pin: harassment escalated for human review (not auto-removed)."""

    review = initial_review(
        message_id="m1",
        classification=ContentClassification.HARASSMENT,
        now=T0,
    )
    after = auto_decide(review, now=T0)
    assert after.status is ReviewStatus.ESCALATED


def test_auto_decide_financial_advice_flagged() -> None:
    """Pin: financial advice is FLAGGED, not removed.

    Transparent moderation: the message stays visible with a
    disclaimer rather than silently deleted.
    """

    review = initial_review(
        message_id="m1",
        classification=ContentClassification.FINANCIAL_ADVICE,
        now=T0,
    )
    after = auto_decide(review, now=T0)
    assert after.status is ReviewStatus.FLAGGED


def test_auto_decide_spam_removed() -> None:
    review = initial_review(
        message_id="m1",
        classification=ContentClassification.SPAM,
        now=T0,
    )
    after = auto_decide(review, now=T0)
    assert after.status is ReviewStatus.REMOVED


def test_auto_decide_already_decided_rejected() -> None:
    """Pin: cannot auto-decide twice."""

    review = initial_review(message_id="m1", classification=ContentClassification.CLEAN, now=T0)
    review = auto_decide(review, now=T0)
    with pytest.raises(ReviewTransitionError):
        auto_decide(review, now=T0)


def test_auto_decide_naive_now_rejected() -> None:
    review = initial_review(message_id="m1", classification=ContentClassification.CLEAN, now=T0)
    with pytest.raises(ValueError, match="now"):
        auto_decide(review, now=datetime(2026, 5, 1))


# --------------------------- moderator_remove --------------------------------


def test_moderator_remove_from_flagged() -> None:
    review = initial_review(
        message_id="m1",
        classification=ContentClassification.FINANCIAL_ADVICE,
        now=T0,
    )
    review = auto_decide(review, now=T0)  # FLAGGED
    review = moderator_remove(review, moderator="alice", now=T0 + timedelta(minutes=2))
    assert review.status is ReviewStatus.REMOVED
    assert review.moderator == "alice"


def test_moderator_remove_from_escalated() -> None:
    review = initial_review(
        message_id="m1",
        classification=ContentClassification.HARASSMENT,
        now=T0,
    )
    review = auto_decide(review, now=T0)  # ESCALATED
    review = moderator_remove(review, moderator="alice", now=T0 + timedelta(minutes=2))
    assert review.status is ReviewStatus.REMOVED


def test_moderator_remove_from_pending_rejected() -> None:
    review = initial_review(message_id="m1", classification=ContentClassification.CLEAN, now=T0)
    with pytest.raises(ReviewTransitionError):
        moderator_remove(review, moderator="alice", now=T0)


def test_moderator_remove_requires_moderator_name() -> None:
    review = initial_review(
        message_id="m1",
        classification=ContentClassification.FINANCIAL_ADVICE,
        now=T0,
    )
    review = auto_decide(review, now=T0)
    with pytest.raises(ValueError, match="moderator"):
        moderator_remove(review, moderator="", now=T0)


# --------------------------- moderator_approve -------------------------------


def test_moderator_approve_from_flagged() -> None:
    """Pin: false-positive escape valve — flagged message can be
    approved on review."""

    review = initial_review(
        message_id="m1",
        classification=ContentClassification.FINANCIAL_ADVICE,
        now=T0,
    )
    review = auto_decide(review, now=T0)  # FLAGGED
    review = moderator_approve(review, moderator="alice", now=T0 + timedelta(minutes=2))
    assert review.status is ReviewStatus.AUTO_APPROVED


def test_moderator_approve_from_escalated() -> None:
    review = initial_review(
        message_id="m1",
        classification=ContentClassification.HARASSMENT,
        now=T0,
    )
    review = auto_decide(review, now=T0)  # ESCALATED
    review = moderator_approve(review, moderator="alice", now=T0 + timedelta(minutes=2))
    assert review.status is ReviewStatus.AUTO_APPROVED


def test_moderator_approve_from_pending_rejected() -> None:
    review = initial_review(message_id="m1", classification=ContentClassification.CLEAN, now=T0)
    with pytest.raises(ReviewTransitionError):
        moderator_approve(review, moderator="alice", now=T0)


# --------------------------- is_visible_to_channel ---------------------------


def test_visible_auto_approved() -> None:
    review = initial_review(message_id="m1", classification=ContentClassification.CLEAN, now=T0)
    review = auto_decide(review, now=T0)
    assert is_visible_to_channel(review) is True


def test_visible_flagged() -> None:
    """Pin: FLAGGED messages stay visible (with disclaimer); NOT removed."""

    review = initial_review(
        message_id="m1",
        classification=ContentClassification.FINANCIAL_ADVICE,
        now=T0,
    )
    review = auto_decide(review, now=T0)
    assert is_visible_to_channel(review) is True


def test_invisible_pending() -> None:
    """Pin: PENDING messages are hidden until auto_decide runs."""

    review = initial_review(message_id="m1", classification=ContentClassification.CLEAN, now=T0)
    assert is_visible_to_channel(review) is False


def test_invisible_escalated() -> None:
    review = initial_review(
        message_id="m1",
        classification=ContentClassification.HARASSMENT,
        now=T0,
    )
    review = auto_decide(review, now=T0)
    assert is_visible_to_channel(review) is False


def test_invisible_removed() -> None:
    review = initial_review(
        message_id="m1",
        classification=ContentClassification.PII_LEAK,
        now=T0,
    )
    review = auto_decide(review, now=T0)
    assert is_visible_to_channel(review) is False


# --------------------------- render ------------------------------------------


def test_render_classification_emoji() -> None:
    out = render_classification(
        ClassificationResult(
            classification=ContentClassification.PII_LEAK,
            flagged_phrases=("email",),
            severity_score=1.0,
        )
    )
    assert "🔓" in out
    assert "pii_leak" in out


def test_render_classification_includes_flagged_phrases() -> None:
    out = render_classification(
        ClassificationResult(
            classification=ContentClassification.FINANCIAL_ADVICE,
            flagged_phrases=("you should buy",),
            severity_score=0.6,
        )
    )
    assert "you should buy" in out


def test_render_classification_no_secret_leak() -> None:
    """Pin: render shows the flagged_phrases, never the raw message
    (in fact the renderer doesn't see the original message at all)."""

    # Simulate a PII match — flagged_phrases is just the labels
    out = render_classification(
        ClassificationResult(
            classification=ContentClassification.PII_LEAK,
            flagged_phrases=("email", "ssn"),
            severity_score=1.0,
        )
    )
    # No actual email or SSN in the output (only the labels)
    assert "@" not in out
    assert "123-45" not in out


def test_render_review_includes_status_emoji() -> None:
    review = initial_review(
        message_id="m1",
        classification=ContentClassification.PII_LEAK,
        now=T0,
    )
    review = auto_decide(review, now=T0)
    out = render_review(review)
    assert "🗑️" in out  # REMOVED emoji
    assert "🔓" in out  # PII_LEAK emoji
    assert "auto" in out  # auto-decision moderator


# --------------------------- e2e flows ---------------------------------------


def test_e2e_pii_message_blocked_immediately() -> None:
    """Real-world: member pastes API key — auto-blocked, channel
    never sees the leak, audit row recorded."""

    fake_msg = "Try this Binance key: BinA" + "Bc" * 30
    classification = classify(fake_msg)
    assert classification.classification is ContentClassification.PII_LEAK

    review = initial_review(
        message_id="m1",
        classification=classification.classification,
        now=T0,
    )
    review = auto_decide(review, now=T0)
    assert review.status is ReviewStatus.REMOVED
    assert is_visible_to_channel(review) is False


def test_e2e_financial_advice_visible_with_flag() -> None:
    """Real-world: member says "you should buy AAPL" — message stays
    visible (transparent moderation) but flagged for human review."""

    classification = classify("you should buy AAPL")
    assert classification.classification is ContentClassification.FINANCIAL_ADVICE

    review = initial_review(
        message_id="m1",
        classification=classification.classification,
        now=T0,
    )
    review = auto_decide(review, now=T0)
    assert review.status is ReviewStatus.FLAGGED
    assert is_visible_to_channel(review) is True


def test_e2e_human_moderator_reviews_flagged() -> None:
    """Real-world: moderator reviews a flagged message + decides."""

    classification = classify("you should buy AAPL")
    review = initial_review(
        message_id="m1",
        classification=classification.classification,
        now=T0,
    )
    review = auto_decide(review, now=T0)  # FLAGGED

    # Moderator decides to remove
    review_removed = moderator_remove(review, moderator="alice", now=T0 + timedelta(minutes=2))
    assert review_removed.status is ReviewStatus.REMOVED
    assert review_removed.moderator == "alice"


def test_e2e_replay_consistency() -> None:
    """Same operations produce equal reviews."""

    def build() -> MessageReview:
        review = initial_review(
            message_id="m1",
            classification=ContentClassification.CLEAN,
            now=T0,
        )
        return auto_decide(review, now=T0)

    a = build()
    b = build()
    assert a == b
