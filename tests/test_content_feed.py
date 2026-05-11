"""Tests for education/content_feed.py — Round-5 Wave 20.H."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from halal_trader.education.content_feed import (
    Channel,
    Episode,
    EpisodeStatus,
    MediaType,
    PlaybackRecord,
    Subscription,
    SubscriptionStatus,
    archive_episode,
    publish_episode,
    recommended_feed,
    render_episode,
    render_playback,
    start_playback,
    subscribe,
    unsubscribe,
    update_position,
)


def _channel(
    channel_id: str = "CH1",
    name: str = "Halal Research Weekly",
    publisher_id: str = "platform-research",
    description: str = "Weekly halal market research.",
    created_on: date = date(2026, 1, 1),
) -> Channel:
    return Channel(
        channel_id=channel_id,
        name=name,
        publisher_id=publisher_id,
        description=description,
        created_on=created_on,
    )


def _episode(
    episode_id: str = "E1",
    channel_id: str = "CH1",
    title: str = "Sukuk markets in 2026",
    media_type: MediaType = MediaType.PODCAST,
    summary: str = "An overview of the sukuk market in 2026.",
    duration_seconds: int = 1800,
    media_uri: str = "s3://bucket/ep1.mp3",
    published_at: datetime | None = None,
    status: EpisodeStatus = EpisodeStatus.DRAFT,
    tags: tuple[str, ...] = ("sukuk",),
) -> Episode:
    if media_type not in (MediaType.PODCAST, MediaType.VIDEO):
        duration_seconds = 0
    return Episode(
        episode_id=episode_id,
        channel_id=channel_id,
        title=title,
        media_type=media_type,
        summary=summary,
        duration_seconds=duration_seconds,
        media_uri=media_uri,
        published_at=published_at,
        status=status,
        tags=tags,
    )


# --- Channel validation --------------------------


def test_channel_valid():
    c = _channel()
    assert c.channel_id == "CH1"


def test_channel_empty_id_rejected():
    with pytest.raises(ValueError):
        _channel(channel_id="")


def test_channel_long_name_rejected():
    with pytest.raises(ValueError):
        _channel(name="x" * 200)


def test_channel_empty_description_rejected():
    with pytest.raises(ValueError):
        _channel(description=" ")


# --- Episode validation ---------------------------


def test_episode_valid_podcast():
    e = _episode()
    assert e.media_type is MediaType.PODCAST
    assert e.status is EpisodeStatus.DRAFT


def test_episode_valid_article_zero_duration():
    e = _episode(media_type=MediaType.ARTICLE)
    assert e.duration_seconds == 0


def test_episode_podcast_zero_duration_rejected():
    with pytest.raises(ValueError):
        Episode(
            episode_id="E1",
            channel_id="CH1",
            title="x",
            media_type=MediaType.PODCAST,
            summary="x",
            duration_seconds=0,
            media_uri="s3://x",
        )


def test_episode_article_nonzero_duration_rejected():
    with pytest.raises(ValueError):
        Episode(
            episode_id="E1",
            channel_id="CH1",
            title="x",
            media_type=MediaType.ARTICLE,
            summary="x",
            duration_seconds=600,
            media_uri="s3://x",
        )


def test_episode_excessive_duration_rejected():
    with pytest.raises(ValueError):
        _episode(duration_seconds=25 * 3600)


def test_episode_empty_id_rejected():
    with pytest.raises(ValueError):
        _episode(episode_id="")


def test_episode_long_title_rejected():
    with pytest.raises(ValueError):
        _episode(title="x" * 300)


def test_episode_published_without_date_rejected():
    with pytest.raises(ValueError):
        _episode(status=EpisodeStatus.PUBLISHED, published_at=None)


def test_episode_draft_with_date_rejected():
    with pytest.raises(ValueError):
        _episode(
            status=EpisodeStatus.DRAFT,
            published_at=datetime(2026, 5, 1),
        )


def test_episode_duplicate_tag_rejected():
    with pytest.raises(ValueError):
        _episode(tags=("sukuk", "sukuk"))


def test_episode_empty_tag_rejected():
    with pytest.raises(ValueError):
        _episode(tags=("sukuk", " "))


def test_episode_long_tag_rejected():
    with pytest.raises(ValueError):
        _episode(tags=("x" * 50,))


def test_episode_immutable():
    e = _episode()
    with pytest.raises(AttributeError):
        e.title = "x"  # type: ignore[misc]


# --- publish_episode ------------------------------


def test_publish_draft_to_published():
    e = _episode()
    p = publish_episode(e, published_at=datetime(2026, 5, 1))
    assert p.status is EpisodeStatus.PUBLISHED
    assert p.published_at == datetime(2026, 5, 1)


def test_publish_non_draft_rejected():
    e = publish_episode(_episode(), published_at=datetime(2026, 5, 1))
    with pytest.raises(ValueError):
        publish_episode(e, published_at=datetime(2026, 5, 2))


def test_publish_moderation_blocks():
    e = _episode(summary="bad word here")
    with pytest.raises(ValueError):
        publish_episode(
            e,
            published_at=datetime(2026, 5, 1),
            is_text_acceptable=lambda s: "bad" not in s,
        )


def test_publish_moderation_passes():
    e = _episode()
    p = publish_episode(
        e,
        published_at=datetime(2026, 5, 1),
        is_text_acceptable=lambda s: True,
    )
    assert p.status is EpisodeStatus.PUBLISHED


# --- archive_episode ------------------------------


def test_archive_published_to_archived():
    e = publish_episode(_episode(), published_at=datetime(2026, 5, 1))
    a = archive_episode(e)
    assert a.status is EpisodeStatus.ARCHIVED


def test_archive_draft_rejected():
    e = _episode()
    with pytest.raises(ValueError):
        archive_episode(e)


def test_archive_terminal():
    e = archive_episode(publish_episode(_episode(), published_at=datetime(2026, 5, 1)))
    with pytest.raises(ValueError):
        archive_episode(e)


# --- Subscription ---------------------------------


def test_subscribe_basic():
    s = subscribe(
        subscription_id="S1",
        user_id="bob",
        channel_id="CH1",
        subscribed_at=datetime(2026, 5, 5),
    )
    assert s.status is SubscriptionStatus.ACTIVE


def test_subscribe_duplicate_rejected():
    s = subscribe(
        subscription_id="S1",
        user_id="bob",
        channel_id="CH1",
        subscribed_at=datetime(2026, 5, 5),
    )
    with pytest.raises(ValueError):
        subscribe(
            subscription_id="S2",
            user_id="bob",
            channel_id="CH1",
            subscribed_at=datetime(2026, 5, 6),
            existing=[s],
        )


def test_subscribe_other_user_same_channel_allowed():
    s = subscribe(
        subscription_id="S1",
        user_id="bob",
        channel_id="CH1",
        subscribed_at=datetime(2026, 5, 5),
    )
    s2 = subscribe(
        subscription_id="S2",
        user_id="charlie",
        channel_id="CH1",
        subscribed_at=datetime(2026, 5, 6),
        existing=[s],
    )
    assert s2.user_id == "charlie"


def test_unsubscribe_active():
    s = subscribe(
        subscription_id="S1",
        user_id="bob",
        channel_id="CH1",
        subscribed_at=datetime(2026, 5, 5),
    )
    u = unsubscribe(s, at=datetime(2026, 6, 1))
    assert u.status is SubscriptionStatus.UNSUBSCRIBED


def test_unsubscribe_already_unsubscribed_rejected():
    s = unsubscribe(
        subscribe(
            subscription_id="S1",
            user_id="bob",
            channel_id="CH1",
            subscribed_at=datetime(2026, 5, 5),
        ),
        at=datetime(2026, 6, 1),
    )
    with pytest.raises(ValueError):
        unsubscribe(s, at=datetime(2026, 6, 2))


def test_resubscribe_after_unsub():
    s_old = unsubscribe(
        subscribe(
            subscription_id="S1",
            user_id="bob",
            channel_id="CH1",
            subscribed_at=datetime(2026, 5, 5),
        ),
        at=datetime(2026, 6, 1),
    )
    s_new = subscribe(
        subscription_id="S2",
        user_id="bob",
        channel_id="CH1",
        subscribed_at=datetime(2026, 7, 1),
        existing=[s_old],
    )
    assert s_new.status is SubscriptionStatus.ACTIVE


def test_unsubscribe_before_start_rejected():
    with pytest.raises(ValueError):
        Subscription(
            subscription_id="S1",
            user_id="bob",
            channel_id="CH1",
            subscribed_at=datetime(2026, 5, 5),
            status=SubscriptionStatus.UNSUBSCRIBED,
            unsubscribed_at=datetime(2026, 4, 1),
        )


# --- Playback ------------------------------------


def test_playback_start():
    r = start_playback(user_id="bob", episode_id="E1", at=datetime(2026, 5, 1))
    assert r.position_seconds == 0
    assert not r.completed


def test_playback_advances_monotonically():
    r = start_playback(user_id="bob", episode_id="E1", at=datetime(2026, 5, 1))
    r = update_position(
        r,
        new_position_seconds=300,
        duration_seconds=1800,
        at=datetime(2026, 5, 1, 0, 5),
    )
    assert r.position_seconds == 300


def test_playback_regression_rejected():
    r = start_playback(user_id="bob", episode_id="E1", at=datetime(2026, 5, 1))
    r = update_position(
        r,
        new_position_seconds=300,
        duration_seconds=1800,
        at=datetime(2026, 5, 1, 0, 5),
    )
    with pytest.raises(ValueError):
        update_position(
            r,
            new_position_seconds=100,
            duration_seconds=1800,
            at=datetime(2026, 5, 1, 0, 10),
        )


def test_playback_above_duration_rejected():
    r = start_playback(user_id="bob", episode_id="E1", at=datetime(2026, 5, 1))
    with pytest.raises(ValueError):
        update_position(
            r,
            new_position_seconds=2000,
            duration_seconds=1800,
            at=datetime(2026, 5, 1),
        )


def test_playback_completion_at_threshold():
    """Pin: completion = position ≥ 95% of duration by default."""
    r = start_playback(user_id="bob", episode_id="E1", at=datetime(2026, 5, 1))
    r = update_position(
        r,
        new_position_seconds=1710,  # 95% of 1800
        duration_seconds=1800,
        at=datetime(2026, 5, 1, 0, 28),
    )
    assert r.completed


def test_playback_completion_sticky():
    """Pin: once completed, stays completed."""
    r = start_playback(user_id="bob", episode_id="E1", at=datetime(2026, 5, 1))
    r = update_position(
        r,
        new_position_seconds=1800,
        duration_seconds=1800,
        at=datetime(2026, 5, 1, 0, 30),
    )
    assert r.completed
    # Hypothetically same-position update — completion sticks.
    r2 = update_position(
        r,
        new_position_seconds=1800,
        duration_seconds=1800,
        at=datetime(2026, 5, 1, 0, 31),
    )
    assert r2.completed


def test_playback_zero_duration_rejected():
    r = start_playback(user_id="bob", episode_id="E1", at=datetime(2026, 5, 1))
    with pytest.raises(ValueError):
        update_position(
            r,
            new_position_seconds=10,
            duration_seconds=0,
            at=datetime(2026, 5, 1),
        )


def test_playback_negative_position_record_rejected():
    with pytest.raises(ValueError):
        PlaybackRecord(
            user_id="bob",
            episode_id="E1",
            position_seconds=-1,
            completed=False,
            last_played_at=datetime(2026, 5, 1),
        )


def test_playback_invalid_threshold_rejected():
    r = start_playback(user_id="bob", episode_id="E1", at=datetime(2026, 5, 1))
    with pytest.raises(ValueError):
        update_position(
            r,
            new_position_seconds=10,
            duration_seconds=100,
            at=datetime(2026, 5, 1),
            completion_threshold_pct=0,
        )


# --- recommended_feed ---------------------------


def test_feed_only_includes_active_subs():
    sub_active = subscribe(
        subscription_id="S1",
        user_id="bob",
        channel_id="CH1",
        subscribed_at=datetime(2026, 4, 1),
    )
    sub_inactive = unsubscribe(
        subscribe(
            subscription_id="S2",
            user_id="bob",
            channel_id="CH2",
            subscribed_at=datetime(2026, 3, 1),
        ),
        at=datetime(2026, 4, 15),
    )
    eps = [
        publish_episode(
            _episode(episode_id="E1", channel_id="CH1"),
            published_at=datetime(2026, 5, 1),
        ),
        publish_episode(
            _episode(episode_id="E2", channel_id="CH2"),
            published_at=datetime(2026, 5, 1),
        ),
    ]
    feed = recommended_feed("bob", [sub_active, sub_inactive], eps, as_of=datetime(2026, 5, 10))
    ids = {e.episode_id for e in feed}
    assert "E1" in ids
    assert "E2" not in ids


def test_feed_only_includes_published():
    sub = subscribe(
        subscription_id="S1",
        user_id="bob",
        channel_id="CH1",
        subscribed_at=datetime(2026, 4, 1),
    )
    eps = [
        publish_episode(
            _episode(episode_id="E1"),
            published_at=datetime(2026, 5, 1),
        ),
        _episode(episode_id="E2"),  # DRAFT
    ]
    feed = recommended_feed("bob", [sub], eps, as_of=datetime(2026, 5, 10))
    assert {e.episode_id for e in feed} == {"E1"}


def test_feed_excludes_future_episodes():
    sub = subscribe(
        subscription_id="S1",
        user_id="bob",
        channel_id="CH1",
        subscribed_at=datetime(2026, 4, 1),
    )
    eps = [
        publish_episode(
            _episode(episode_id="E_future"),
            published_at=datetime(2026, 6, 1),
        )
    ]
    feed = recommended_feed("bob", [sub], eps, as_of=datetime(2026, 5, 1))
    assert feed == ()


def test_feed_sorted_newest_first():
    sub = subscribe(
        subscription_id="S1",
        user_id="bob",
        channel_id="CH1",
        subscribed_at=datetime(2026, 4, 1),
    )
    eps = [
        publish_episode(
            _episode(episode_id="E_old"),
            published_at=datetime(2026, 4, 1),
        ),
        publish_episode(
            _episode(episode_id="E_new"),
            published_at=datetime(2026, 5, 1),
        ),
    ]
    feed = recommended_feed("bob", [sub], eps, as_of=datetime(2026, 5, 10))
    assert feed[0].episode_id == "E_new"


def test_feed_caps_top_n():
    sub = subscribe(
        subscription_id="S1",
        user_id="bob",
        channel_id="CH1",
        subscribed_at=datetime(2026, 4, 1),
    )
    eps = [
        publish_episode(
            _episode(episode_id=f"E{i}"),
            published_at=datetime(2026, 5, i + 1),
        )
        for i in range(10)
    ]
    feed = recommended_feed("bob", [sub], eps, as_of=datetime(2026, 6, 1), top_n=3)
    assert len(feed) == 3


def test_feed_no_subs_returns_empty():
    feed = recommended_feed("bob", [], [], as_of=datetime(2026, 5, 1))
    assert feed == ()


def test_feed_invalid_top_n_rejected():
    with pytest.raises(ValueError):
        recommended_feed("bob", [], [], as_of=datetime(2026, 5, 1), top_n=0)


# --- Render ------------------------------------


def test_render_episode_status_emoji():
    e = _episode()
    out = render_episode(e)
    assert "📝" in out
    assert "🎙️" in out


def test_render_episode_duration_for_time_based():
    e = _episode(duration_seconds=1800)
    out = render_episode(e)
    assert "30m" in out


def test_render_episode_no_duration_for_article():
    e = _episode(media_type=MediaType.ARTICLE, title="A Plain Title")
    out = render_episode(e)
    # No "(Nm)" duration marker for articles.
    assert "m)" not in out


def test_render_playback_no_secret_leak():
    r = start_playback(
        user_id="alice@example.com",
        episode_id="E1",
        at=datetime(2026, 5, 1),
    )
    out = render_playback(r, duration_seconds=1800)
    assert "alice@example.com" not in out


def test_render_playback_completed_marker():
    r = start_playback(user_id="bob", episode_id="E1", at=datetime(2026, 5, 1))
    r = update_position(
        r,
        new_position_seconds=1800,
        duration_seconds=1800,
        at=datetime(2026, 5, 1, 0, 30),
    )
    out = render_playback(r, duration_seconds=1800)
    assert "✅" in out
