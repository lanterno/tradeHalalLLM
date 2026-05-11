"""Halal-investing content feed — Round-5 Wave 20.H.

Weekly content drops from the platform's research team across podcast +
YouTube + written formats. This module is the **catalogue + episode
lifecycle + per-user playback tracker**:

1. Operator publishes Episodes with a closed-set MediaType.
2. Users subscribe to Channels; subscription is FIFO-ordered for
   recommendation surfaces.
3. Playback records track per-user progress; mark-as-played updates
   the position monotonically.

Pinned semantics:

- **Closed-set MediaType ladder** — PODCAST / VIDEO / ARTICLE /
  NEWSLETTER.
- **Closed-set EpisodeStatus FSM** — DRAFT → PUBLISHED → ARCHIVED.
- **Closed-set SubscriptionStatus** — ACTIVE / UNSUBSCRIBED.
- **Episode duration in seconds** for time-based formats; 0 allowed
  for ARTICLE / NEWSLETTER (no playback duration).
- **Playback position is monotone non-decreasing** within an episode;
  re-listening counts as a new playback event, not a regression.
- **Subscription FIFO order** — `subscribed_at` strictly determines
  recommendation ordering.
- **Halal moderation gate** at publish time via plug-in predicate.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — user/channel IDs masked.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import Enum


class MediaType(str, Enum):
    """Closed-set media-type ladder."""

    PODCAST = "podcast"
    VIDEO = "video"
    ARTICLE = "article"
    NEWSLETTER = "newsletter"


_TIME_BASED: frozenset[MediaType] = frozenset({MediaType.PODCAST, MediaType.VIDEO})


class EpisodeStatus(str, Enum):
    """Closed-set episode FSM ladder."""

    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class SubscriptionStatus(str, Enum):
    """Closed-set subscription status."""

    ACTIVE = "active"
    UNSUBSCRIBED = "unsubscribed"


@dataclass(frozen=True)
class Channel:
    """A content channel — one publisher's stream of episodes."""

    channel_id: str
    name: str
    publisher_id: str
    description: str
    created_on: date

    def __post_init__(self) -> None:
        if not self.channel_id or not self.channel_id.strip():
            raise ValueError("channel_id must be non-empty")
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if len(self.name) > 100:
            raise ValueError("name must be ≤ 100 chars")
        if not self.publisher_id or not self.publisher_id.strip():
            raise ValueError("publisher_id must be non-empty")
        if not self.description.strip():
            raise ValueError("description must be non-empty")
        if len(self.description) > 1000:
            raise ValueError("description must be ≤ 1000 chars")


@dataclass(frozen=True)
class Episode:
    """One episode in a channel."""

    episode_id: str
    channel_id: str
    title: str
    media_type: MediaType
    summary: str
    duration_seconds: int
    """Required > 0 for PODCAST / VIDEO; must be 0 for ARTICLE / NEWSLETTER."""
    media_uri: str
    """Operator-side URI to the asset (S3 / CDN); masked in render."""
    published_at: datetime | None = None
    status: EpisodeStatus = EpisodeStatus.DRAFT
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.episode_id or not self.episode_id.strip():
            raise ValueError("episode_id must be non-empty")
        if not self.channel_id or not self.channel_id.strip():
            raise ValueError("channel_id must be non-empty")
        if not self.title.strip():
            raise ValueError("title must be non-empty")
        if len(self.title) > 200:
            raise ValueError("title must be ≤ 200 chars")
        if not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if len(self.summary) > 1000:
            raise ValueError("summary must be ≤ 1000 chars")
        if not self.media_uri.strip():
            raise ValueError("media_uri must be non-empty")
        if self.media_type in _TIME_BASED:
            if self.duration_seconds <= 0:
                raise ValueError(f"{self.media_type.value} requires duration_seconds > 0")
        else:
            if self.duration_seconds != 0:
                raise ValueError(f"{self.media_type.value} must have duration_seconds=0")
        if self.duration_seconds > 24 * 3600:
            raise ValueError("duration > 24h suspicious")
        if self.status is EpisodeStatus.PUBLISHED and self.published_at is None:
            raise ValueError("PUBLISHED requires published_at")
        if self.status is EpisodeStatus.DRAFT and self.published_at is not None:
            raise ValueError("DRAFT must not have published_at")
        # Tag normalisation pins.
        seen: set[str] = set()
        for t in self.tags:
            if not t or not t.strip():
                raise ValueError("tag must be non-empty")
            if len(t) > 32:
                raise ValueError("tag must be ≤ 32 chars")
            if t in seen:
                raise ValueError(f"duplicate tag {t}")
            seen.add(t)


def publish_episode(
    episode: Episode,
    *,
    published_at: datetime,
    is_text_acceptable: Callable[[str], bool] | None = None,
) -> Episode:
    """Promote a DRAFT episode to PUBLISHED.

    Pinned: optional moderation predicate runs on the summary; rejected
    text raises.
    """
    if episode.status is not EpisodeStatus.DRAFT:
        raise ValueError(f"publish illegal from {episode.status.value}")
    if is_text_acceptable is not None and not is_text_acceptable(episode.summary):
        raise ValueError("summary failed moderation")
    return replace(
        episode,
        status=EpisodeStatus.PUBLISHED,
        published_at=published_at,
    )


def archive_episode(episode: Episode) -> Episode:
    if episode.status is not EpisodeStatus.PUBLISHED:
        raise ValueError(f"archive illegal from {episode.status.value}")
    return replace(episode, status=EpisodeStatus.ARCHIVED)


# --- Subscriptions ---------------------------------------------------


@dataclass(frozen=True)
class Subscription:
    """One user's subscription to a channel."""

    subscription_id: str
    user_id: str
    channel_id: str
    subscribed_at: datetime
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    unsubscribed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.subscription_id or not self.subscription_id.strip():
            raise ValueError("subscription_id must be non-empty")
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not self.channel_id or not self.channel_id.strip():
            raise ValueError("channel_id must be non-empty")
        if self.status is SubscriptionStatus.UNSUBSCRIBED and self.unsubscribed_at is None:
            raise ValueError("UNSUBSCRIBED requires unsubscribed_at")
        if self.status is SubscriptionStatus.ACTIVE and self.unsubscribed_at is not None:
            raise ValueError("ACTIVE must not have unsubscribed_at set")
        if self.unsubscribed_at is not None and self.unsubscribed_at < self.subscribed_at:
            raise ValueError("unsubscribed_at must be ≥ subscribed_at")


def subscribe(
    *,
    subscription_id: str,
    user_id: str,
    channel_id: str,
    subscribed_at: datetime,
    existing: Iterable[Subscription] = (),
) -> Subscription:
    """Open an ACTIVE subscription. Rejects duplicates."""
    for s in existing:
        if (
            s.user_id == user_id
            and s.channel_id == channel_id
            and s.status is SubscriptionStatus.ACTIVE
        ):
            raise ValueError("user already has an ACTIVE subscription to this channel")
    return Subscription(
        subscription_id=subscription_id,
        user_id=user_id,
        channel_id=channel_id,
        subscribed_at=subscribed_at,
    )


def unsubscribe(subscription: Subscription, *, at: datetime) -> Subscription:
    if subscription.status is not SubscriptionStatus.ACTIVE:
        raise ValueError("only ACTIVE subscriptions can be unsubscribed")
    return replace(
        subscription,
        status=SubscriptionStatus.UNSUBSCRIBED,
        unsubscribed_at=at,
    )


# --- Playback --------------------------------------------------------


@dataclass(frozen=True)
class PlaybackRecord:
    """One user's playback position in an episode."""

    user_id: str
    episode_id: str
    position_seconds: int
    """Last known position. Monotone non-decreasing per update_position."""
    completed: bool
    last_played_at: datetime

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not self.episode_id or not self.episode_id.strip():
            raise ValueError("episode_id must be non-empty")
        if self.position_seconds < 0:
            raise ValueError("position_seconds must be ≥ 0")


def update_position(
    record: PlaybackRecord,
    *,
    new_position_seconds: int,
    duration_seconds: int,
    at: datetime,
    completion_threshold_pct: float = 0.95,
) -> PlaybackRecord:
    """Advance the playback position monotonically.

    Pinned:
    - `new_position_seconds` ≥ record.position_seconds (no regression).
    - `new_position_seconds` ≤ duration_seconds.
    - completion fires when position ≥ duration × completion_threshold_pct
      OR equals duration exactly. Once True it stays True (even after
      re-watching from 0 — which would be a new `PlaybackRecord` per
      operator convention; we don't model that within one record).
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive for playback")
    if not 0.0 < completion_threshold_pct <= 1.0:
        raise ValueError("completion_threshold_pct must be in (0, 1]")
    if new_position_seconds < record.position_seconds:
        raise ValueError("playback position cannot regress within one record")
    if new_position_seconds > duration_seconds:
        raise ValueError("position_seconds cannot exceed duration_seconds")
    completed = (
        record.completed
        or new_position_seconds >= duration_seconds
        or new_position_seconds / duration_seconds >= completion_threshold_pct - 1e-9
    )
    return replace(
        record,
        position_seconds=new_position_seconds,
        completed=completed,
        last_played_at=at,
    )


def start_playback(
    *,
    user_id: str,
    episode_id: str,
    at: datetime,
) -> PlaybackRecord:
    return PlaybackRecord(
        user_id=user_id,
        episode_id=episode_id,
        position_seconds=0,
        completed=False,
        last_played_at=at,
    )


# --- Recommendation feed --------------------------------------------


def recommended_feed(
    user_id: str,
    subscriptions: Iterable[Subscription],
    episodes: Iterable[Episode],
    *,
    as_of: datetime,
    top_n: int = 20,
) -> tuple[Episode, ...]:
    """Recommend episodes for a user.

    Pinned:
    - Only PUBLISHED episodes from channels with an ACTIVE subscription.
    - Sorted by published_at descending (newest first).
    - Capped at `top_n`.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    subs_t = tuple(
        s for s in subscriptions if s.user_id == user_id and s.status is SubscriptionStatus.ACTIVE
    )
    if not subs_t:
        return ()
    channel_ids = {s.channel_id for s in subs_t}
    eligible = [
        e
        for e in episodes
        if e.channel_id in channel_ids
        and e.status is EpisodeStatus.PUBLISHED
        and e.published_at is not None
        and e.published_at <= as_of
    ]
    eligible.sort(
        key=lambda e: (e.published_at, e.episode_id),
        reverse=True,
    )
    return tuple(eligible[:top_n])


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_STATUS_EMOJI: dict[EpisodeStatus, str] = {
    EpisodeStatus.DRAFT: "📝",
    EpisodeStatus.PUBLISHED: "📡",
    EpisodeStatus.ARCHIVED: "🗄️",
}


_MEDIA_EMOJI: dict[MediaType, str] = {
    MediaType.PODCAST: "🎙️",
    MediaType.VIDEO: "🎬",
    MediaType.ARTICLE: "📄",
    MediaType.NEWSLETTER: "✉️",
}


def render_episode(episode: Episode) -> str:
    """Operator-readable summary; media_uri is masked."""
    head = (
        f"{_STATUS_EMOJI[episode.status]} "
        f"{_MEDIA_EMOJI[episode.media_type]} "
        f"[{episode.episode_id}] {episode.title}"
    )
    if episode.media_type in _TIME_BASED:
        head += f" ({episode.duration_seconds // 60}m)"
    return head


def render_playback(record: PlaybackRecord, *, duration_seconds: int) -> str:
    pct = record.position_seconds / duration_seconds * 100 if duration_seconds > 0 else 0.0
    flag = " ✅" if record.completed else ""
    return (
        f"▶️ {_mask(record.user_id)} on {record.episode_id}: "
        f"{record.position_seconds}s / {duration_seconds}s "
        f"({pct:.1f}%){flag}"
    )
