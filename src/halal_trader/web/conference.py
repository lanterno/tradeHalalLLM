"""Annual halal fintech conference planning engine.

The roadmap pins Wave 10.E: "Bring together halal fintech founders,
operators, scholars, academics. Platform sponsors / hosts. Cements
brand position." This module is the **pure-Python planning state
machine** the conference operations team consults to track speaker
invitations, schedule sessions without conflict, and manage sponsor
tiers.

Picked a focused planning state machine over a generic
"event-management SaaS" approach because (a) speaker invitation
lifecycle (invited → accepted/declined → confirmed) needs an
audit trail for "did Mufti X confirm before we printed the
schedule?" — a missed-confirmation that lands in the printed
program and the speaker doesn't show up is the worst-case failure;
(b) session scheduling needs hard time-conflict detection — booking
two sessions for Mufti X in overlapping slots is an error class
that must fail at planning time, not at the conference itself; (c)
sponsor tiers (PLATINUM / GOLD / SILVER / BRONZE) drive logo placement
+ booth size + speaking slots; encoding the tier ladder once means
the printed program + signage + website all consult the same source.

Pinned semantics:
- **Speaker invitation lifecycle: INVITED → ACCEPTED / DECLINED →
  CONFIRMED.** Forward-only; only ACCEPTED can move to CONFIRMED.
  CONFIRMED is what the printed program keys on.
- **Session conflict detection.** Sessions for the same speaker in
  overlapping time windows raise SessionConflictError; sessions in
  the same room overlap raise too.
- **Closed-set SponsorTier ladder.** PLATINUM > GOLD > SILVER >
  BRONZE; tier drives benefits matrix.
- **Render output never includes speaker contact emails or sponsor
  invoice amounts.** Speaker handle + sponsor display name only;
  mirrors no-secret patterns of upstream waves.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SpeakerStatus(str, Enum):
    """Speaker invitation lifecycle status.

    Pinned string values for JSON / DB stability.
    """

    INVITED = "invited"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    CONFIRMED = "confirmed"  # Travel + slides confirmed; safe to print
    WITHDRAWN = "withdrawn"  # Was confirmed but pulled out — terminal


class SponsorTier(str, Enum):
    """Sponsor tier ladder, ordered PLATINUM > GOLD > SILVER > BRONZE."""

    PLATINUM = "platinum"
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"


_TIER_ORDER: dict[SponsorTier, int] = {
    SponsorTier.BRONZE: 0,
    SponsorTier.SILVER: 1,
    SponsorTier.GOLD: 2,
    SponsorTier.PLATINUM: 3,
}


class SpeakerKind(str, Enum):
    """Speaker classification.

    Pinned values; the conference invitation rotation aims for a
    balance across kinds (the operator-side "did we book enough
    scholars vs founders?" check consults this enum).
    """

    SCHOLAR = "scholar"
    FOUNDER = "founder"
    OPERATOR = "operator"
    ACADEMIC = "academic"
    REGULATOR = "regulator"


class SpeakerTransitionError(Exception):
    """Raised when a speaker status transition is invalid."""

    def __init__(self, current: SpeakerStatus, attempted: SpeakerStatus) -> None:
        super().__init__(f"cannot transition from {current.value} to {attempted.value}")
        self.current = current
        self.attempted = attempted


class SessionConflictError(Exception):
    """Raised when a session would conflict with an existing booking."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Speaker:
    """One invited speaker.

    `display_handle` is the public-facing name (Mufti X / Dr Y / etc).
    `kind` classifies the speaker for rotation balance. The dataclass
    deliberately doesn't carry email / phone — those are operator-side
    state, kept structurally out of the render path.
    """

    speaker_id: str
    display_handle: str
    kind: SpeakerKind
    status: SpeakerStatus
    invited_at: datetime
    decided_at: datetime | None = None  # When status moved past INVITED

    def __post_init__(self) -> None:
        if not self.speaker_id or not self.speaker_id.strip():
            raise ValueError("speaker_id must be non-empty")
        if not self.display_handle or not self.display_handle.strip():
            raise ValueError("display_handle must be non-empty")
        if self.invited_at.tzinfo is None:
            raise ValueError("invited_at must be timezone-aware")
        if self.decided_at is not None:
            if self.decided_at.tzinfo is None:
                raise ValueError("decided_at must be timezone-aware when set")
            if self.decided_at < self.invited_at:
                raise ValueError("decided_at must be >= invited_at")
        # INVITED status must NOT have decided_at; non-INVITED requires it
        if self.status is SpeakerStatus.INVITED:
            if self.decided_at is not None:
                raise ValueError("INVITED status must not have decided_at")
        else:
            if self.decided_at is None:
                raise ValueError(f"{self.status.value} status requires decided_at")


def invite_speaker(
    *,
    speaker_id: str,
    display_handle: str,
    kind: SpeakerKind,
    now: datetime,
) -> Speaker:
    """Build a fresh INVITED speaker record."""

    if not speaker_id or not speaker_id.strip():
        raise ValueError("speaker_id must be non-empty")
    if not display_handle or not display_handle.strip():
        raise ValueError("display_handle must be non-empty")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return Speaker(
        speaker_id=speaker_id,
        display_handle=display_handle,
        kind=kind,
        status=SpeakerStatus.INVITED,
        invited_at=now,
    )


def accept_invitation(speaker: Speaker, *, now: datetime) -> Speaker:
    """Move INVITED → ACCEPTED."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if speaker.status is not SpeakerStatus.INVITED:
        raise SpeakerTransitionError(speaker.status, SpeakerStatus.ACCEPTED)
    return Speaker(
        speaker_id=speaker.speaker_id,
        display_handle=speaker.display_handle,
        kind=speaker.kind,
        status=SpeakerStatus.ACCEPTED,
        invited_at=speaker.invited_at,
        decided_at=now,
    )


def decline_invitation(speaker: Speaker, *, now: datetime) -> Speaker:
    """Move INVITED → DECLINED. Terminal."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if speaker.status is not SpeakerStatus.INVITED:
        raise SpeakerTransitionError(speaker.status, SpeakerStatus.DECLINED)
    return Speaker(
        speaker_id=speaker.speaker_id,
        display_handle=speaker.display_handle,
        kind=speaker.kind,
        status=SpeakerStatus.DECLINED,
        invited_at=speaker.invited_at,
        decided_at=now,
    )


def confirm_speaker(speaker: Speaker, *, now: datetime) -> Speaker:
    """Move ACCEPTED → CONFIRMED.

    CONFIRMED is what the printed program keys on; only ACCEPTED
    speakers can be confirmed (the load-bearing print-safety pin).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if speaker.status is not SpeakerStatus.ACCEPTED:
        raise SpeakerTransitionError(speaker.status, SpeakerStatus.CONFIRMED)
    return Speaker(
        speaker_id=speaker.speaker_id,
        display_handle=speaker.display_handle,
        kind=speaker.kind,
        status=SpeakerStatus.CONFIRMED,
        invited_at=speaker.invited_at,
        decided_at=now,
    )


def withdraw_speaker(speaker: Speaker, *, now: datetime) -> Speaker:
    """Move CONFIRMED → WITHDRAWN. Terminal.

    Used when a confirmed speaker pulls out late and the operator
    needs an audit row showing the change happened post-confirmation.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if speaker.status is not SpeakerStatus.CONFIRMED:
        raise SpeakerTransitionError(speaker.status, SpeakerStatus.WITHDRAWN)
    return Speaker(
        speaker_id=speaker.speaker_id,
        display_handle=speaker.display_handle,
        kind=speaker.kind,
        status=SpeakerStatus.WITHDRAWN,
        invited_at=speaker.invited_at,
        decided_at=now,
    )


def is_print_safe(speaker: Speaker) -> bool:
    """True if speaker is safe to include in the printed program.

    Only CONFIRMED speakers print-safe. ACCEPTED is "yes, attending"
    but slides / travel not yet confirmed; WITHDRAWN is post-confirmation
    pull-out and must be removed from the program.
    """

    return speaker.status is SpeakerStatus.CONFIRMED


@dataclass(frozen=True)
class Session:
    """One conference session.

    `room` is a string room identifier (e.g. "main_hall", "track_a").
    `starts_at` and `ends_at` are tz-aware UTC bounds; `speaker_ids`
    is a frozenset of speaker IDs presenting in this session
    (panels can have multiple speakers).
    """

    session_id: str
    title: str
    room: str
    starts_at: datetime
    ends_at: datetime
    speaker_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.room or not self.room.strip():
            raise ValueError("room must be non-empty")
        if self.starts_at.tzinfo is None:
            raise ValueError("starts_at must be timezone-aware")
        if self.ends_at.tzinfo is None:
            raise ValueError("ends_at must be timezone-aware")
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be after starts_at")
        if not self.speaker_ids:
            raise ValueError("speaker_ids must be non-empty")


def _windows_overlap(
    a_start: datetime,
    a_end: datetime,
    b_start: datetime,
    b_end: datetime,
) -> bool:
    """True if two half-open [start, end) windows overlap.

    Adjacent windows (a_end == b_start) do NOT overlap — pinned so
    a session ending at 14:00 and another starting at 14:00 in the
    same room are valid back-to-back bookings.
    """

    return a_start < b_end and b_start < a_end


def assert_no_conflict(
    new_session: Session,
    existing_sessions: Iterable[Session],
) -> None:
    """Raise SessionConflictError if `new_session` conflicts with any
    existing session.

    Two kinds of conflict:
    1. Same speaker booked in overlapping windows (a speaker can't
       present two sessions at once).
    2. Same room booked in overlapping windows (one room, one
       session at a time).
    """

    for existing in existing_sessions:
        if existing.session_id == new_session.session_id:
            raise SessionConflictError(f"session_id {new_session.session_id!r} already exists")
        # Time overlap precondition for both conflict checks
        if not _windows_overlap(
            new_session.starts_at,
            new_session.ends_at,
            existing.starts_at,
            existing.ends_at,
        ):
            continue
        # Speaker conflict
        shared_speakers = new_session.speaker_ids & existing.speaker_ids
        if shared_speakers:
            who = ", ".join(sorted(shared_speakers))
            raise SessionConflictError(
                f"speakers {who} double-booked: {existing.session_id} vs {new_session.session_id}"
            )
        # Room conflict
        if existing.room == new_session.room:
            raise SessionConflictError(
                f"room {new_session.room!r} double-booked: "
                f"{existing.session_id} vs {new_session.session_id}"
            )


@dataclass(frozen=True)
class Sponsor:
    """One conference sponsor."""

    sponsor_id: str
    display_name: str
    tier: SponsorTier
    onboarded_at: datetime

    def __post_init__(self) -> None:
        if not self.sponsor_id or not self.sponsor_id.strip():
            raise ValueError("sponsor_id must be non-empty")
        if not self.display_name or not self.display_name.strip():
            raise ValueError("display_name must be non-empty")
        if self.onboarded_at.tzinfo is None:
            raise ValueError("onboarded_at must be timezone-aware")


def tier_outranks(a: SponsorTier, b: SponsorTier) -> bool:
    """True if tier `a` is strictly higher than tier `b`."""

    return _TIER_ORDER[a] > _TIER_ORDER[b]


def sponsors_at_or_above(
    sponsors: Iterable[Sponsor], *, minimum: SponsorTier
) -> tuple[Sponsor, ...]:
    """Return sponsors at or above the minimum tier (sorted highest first)."""

    qualified = [s for s in sponsors if _TIER_ORDER[s.tier] >= _TIER_ORDER[minimum]]
    return tuple(sorted(qualified, key=lambda s: -_TIER_ORDER[s.tier]))


def speaker_kind_balance(
    speakers: Iterable[Speaker],
) -> dict[SpeakerKind, int]:
    """Return count per SpeakerKind for confirmed speakers only.

    Operators consult this to verify the conference has the
    documented balance (scholars + founders + operators + academics
    + regulators).
    """

    counts: dict[SpeakerKind, int] = {kind: 0 for kind in SpeakerKind}
    for sp in speakers:
        if sp.status is SpeakerStatus.CONFIRMED:
            counts[sp.kind] += 1
    return counts


_STATUS_EMOJI: dict[SpeakerStatus, str] = {
    SpeakerStatus.INVITED: "📧",
    SpeakerStatus.ACCEPTED: "✅",
    SpeakerStatus.DECLINED: "❌",
    SpeakerStatus.CONFIRMED: "🎤",
    SpeakerStatus.WITHDRAWN: "🚫",
}


_KIND_EMOJI: dict[SpeakerKind, str] = {
    SpeakerKind.SCHOLAR: "📚",
    SpeakerKind.FOUNDER: "🚀",
    SpeakerKind.OPERATOR: "⚙️",
    SpeakerKind.ACADEMIC: "🎓",
    SpeakerKind.REGULATOR: "⚖️",
}


_TIER_EMOJI: dict[SponsorTier, str] = {
    SponsorTier.PLATINUM: "💎",
    SponsorTier.GOLD: "🥇",
    SponsorTier.SILVER: "🥈",
    SponsorTier.BRONZE: "🥉",
}


def render_speaker(speaker: Speaker) -> str:
    """Format a speaker for ops display.

    No-secret-leak: never includes contact emails / phone — those
    aren't on the dataclass.
    """

    status_emoji = _STATUS_EMOJI[speaker.status]
    kind_emoji = _KIND_EMOJI[speaker.kind]
    return (
        f"{status_emoji}{kind_emoji} {speaker.display_handle} "
        f"({speaker.kind.value}) — {speaker.status.value}"
    )


def render_session(session: Session) -> str:
    """Format a session for ops display."""

    speakers_str = ", ".join(sorted(session.speaker_ids))
    return (
        f"📅 {session.title}\n"
        f"  room: {session.room}\n"
        f"  time: {session.starts_at.isoformat()} → {session.ends_at.isoformat()}\n"
        f"  speakers: {speakers_str}"
    )


def render_sponsor(sponsor: Sponsor) -> str:
    """Format a sponsor for ops display.

    No-secret-leak: never includes invoice amount / contact email.
    """

    emoji = _TIER_EMOJI[sponsor.tier]
    return f"{emoji} {sponsor.display_name} ({sponsor.tier.value})"


__all__ = [
    "Session",
    "SessionConflictError",
    "Speaker",
    "SpeakerKind",
    "SpeakerStatus",
    "SpeakerTransitionError",
    "Sponsor",
    "SponsorTier",
    "accept_invitation",
    "assert_no_conflict",
    "confirm_speaker",
    "decline_invitation",
    "invite_speaker",
    "is_print_safe",
    "render_session",
    "render_speaker",
    "render_sponsor",
    "speaker_kind_balance",
    "sponsors_at_or_above",
    "tier_outranks",
    "withdraw_speaker",
]
