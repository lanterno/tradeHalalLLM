"""Earnings watch party coordinator — Round-5 Wave 17.G.

Live community events around halal-ticker quarterly earnings:
operator schedules an event, users RSVP, capacity is enforced, the
event lifecycle is tracked. Audio/video transport is out of scope —
this module handles **scheduling + RSVP + capacity + lifecycle**.

Pinned semantics:

- **Closed-set EventStatus FSM** — SCHEDULED → LIVE → ENDED, with
  CANCELLED as alternate terminal from SCHEDULED or LIVE.
- **Closed-set RSVPStatus FSM** — INVITED → RSVPED → ATTENDED, with
  DECLINED as alternate terminal from INVITED or RSVPED.
- **Capacity enforced at RSVP time**; over-capacity → raise.
- **Ticker must be on the operator's halal list** (caller predicate).
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum


class EventStatus(str, Enum):
    """Closed-set event FSM ladder."""

    SCHEDULED = "scheduled"
    LIVE = "live"
    ENDED = "ended"
    CANCELLED = "cancelled"


class RSVPStatus(str, Enum):
    """Closed-set RSVP FSM ladder."""

    INVITED = "invited"
    RSVPED = "rsvped"
    DECLINED = "declined"
    ATTENDED = "attended"


@dataclass(frozen=True)
class WatchParty:
    """One scheduled watch party."""

    party_id: str
    ticker: str
    host_id: str
    title: str
    starts_at: datetime
    ends_at: datetime
    capacity: int
    status: EventStatus = EventStatus.SCHEDULED
    cancelled_reason: str = ""

    def __post_init__(self) -> None:
        if not self.party_id or not self.party_id.strip():
            raise ValueError("party_id must be non-empty")
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        if not self.host_id or not self.host_id.strip():
            raise ValueError("host_id must be non-empty")
        if not self.title.strip():
            raise ValueError("title must be non-empty")
        if len(self.title) > 200:
            raise ValueError("title must be ≤ 200 chars")
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be > starts_at")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.capacity > 10_000:
            raise ValueError("capacity > 10_000 suspicious")
        if self.status is EventStatus.CANCELLED and not self.cancelled_reason.strip():
            raise ValueError("CANCELLED requires cancelled_reason")
        if self.status is not EventStatus.CANCELLED and self.cancelled_reason.strip():
            raise ValueError("cancelled_reason only set when CANCELLED")


@dataclass(frozen=True)
class RSVPRecord:
    """One user's RSVP record for a party."""

    rsvp_id: str
    party_id: str
    user_id: str
    status: RSVPStatus = RSVPStatus.INVITED
    rsvped_at: datetime | None = None
    attended_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.rsvp_id or not self.rsvp_id.strip():
            raise ValueError("rsvp_id must be non-empty")
        if not self.party_id or not self.party_id.strip():
            raise ValueError("party_id must be non-empty")
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if self.status is RSVPStatus.RSVPED and self.rsvped_at is None:
            raise ValueError("RSVPED requires rsvped_at")
        if self.status is RSVPStatus.ATTENDED and self.attended_at is None:
            raise ValueError("ATTENDED requires attended_at")


def schedule_party(
    *,
    party_id: str,
    ticker: str,
    host_id: str,
    title: str,
    starts_at: datetime,
    ends_at: datetime,
    capacity: int,
    is_ticker_halal: Callable[[str], bool],
) -> WatchParty:
    """Construct + validate a SCHEDULED party.

    Pinned: rejects if ticker fails the halal screen predicate.
    """
    if not is_ticker_halal(ticker):
        raise ValueError(f"ticker {ticker} is not halal-compliant")
    return WatchParty(
        party_id=party_id,
        ticker=ticker,
        host_id=host_id,
        title=title,
        starts_at=starts_at,
        ends_at=ends_at,
        capacity=capacity,
    )


_LEGAL_PARTY_TRANSITIONS: dict[EventStatus, set[EventStatus]] = {
    EventStatus.SCHEDULED: {EventStatus.LIVE, EventStatus.CANCELLED},
    EventStatus.LIVE: {EventStatus.ENDED, EventStatus.CANCELLED},
    EventStatus.ENDED: set(),
    EventStatus.CANCELLED: set(),
}


def go_live(party: WatchParty) -> WatchParty:
    if party.status is not EventStatus.SCHEDULED:
        raise ValueError(f"go_live illegal from {party.status.value}")
    return replace(party, status=EventStatus.LIVE)


def end_party(party: WatchParty) -> WatchParty:
    if party.status is not EventStatus.LIVE:
        raise ValueError(f"end_party illegal from {party.status.value}")
    return replace(party, status=EventStatus.ENDED)


def cancel_party(party: WatchParty, *, reason: str) -> WatchParty:
    if EventStatus.CANCELLED not in _LEGAL_PARTY_TRANSITIONS[party.status]:
        raise ValueError(f"cancel_party illegal from {party.status.value}")
    if not reason.strip():
        raise ValueError("reason must be non-empty")
    if len(reason) > 500:
        raise ValueError("reason ≤ 500 chars")
    return replace(party, status=EventStatus.CANCELLED, cancelled_reason=reason)


# --- RSVP -----------------------------------------------------


_LEGAL_RSVP_TRANSITIONS: dict[RSVPStatus, set[RSVPStatus]] = {
    RSVPStatus.INVITED: {RSVPStatus.RSVPED, RSVPStatus.DECLINED},
    RSVPStatus.RSVPED: {RSVPStatus.ATTENDED, RSVPStatus.DECLINED},
    RSVPStatus.DECLINED: set(),
    RSVPStatus.ATTENDED: set(),
}


def current_rsvped_count(rsvps: Iterable[RSVPRecord]) -> int:
    """Count users whose status is RSVPED or ATTENDED for that party."""
    return sum(1 for r in rsvps if r.status in (RSVPStatus.RSVPED, RSVPStatus.ATTENDED))


def rsvp(
    party: WatchParty,
    rsvps: Iterable[RSVPRecord],
    *,
    rsvp_id: str,
    user_id: str,
    rsvped_at: datetime,
) -> RSVPRecord:
    """Create a new RSVPED record after capacity + status checks.

    Pinned:
    - Party must be SCHEDULED or LIVE.
    - Capacity not exceeded post-add.
    - User cannot already have an active RSVP for this party.
    """
    if party.status not in (EventStatus.SCHEDULED, EventStatus.LIVE):
        raise ValueError(f"cannot rsvp to a {party.status.value} party")
    rsvps_t = tuple(rsvps)
    party_rsvps = [r for r in rsvps_t if r.party_id == party.party_id]
    if any(r.user_id == user_id and r.status != RSVPStatus.DECLINED for r in party_rsvps):
        raise ValueError(f"user {user_id} already RSVPed/invited")
    current = current_rsvped_count(party_rsvps)
    if current + 1 > party.capacity:
        raise ValueError(f"capacity {party.capacity} exceeded")
    return RSVPRecord(
        rsvp_id=rsvp_id,
        party_id=party.party_id,
        user_id=user_id,
        status=RSVPStatus.RSVPED,
        rsvped_at=rsvped_at,
    )


def mark_attended(record: RSVPRecord, *, at: datetime) -> RSVPRecord:
    if RSVPStatus.ATTENDED not in _LEGAL_RSVP_TRANSITIONS[record.status]:
        raise ValueError(f"mark_attended illegal from {record.status.value}")
    return replace(record, status=RSVPStatus.ATTENDED, attended_at=at)


def decline_rsvp(record: RSVPRecord) -> RSVPRecord:
    if RSVPStatus.DECLINED not in _LEGAL_RSVP_TRANSITIONS[record.status]:
        raise ValueError(f"decline_rsvp illegal from {record.status.value}")
    return replace(record, status=RSVPStatus.DECLINED)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_STATUS_EMOJI: dict[EventStatus, str] = {
    EventStatus.SCHEDULED: "📅",
    EventStatus.LIVE: "🔴",
    EventStatus.ENDED: "✅",
    EventStatus.CANCELLED: "🚫",
}


def render_party(party: WatchParty, *, n_rsvps: int | None = None) -> str:
    head = (
        f"{_STATUS_EMOJI[party.status]} {party.party_id} "
        f"[{party.status.value}] {party.ticker}: {party.title}\n"
        f"  Host: {_mask(party.host_id)} | "
        f"{party.starts_at.isoformat()} → {party.ends_at.isoformat()} | "
        f"capacity {party.capacity}"
    )
    if n_rsvps is not None:
        head += f" | RSVPed {n_rsvps}"
    if party.status is EventStatus.CANCELLED:
        head += f"\n  Cancelled: {party.cancelled_reason}"
    return head
