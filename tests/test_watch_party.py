"""Tests for community/watch_party.py — Round-5 Wave 17.G."""

from __future__ import annotations

from datetime import datetime

import pytest

from halal_trader.community.watch_party import (
    EventStatus,
    RSVPRecord,
    RSVPStatus,
    WatchParty,
    cancel_party,
    current_rsvped_count,
    decline_rsvp,
    end_party,
    go_live,
    mark_attended,
    render_party,
    rsvp,
    schedule_party,
)


def _always_halal(_: str) -> bool:
    return True


def _party(
    party_id: str = "WP1",
    ticker: str = "AAPL",
    host_id: str = "host-alice",
    title: str = "AAPL Q2 Earnings Watch",
    starts_at: datetime = datetime(2026, 7, 30, 16, 0),
    ends_at: datetime = datetime(2026, 7, 30, 18, 0),
    capacity: int = 50,
    status: EventStatus = EventStatus.SCHEDULED,
    cancelled_reason: str = "",
) -> WatchParty:
    return WatchParty(
        party_id=party_id,
        ticker=ticker,
        host_id=host_id,
        title=title,
        starts_at=starts_at,
        ends_at=ends_at,
        capacity=capacity,
        status=status,
        cancelled_reason=cancelled_reason,
    )


# --- WatchParty validation ----------------------------


def test_party_valid():
    p = _party()
    assert p.status is EventStatus.SCHEDULED


def test_party_empty_id_rejected():
    with pytest.raises(ValueError):
        _party(party_id="")


def test_party_empty_ticker_rejected():
    with pytest.raises(ValueError):
        _party(ticker=" ")


def test_party_long_title_rejected():
    with pytest.raises(ValueError):
        _party(title="x" * 300)


def test_party_ends_before_starts_rejected():
    with pytest.raises(ValueError):
        _party(
            starts_at=datetime(2026, 7, 30, 18, 0),
            ends_at=datetime(2026, 7, 30, 16, 0),
        )


def test_party_zero_capacity_rejected():
    with pytest.raises(ValueError):
        _party(capacity=0)


def test_party_excessive_capacity_rejected():
    with pytest.raises(ValueError):
        _party(capacity=20_000)


def test_party_cancelled_without_reason_rejected():
    with pytest.raises(ValueError):
        _party(status=EventStatus.CANCELLED, cancelled_reason="")


def test_party_reason_on_non_cancelled_rejected():
    with pytest.raises(ValueError):
        _party(status=EventStatus.SCHEDULED, cancelled_reason="oops")


def test_party_immutable():
    p = _party()
    with pytest.raises(AttributeError):
        p.capacity = 100  # type: ignore[misc]


# --- schedule_party ----------------------------------


def test_schedule_clean():
    p = schedule_party(
        party_id="WP1",
        ticker="AAPL",
        host_id="alice",
        title="Q2",
        starts_at=datetime(2026, 7, 30, 16, 0),
        ends_at=datetime(2026, 7, 30, 18, 0),
        capacity=50,
        is_ticker_halal=_always_halal,
    )
    assert p.status is EventStatus.SCHEDULED


def test_schedule_haram_ticker_rejected():
    with pytest.raises(ValueError):
        schedule_party(
            party_id="WP1",
            ticker="MO",
            host_id="alice",
            title="Q2",
            starts_at=datetime(2026, 7, 30, 16, 0),
            ends_at=datetime(2026, 7, 30, 18, 0),
            capacity=50,
            is_ticker_halal=lambda t: t != "MO",
        )


# --- FSM transitions ---------------------------------


def test_go_live_from_scheduled():
    p = _party()
    p2 = go_live(p)
    assert p2.status is EventStatus.LIVE


def test_go_live_from_live_rejected():
    p = go_live(_party())
    with pytest.raises(ValueError):
        go_live(p)


def test_end_from_live():
    p = go_live(_party())
    p2 = end_party(p)
    assert p2.status is EventStatus.ENDED


def test_end_from_scheduled_rejected():
    p = _party()
    with pytest.raises(ValueError):
        end_party(p)


def test_ended_terminal():
    p = end_party(go_live(_party()))
    with pytest.raises(ValueError):
        go_live(p)


def test_cancel_from_scheduled():
    p = _party()
    c = cancel_party(p, reason="host unavailable")
    assert c.status is EventStatus.CANCELLED
    assert c.cancelled_reason == "host unavailable"


def test_cancel_from_live():
    p = go_live(_party())
    c = cancel_party(p, reason="technical failure")
    assert c.status is EventStatus.CANCELLED


def test_cancel_from_ended_rejected():
    p = end_party(go_live(_party()))
    with pytest.raises(ValueError):
        cancel_party(p, reason="why")


def test_cancel_empty_reason_rejected():
    p = _party()
    with pytest.raises(ValueError):
        cancel_party(p, reason=" ")


def test_cancel_long_reason_rejected():
    p = _party()
    with pytest.raises(ValueError):
        cancel_party(p, reason="x" * 600)


# --- RSVPRecord validation --------------------------


def test_rsvp_record_valid():
    r = RSVPRecord(
        rsvp_id="R1",
        party_id="WP1",
        user_id="bob",
        status=RSVPStatus.RSVPED,
        rsvped_at=datetime(2026, 7, 20),
    )
    assert r.status is RSVPStatus.RSVPED


def test_rsvp_record_rsvped_without_date_rejected():
    with pytest.raises(ValueError):
        RSVPRecord(
            rsvp_id="R1",
            party_id="WP1",
            user_id="bob",
            status=RSVPStatus.RSVPED,
            rsvped_at=None,
        )


def test_rsvp_record_attended_without_date_rejected():
    with pytest.raises(ValueError):
        RSVPRecord(
            rsvp_id="R1",
            party_id="WP1",
            user_id="bob",
            status=RSVPStatus.ATTENDED,
            rsvped_at=datetime(2026, 7, 20),
            attended_at=None,
        )


# --- rsvp + capacity --------------------------------


def test_rsvp_basic():
    p = _party()
    r = rsvp(
        p,
        [],
        rsvp_id="R1",
        user_id="bob",
        rsvped_at=datetime(2026, 7, 20),
    )
    assert r.status is RSVPStatus.RSVPED


def test_rsvp_capacity_enforced():
    p = _party(capacity=2)
    rsvps = [
        rsvp(p, [], rsvp_id="R1", user_id="bob", rsvped_at=datetime(2026, 7, 20)),
    ]
    rsvps.append(rsvp(p, rsvps, rsvp_id="R2", user_id="charlie", rsvped_at=datetime(2026, 7, 21)))
    with pytest.raises(ValueError):
        rsvp(p, rsvps, rsvp_id="R3", user_id="dave", rsvped_at=datetime(2026, 7, 22))


def test_rsvp_duplicate_user_rejected():
    p = _party()
    rsvps = [
        rsvp(p, [], rsvp_id="R1", user_id="bob", rsvped_at=datetime(2026, 7, 20)),
    ]
    with pytest.raises(ValueError):
        rsvp(p, rsvps, rsvp_id="R2", user_id="bob", rsvped_at=datetime(2026, 7, 21))


def test_rsvp_to_cancelled_rejected():
    p = cancel_party(_party(), reason="host out")
    with pytest.raises(ValueError):
        rsvp(p, [], rsvp_id="R1", user_id="bob", rsvped_at=datetime(2026, 7, 20))


def test_rsvp_to_ended_rejected():
    p = end_party(go_live(_party()))
    with pytest.raises(ValueError):
        rsvp(p, [], rsvp_id="R1", user_id="bob", rsvped_at=datetime(2026, 7, 20))


def test_rsvp_to_live_allowed():
    p = go_live(_party())
    r = rsvp(p, [], rsvp_id="R1", user_id="bob", rsvped_at=datetime(2026, 7, 30, 16, 5))
    assert r.status is RSVPStatus.RSVPED


def test_rsvp_declined_user_can_re_rsvp():
    """A user who DECLINED can later RSVP again."""
    p = _party()
    first = rsvp(p, [], rsvp_id="R1", user_id="bob", rsvped_at=datetime(2026, 7, 20))
    declined = decline_rsvp(first)
    second = rsvp(p, [declined], rsvp_id="R2", user_id="bob", rsvped_at=datetime(2026, 7, 22))
    assert second.status is RSVPStatus.RSVPED


# --- current_rsvped_count ---------------------------


def test_count_includes_attended():
    records = [
        RSVPRecord(
            rsvp_id="R1",
            party_id="WP1",
            user_id="bob",
            status=RSVPStatus.RSVPED,
            rsvped_at=datetime(2026, 7, 20),
        ),
        RSVPRecord(
            rsvp_id="R2",
            party_id="WP1",
            user_id="charlie",
            status=RSVPStatus.ATTENDED,
            rsvped_at=datetime(2026, 7, 20),
            attended_at=datetime(2026, 7, 30),
        ),
        RSVPRecord(
            rsvp_id="R3",
            party_id="WP1",
            user_id="dave",
            status=RSVPStatus.DECLINED,
        ),
    ]
    assert current_rsvped_count(records) == 2


# --- mark_attended + decline_rsvp -----------------


def test_mark_attended_from_rsvped():
    r = RSVPRecord(
        rsvp_id="R1",
        party_id="WP1",
        user_id="bob",
        status=RSVPStatus.RSVPED,
        rsvped_at=datetime(2026, 7, 20),
    )
    r2 = mark_attended(r, at=datetime(2026, 7, 30, 16, 30))
    assert r2.status is RSVPStatus.ATTENDED


def test_mark_attended_from_invited_rejected():
    r = RSVPRecord(rsvp_id="R1", party_id="WP1", user_id="bob")
    with pytest.raises(ValueError):
        mark_attended(r, at=datetime(2026, 7, 30))


def test_decline_from_invited():
    r = RSVPRecord(rsvp_id="R1", party_id="WP1", user_id="bob")
    r2 = decline_rsvp(r)
    assert r2.status is RSVPStatus.DECLINED


def test_decline_from_attended_rejected():
    r = RSVPRecord(
        rsvp_id="R1",
        party_id="WP1",
        user_id="bob",
        status=RSVPStatus.ATTENDED,
        rsvped_at=datetime(2026, 7, 20),
        attended_at=datetime(2026, 7, 30),
    )
    with pytest.raises(ValueError):
        decline_rsvp(r)


# --- Render ----------------------------------------


def test_render_party_no_secret_leak():
    p = _party(host_id="alice@example.com")
    out = render_party(p)
    assert "alice@example.com" not in out


def test_render_party_status_emoji():
    p = _party()
    out = render_party(p)
    assert "📅" in out


def test_render_party_with_rsvp_count():
    p = _party()
    out = render_party(p, n_rsvps=12)
    assert "RSVPed 12" in out


def test_render_party_cancelled():
    p = cancel_party(_party(), reason="host out")
    out = render_party(p)
    assert "Cancelled" in out
    assert "host out" in out
