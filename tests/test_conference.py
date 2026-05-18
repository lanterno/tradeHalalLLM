"""Tests for `halal_trader.web.conference` (Wave 10.E).

Covers: speaker invitation lifecycle, session-conflict detection
(speaker + room), sponsor tier ladder, no-secret render.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.web.conference import (
    Session,
    SessionConflictError,
    Speaker,
    SpeakerKind,
    SpeakerStatus,
    SpeakerTransitionError,
    Sponsor,
    SponsorTier,
    accept_invitation,
    assert_no_conflict,
    confirm_speaker,
    decline_invitation,
    invite_speaker,
    is_print_safe,
    render_session,
    render_speaker,
    render_sponsor,
    speaker_kind_balance,
    sponsors_at_or_above,
    tier_outranks,
    withdraw_speaker,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_speaker_status_string_values_pinned() -> None:
    assert SpeakerStatus.INVITED.value == "invited"
    assert SpeakerStatus.ACCEPTED.value == "accepted"
    assert SpeakerStatus.DECLINED.value == "declined"
    assert SpeakerStatus.CONFIRMED.value == "confirmed"
    assert SpeakerStatus.WITHDRAWN.value == "withdrawn"


def test_sponsor_tier_string_values_pinned() -> None:
    assert SponsorTier.PLATINUM.value == "platinum"
    assert SponsorTier.GOLD.value == "gold"
    assert SponsorTier.SILVER.value == "silver"
    assert SponsorTier.BRONZE.value == "bronze"


def test_speaker_kind_string_values_pinned() -> None:
    assert SpeakerKind.SCHOLAR.value == "scholar"
    assert SpeakerKind.FOUNDER.value == "founder"
    assert SpeakerKind.OPERATOR.value == "operator"
    assert SpeakerKind.ACADEMIC.value == "academic"
    assert SpeakerKind.REGULATOR.value == "regulator"


# --------------------------- Speaker validation ------------------------------


def test_speaker_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="speaker_id"):
        Speaker(
            speaker_id="",
            display_handle="Mufti X",
            kind=SpeakerKind.SCHOLAR,
            status=SpeakerStatus.INVITED,
            invited_at=T0,
        )


def test_speaker_rejects_empty_handle() -> None:
    with pytest.raises(ValueError, match="display_handle"):
        Speaker(
            speaker_id="sp1",
            display_handle="",
            kind=SpeakerKind.SCHOLAR,
            status=SpeakerStatus.INVITED,
            invited_at=T0,
        )


def test_speaker_rejects_naive_invited_at() -> None:
    with pytest.raises(ValueError, match="invited_at"):
        Speaker(
            speaker_id="sp1",
            display_handle="Mufti X",
            kind=SpeakerKind.SCHOLAR,
            status=SpeakerStatus.INVITED,
            invited_at=datetime(2026, 5, 1),
        )


def test_speaker_invited_must_not_have_decided_at() -> None:
    """Pin: INVITED status implies decided_at is None."""

    with pytest.raises(ValueError, match="decided_at"):
        Speaker(
            speaker_id="sp1",
            display_handle="Mufti X",
            kind=SpeakerKind.SCHOLAR,
            status=SpeakerStatus.INVITED,
            invited_at=T0,
            decided_at=T0,
        )


def test_speaker_non_invited_requires_decided_at() -> None:
    """Pin: ACCEPTED / DECLINED / CONFIRMED require decided_at."""

    with pytest.raises(ValueError, match="decided_at"):
        Speaker(
            speaker_id="sp1",
            display_handle="Mufti X",
            kind=SpeakerKind.SCHOLAR,
            status=SpeakerStatus.CONFIRMED,
            invited_at=T0,
            decided_at=None,
        )


def test_speaker_decided_before_invited_rejected() -> None:
    with pytest.raises(ValueError, match="decided_at"):
        Speaker(
            speaker_id="sp1",
            display_handle="Mufti X",
            kind=SpeakerKind.SCHOLAR,
            status=SpeakerStatus.ACCEPTED,
            invited_at=T0,
            decided_at=T0 - timedelta(days=1),
        )


def test_speaker_is_frozen() -> None:
    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    with pytest.raises(FrozenInstanceError):
        s.status = SpeakerStatus.ACCEPTED  # type: ignore[misc]


# --------------------------- invite_speaker ----------------------------------


def test_invite_basic() -> None:
    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    assert s.status is SpeakerStatus.INVITED
    assert s.decided_at is None


def test_invite_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="speaker_id"):
        invite_speaker(
            speaker_id="",
            display_handle="Mufti X",
            kind=SpeakerKind.SCHOLAR,
            now=T0,
        )


def test_invite_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        invite_speaker(
            speaker_id="sp1",
            display_handle="Mufti X",
            kind=SpeakerKind.SCHOLAR,
            now=datetime(2026, 5, 1),
        )


# --------------------------- accept_invitation -------------------------------


def test_accept_from_invited() -> None:
    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    s = accept_invitation(s, now=T0 + timedelta(days=2))
    assert s.status is SpeakerStatus.ACCEPTED


def test_accept_from_non_invited_rejected() -> None:
    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    s = accept_invitation(s, now=T0 + timedelta(days=2))
    with pytest.raises(SpeakerTransitionError):
        accept_invitation(s, now=T0 + timedelta(days=3))


def test_accept_naive_now_rejected() -> None:
    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    with pytest.raises(ValueError, match="now"):
        accept_invitation(s, now=datetime(2026, 5, 1))


# --------------------------- decline_invitation ------------------------------


def test_decline_from_invited() -> None:
    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    s = decline_invitation(s, now=T0 + timedelta(days=1))
    assert s.status is SpeakerStatus.DECLINED


def test_decline_from_accepted_rejected() -> None:
    """Pin: cannot decline an already-accepted invitation; must be
    explicit withdraw operation if speaker pulls out."""

    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    s = accept_invitation(s, now=T0)
    with pytest.raises(SpeakerTransitionError):
        decline_invitation(s, now=T0)


# --------------------------- confirm_speaker ---------------------------------


def test_confirm_from_accepted() -> None:
    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    s = accept_invitation(s, now=T0)
    s = confirm_speaker(s, now=T0 + timedelta(days=10))
    assert s.status is SpeakerStatus.CONFIRMED


def test_confirm_from_invited_rejected() -> None:
    """Pin: cannot skip from INVITED → CONFIRMED."""

    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    with pytest.raises(SpeakerTransitionError):
        confirm_speaker(s, now=T0)


def test_confirm_from_declined_rejected() -> None:
    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    s = decline_invitation(s, now=T0)
    with pytest.raises(SpeakerTransitionError):
        confirm_speaker(s, now=T0)


# --------------------------- withdraw_speaker --------------------------------


def test_withdraw_from_confirmed() -> None:
    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    s = accept_invitation(s, now=T0)
    s = confirm_speaker(s, now=T0)
    s = withdraw_speaker(s, now=T0 + timedelta(days=30))
    assert s.status is SpeakerStatus.WITHDRAWN


def test_withdraw_from_accepted_rejected() -> None:
    """Pin: WITHDRAWN is for confirmed-then-pulled-out only."""

    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    s = accept_invitation(s, now=T0)
    with pytest.raises(SpeakerTransitionError):
        withdraw_speaker(s, now=T0)


# --------------------------- is_print_safe -----------------------------------


def test_print_safe_only_for_confirmed() -> None:
    """Pin: only CONFIRMED is print-safe.

    INVITED / ACCEPTED / DECLINED / WITHDRAWN are NOT — printing
    a speaker who isn't confirmed could result in the speaker not
    showing up at the conference.
    """

    invited = invite_speaker(
        speaker_id="sp1",
        display_handle="x",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    assert is_print_safe(invited) is False

    accepted = accept_invitation(invited, now=T0)
    assert is_print_safe(accepted) is False  # "yes" but no slides yet

    declined = decline_invitation(invited, now=T0)
    assert is_print_safe(declined) is False

    confirmed = confirm_speaker(accepted, now=T0)
    assert is_print_safe(confirmed) is True

    withdrawn = withdraw_speaker(confirmed, now=T0)
    assert is_print_safe(withdrawn) is False


# --------------------------- Session ----------------------------------------


def _session(**overrides: object) -> Session:
    base: dict[str, object] = {
        "session_id": "ses1",
        "title": "Halal Algorithmic Trading",
        "room": "main_hall",
        "starts_at": T0,
        "ends_at": T0 + timedelta(hours=1),
        "speaker_ids": frozenset({"sp1"}),
    }
    base.update(overrides)
    return Session(**base)  # type: ignore[arg-type]


def test_session_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="session_id"):
        _session(session_id="")


def test_session_rejects_empty_title() -> None:
    with pytest.raises(ValueError, match="title"):
        _session(title="")


def test_session_rejects_empty_room() -> None:
    with pytest.raises(ValueError, match="room"):
        _session(room="")


def test_session_rejects_naive_starts_at() -> None:
    with pytest.raises(ValueError, match="starts_at"):
        _session(starts_at=datetime(2026, 5, 1))


def test_session_rejects_ends_at_before_starts() -> None:
    with pytest.raises(ValueError, match="ends_at"):
        _session(ends_at=T0 - timedelta(seconds=1))


def test_session_rejects_ends_at_equal_starts() -> None:
    with pytest.raises(ValueError, match="ends_at"):
        _session(ends_at=T0)


def test_session_rejects_empty_speakers() -> None:
    with pytest.raises(ValueError, match="speaker_ids"):
        _session(speaker_ids=frozenset())


def test_session_is_frozen() -> None:
    s = _session()
    with pytest.raises(FrozenInstanceError):
        s.title = "other"  # type: ignore[misc]


# --------------------------- assert_no_conflict ------------------------------


def test_no_conflict_with_empty_existing() -> None:
    new = _session()
    assert_no_conflict(new, [])  # no raise


def test_no_conflict_with_non_overlapping_time() -> None:
    """Two sessions in same room, different times → no conflict."""

    morning = _session(
        session_id="morning",
        starts_at=T0,
        ends_at=T0 + timedelta(hours=1),
    )
    afternoon = _session(
        session_id="afternoon",
        starts_at=T0 + timedelta(hours=2),
        ends_at=T0 + timedelta(hours=3),
    )
    assert_no_conflict(afternoon, [morning])


def test_no_conflict_with_back_to_back_sessions() -> None:
    """Pin: adjacent windows (a.end == b.start) do NOT conflict.

    Same room, back-to-back (no gap) is valid scheduling.
    """

    first = _session(
        session_id="first",
        starts_at=T0,
        ends_at=T0 + timedelta(hours=1),
    )
    second = _session(
        session_id="second",
        starts_at=T0 + timedelta(hours=1),  # exactly when first ends
        ends_at=T0 + timedelta(hours=2),
        speaker_ids=frozenset({"sp_other"}),
    )
    assert_no_conflict(second, [first])


def test_speaker_double_booked_conflict() -> None:
    """Pin: same speaker in overlapping windows → SessionConflictError."""

    track_a = _session(
        session_id="track_a_morning",
        room="track_a",
        starts_at=T0,
        ends_at=T0 + timedelta(hours=1),
        speaker_ids=frozenset({"sp1"}),
    )
    track_b = _session(
        session_id="track_b_morning",
        room="track_b",  # different room
        starts_at=T0 + timedelta(minutes=30),  # overlaps
        ends_at=T0 + timedelta(hours=1, minutes=30),
        speaker_ids=frozenset({"sp1"}),  # same speaker
    )
    with pytest.raises(SessionConflictError, match="sp1"):
        assert_no_conflict(track_b, [track_a])


def test_room_double_booked_conflict() -> None:
    """Pin: same room in overlapping windows → SessionConflictError."""

    a = _session(
        session_id="a",
        room="main_hall",
        starts_at=T0,
        ends_at=T0 + timedelta(hours=1),
        speaker_ids=frozenset({"sp1"}),
    )
    b = _session(
        session_id="b",
        room="main_hall",  # same room
        starts_at=T0 + timedelta(minutes=30),
        ends_at=T0 + timedelta(hours=1, minutes=30),
        speaker_ids=frozenset({"sp_other"}),  # different speaker
    )
    with pytest.raises(SessionConflictError, match="main_hall"):
        assert_no_conflict(b, [a])


def test_duplicate_session_id_rejected() -> None:
    """Pin: a session with the same ID as existing is a conflict."""

    existing = _session(session_id="dup")
    new = _session(
        session_id="dup",  # same ID
        room="other_room",
        starts_at=T0 + timedelta(hours=10),
        ends_at=T0 + timedelta(hours=11),
        speaker_ids=frozenset({"sp_other"}),
    )
    with pytest.raises(SessionConflictError, match="already exists"):
        assert_no_conflict(new, [existing])


def test_panel_with_multiple_speakers_one_conflict() -> None:
    """Pin: a session with multiple speakers conflicts if ANY speaker
    is double-booked (not just primary)."""

    panel = _session(
        session_id="panel",
        speaker_ids=frozenset({"sp1", "sp2", "sp3"}),
        room="panel_hall",
    )
    other = _session(
        session_id="other",
        room="track_x",
        starts_at=T0 + timedelta(minutes=30),
        ends_at=T0 + timedelta(hours=1, minutes=30),
        speaker_ids=frozenset({"sp2"}),  # member of panel
    )
    with pytest.raises(SessionConflictError, match="sp2"):
        assert_no_conflict(other, [panel])


# --------------------------- Sponsor + tier ladder ---------------------------


def test_sponsor_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="sponsor_id"):
        Sponsor(
            sponsor_id="",
            display_name="Foo Capital",
            tier=SponsorTier.GOLD,
            onboarded_at=T0,
        )


def test_sponsor_rejects_empty_display_name() -> None:
    with pytest.raises(ValueError, match="display_name"):
        Sponsor(
            sponsor_id="sp1",
            display_name="",
            tier=SponsorTier.GOLD,
            onboarded_at=T0,
        )


def test_sponsor_rejects_naive_onboarded_at() -> None:
    with pytest.raises(ValueError, match="onboarded_at"):
        Sponsor(
            sponsor_id="sp1",
            display_name="Foo",
            tier=SponsorTier.GOLD,
            onboarded_at=datetime(2026, 5, 1),
        )


def test_sponsor_is_frozen() -> None:
    sp = Sponsor(
        sponsor_id="sp1",
        display_name="Foo",
        tier=SponsorTier.GOLD,
        onboarded_at=T0,
    )
    with pytest.raises(FrozenInstanceError):
        sp.tier = SponsorTier.PLATINUM  # type: ignore[misc]


def test_tier_outranks_platinum_above_gold() -> None:
    assert tier_outranks(SponsorTier.PLATINUM, SponsorTier.GOLD) is True
    assert tier_outranks(SponsorTier.GOLD, SponsorTier.PLATINUM) is False


def test_tier_outranks_gold_above_silver() -> None:
    assert tier_outranks(SponsorTier.GOLD, SponsorTier.SILVER) is True


def test_tier_outranks_strict() -> None:
    """Pin: tier_outranks is STRICTLY greater (same tier returns False)."""

    assert tier_outranks(SponsorTier.GOLD, SponsorTier.GOLD) is False


def test_sponsors_at_or_above_filters_correctly() -> None:
    sponsors = [
        Sponsor(
            sponsor_id="s1",
            display_name="Plat",
            tier=SponsorTier.PLATINUM,
            onboarded_at=T0,
        ),
        Sponsor(
            sponsor_id="s2",
            display_name="Gold",
            tier=SponsorTier.GOLD,
            onboarded_at=T0,
        ),
        Sponsor(
            sponsor_id="s3",
            display_name="Silver",
            tier=SponsorTier.SILVER,
            onboarded_at=T0,
        ),
        Sponsor(
            sponsor_id="s4",
            display_name="Bronze",
            tier=SponsorTier.BRONZE,
            onboarded_at=T0,
        ),
    ]
    qualified = sponsors_at_or_above(sponsors, minimum=SponsorTier.GOLD)
    ids = {s.sponsor_id for s in qualified}
    assert ids == {"s1", "s2"}


def test_sponsors_at_or_above_returns_sorted_descending() -> None:
    """Pin: sponsors_at_or_above returns highest tier first."""

    sponsors = [
        Sponsor(
            sponsor_id="s_silver",
            display_name="Silver",
            tier=SponsorTier.SILVER,
            onboarded_at=T0,
        ),
        Sponsor(
            sponsor_id="s_plat",
            display_name="Plat",
            tier=SponsorTier.PLATINUM,
            onboarded_at=T0,
        ),
        Sponsor(
            sponsor_id="s_gold",
            display_name="Gold",
            tier=SponsorTier.GOLD,
            onboarded_at=T0,
        ),
    ]
    qualified = sponsors_at_or_above(sponsors, minimum=SponsorTier.SILVER)
    tiers = [s.tier for s in qualified]
    assert tiers == [SponsorTier.PLATINUM, SponsorTier.GOLD, SponsorTier.SILVER]


# --------------------------- speaker_kind_balance ----------------------------


def test_kind_balance_only_counts_confirmed() -> None:
    """Pin: balance counts CONFIRMED speakers only.

    INVITED / ACCEPTED / DECLINED don't count — operators want to
    know who's actually speaking, not who was invited.
    """

    speakers = [
        # Confirmed scholar
        confirm_speaker(
            accept_invitation(
                invite_speaker(
                    speaker_id="sp_sch",
                    display_handle="Mufti",
                    kind=SpeakerKind.SCHOLAR,
                    now=T0,
                ),
                now=T0,
            ),
            now=T0,
        ),
        # Just-invited founder (doesn't count yet)
        invite_speaker(
            speaker_id="sp_fnd",
            display_handle="Founder X",
            kind=SpeakerKind.FOUNDER,
            now=T0,
        ),
        # Declined regulator (definitely doesn't count)
        decline_invitation(
            invite_speaker(
                speaker_id="sp_reg",
                display_handle="Reg Y",
                kind=SpeakerKind.REGULATOR,
                now=T0,
            ),
            now=T0,
        ),
    ]
    balance = speaker_kind_balance(speakers)
    assert balance[SpeakerKind.SCHOLAR] == 1
    assert balance[SpeakerKind.FOUNDER] == 0
    assert balance[SpeakerKind.REGULATOR] == 0


def test_kind_balance_includes_all_kinds() -> None:
    """Pin: balance dict has an entry for every kind, even zero."""

    balance = speaker_kind_balance([])
    for kind in SpeakerKind:
        assert kind in balance
        assert balance[kind] == 0


# --------------------------- render ------------------------------------------


def test_render_speaker_emoji_per_kind() -> None:
    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    out = render_speaker(s)
    assert "📚" in out  # scholar emoji
    assert "📧" in out  # invited emoji
    assert "Mufti X" in out


def test_render_speaker_emoji_per_status() -> None:
    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    s = accept_invitation(s, now=T0)
    s = confirm_speaker(s, now=T0)
    out = render_speaker(s)
    assert "🎤" in out  # confirmed


def test_render_speaker_no_secret_leak() -> None:
    """Pin: render never includes contact email / phone — those
    aren't on the dataclass at all."""

    s = invite_speaker(
        speaker_id="sp1",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    out = render_speaker(s)
    assert "@" not in out  # no email-shape
    assert "phone" not in out.lower()
    assert "+1" not in out


def test_render_session_includes_title_and_room() -> None:
    s = _session()
    out = render_session(s)
    assert "Halal Algorithmic Trading" in out
    assert "main_hall" in out


def test_render_sponsor_emoji_per_tier() -> None:
    sponsor = Sponsor(
        sponsor_id="s1",
        display_name="Foo Capital",
        tier=SponsorTier.PLATINUM,
        onboarded_at=T0,
    )
    out = render_sponsor(sponsor)
    assert "💎" in out  # platinum
    assert "Foo Capital" in out


def test_render_sponsor_no_secret_leak() -> None:
    """Pin: render never includes invoice amount / contact email."""

    sponsor = Sponsor(
        sponsor_id="s1",
        display_name="Foo Capital",
        tier=SponsorTier.PLATINUM,
        onboarded_at=T0,
    )
    out = render_sponsor(sponsor)
    assert "$" not in out
    assert "USD" not in out
    assert "invoice" not in out.lower()
    assert "@" not in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_speaker_full_lifecycle() -> None:
    """Real-world: scholar invitation → accepted → confirmed → safe to print."""

    s = invite_speaker(
        speaker_id="sp_mufti",
        display_handle="Mufti Faraz Adam",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    assert is_print_safe(s) is False
    s = accept_invitation(s, now=T0 + timedelta(days=14))
    assert is_print_safe(s) is False  # still need confirmation
    s = confirm_speaker(s, now=T0 + timedelta(days=60))
    assert is_print_safe(s) is True


def test_e2e_speaker_pulls_out_after_confirmation() -> None:
    """Real-world: confirmed speaker withdraws 1 month before conference."""

    s = invite_speaker(
        speaker_id="sp_mufti",
        display_handle="Mufti X",
        kind=SpeakerKind.SCHOLAR,
        now=T0,
    )
    s = accept_invitation(s, now=T0)
    s = confirm_speaker(s, now=T0)
    s = withdraw_speaker(s, now=T0 + timedelta(days=180))
    assert s.status is SpeakerStatus.WITHDRAWN
    assert is_print_safe(s) is False  # must be removed from program


def test_e2e_no_double_booking_in_full_schedule() -> None:
    """Real-world: build a 4-session schedule, verify no conflicts."""

    sessions: list[Session] = []
    # 9-10am main hall: opening keynote
    sessions.append(
        Session(
            session_id="opening",
            title="Opening Keynote",
            room="main_hall",
            starts_at=T0,
            ends_at=T0 + timedelta(hours=1),
            speaker_ids=frozenset({"sp_keynote"}),
        )
    )
    # 10-11am main hall: scholar panel (back-to-back, no conflict)
    new_session = Session(
        session_id="panel",
        title="Scholar Panel",
        room="main_hall",
        starts_at=T0 + timedelta(hours=1),  # exactly when keynote ends
        ends_at=T0 + timedelta(hours=2),
        speaker_ids=frozenset({"sp_mufti1", "sp_mufti2"}),
    )
    assert_no_conflict(new_session, sessions)
    sessions.append(new_session)

    # 11-12 track A: founder talk (different room, different speaker)
    new_session = Session(
        session_id="founder_track",
        title="Founder Track",
        room="track_a",
        starts_at=T0 + timedelta(hours=2),
        ends_at=T0 + timedelta(hours=3),
        speaker_ids=frozenset({"sp_founder"}),
    )
    assert_no_conflict(new_session, sessions)
    sessions.append(new_session)


def test_e2e_speaker_double_booked_caught() -> None:
    """Pin: planning system catches the obvious double-booking error."""

    sessions = [
        Session(
            session_id="track_a",
            title="Track A",
            room="track_a",
            starts_at=T0,
            ends_at=T0 + timedelta(hours=1),
            speaker_ids=frozenset({"sp_mufti"}),
        ),
    ]
    bad_session = Session(
        session_id="track_b",
        title="Track B",
        room="track_b",
        starts_at=T0 + timedelta(minutes=30),
        ends_at=T0 + timedelta(hours=1, minutes=30),
        speaker_ids=frozenset({"sp_mufti"}),  # same speaker
    )
    with pytest.raises(SessionConflictError, match="sp_mufti"):
        assert_no_conflict(bad_session, sessions)


def test_e2e_replay_consistency() -> None:
    """Same operations produce equal speaker states."""

    def build() -> Speaker:
        s = invite_speaker(
            speaker_id="sp1",
            display_handle="Mufti X",
            kind=SpeakerKind.SCHOLAR,
            now=T0,
        )
        s = accept_invitation(s, now=T0)
        return confirm_speaker(s, now=T0)

    a = build()
    b = build()
    assert a == b
