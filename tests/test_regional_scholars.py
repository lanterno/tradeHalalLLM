"""Tests for halal/regional_scholars.py — Round-5 Wave 23.J."""

from __future__ import annotations

import pytest

from halal_trader.halal.regional_scholars import (
    Madhhab,
    Region,
    Scholar,
    ScholarStatus,
    UserProfile,
    has_madhhab_compatible,
    render_routing,
    render_scholar,
    route_user,
    scholars_for_region,
    transition_status,
)


def _scholar(
    scholar_id: str = "sch-1",
    display_name: str = "Imam Ali",
    primary_region: Region = Region.GULF,
    alternative_regions: tuple[Region, ...] = (),
    madhhabs: tuple[Madhhab, ...] = (Madhhab.HANBALI,),
    is_global: bool = False,
    status: ScholarStatus = ScholarStatus.ACTIVE,
    bio_summary: str = "",
) -> Scholar:
    return Scholar(
        scholar_id=scholar_id,
        display_name=display_name,
        primary_region=primary_region,
        alternative_regions=alternative_regions,
        madhhabs=madhhabs,
        is_global=is_global,
        status=status,
        bio_summary=bio_summary,
    )


def _user(
    user_id: str = "alice",
    region: Region = Region.GULF,
    preferred_madhhab: Madhhab | None = None,
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        region=region,
        preferred_madhhab=preferred_madhhab,
    )


# --- Scholar validation -----------------------


def test_scholar_valid():
    s = _scholar()
    assert s.primary_region is Region.GULF


def test_scholar_empty_id_rejected():
    with pytest.raises(ValueError):
        _scholar(scholar_id="")


def test_scholar_long_name_rejected():
    with pytest.raises(ValueError):
        _scholar(display_name="x" * 300)


def test_scholar_primary_in_alternatives_rejected():
    with pytest.raises(ValueError):
        _scholar(
            primary_region=Region.GULF,
            alternative_regions=(Region.GULF, Region.LEVANT),
        )


def test_scholar_duplicate_alternative_rejected():
    with pytest.raises(ValueError):
        _scholar(alternative_regions=(Region.LEVANT, Region.LEVANT))


def test_scholar_duplicate_madhhab_rejected():
    with pytest.raises(ValueError):
        _scholar(madhhabs=(Madhhab.HANAFI, Madhhab.HANAFI))


def test_scholar_long_bio_rejected():
    with pytest.raises(ValueError):
        _scholar(bio_summary="x" * 1500)


def test_scholar_immutable():
    s = _scholar()
    with pytest.raises(AttributeError):
        s.status = ScholarStatus.INACTIVE  # type: ignore[misc]


def test_scholar_serves_primary():
    s = _scholar(primary_region=Region.GULF)
    assert s.serves(Region.GULF)


def test_scholar_serves_alternative():
    s = _scholar(
        primary_region=Region.GULF,
        alternative_regions=(Region.LEVANT,),
    )
    assert s.serves(Region.LEVANT)


def test_scholar_does_not_serve_other():
    s = _scholar(primary_region=Region.GULF)
    assert not s.serves(Region.SOUTHEAST_ASIA)


# --- UserProfile validation -------------------


def test_user_valid():
    u = _user()
    assert u.user_id == "alice"


def test_user_empty_id_rejected():
    with pytest.raises(ValueError):
        _user(user_id="")


# --- transition_status -----------------------


def test_transition_active_to_inactive():
    s = _scholar(status=ScholarStatus.ACTIVE)
    s2 = transition_status(s, new_status=ScholarStatus.INACTIVE)
    assert s2.status is ScholarStatus.INACTIVE


def test_transition_active_to_deceased():
    s = _scholar(status=ScholarStatus.ACTIVE)
    s2 = transition_status(s, new_status=ScholarStatus.DECEASED)
    assert s2.status is ScholarStatus.DECEASED


def test_transition_inactive_to_active():
    s = _scholar(status=ScholarStatus.INACTIVE)
    s2 = transition_status(s, new_status=ScholarStatus.ACTIVE)
    assert s2.status is ScholarStatus.ACTIVE


def test_transition_deceased_terminal():
    s = _scholar(status=ScholarStatus.DECEASED)
    with pytest.raises(ValueError):
        transition_status(s, new_status=ScholarStatus.ACTIVE)


def test_transition_inactive_to_deceased():
    s = _scholar(status=ScholarStatus.INACTIVE)
    s2 = transition_status(s, new_status=ScholarStatus.DECEASED)
    assert s2.status is ScholarStatus.DECEASED


# --- route_user ------------------------------


def test_route_primary_region_match_wins():
    gulf_scholar = _scholar(scholar_id="gulf-1", primary_region=Region.GULF)
    other_scholar = _scholar(
        scholar_id="asia-1",
        primary_region=Region.SOUTHEAST_ASIA,
    )
    user = _user(region=Region.GULF)
    out = route_user(user, [other_scholar, gulf_scholar])
    assert out[0].scholar_id == "gulf-1"


def test_route_alternative_region_match_secondary():
    gulf_primary = _scholar(scholar_id="gulf-1", primary_region=Region.GULF)
    gulf_alt = _scholar(
        scholar_id="alt-1",
        primary_region=Region.LEVANT,
        alternative_regions=(Region.GULF,),
    )
    user = _user(region=Region.GULF)
    out = route_user(user, [gulf_alt, gulf_primary])
    # gulf_primary should rank above gulf_alt.
    assert out[0].scholar_id == "gulf-1"
    assert out[1].scholar_id == "alt-1"


def test_route_global_fallback():
    """No regional match, but global scholar fills."""
    global_scholar = _scholar(
        scholar_id="global-1",
        primary_region=Region.GULF,
        is_global=True,
    )
    user = _user(region=Region.SOUTHEAST_ASIA)
    out = route_user(user, [global_scholar])
    # Region doesn't match, but is_global → returned.
    assert len(out) == 1
    assert out[0].scholar_id == "global-1"


def test_route_excludes_inactive():
    inactive = _scholar(scholar_id="x", primary_region=Region.GULF, status=ScholarStatus.INACTIVE)
    user = _user(region=Region.GULF)
    out = route_user(user, [inactive])
    assert out == ()


def test_route_excludes_deceased():
    deceased = _scholar(
        scholar_id="x",
        primary_region=Region.GULF,
        status=ScholarStatus.DECEASED,
    )
    user = _user(region=Region.GULF)
    out = route_user(user, [deceased])
    assert out == ()


def test_route_excludes_unrelated_non_global():
    """A scholar with no overlap + not global is dropped."""
    s = _scholar(scholar_id="x", primary_region=Region.SOUTH_ASIA, is_global=False)
    user = _user(region=Region.GULF)
    out = route_user(user, [s])
    assert out == ()


def test_route_madhhab_match_breaks_tie():
    """Two same-region scholars: madhhab match wins."""
    hanbali = _scholar(
        scholar_id="hanbali",
        primary_region=Region.GULF,
        madhhabs=(Madhhab.HANBALI,),
    )
    maliki = _scholar(
        scholar_id="maliki",
        primary_region=Region.GULF,
        madhhabs=(Madhhab.MALIKI,),
    )
    user = _user(region=Region.GULF, preferred_madhhab=Madhhab.HANBALI)
    out = route_user(user, [maliki, hanbali])
    assert out[0].scholar_id == "hanbali"


def test_route_deterministic_tie_break_by_id():
    """Same region + same madhhab → scholar_id sorts."""
    s1 = _scholar(scholar_id="alpha", primary_region=Region.GULF)
    s2 = _scholar(scholar_id="beta", primary_region=Region.GULF)
    user = _user(region=Region.GULF)
    out = route_user(user, [s2, s1])
    assert out[0].scholar_id == "alpha"


def test_route_top_n_cap():
    scholars = [_scholar(scholar_id=f"s-{i}", primary_region=Region.GULF) for i in range(10)]
    user = _user(region=Region.GULF)
    out = route_user(user, scholars, top_n=3)
    assert len(out) == 3


def test_route_invalid_top_n_rejected():
    user = _user()
    with pytest.raises(ValueError):
        route_user(user, [], top_n=0)


def test_route_empty_input_empty_output():
    user = _user()
    assert route_user(user, []) == ()


# --- scholars_for_region ---------------------


def test_scholars_for_region_includes_primary_and_alt():
    s1 = _scholar(scholar_id="s1", primary_region=Region.GULF)
    s2 = _scholar(
        scholar_id="s2",
        primary_region=Region.LEVANT,
        alternative_regions=(Region.GULF,),
    )
    s3 = _scholar(scholar_id="s3", primary_region=Region.SOUTHEAST_ASIA)
    out = scholars_for_region(Region.GULF, [s1, s2, s3])
    ids = {s.scholar_id for s in out}
    assert ids == {"s1", "s2"}


def test_scholars_for_region_excludes_inactive():
    s = _scholar(
        scholar_id="x",
        primary_region=Region.GULF,
        status=ScholarStatus.INACTIVE,
    )
    out = scholars_for_region(Region.GULF, [s])
    assert out == ()


def test_scholars_for_region_sorted_by_id():
    s1 = _scholar(scholar_id="zoo", primary_region=Region.GULF)
    s2 = _scholar(scholar_id="ali", primary_region=Region.GULF)
    out = scholars_for_region(Region.GULF, [s1, s2])
    assert [s.scholar_id for s in out] == ["ali", "zoo"]


# --- has_madhhab_compatible ------------------


def test_has_madhhab_compatible_true():
    s = _scholar(primary_region=Region.GULF, madhhabs=(Madhhab.HANBALI,))
    assert has_madhhab_compatible(Region.GULF, Madhhab.HANBALI, [s])


def test_has_madhhab_compatible_false_wrong_madhhab():
    s = _scholar(primary_region=Region.GULF, madhhabs=(Madhhab.MALIKI,))
    assert not has_madhhab_compatible(Region.GULF, Madhhab.HANBALI, [s])


def test_has_madhhab_compatible_false_wrong_region():
    s = _scholar(primary_region=Region.SOUTH_ASIA, madhhabs=(Madhhab.HANBALI,))
    assert not has_madhhab_compatible(Region.GULF, Madhhab.HANBALI, [s])


def test_has_madhhab_compatible_skips_inactive():
    s = _scholar(
        primary_region=Region.GULF,
        madhhabs=(Madhhab.HANBALI,),
        status=ScholarStatus.INACTIVE,
    )
    assert not has_madhhab_compatible(Region.GULF, Madhhab.HANBALI, [s])


# --- Render ----------------------------------


def test_render_scholar_includes_region_marker():
    s = _scholar(primary_region=Region.SOUTHEAST_ASIA)
    out = render_scholar(s)
    assert "southeast_asia" in out


def test_render_scholar_marks_global():
    s = _scholar(is_global=True)
    out = render_scholar(s)
    assert "🌐" in out


def test_render_scholar_includes_madhhab():
    s = _scholar(madhhabs=(Madhhab.HANBALI, Madhhab.MALIKI))
    out = render_scholar(s)
    assert "hanbali" in out
    assert "maliki" in out


def test_render_routing_empty():
    out = render_routing(_user(), [])
    assert "0 scholar" in out


def test_render_routing_lists_matches():
    s = _scholar()
    out = render_routing(_user(), [s])
    assert "Route for" in out
    assert "Imam Ali" in out


def test_render_routing_includes_preferred_madhhab_when_set():
    s = _scholar()
    user = _user(preferred_madhhab=Madhhab.HANBALI)
    out = render_routing(user, [s])
    assert "madhhab=hanbali" in out


def test_render_routing_omits_madhhab_when_unset():
    s = _scholar()
    user = _user(preferred_madhhab=None)
    out = render_routing(user, [s])
    assert "madhhab" not in out
