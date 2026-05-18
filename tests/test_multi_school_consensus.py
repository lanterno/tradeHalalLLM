"""Tests for the multi-school Shariah consensus engine."""

from __future__ import annotations

import pytest

from halal_trader.halal.multi_school_consensus import (
    SUNNI_SCHOOLS,
    ConsensusMode,
    ConsensusReport,
    School,
    SchoolPosition,
    SchoolVerdict,
    build_report,
    disagreement_summary,
    render_position,
    render_report,
    tradable_under_consensus,
)


def _pos(
    school: School,
    verdict: SchoolVerdict,
    reasoning: str = "test reasoning",
    scholar_handle: str | None = None,
) -> SchoolPosition:
    return SchoolPosition(
        school=school,
        verdict=verdict,
        reasoning=reasoning,
        scholar_handle=scholar_handle,
    )


# --- Enum string-value pins ---------------------------------------------------


def test_school_string_values():
    assert School.HANAFI.value == "hanafi"
    assert School.SHAFII.value == "shafii"
    assert School.MALIKI.value == "maliki"
    assert School.HANBALI.value == "hanbali"
    assert School.JAFARI.value == "jafari"


def test_school_verdict_string_values():
    assert SchoolVerdict.PERMISSIBLE.value == "permissible"
    assert SchoolVerdict.IMPERMISSIBLE.value == "impermissible"
    assert SchoolVerdict.ABSTAIN.value == "abstain"


def test_consensus_mode_string_values():
    assert ConsensusMode.UNANIMOUS.value == "unanimous"
    assert ConsensusMode.MAJORITY.value == "majority"
    assert ConsensusMode.ANY.value == "any"


def test_sunni_schools_set_pin():
    """Pin: SUNNI_SCHOOLS contains exactly Hanafi/Shafi'i/Maliki/Hanbali."""
    assert SUNNI_SCHOOLS == frozenset({School.HANAFI, School.SHAFII, School.MALIKI, School.HANBALI})
    assert School.JAFARI not in SUNNI_SCHOOLS


# --- SchoolPosition validation -----------------------------------------------


def test_position_immutable():
    p = _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE)
    with pytest.raises(Exception):
        p.verdict = SchoolVerdict.IMPERMISSIBLE  # type: ignore[misc]


def test_empty_reasoning_rejected():
    with pytest.raises(ValueError, match="reasoning"):
        _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE, reasoning="")


def test_whitespace_reasoning_rejected():
    with pytest.raises(ValueError, match="reasoning"):
        _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE, reasoning="   ")


def test_optional_scholar_handle_omitted():
    p = _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE)
    assert p.scholar_handle is None


def test_optional_scholar_handle_provided():
    p = _pos(
        School.HANAFI,
        SchoolVerdict.PERMISSIBLE,
        scholar_handle="mufti_taqi_usmani",
    )
    assert p.scholar_handle == "mufti_taqi_usmani"


def test_empty_scholar_handle_rejected():
    """Pin: scholar_handle must be None or non-empty (no empty string)."""
    with pytest.raises(ValueError, match="scholar_handle"):
        _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE, scholar_handle="")


# --- ConsensusReport validation ----------------------------------------------


def test_count_consistency_invariant():
    """Pin: counts must equal positions length."""
    with pytest.raises(ValueError, match="counts"):
        ConsensusReport(
            positions=(_pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),),
            permissible_count=2,  # mismatch
            impermissible_count=0,
            abstain_count=0,
        )


def test_negative_count_rejected():
    with pytest.raises(ValueError, match="permissible_count"):
        ConsensusReport(
            positions=(),
            permissible_count=-1,
            impermissible_count=0,
            abstain_count=0,
        )


def test_duplicate_school_rejected():
    """Pin: a school cannot appear twice in one report."""
    with pytest.raises(ValueError, match="duplicate position for school"):
        ConsensusReport(
            positions=(
                _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
                _pos(School.HANAFI, SchoolVerdict.IMPERMISSIBLE),
            ),
            permissible_count=1,
            impermissible_count=1,
            abstain_count=0,
        )


def test_report_immutable():
    r = build_report([_pos(School.HANAFI, SchoolVerdict.PERMISSIBLE)])
    with pytest.raises(Exception):
        r.permissible_count = 99  # type: ignore[misc]


# --- ConsensusReport properties ----------------------------------------------


def test_total_engaged_excludes_abstain():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
            _pos(School.MALIKI, SchoolVerdict.ABSTAIN),
        ]
    )
    assert r.total_engaged == 2


def test_unanimous_permissible_when_all_say_yes():
    r = build_report(
        [
            _pos(s, SchoolVerdict.PERMISSIBLE)
            for s in (School.HANAFI, School.SHAFII, School.MALIKI, School.HANBALI)
        ]
    )
    assert r.is_unanimous_permissible is True
    assert r.is_unanimous_impermissible is False


def test_unanimous_permissible_with_abstain_still_unanimous():
    """Pin: ABSTAIN doesn't break unanimity (school just hasn't opined)."""
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.ABSTAIN),
        ]
    )
    assert r.is_unanimous_permissible is True


def test_unanimous_permissible_false_with_one_impermissible():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    assert r.is_unanimous_permissible is False


def test_unanimous_impermissible_when_all_say_no():
    r = build_report(
        [
            _pos(s, SchoolVerdict.IMPERMISSIBLE)
            for s in (School.HANAFI, School.SHAFII, School.MALIKI, School.HANBALI)
        ]
    )
    assert r.is_unanimous_impermissible is True


def test_unanimous_requires_at_least_one_engagement():
    """Pin: an empty/all-abstain report is NOT unanimous either way."""
    r = build_report([])
    assert r.is_unanimous_permissible is False
    assert r.is_unanimous_impermissible is False
    r2 = build_report([_pos(School.HANAFI, SchoolVerdict.ABSTAIN)])
    assert r2.is_unanimous_permissible is False
    assert r2.is_unanimous_impermissible is False


def test_majority_permissible_strict_greater_than():
    """Pin: 2 permissible vs 1 impermissible = majority; tied is NOT."""
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE),
            _pos(School.MALIKI, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    assert r.is_majority_permissible is True


def test_tied_is_not_majority():
    """Pin: 1-1 tie is NOT a majority (strict greater-than)."""
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    assert r.is_majority_permissible is False


def test_split_when_both_sides_present():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    assert r.is_split is True


def test_not_split_when_one_sided():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE),
        ]
    )
    assert r.is_split is False


# --- Sunni-consensus property ------------------------------------------------


def test_sunni_consensus_requires_all_four_sunni():
    """Pin: is_sunni_consensus requires all 4 Sunni schools to opine PERMISSIBLE."""
    r = build_report(
        [
            _pos(s, SchoolVerdict.PERMISSIBLE)
            for s in (School.HANAFI, School.SHAFII, School.MALIKI, School.HANBALI)
        ]
    )
    assert r.is_sunni_consensus is True


def test_sunni_consensus_excludes_jafari_from_required():
    """Pin: Ja'fari abstaining doesn't break Sunni consensus."""
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE),
            _pos(School.MALIKI, SchoolVerdict.PERMISSIBLE),
            _pos(School.HANBALI, SchoolVerdict.PERMISSIBLE),
        ]
    )
    assert r.is_sunni_consensus is True


def test_sunni_consensus_false_with_only_three_sunni():
    """Pin: missing one Sunni school breaks the consensus property."""
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE),
            _pos(School.MALIKI, SchoolVerdict.PERMISSIBLE),
        ]
    )
    assert r.is_sunni_consensus is False


def test_sunni_consensus_false_with_one_impermissible():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.IMPERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE),
            _pos(School.MALIKI, SchoolVerdict.PERMISSIBLE),
            _pos(School.HANBALI, SchoolVerdict.PERMISSIBLE),
        ]
    )
    assert r.is_sunni_consensus is False


def test_sunni_consensus_false_with_abstain():
    """Pin: ABSTAIN is not engagement — Sunni consensus needs verdicts."""
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.ABSTAIN),
            _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE),
            _pos(School.MALIKI, SchoolVerdict.PERMISSIBLE),
            _pos(School.HANBALI, SchoolVerdict.PERMISSIBLE),
        ]
    )
    assert r.is_sunni_consensus is False


# --- build_report -------------------------------------------------------------


def test_build_report_empty():
    r = build_report([])
    assert r.positions == ()
    assert r.permissible_count == 0
    assert r.impermissible_count == 0
    assert r.abstain_count == 0


def test_build_report_counts_correctly():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE),
            _pos(School.MALIKI, SchoolVerdict.IMPERMISSIBLE),
            _pos(School.HANBALI, SchoolVerdict.ABSTAIN),
        ]
    )
    assert r.permissible_count == 2
    assert r.impermissible_count == 1
    assert r.abstain_count == 1


def test_build_report_sorts_by_school_enum_order():
    """Pin: positions sorted in canonical school-enum order."""
    r = build_report(
        [
            _pos(School.JAFARI, SchoolVerdict.PERMISSIBLE),
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.HANBALI, SchoolVerdict.PERMISSIBLE),
        ]
    )
    schools = [p.school for p in r.positions]
    assert schools == [School.HANAFI, School.HANBALI, School.JAFARI]


# --- tradable_under_consensus ------------------------------------------------


def test_unanimous_mode_blocks_split():
    """Pin: UNANIMOUS rejects any split."""
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    assert tradable_under_consensus(r, mode=ConsensusMode.UNANIMOUS) is False


def test_unanimous_mode_allows_unanimous_permissible():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE),
        ]
    )
    assert tradable_under_consensus(r, mode=ConsensusMode.UNANIMOUS) is True


def test_unanimous_mode_blocks_empty():
    """Pin: UNANIMOUS requires at least one engagement (no engaged → False)."""
    r = build_report([])
    assert tradable_under_consensus(r, mode=ConsensusMode.UNANIMOUS) is False


def test_unanimous_mode_blocks_all_abstain():
    r = build_report([_pos(School.HANAFI, SchoolVerdict.ABSTAIN)])
    assert tradable_under_consensus(r, mode=ConsensusMode.UNANIMOUS) is False


def test_majority_mode_allows_split_with_majority_permissible():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE),
            _pos(School.MALIKI, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    assert tradable_under_consensus(r, mode=ConsensusMode.MAJORITY) is True


def test_majority_mode_blocks_tied():
    """Pin: MAJORITY needs strict greater-than, not tied."""
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    assert tradable_under_consensus(r, mode=ConsensusMode.MAJORITY) is False


def test_any_mode_allows_with_one_permissible():
    """Pin: ANY mode allows even minority permissible."""
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
            _pos(School.MALIKI, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    assert tradable_under_consensus(r, mode=ConsensusMode.ANY) is True


def test_any_mode_blocks_no_permissible():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.IMPERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    assert tradable_under_consensus(r, mode=ConsensusMode.ANY) is False


def test_default_mode_is_majority():
    """Pin: default ConsensusMode is MAJORITY."""
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    # Tied 1-1: MAJORITY would block; UNANIMOUS would also block; ANY would allow
    assert tradable_under_consensus(r) is False  # default is MAJORITY


# --- disagreement_summary ----------------------------------------------------


def test_disagreement_returns_minority_side():
    """Majority permissible → minority is the impermissible voters."""
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE),
            _pos(School.MALIKI, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    assert disagreement_summary(r) == (School.MALIKI,)


def test_disagreement_returns_minority_when_majority_impermissible():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.IMPERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
            _pos(School.MALIKI, SchoolVerdict.PERMISSIBLE),
        ]
    )
    assert disagreement_summary(r) == (School.MALIKI,)


def test_disagreement_tied_returns_impermissible_side_by_convention():
    """Pin: tied → return IMPERMISSIBLE side (conservative read)."""
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    assert disagreement_summary(r) == (School.SHAFII,)


def test_disagreement_unanimous_permissible_returns_empty():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE),
        ]
    )
    assert disagreement_summary(r) == ()


def test_disagreement_unanimous_impermissible_returns_empty():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.IMPERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    assert disagreement_summary(r) == ()


# --- Render -------------------------------------------------------------------


def test_render_position_includes_emoji_and_label():
    p = _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE, "permitted by majority")
    out = render_position(p)
    assert "✅" in out
    assert "Hanafi" in out
    assert "permissible" in out
    assert "permitted by majority" in out


def test_render_position_includes_scholar_handle_when_set():
    p = _pos(
        School.HANBALI,
        SchoolVerdict.IMPERMISSIBLE,
        "raises riba concerns",
        scholar_handle="mufti_taqi",
    )
    out = render_position(p)
    assert "mufti_taqi" in out


def test_render_position_omits_scholar_handle_when_none():
    p = _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE)
    out = render_position(p)
    assert "via" not in out


def test_render_position_abstain_emoji():
    p = _pos(School.HANAFI, SchoolVerdict.ABSTAIN, "no formal opinion yet")
    out = render_position(p)
    assert "❔" in out


def test_render_report_tradable():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE),
        ]
    )
    out = render_report(r, mode=ConsensusMode.UNANIMOUS)
    assert "✅ TRADABLE" in out
    assert "unanimous" in out


def test_render_report_not_tradable_split():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    out = render_report(r, mode=ConsensusMode.UNANIMOUS)
    assert "❌ NOT TRADABLE" in out
    assert "schools disagree" in out


def test_render_report_includes_per_school_lines():
    r = build_report(
        [
            _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
            _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
        ]
    )
    out = render_report(r)
    assert "Hanafi" in out
    assert "Shafi'i" in out


def test_render_report_no_secret_leak():
    """Pin: render output never includes scholar contact emails or
    transcripts."""
    p = _pos(
        School.HANAFI,
        SchoolVerdict.PERMISSIBLE,
        "see Standard 21",
        scholar_handle="mufti_taqi",
    )
    r = build_report([p])
    out = render_report(r)
    forbidden = ["@", "zoom.us", "meet.google", "private_email", "+1-"]
    for word in forbidden:
        assert word not in out


# --- E2E flows ----------------------------------------------------------------


def test_e2e_aapl_unanimous_permissible():
    """All four Sunni schools agree: AAPL is halal — unanimous PERMISSIBLE."""
    positions = [
        _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE, "passes debt + revenue"),
        _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE, "passes debt + revenue"),
        _pos(School.MALIKI, SchoolVerdict.PERMISSIBLE, "passes debt + revenue"),
        _pos(School.HANBALI, SchoolVerdict.PERMISSIBLE, "passes debt + revenue"),
    ]
    r = build_report(positions)
    assert r.is_unanimous_permissible
    assert r.is_sunni_consensus
    assert tradable_under_consensus(r, mode=ConsensusMode.UNANIMOUS)
    assert tradable_under_consensus(r, mode=ConsensusMode.MAJORITY)
    assert tradable_under_consensus(r, mode=ConsensusMode.ANY)


def test_e2e_disputed_name_split_block_unanimous_allow_majority():
    """A name where 3 schools say yes + 1 says no — operator pick of mode matters."""
    positions = [
        _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE, "constructs are valid"),
        _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE, "follow majority"),
        _pos(School.MALIKI, SchoolVerdict.PERMISSIBLE, "follow majority"),
        _pos(School.HANBALI, SchoolVerdict.IMPERMISSIBLE, "structure issue"),
    ]
    r = build_report(positions)
    assert r.is_split
    assert r.is_majority_permissible
    assert not r.is_unanimous_permissible
    # Saudi-style strict operator (UNANIMOUS) → blocked
    assert tradable_under_consensus(r, mode=ConsensusMode.UNANIMOUS) is False
    # Pakistani-style operator (MAJORITY) → allowed
    assert tradable_under_consensus(r, mode=ConsensusMode.MAJORITY) is True
    # Disagreement summary surfaces Hanbali
    assert disagreement_summary(r) == (School.HANBALI,)


def test_e2e_haram_name_unanimous_block():
    """Tobacco company: every school says no."""
    positions = [
        _pos(s, SchoolVerdict.IMPERMISSIBLE, "tobacco sector excluded")
        for s in (School.HANAFI, School.SHAFII, School.MALIKI, School.HANBALI)
    ]
    r = build_report(positions)
    assert r.is_unanimous_impermissible
    for mode in ConsensusMode:
        assert tradable_under_consensus(r, mode=mode) is False


def test_e2e_novel_fintech_partial_abstain():
    """Brand-new instrument: only 2 schools have opined; 2 abstain.
    UNANIMOUS allows (engaged schools all say yes); but is_sunni_consensus
    is False (not all 4 Sunni opined)."""
    positions = [
        _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE, "Wa'd structure ok"),
        _pos(School.SHAFII, SchoolVerdict.PERMISSIBLE, "structurally clean"),
        _pos(School.MALIKI, SchoolVerdict.ABSTAIN, "no formal opinion yet"),
        _pos(School.HANBALI, SchoolVerdict.ABSTAIN, "no formal opinion yet"),
    ]
    r = build_report(positions)
    assert r.is_unanimous_permissible  # engaged schools agree
    assert not r.is_sunni_consensus  # but not all 4 Sunni engaged
    assert tradable_under_consensus(r, mode=ConsensusMode.UNANIMOUS)


def test_e2e_replay_consistency():
    """Pin: same positions → equal report."""
    positions = [
        _pos(School.HANAFI, SchoolVerdict.PERMISSIBLE),
        _pos(School.SHAFII, SchoolVerdict.IMPERMISSIBLE),
    ]
    r1 = build_report(positions)
    r2 = build_report(positions)
    assert r1 == r2
