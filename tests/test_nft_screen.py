"""Tests for halal/nft_screen.py — Round-5 Wave 22.E."""

from __future__ import annotations

import pytest

from halal_trader.halal.nft_screen import (
    PROHIBITED_SUBJECTS,
    NftAssessment,
    NftInputs,
    NftIssue,
    SubjectMatter,
    render_assessment,
    screen_nft,
)


def _inputs(**overrides) -> NftInputs:
    base = {
        "nft_id": "NFT-001",
        "title": "Calligraphy Series #5",
        "subject_matter": SubjectMatter.CALLIGRAPHY,
        "creator_handle": "artist-handle",
        "provenance_verified": True,
        "has_known_theft_in_chain": False,
        "utility_purpose": "art",
        "embeds_financing_contract": False,
        "represents_haram_underlying_asset": False,
    }
    base.update(overrides)
    return NftInputs(**base)


# --- Validation ---------------------------------------------------


def test_subject_matter_string_values():
    assert SubjectMatter.CALLIGRAPHY.value == "calligraphy"
    assert SubjectMatter.GAMBLING_THEME.value == "gambling_theme"
    assert SubjectMatter.NUDITY.value == "nudity"


def test_issue_string_values():
    assert NftIssue.PROHIBITED_SUBJECT.value == "prohibited_subject"
    assert NftIssue.PROVENANCE_UNVERIFIED.value == "provenance_unverified"
    assert NftIssue.PROVENANCE_HAS_THEFT.value == "provenance_has_theft"
    assert NftIssue.UTILITY_IS_PROHIBITED.value == "utility_is_prohibited"
    assert NftIssue.EMBEDDED_FINANCING.value == "embedded_financing"
    assert NftIssue.FRACTIONAL_HARAM_ASSET.value == "fractional_haram_asset"
    assert NftIssue.NO_CREATOR_DISCLOSED.value == "no_creator_disclosed"


def test_prohibited_subjects_set_pin():
    expected = {
        SubjectMatter.IDOLATRY,
        SubjectMatter.GAMBLING_THEME,
        SubjectMatter.MUSIC_INSTRUMENT_HARAM,
        SubjectMatter.HARAM_ANIMAL,
        SubjectMatter.NUDITY,
        SubjectMatter.PROHIBITED_BEVERAGE,
    }
    assert PROHIBITED_SUBJECTS == frozenset(expected)


def test_inputs_empty_id_rejected():
    with pytest.raises(ValueError):
        _inputs(nft_id="")


def test_inputs_empty_title_rejected():
    with pytest.raises(ValueError):
        _inputs(title=" ")


def test_inputs_empty_utility_rejected():
    with pytest.raises(ValueError):
        _inputs(utility_purpose="")


# --- Screen -------------------------------------------------------


def test_clean_calligraphy_passes():
    a = screen_nft(_inputs())
    assert a.is_compliant


def test_idolatry_blocked():
    a = screen_nft(_inputs(subject_matter=SubjectMatter.IDOLATRY))
    assert NftIssue.PROHIBITED_SUBJECT in a.issues


def test_gambling_theme_blocked():
    a = screen_nft(_inputs(subject_matter=SubjectMatter.GAMBLING_THEME))
    assert NftIssue.PROHIBITED_SUBJECT in a.issues


def test_nudity_blocked():
    a = screen_nft(_inputs(subject_matter=SubjectMatter.NUDITY))
    assert NftIssue.PROHIBITED_SUBJECT in a.issues


def test_haram_animal_blocked():
    a = screen_nft(_inputs(subject_matter=SubjectMatter.HARAM_ANIMAL))
    assert NftIssue.PROHIBITED_SUBJECT in a.issues


def test_prohibited_beverage_blocked():
    a = screen_nft(_inputs(subject_matter=SubjectMatter.PROHIBITED_BEVERAGE))
    assert NftIssue.PROHIBITED_SUBJECT in a.issues


def test_unverified_provenance_blocked():
    a = screen_nft(_inputs(provenance_verified=False))
    assert NftIssue.PROVENANCE_UNVERIFIED in a.issues


def test_theft_in_chain_blocked():
    a = screen_nft(_inputs(has_known_theft_in_chain=True))
    assert NftIssue.PROVENANCE_HAS_THEFT in a.issues


def test_gambling_utility_blocked():
    a = screen_nft(_inputs(utility_purpose="gambling"))
    assert NftIssue.UTILITY_IS_PROHIBITED in a.issues


def test_lottery_utility_blocked():
    a = screen_nft(_inputs(utility_purpose="lottery"))
    assert NftIssue.UTILITY_IS_PROHIBITED in a.issues


def test_interest_bearing_utility_blocked():
    a = screen_nft(_inputs(utility_purpose="interest_bearing"))
    assert NftIssue.UTILITY_IS_PROHIBITED in a.issues


def test_utility_case_insensitive():
    a = screen_nft(_inputs(utility_purpose="GAMBLING"))
    assert NftIssue.UTILITY_IS_PROHIBITED in a.issues


def test_embedded_financing_blocked():
    a = screen_nft(_inputs(embeds_financing_contract=True))
    assert NftIssue.EMBEDDED_FINANCING in a.issues


def test_fractional_haram_blocked():
    a = screen_nft(_inputs(represents_haram_underlying_asset=True))
    assert NftIssue.FRACTIONAL_HARAM_ASSET in a.issues


def test_no_creator_blocked():
    a = screen_nft(_inputs(creator_handle=" "))
    assert NftIssue.NO_CREATOR_DISCLOSED in a.issues


def test_multiple_issues_combined():
    a = screen_nft(
        _inputs(
            subject_matter=SubjectMatter.IDOLATRY,
            provenance_verified=False,
            has_known_theft_in_chain=True,
        )
    )
    assert {
        NftIssue.PROHIBITED_SUBJECT,
        NftIssue.PROVENANCE_UNVERIFIED,
        NftIssue.PROVENANCE_HAS_THEFT,
    } <= a.issues


# --- Permitted subjects pass ------------------------------------------


def test_calligraphy_passes():
    a = screen_nft(_inputs(subject_matter=SubjectMatter.CALLIGRAPHY))
    assert a.is_compliant


def test_abstract_art_passes():
    a = screen_nft(_inputs(subject_matter=SubjectMatter.ABSTRACT_ART))
    assert a.is_compliant


def test_nature_passes():
    a = screen_nft(_inputs(subject_matter=SubjectMatter.NATURE))
    assert a.is_compliant


def test_sukuk_representation_passes():
    a = screen_nft(_inputs(subject_matter=SubjectMatter.SUKUK_REPRESENTATION))
    assert a.is_compliant


# --- Assessment invariants -----------------------------------------


def test_assessment_compliant_with_issues_rejected():
    with pytest.raises(ValueError):
        NftAssessment(
            nft_id="x",
            subject_matter=SubjectMatter.CALLIGRAPHY,
            issues=frozenset({NftIssue.PROHIBITED_SUBJECT}),
            is_compliant=True,
        )


def test_assessment_non_compliant_without_issues_rejected():
    with pytest.raises(ValueError):
        NftAssessment(
            nft_id="x",
            subject_matter=SubjectMatter.CALLIGRAPHY,
            issues=frozenset(),
            is_compliant=False,
        )


# --- Render -------------------------------------------------------


def test_render_clean():
    inp = _inputs()
    a = screen_nft(inp)
    out = render_assessment(inp, a)
    assert "✅" in out
    assert "Calligraphy" in out
    assert "calligraphy" in out


def test_render_violations():
    inp = _inputs(subject_matter=SubjectMatter.IDOLATRY)
    a = screen_nft(inp)
    out = render_assessment(inp, a)
    assert "❌" in out
    assert "prohibited_subject" in out


def test_render_no_secret_leak():
    inp = _inputs()
    a = screen_nft(inp)
    out = render_assessment(inp, a)
    for token in (
        "@",
        "zoom.us",
        "meet.google",
        "private_email",
        "+1-",
        "Authorization",
        "wallet_address",
        "private_key",
    ):
        assert token not in out


# --- E2E -----------------------------------------------------


def test_e2e_calligraphy_sale_clean():
    inp = _inputs(
        subject_matter=SubjectMatter.CALLIGRAPHY,
        creator_handle="artist-001",
        provenance_verified=True,
    )
    a = screen_nft(inp)
    assert a.is_compliant


def test_e2e_gambling_themed_blocked():
    inp = _inputs(
        subject_matter=SubjectMatter.GAMBLING_THEME,
        utility_purpose="gambling",
    )
    a = screen_nft(inp)
    assert not a.is_compliant
    assert NftIssue.PROHIBITED_SUBJECT in a.issues
    assert NftIssue.UTILITY_IS_PROHIBITED in a.issues


def test_replay_consistency():
    inp = _inputs()
    a = screen_nft(inp)
    b = screen_nft(inp)
    assert a == b
