"""Tests for sentiment/sec_diff.py — Round-5 Wave 11.E."""

from __future__ import annotations

import pytest

from halal_trader.sentiment.sec_diff import (
    ChangeKind,
    DiffPolicy,
    FilingDiff,
    FilingSection,
    SectionDiff,
    diff_filing,
    diff_section,
    render_diff,
)

# --- Validation -------------------------------------------------


def test_section_string_values():
    assert FilingSection.RISK_FACTORS.value == "risk_factors"
    assert FilingSection.MD_A.value == "md_a"
    assert FilingSection.ITEM_5_02.value == "item_5_02"


def test_change_kind_string_values():
    assert ChangeKind.ADDED.value == "added"
    assert ChangeKind.REMOVED.value == "removed"
    assert ChangeKind.MODIFIED.value == "modified"
    assert ChangeKind.UNCHANGED.value == "unchanged"


def test_default_policy():
    p = DiffPolicy()
    assert p.material_overlap_threshold == 0.30
    assert p.min_section_chars == 50


def test_policy_zero_threshold_rejected():
    with pytest.raises(ValueError):
        DiffPolicy(material_overlap_threshold=0.0)


def test_policy_one_threshold_rejected():
    with pytest.raises(ValueError):
        DiffPolicy(material_overlap_threshold=1.0)


def test_policy_zero_min_chars_rejected():
    with pytest.raises(ValueError):
        DiffPolicy(min_section_chars=0)


def test_section_diff_negative_count_rejected():
    with pytest.raises(ValueError):
        SectionDiff(
            section=FilingSection.RISK_FACTORS,
            change_kind=ChangeKind.MODIFIED,
            overlap_ratio=0.5,
            is_material=False,
            sentences_added=-1,
            sentences_removed=0,
        )


def test_section_diff_overlap_outside_unit_rejected():
    with pytest.raises(ValueError):
        SectionDiff(
            section=FilingSection.RISK_FACTORS,
            change_kind=ChangeKind.MODIFIED,
            overlap_ratio=1.5,
            is_material=False,
            sentences_added=0,
            sentences_removed=0,
        )


# --- Section diff -----------------------------------------------


_LONG_TEXT_A = (
    "We face risks from increased competition. "
    "Macroeconomic headwinds could materially affect operations. "
    "Regulatory changes in the EU may increase compliance cost."
)

_LONG_TEXT_B = _LONG_TEXT_A + " A new ransomware attack disrupted operations."


def test_unchanged_when_text_identical():
    d = diff_section(FilingSection.RISK_FACTORS, _LONG_TEXT_A, _LONG_TEXT_A)
    assert d.change_kind is ChangeKind.UNCHANGED
    assert not d.is_material


def test_modified_when_minor_change():
    d = diff_section(FilingSection.RISK_FACTORS, _LONG_TEXT_A, _LONG_TEXT_B)
    assert d.change_kind is ChangeKind.MODIFIED
    assert d.sentences_added >= 1


def test_added_when_section_is_new():
    d = diff_section(FilingSection.RISK_FACTORS, "", _LONG_TEXT_A)
    assert d.change_kind is ChangeKind.ADDED
    assert d.is_material


def test_removed_when_section_disappears():
    d = diff_section(FilingSection.RISK_FACTORS, _LONG_TEXT_A, "")
    assert d.change_kind is ChangeKind.REMOVED
    assert d.is_material


def test_unchanged_when_both_empty():
    d = diff_section(FilingSection.RISK_FACTORS, "", "")
    assert d.change_kind is ChangeKind.UNCHANGED
    assert not d.is_material


def test_material_when_low_overlap():
    """A 100% rewrite (zero overlap) is MATERIAL."""
    a = "Sentence one is here. Sentence two is here. Sentence three is here. Sentence four extra."
    b = (
        "Wholly new content here for sure. Different topic appears now. "
        "Unrelated paragraph follows below."
    )
    d = diff_section(FilingSection.RISK_FACTORS, a, b)
    assert d.is_material is True
    assert d.change_kind is ChangeKind.MODIFIED


def test_not_material_when_high_overlap():
    """A small change (single sentence added) preserves overlap above threshold."""
    a = "First. Second. Third. Fourth. Fifth. Sixth. Seventh. Eighth. Ninth. Tenth."
    b = a + " Eleventh."
    d = diff_section(FilingSection.RISK_FACTORS, a, b)
    assert d.is_material is False


def test_overlap_ratio_in_unit_range():
    d = diff_section(FilingSection.RISK_FACTORS, _LONG_TEXT_A, _LONG_TEXT_B)
    assert 0.0 <= d.overlap_ratio <= 1.0


# --- Custom policy ----------------------------------------------


def test_custom_threshold_changes_material_classification():
    a = "First. Second. Third. Fourth. Fifth."
    b = "First. Second. NEW1. NEW2. NEW3."
    strict = diff_section(
        FilingSection.RISK_FACTORS,
        a,
        b,
        policy=DiffPolicy(material_overlap_threshold=0.5, min_section_chars=10),
    )
    relaxed = diff_section(
        FilingSection.RISK_FACTORS,
        a,
        b,
        policy=DiffPolicy(material_overlap_threshold=0.10, min_section_chars=10),
    )
    assert strict.is_material
    assert not relaxed.is_material


# --- Filing diff -------------------------------------------------


def test_diff_filing_handles_multiple_sections():
    prev = {
        FilingSection.RISK_FACTORS: _LONG_TEXT_A,
        FilingSection.MD_A: "Operating results were strong. Revenue grew 10%.",
    }
    curr = {
        FilingSection.RISK_FACTORS: _LONG_TEXT_B,
        FilingSection.MD_A: "Operating results were strong. Revenue grew 10%.",
        FilingSection.ITEM_5_02: "CEO resigned effective immediately. Board appointed interim.",
    }
    fd = diff_filing(prev, curr)
    by_section = {d.section: d for d in fd.section_diffs}
    assert by_section[FilingSection.MD_A].change_kind is ChangeKind.UNCHANGED
    assert by_section[FilingSection.RISK_FACTORS].change_kind is ChangeKind.MODIFIED
    assert by_section[FilingSection.ITEM_5_02].change_kind is ChangeKind.ADDED


def test_filing_has_material_changes():
    prev = {FilingSection.RISK_FACTORS: ""}
    curr = {FilingSection.RISK_FACTORS: _LONG_TEXT_A}
    fd = diff_filing(prev, curr)
    assert fd.has_material_changes()


def test_filing_no_material_when_unchanged():
    prev = {FilingSection.RISK_FACTORS: _LONG_TEXT_A}
    curr = {FilingSection.RISK_FACTORS: _LONG_TEXT_A}
    fd = diff_filing(prev, curr)
    assert not fd.has_material_changes()


def test_filing_diffs_sorted_by_section_value():
    prev = {
        FilingSection.MD_A: "x" * 100,
        FilingSection.RISK_FACTORS: "y" * 100,
    }
    curr = {FilingSection.MD_A: "x" * 100, FilingSection.RISK_FACTORS: "y" * 100}
    fd = diff_filing(prev, curr)
    sections = [d.section.value for d in fd.section_diffs]
    assert sections == sorted(sections)


# --- Render -----------------------------------------------------


def test_render_filing_diff():
    prev = {FilingSection.RISK_FACTORS: _LONG_TEXT_A}
    curr = {FilingSection.RISK_FACTORS: _LONG_TEXT_B}
    fd = diff_filing(prev, curr)
    out = render_diff(fd)
    assert "Filing diff" in out
    assert "risk_factors" in out


def test_render_empty_filing_diff():
    fd = FilingDiff(section_diffs=())
    out = render_diff(fd)
    assert "no sections" in out


def test_render_marks_material():
    prev = {FilingSection.RISK_FACTORS: ""}
    curr = {FilingSection.RISK_FACTORS: _LONG_TEXT_A}
    fd = diff_filing(prev, curr)
    out = render_diff(fd)
    assert "‼" in out


def test_render_no_secret_leak():
    prev = {FilingSection.RISK_FACTORS: _LONG_TEXT_A}
    curr = {FilingSection.RISK_FACTORS: _LONG_TEXT_B}
    fd = diff_filing(prev, curr)
    out = render_diff(fd)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ------------------------------------------------------


def test_e2e_8k_executive_change_material():
    """A new ITEM_5_02 (officer change) is unambiguously material."""
    prev: dict[FilingSection, str] = {}
    curr = {
        FilingSection.ITEM_5_02: (
            "On January 5 2026, the CEO resigned effective immediately. "
            "The Board appointed the CFO as interim CEO. "
            "A search for a permanent CEO has begun."
        )
    }
    fd = diff_filing(prev, curr)
    assert fd.has_material_changes()


def test_replay_consistency():
    a = diff_section(FilingSection.RISK_FACTORS, _LONG_TEXT_A, _LONG_TEXT_B)
    b = diff_section(FilingSection.RISK_FACTORS, _LONG_TEXT_A, _LONG_TEXT_B)
    assert a == b
