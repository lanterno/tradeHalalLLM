"""Tests for core/yearend_checklist.py — Round-5 Wave 18.J."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.core.yearend_checklist import (
    ChecklistTask,
    TaskCategory,
    build_checklist,
    filter_by_category,
    render_checklist,
    upcoming,
)
from halal_trader.halal.jurisdiction_router import Jurisdiction

# --- Validation -----------------------------------------------------


def test_category_string_values():
    assert TaskCategory.TAX.value == "tax"
    assert TaskCategory.ZAKAT.value == "zakat"
    assert TaskCategory.PURIFICATION.value == "purification"
    assert TaskCategory.RECORDS.value == "records"
    assert TaskCategory.COMPLIANCE.value == "compliance"


def test_task_empty_id_rejected():
    with pytest.raises(ValueError):
        ChecklistTask(
            task_id="",
            title="x",
            category=TaskCategory.TAX,
            jurisdictions=frozenset({Jurisdiction.USA}),
            deadline=date(2026, 12, 31),
        )


def test_task_empty_title_rejected():
    with pytest.raises(ValueError):
        ChecklistTask(
            task_id="X",
            title="",
            category=TaskCategory.TAX,
            jurisdictions=frozenset({Jurisdiction.USA}),
            deadline=date(2026, 12, 31),
        )


# --- Build ------------------------------------------------------


def test_build_us_jurisdiction_includes_us_tasks():
    checklist = build_checklist(year=2026, jurisdictions=[Jurisdiction.USA])
    titles = [t.title for t in checklist]
    assert any("1099-B" in t for t in titles)
    assert any("Form 8949" in t for t in titles)


def test_build_uk_jurisdiction_includes_uk_tasks():
    checklist = build_checklist(year=2026, jurisdictions=[Jurisdiction.UK])
    titles = [t.title for t in checklist]
    assert any("UK CGT" in t for t in titles)


def test_build_saudi_includes_saudi_only_compliance_task():
    checklist = build_checklist(year=2026, jurisdictions=[Jurisdiction.SAUDI_ARABIA])
    titles = [t.title for t in checklist]
    assert any("Saudi CMA" in t for t in titles)


def test_build_excludes_jurisdiction_specific_when_not_requested():
    checklist = build_checklist(year=2026, jurisdictions=[Jurisdiction.USA])
    titles = [t.title for t in checklist]
    assert not any("UK CGT" in t for t in titles)


def test_build_zakat_task_for_muslim_majority_jurisdictions():
    checklist = build_checklist(year=2026, jurisdictions=[Jurisdiction.SAUDI_ARABIA])
    cats = [t.category for t in checklist]
    assert TaskCategory.ZAKAT in cats


def test_build_zakat_not_required_for_non_muslim_jurisdiction():
    """USA-only operator typically isn't required to file Zakat (it's voluntary)."""
    checklist = build_checklist(year=2026, jurisdictions=[Jurisdiction.USA])
    zakat_tasks = filter_by_category(checklist, TaskCategory.ZAKAT)
    # Zakat task is jurisdiction-keyed; USA isn't in the Zakat task's
    # jurisdiction set, so it's filtered out.
    assert len(zakat_tasks) == 0


def test_build_sorted_by_deadline():
    checklist = build_checklist(
        year=2026,
        jurisdictions=[Jurisdiction.USA, Jurisdiction.SAUDI_ARABIA],
    )
    deadlines = [t.deadline for t in checklist]
    assert deadlines == sorted(deadlines)


def test_build_empty_jurisdictions_rejected():
    with pytest.raises(ValueError):
        build_checklist(year=2026, jurisdictions=[])


def test_build_invalid_year_rejected():
    with pytest.raises(ValueError):
        build_checklist(year=1800, jurisdictions=[Jurisdiction.USA])


def test_build_with_extra_tasks():
    extra = [
        ChecklistTask(
            task_id="CUSTOM-001",
            title="Custom task",
            category=TaskCategory.RECORDS,
            jurisdictions=frozenset({Jurisdiction.USA}),
            deadline=date(2027, 1, 1),
        ),
    ]
    checklist = build_checklist(year=2026, jurisdictions=[Jurisdiction.USA], extra=extra)
    assert any(t.task_id == "CUSTOM-001" for t in checklist)


# --- Filters ----------------------------------------------------


def test_filter_by_category():
    checklist = build_checklist(year=2026, jurisdictions=[Jurisdiction.USA])
    tax_tasks = filter_by_category(checklist, TaskCategory.TAX)
    assert all(t.category is TaskCategory.TAX for t in tax_tasks)


def test_upcoming_within_window():
    checklist = build_checklist(year=2026, jurisdictions=[Jurisdiction.USA])
    today = date(2026, 12, 1)
    upcoming_tasks = upcoming(checklist, today=today, days=60)
    for t in upcoming_tasks:
        delta = (t.deadline - today).days
        assert 0 <= delta <= 60


def test_upcoming_negative_days_rejected():
    checklist = build_checklist(year=2026, jurisdictions=[Jurisdiction.USA])
    with pytest.raises(ValueError):
        upcoming(checklist, today=date(2026, 12, 1), days=-1)


def test_upcoming_excludes_past():
    checklist = build_checklist(year=2026, jurisdictions=[Jurisdiction.USA])
    today = date(2026, 12, 28)
    # YE-007 (US wash-sale cutoff) deadline is 2026-12-27 — past as of today
    upcoming_tasks = upcoming(checklist, today=today, days=60)
    ids = [t.task_id for t in upcoming_tasks]
    assert "YE-007" not in ids


# --- Render ---------------------------------------------------


def test_render_empty():
    out = render_checklist([])
    assert "empty" in out


def test_render_lists_tasks():
    checklist = build_checklist(year=2026, jurisdictions=[Jurisdiction.USA])
    out = render_checklist(checklist)
    assert "Year-end checklist" in out
    assert "□" in out


# --- E2E -------------------------------------------------


def test_e2e_diaspora_us_saudi_operator():
    """US-Saudi diaspora operator: gets US tax tasks + Saudi compliance + Zakat."""
    checklist = build_checklist(
        year=2026,
        jurisdictions=[Jurisdiction.USA, Jurisdiction.SAUDI_ARABIA],
    )
    cats = {t.category for t in checklist}
    # Should have all 5 categories represented
    assert TaskCategory.TAX in cats
    assert TaskCategory.ZAKAT in cats
    assert TaskCategory.PURIFICATION in cats
    assert TaskCategory.RECORDS in cats
    assert TaskCategory.COMPLIANCE in cats


def test_replay_consistency():
    a = build_checklist(year=2026, jurisdictions=[Jurisdiction.USA])
    b = build_checklist(year=2026, jurisdictions=[Jurisdiction.USA])
    assert a == b
