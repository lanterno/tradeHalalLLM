"""Tests for ops/iso27001_mapping.py — Round-5 Wave 19.F."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.ops.iso27001_mapping import (
    AnnexAControl,
    AnnexATheme,
    CatalogSummary,
    ControlMapping,
    MappingGap,
    MappingStatus,
    evidence_completeness,
    find_gaps,
    render_gap,
    render_summary,
    summarise_catalog,
)


def _ctrl(
    control_id: str = "A.5.1",
    theme: AnnexATheme = AnnexATheme.ORGANISATIONAL,
    title: str = "Policies for information security",
) -> AnnexAControl:
    return AnnexAControl(control_id=control_id, theme=theme, title=title)


def _mapping(
    mapping_id: str = "M1",
    annex_control_id: str = "A.5.1",
    platform_ref: str = "PLAT-001",
    rationale: str = "InfoSec policy lives in /docs/security.",
    status: MappingStatus = MappingStatus.IMPLEMENTED,
    last_evidenced_on: date | None = None,
) -> ControlMapping:
    return ControlMapping(
        mapping_id=mapping_id,
        annex_control_id=annex_control_id,
        platform_control_ref=platform_ref,
        rationale=rationale,
        status=status,
        last_evidenced_on=last_evidenced_on,
    )


# --- AnnexAControl validation -------------------------------------


def test_control_valid():
    c = _ctrl()
    assert c.control_id == "A.5.1"


def test_control_must_start_with_A():
    with pytest.raises(ValueError):
        _ctrl(control_id="B.5.1")


def test_control_empty_id_rejected():
    with pytest.raises(ValueError):
        _ctrl(control_id="")


def test_control_empty_title_rejected():
    with pytest.raises(ValueError):
        _ctrl(title=" ")


# --- ControlMapping validation -------------------------------------


def test_mapping_valid():
    m = _mapping()
    assert m.status is MappingStatus.IMPLEMENTED


def test_mapping_empty_rationale_rejected():
    with pytest.raises(ValueError):
        _mapping(rationale=" ")


def test_mapping_long_rationale_rejected():
    with pytest.raises(ValueError):
        _mapping(rationale="x" * 1500)


def test_mapping_evidenced_without_date_rejected():
    with pytest.raises(ValueError):
        _mapping(status=MappingStatus.EVIDENCED, last_evidenced_on=None)


def test_mapping_non_evidenced_with_date_rejected():
    with pytest.raises(ValueError):
        _mapping(
            status=MappingStatus.IMPLEMENTED,
            last_evidenced_on=date(2026, 1, 1),
        )


def test_mapping_immutable():
    m = _mapping()
    with pytest.raises(AttributeError):
        m.status = MappingStatus.EVIDENCED  # type: ignore[misc]


# --- find_gaps — coverage paths ------------------------------------


def test_find_gaps_unmapped():
    catalog = [_ctrl(control_id="A.5.1")]
    gaps = find_gaps(catalog, [], as_of=date(2026, 5, 11))
    assert len(gaps) == 1
    assert gaps[0].gap_kind == "unmapped"


def test_find_gaps_not_implemented():
    catalog = [_ctrl()]
    mappings = [_mapping(status=MappingStatus.MAPPED)]
    gaps = find_gaps(catalog, mappings, as_of=date(2026, 5, 11))
    assert len(gaps) == 1
    assert gaps[0].gap_kind == "not_implemented"


def test_find_gaps_no_evidence_when_implemented():
    catalog = [_ctrl()]
    mappings = [_mapping(status=MappingStatus.IMPLEMENTED)]
    gaps = find_gaps(catalog, mappings, as_of=date(2026, 5, 11))
    assert len(gaps) == 1
    assert gaps[0].gap_kind == "no_evidence"


def test_find_gaps_clean_evidenced():
    catalog = [_ctrl()]
    mappings = [
        _mapping(
            status=MappingStatus.EVIDENCED,
            last_evidenced_on=date(2026, 5, 1),
        )
    ]
    gaps = find_gaps(catalog, mappings, as_of=date(2026, 5, 11))
    assert gaps == ()


def test_find_gaps_stale_when_old_evidence():
    catalog = [_ctrl()]
    mappings = [
        _mapping(
            status=MappingStatus.EVIDENCED,
            last_evidenced_on=date(2024, 1, 1),  # > 1 year ago
        )
    ]
    gaps = find_gaps(catalog, mappings, as_of=date(2026, 5, 11), stale_days=365)
    assert len(gaps) == 1
    assert gaps[0].gap_kind == "stale"


def test_find_gaps_not_applicable_no_gap():
    catalog = [_ctrl()]
    mappings = [_mapping(status=MappingStatus.NOT_APPLICABLE)]
    gaps = find_gaps(catalog, mappings, as_of=date(2026, 5, 11))
    assert gaps == ()


def test_find_gaps_invalid_stale_days_rejected():
    with pytest.raises(ValueError):
        find_gaps([], [], as_of=date(2026, 5, 11), stale_days=0)


def test_find_gaps_highest_status_wins():
    """Pin: multiple mappings to the same control → highest status wins."""
    catalog = [_ctrl()]
    mappings = [
        _mapping(mapping_id="M1", status=MappingStatus.MAPPED),
        _mapping(
            mapping_id="M2",
            status=MappingStatus.EVIDENCED,
            last_evidenced_on=date(2026, 5, 1),
        ),
    ]
    gaps = find_gaps(catalog, mappings, as_of=date(2026, 5, 11))
    assert gaps == ()


def test_find_gaps_multiple_controls():
    catalog = [
        _ctrl(control_id="A.5.1"),
        _ctrl(control_id="A.5.2", theme=AnnexATheme.ORGANISATIONAL),
        _ctrl(control_id="A.7.1", theme=AnnexATheme.PHYSICAL),
    ]
    mappings = [
        _mapping(
            annex_control_id="A.5.1",
            status=MappingStatus.EVIDENCED,
            last_evidenced_on=date(2026, 5, 1),
        ),
        _mapping(
            mapping_id="M2",
            annex_control_id="A.5.2",
            status=MappingStatus.MAPPED,
        ),
        # A.7.1 unmapped.
    ]
    gaps = find_gaps(catalog, mappings, as_of=date(2026, 5, 11))
    by_id = {g.annex_control_id: g.gap_kind for g in gaps}
    assert "A.5.1" not in by_id
    assert by_id["A.5.2"] == "not_implemented"
    assert by_id["A.7.1"] == "unmapped"


# --- summarise_catalog ------------------------------------------


def test_summary_counts():
    catalog = [
        _ctrl(control_id="A.5.1"),
        _ctrl(control_id="A.5.2"),
        _ctrl(control_id="A.5.3"),
        _ctrl(control_id="A.5.4"),
        _ctrl(control_id="A.5.5"),
    ]
    mappings = [
        _mapping(
            annex_control_id="A.5.1",
            status=MappingStatus.EVIDENCED,
            last_evidenced_on=date(2026, 5, 1),
        ),
        _mapping(
            mapping_id="M2",
            annex_control_id="A.5.2",
            status=MappingStatus.IMPLEMENTED,
        ),
        _mapping(
            mapping_id="M3",
            annex_control_id="A.5.3",
            status=MappingStatus.MAPPED,
        ),
        _mapping(
            mapping_id="M4",
            annex_control_id="A.5.4",
            status=MappingStatus.NOT_APPLICABLE,
        ),
        # A.5.5 unmapped.
    ]
    summary = summarise_catalog(catalog, mappings, as_of=date(2026, 5, 11))
    assert summary.n_controls == 5
    assert summary.n_evidenced == 1
    assert summary.n_implemented == 1
    assert summary.n_mapped == 1
    assert summary.n_not_applicable == 1
    assert summary.n_unmapped == 1


def test_summary_stale_counted_separately():
    catalog = [_ctrl(control_id="A.5.1")]
    mappings = [
        _mapping(
            status=MappingStatus.EVIDENCED,
            last_evidenced_on=date(2024, 1, 1),
        )
    ]
    summary = summarise_catalog(catalog, mappings, as_of=date(2026, 5, 11))
    assert summary.n_stale == 1
    assert summary.n_evidenced == 0


# --- evidence_completeness -------------------------------------


def test_evidence_completeness_zero_for_empty():
    summary = CatalogSummary(
        n_controls=10,
        n_not_applicable=10,
        n_mapped=0,
        n_implemented=0,
        n_evidenced=0,
        n_unmapped=0,
        n_stale=0,
    )
    assert evidence_completeness(summary) == 0.0


def test_evidence_completeness_half_for_50pct():
    summary = CatalogSummary(
        n_controls=10,
        n_not_applicable=0,
        n_mapped=2,
        n_implemented=3,
        n_evidenced=5,
        n_unmapped=0,
        n_stale=0,
    )
    assert evidence_completeness(summary) == pytest.approx(0.5)


def test_evidence_completeness_full():
    summary = CatalogSummary(
        n_controls=4,
        n_not_applicable=1,
        n_mapped=0,
        n_implemented=0,
        n_evidenced=3,
        n_unmapped=0,
        n_stale=0,
    )
    # Denom = 4 - 1 = 3; evidenced = 3 → 1.0.
    assert evidence_completeness(summary) == 1.0


# --- Render ----------------------------------------------------


def test_render_gap_emoji_per_kind():
    for kind in ("unmapped", "not_implemented", "no_evidence", "stale"):
        gap = MappingGap(
            annex_control_id="A.5.1",
            theme=AnnexATheme.ORGANISATIONAL,
            gap_kind=kind,
        )
        out = render_gap(gap)
        assert kind in out


def test_render_summary_format():
    summary = summarise_catalog(
        [_ctrl()],
        [_mapping(status=MappingStatus.EVIDENCED, last_evidenced_on=date(2026, 5, 1))],
        as_of=date(2026, 5, 11),
    )
    out = render_summary(summary)
    assert "ISO 27001" in out
    assert "evidenced=1" in out
