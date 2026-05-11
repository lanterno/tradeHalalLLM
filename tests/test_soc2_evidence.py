"""Tests for ops/soc2_evidence.py — Round-5 Wave 19.E."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.ops.soc2_evidence import (
    AuditWindow,
    BundleStatus,
    BundleSummary,
    Control,
    ControlCoverage,
    EvidenceArtefact,
    EvidenceBundle,
    EvidenceKind,
    TrustServiceCategory,
    add_artefact,
    compute_coverage,
    coverage_for_catalog,
    integrity_of,
    render_coverage,
    render_summary,
    summarise_bundle,
    transition_bundle,
)


def _control(
    control_id: str = "CC-1",
    category: TrustServiceCategory = TrustServiceCategory.SECURITY,
    expected_count: int = 12,
    kinds: tuple[EvidenceKind, ...] = (EvidenceKind.ACCESS_LOG,),
) -> Control:
    return Control(
        control_id=control_id,
        category=category,
        title="Test control",
        expected_kinds=kinds,
        expected_periodic_count=expected_count,
    )


def _artefact(
    artefact_id: str = "A1",
    control_id: str = "CC-1",
    kind: EvidenceKind = EvidenceKind.ACCESS_LOG,
    collected_on: date = date(2026, 5, 1),
    uri: str = "s3://bucket/key",
    integrity_hash: str = "a" * 64,
) -> EvidenceArtefact:
    return EvidenceArtefact(
        artefact_id=artefact_id,
        control_id=control_id,
        kind=kind,
        collected_on=collected_on,
        uri=uri,
        integrity_hash=integrity_hash,
    )


def _window() -> AuditWindow:
    return AuditWindow(start=date(2026, 1, 1), end=date(2026, 12, 31))


# --- Control validation ---------------------------------------


def test_control_valid():
    c = _control()
    assert c.expected_periodic_count == 12


def test_control_empty_id_rejected():
    with pytest.raises(ValueError):
        _control(control_id="")


def test_control_empty_kinds_rejected():
    with pytest.raises(ValueError):
        Control(
            control_id="CC-1",
            category=TrustServiceCategory.SECURITY,
            title="x",
            expected_kinds=(),
            expected_periodic_count=12,
        )


def test_control_duplicate_kinds_rejected():
    with pytest.raises(ValueError):
        Control(
            control_id="CC-1",
            category=TrustServiceCategory.SECURITY,
            title="x",
            expected_kinds=(EvidenceKind.ACCESS_LOG, EvidenceKind.ACCESS_LOG),
            expected_periodic_count=12,
        )


def test_control_zero_periodic_rejected():
    with pytest.raises(ValueError):
        _control(expected_count=0)


# --- EvidenceArtefact validation ------------------------------


def test_artefact_valid():
    a = _artefact()
    assert a.integrity_hash == "a" * 64


def test_artefact_invalid_hash_length_rejected():
    with pytest.raises(ValueError):
        _artefact(integrity_hash="short")


def test_artefact_empty_uri_rejected():
    with pytest.raises(ValueError):
        _artefact(uri=" ")


def test_artefact_immutable():
    a = _artefact()
    with pytest.raises(AttributeError):
        a.uri = "x"  # type: ignore[misc]


def test_integrity_of_helper():
    h = integrity_of(b"test")
    assert len(h) == 64
    assert integrity_of(b"test") == integrity_of(b"test")


# --- AuditWindow ----------------------------------------------


def test_window_valid():
    w = _window()
    assert w.days() == 365


def test_window_inverted_rejected():
    with pytest.raises(ValueError):
        AuditWindow(start=date(2026, 12, 31), end=date(2026, 1, 1))


def test_window_contains():
    w = _window()
    assert w.contains(date(2026, 6, 15))
    assert w.contains(date(2026, 1, 1))
    assert w.contains(date(2026, 12, 31))
    assert not w.contains(date(2025, 12, 31))


# --- compute_coverage ----------------------------------------


def test_coverage_full_complete():
    control = _control(expected_count=3)
    artefacts = [
        _artefact(artefact_id=f"A{i}", collected_on=date(2026, m, 1))
        for i, m in enumerate((1, 6, 12))
    ]
    cov = compute_coverage(control, artefacts, window=_window())
    assert cov.actual_count == 3
    assert cov.coverage_ratio == 1.0
    assert cov.is_complete
    assert cov.missing_kinds == ()


def test_coverage_partial_below_threshold():
    control = _control(expected_count=12)
    artefacts = [_artefact()]
    cov = compute_coverage(control, artefacts, window=_window())
    assert cov.actual_count == 1
    assert cov.coverage_ratio < 0.80
    assert not cov.is_complete


def test_coverage_excludes_outside_window():
    control = _control(expected_count=2)
    artefacts = [
        _artefact(artefact_id="A1", collected_on=date(2025, 1, 1)),
        _artefact(artefact_id="A2", collected_on=date(2026, 6, 1)),
    ]
    cov = compute_coverage(control, artefacts, window=_window())
    assert cov.actual_count == 1


def test_coverage_filters_by_control_id():
    control = _control(control_id="CC-1", expected_count=1)
    artefacts = [
        _artefact(artefact_id="A1", control_id="CC-2"),
    ]
    cov = compute_coverage(control, artefacts, window=_window())
    assert cov.actual_count == 0


def test_coverage_ratio_capped_at_one():
    control = _control(expected_count=2)
    artefacts = [
        _artefact(artefact_id=f"A{i}", collected_on=date(2026, i + 1, 1)) for i in range(5)
    ]
    cov = compute_coverage(control, artefacts, window=_window())
    assert cov.coverage_ratio == 1.0


def test_coverage_missing_kinds_pinned():
    control = _control(
        expected_count=2,
        kinds=(EvidenceKind.ACCESS_LOG, EvidenceKind.CHANGE_RECORD),
    )
    artefacts = [
        _artefact(kind=EvidenceKind.ACCESS_LOG, collected_on=date(2026, 1, 1)),
        _artefact(
            artefact_id="A2",
            kind=EvidenceKind.ACCESS_LOG,
            collected_on=date(2026, 7, 1),
        ),
    ]
    cov = compute_coverage(control, artefacts, window=_window())
    assert EvidenceKind.CHANGE_RECORD in cov.missing_kinds
    assert not cov.is_complete


def test_coverage_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        compute_coverage(_control(), [], window=_window(), completeness_threshold=0.0)


# --- coverage_for_catalog ----------------------------------


def test_coverage_for_catalog_empty_rejected():
    with pytest.raises(ValueError):
        coverage_for_catalog([], [], window=_window())


def test_coverage_for_catalog_duplicate_rejected():
    bad = [_control(control_id="CC-1"), _control(control_id="CC-1")]
    with pytest.raises(ValueError):
        coverage_for_catalog(bad, [], window=_window())


def test_coverage_for_catalog_one_row_per_control():
    catalog = [
        _control(control_id="CC-1"),
        _control(control_id="CC-2"),
    ]
    out = coverage_for_catalog(catalog, [], window=_window())
    assert len(out) == 2


# --- EvidenceBundle ----------------------------------------


def _bundle(
    catalog: tuple[Control, ...] | None = None,
    artefacts: tuple[EvidenceArtefact, ...] = (),
    status: BundleStatus = BundleStatus.DRAFT,
    submitted_on: date | None = None,
    audited_on: date | None = None,
) -> EvidenceBundle:
    if catalog is None:
        catalog = (_control(),)
    return EvidenceBundle(
        bundle_id="B1",
        window=_window(),
        catalog=catalog,
        artefacts=artefacts,
        status=status,
        submitted_on=submitted_on,
        audited_on=audited_on,
    )


def test_bundle_valid():
    b = _bundle()
    assert b.status is BundleStatus.DRAFT


def test_bundle_empty_catalog_rejected():
    with pytest.raises(ValueError):
        EvidenceBundle(
            bundle_id="B1",
            window=_window(),
            catalog=(),
            artefacts=(),
        )


def test_bundle_artefact_references_unknown_control_rejected():
    a = _artefact(control_id="CC-X")
    with pytest.raises(ValueError):
        _bundle(artefacts=(a,))


def test_bundle_duplicate_artefact_id_rejected():
    a1 = _artefact(artefact_id="A1")
    a2 = _artefact(artefact_id="A1", collected_on=date(2026, 7, 1))
    with pytest.raises(ValueError):
        _bundle(artefacts=(a1, a2))


def test_bundle_submitted_without_date_rejected():
    with pytest.raises(ValueError):
        _bundle(status=BundleStatus.SUBMITTED, submitted_on=None)


def test_bundle_audited_without_date_rejected():
    with pytest.raises(ValueError):
        _bundle(
            status=BundleStatus.AUDITED,
            submitted_on=date(2026, 1, 5),
            audited_on=None,
        )


def test_bundle_audited_before_submitted_rejected():
    with pytest.raises(ValueError):
        _bundle(
            status=BundleStatus.AUDITED,
            submitted_on=date(2026, 6, 1),
            audited_on=date(2026, 1, 1),
        )


# --- transition_bundle ------------------------------------


def test_transition_draft_to_submitted():
    b = _bundle()
    b2 = transition_bundle(b, new_status=BundleStatus.SUBMITTED, at=date(2026, 6, 1))
    assert b2.status is BundleStatus.SUBMITTED
    assert b2.submitted_on == date(2026, 6, 1)


def test_transition_submitted_to_audited():
    b = transition_bundle(_bundle(), new_status=BundleStatus.SUBMITTED, at=date(2026, 6, 1))
    b2 = transition_bundle(b, new_status=BundleStatus.AUDITED, at=date(2026, 7, 1))
    assert b2.status is BundleStatus.AUDITED
    assert b2.audited_on == date(2026, 7, 1)


def test_transition_submitted_to_rejected():
    b = transition_bundle(_bundle(), new_status=BundleStatus.SUBMITTED, at=date(2026, 6, 1))
    b2 = transition_bundle(b, new_status=BundleStatus.REJECTED, at=date(2026, 7, 1))
    assert b2.status is BundleStatus.REJECTED


def test_transition_audited_terminal():
    b = transition_bundle(_bundle(), new_status=BundleStatus.SUBMITTED, at=date(2026, 6, 1))
    b = transition_bundle(b, new_status=BundleStatus.AUDITED, at=date(2026, 7, 1))
    with pytest.raises(ValueError):
        transition_bundle(b, new_status=BundleStatus.SUBMITTED, at=date(2026, 8, 1))


def test_transition_skip_to_audited_rejected():
    b = _bundle()
    with pytest.raises(ValueError):
        transition_bundle(b, new_status=BundleStatus.AUDITED, at=date(2026, 7, 1))


# --- add_artefact ----------------------------------------


def test_add_artefact_in_draft():
    b = _bundle()
    b2 = add_artefact(b, _artefact())
    assert len(b2.artefacts) == 1


def test_add_artefact_outside_draft_rejected():
    b = transition_bundle(_bundle(), new_status=BundleStatus.SUBMITTED, at=date(2026, 6, 1))
    with pytest.raises(ValueError):
        add_artefact(b, _artefact())


# --- summarise_bundle ------------------------------------


def test_summary_complete_bundle():
    catalog = (_control(expected_count=2),)
    artefacts = (
        _artefact(artefact_id="A1", collected_on=date(2026, 1, 1)),
        _artefact(artefact_id="A2", collected_on=date(2026, 6, 1)),
    )
    b = _bundle(catalog=catalog, artefacts=artefacts)
    summary = summarise_bundle(b)
    assert summary.n_complete == 1
    assert summary.n_incomplete == 0
    assert summary.coverage_avg == 1.0


def test_summary_incomplete_bundle():
    catalog = (_control(expected_count=12),)
    artefacts = (_artefact(),)
    b = _bundle(catalog=catalog, artefacts=artefacts)
    summary = summarise_bundle(b)
    assert summary.n_complete == 0
    assert summary.n_incomplete == 1
    assert "CC-1" in summary.incomplete_control_ids


def test_summary_status_propagated():
    b = transition_bundle(_bundle(), new_status=BundleStatus.SUBMITTED, at=date(2026, 6, 1))
    summary = summarise_bundle(b)
    assert summary.status is BundleStatus.SUBMITTED


# --- Render ---------------------------------------------


def test_render_coverage_complete_emoji():
    cov = ControlCoverage(
        control_id="CC-1",
        category=TrustServiceCategory.SECURITY,
        expected_count=3,
        actual_count=3,
        coverage_ratio=1.0,
        missing_kinds=(),
        is_complete=True,
    )
    out = render_coverage(cov)
    assert "✅" in out


def test_render_coverage_incomplete_emoji():
    cov = ControlCoverage(
        control_id="CC-1",
        category=TrustServiceCategory.SECURITY,
        expected_count=12,
        actual_count=5,
        coverage_ratio=5 / 12,
        missing_kinds=(EvidenceKind.CHANGE_RECORD,),
        is_complete=False,
    )
    out = render_coverage(cov)
    assert "🟡" in out
    assert "missing=" in out


def test_render_summary_status_emoji():
    summary = BundleSummary(
        bundle_id="B1",
        status=BundleStatus.AUDITED,
        n_controls=5,
        n_complete=5,
        n_incomplete=0,
        coverage_avg=1.0,
        incomplete_control_ids=(),
    )
    out = render_summary(summary)
    assert "✅" in out
