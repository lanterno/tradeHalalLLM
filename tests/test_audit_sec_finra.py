"""Tests for ops/audit_sec_finra.py — Round-5 Wave 19.A."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.ops.audit_sec_finra import (
    REQUIRED_ARTEFACTS,
    Artefact,
    ArtefactKind,
    AuditBundle,
    build_bundle,
    hash_payload,
    render_bundle,
)


def _hash() -> str:
    return hash_payload(b"sample-payload")


def _artefact(kind: ArtefactKind, title: str = "x") -> Artefact:
    return Artefact(
        kind=kind,
        title=title,
        payload_digest=_hash(),
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        record_count=100,
    )


# --- Validation -----------------------------------------------


def test_artefact_kind_string_values():
    assert ArtefactKind.TRADE_BLOTTER.value == "trade_blotter"
    assert ArtefactKind.ORDER_ROUTING_LOG.value == "order_routing_log"
    assert ArtefactKind.OPERATOR_CERTIFICATION.value == "operator_certification"


def test_required_artefacts_pin():
    assert REQUIRED_ARTEFACTS == frozenset(
        {
            ArtefactKind.TRADE_BLOTTER,
            ArtefactKind.ORDER_ROUTING_LOG,
            ArtefactKind.RISK_POLICY,
            ArtefactKind.OPERATOR_CERTIFICATION,
        }
    )


def test_artefact_empty_title_rejected():
    with pytest.raises(ValueError):
        Artefact(
            kind=ArtefactKind.TRADE_BLOTTER,
            title="",
            payload_digest=_hash(),
            period_start=date(2026, 1, 1),
            period_end=date(2026, 12, 31),
            record_count=10,
        )


def test_artefact_invalid_digest_rejected():
    with pytest.raises(ValueError):
        Artefact(
            kind=ArtefactKind.TRADE_BLOTTER,
            title="x",
            payload_digest="abc",
            period_start=date(2026, 1, 1),
            period_end=date(2026, 12, 31),
            record_count=10,
        )


def test_artefact_period_inversion_rejected():
    with pytest.raises(ValueError):
        Artefact(
            kind=ArtefactKind.TRADE_BLOTTER,
            title="x",
            payload_digest=_hash(),
            period_start=date(2026, 12, 31),
            period_end=date(2026, 1, 1),
            record_count=10,
        )


def test_artefact_negative_record_count_rejected():
    with pytest.raises(ValueError):
        Artefact(
            kind=ArtefactKind.TRADE_BLOTTER,
            title="x",
            payload_digest=_hash(),
            period_start=date(2026, 1, 1),
            period_end=date(2026, 12, 31),
            record_count=-1,
        )


def test_bundle_email_handle_rejected():
    with pytest.raises(ValueError):
        AuditBundle(
            operator_handle="ops@example.com",
            audit_period_start=date(2026, 1, 1),
            audit_period_end=date(2026, 12, 31),
            artefacts=(),
            merkle_root="0" * 64,
            is_complete=False,
            missing_required=frozenset(REQUIRED_ARTEFACTS),
        )


def test_bundle_invariant_complete_with_missing_rejected():
    with pytest.raises(ValueError):
        AuditBundle(
            operator_handle="op-1",
            audit_period_start=date(2026, 1, 1),
            audit_period_end=date(2026, 12, 31),
            artefacts=(),
            merkle_root="0" * 64,
            is_complete=True,
            missing_required=frozenset({ArtefactKind.TRADE_BLOTTER}),
        )


def test_bundle_invariant_incomplete_without_missing_rejected():
    with pytest.raises(ValueError):
        AuditBundle(
            operator_handle="op-1",
            audit_period_start=date(2026, 1, 1),
            audit_period_end=date(2026, 12, 31),
            artefacts=(),
            merkle_root="0" * 64,
            is_complete=False,
            missing_required=frozenset(),
        )


# --- build_bundle ---------------------------------------------


def test_build_complete_bundle():
    artefacts = [_artefact(k) for k in REQUIRED_ARTEFACTS]
    bundle = build_bundle(
        operator_handle="op-1",
        audit_period_start=date(2026, 1, 1),
        audit_period_end=date(2026, 12, 31),
        artefacts=artefacts,
    )
    assert bundle.is_complete
    assert bundle.missing_required == frozenset()


def test_build_incomplete_bundle_missing_artefact():
    """Missing the trade blotter → incomplete + flagged."""
    artefacts = [
        _artefact(ArtefactKind.ORDER_ROUTING_LOG),
        _artefact(ArtefactKind.RISK_POLICY),
        _artefact(ArtefactKind.OPERATOR_CERTIFICATION),
    ]
    bundle = build_bundle(
        operator_handle="op-1",
        audit_period_start=date(2026, 1, 1),
        audit_period_end=date(2026, 12, 31),
        artefacts=artefacts,
    )
    assert not bundle.is_complete
    assert ArtefactKind.TRADE_BLOTTER in bundle.missing_required


def test_build_artefact_outside_window_rejected():
    artefact = Artefact(
        kind=ArtefactKind.TRADE_BLOTTER,
        title="x",
        payload_digest=_hash(),
        period_start=date(2025, 1, 1),  # before window
        period_end=date(2025, 12, 31),
        record_count=10,
    )
    with pytest.raises(ValueError):
        build_bundle(
            operator_handle="op-1",
            audit_period_start=date(2026, 1, 1),
            audit_period_end=date(2026, 12, 31),
            artefacts=[artefact],
        )


def test_merkle_root_changes_with_artefact_set():
    """Different artefacts → different Merkle roots."""
    bundle_a = build_bundle(
        operator_handle="op-1",
        audit_period_start=date(2026, 1, 1),
        audit_period_end=date(2026, 12, 31),
        artefacts=[_artefact(k) for k in REQUIRED_ARTEFACTS],
    )
    bundle_b = build_bundle(
        operator_handle="op-1",
        audit_period_start=date(2026, 1, 1),
        audit_period_end=date(2026, 12, 31),
        artefacts=[_artefact(k) for k in list(REQUIRED_ARTEFACTS)[:2]],
    )
    assert bundle_a.merkle_root != bundle_b.merkle_root


def test_merkle_root_stable_for_same_artefacts():
    a1 = build_bundle(
        operator_handle="op-1",
        audit_period_start=date(2026, 1, 1),
        audit_period_end=date(2026, 12, 31),
        artefacts=[_artefact(k) for k in REQUIRED_ARTEFACTS],
    )
    a2 = build_bundle(
        operator_handle="op-1",
        audit_period_start=date(2026, 1, 1),
        audit_period_end=date(2026, 12, 31),
        artefacts=[_artefact(k) for k in REQUIRED_ARTEFACTS],
    )
    assert a1.merkle_root == a2.merkle_root


def test_hash_payload_returns_64_hex():
    h = hash_payload(b"hello")
    assert len(h) == 64


# --- Render --------------------------------------------------


def test_render_complete_bundle():
    bundle = build_bundle(
        operator_handle="op-1",
        audit_period_start=date(2026, 1, 1),
        audit_period_end=date(2026, 12, 31),
        artefacts=[_artefact(k) for k in REQUIRED_ARTEFACTS],
    )
    out = render_bundle(bundle)
    assert "✅" in out
    assert "op-1" in out


def test_render_incomplete_bundle_marks_missing():
    bundle = build_bundle(
        operator_handle="op-1",
        audit_period_start=date(2026, 1, 1),
        audit_period_end=date(2026, 12, 31),
        artefacts=[_artefact(ArtefactKind.OPERATOR_CERTIFICATION)],
    )
    out = render_bundle(bundle)
    assert "⚠️" in out
    assert "missing required" in out


def test_render_no_secret_leak():
    bundle = build_bundle(
        operator_handle="op-1",
        audit_period_start=date(2026, 1, 1),
        audit_period_end=date(2026, 12, 31),
        artefacts=[_artefact(k) for k in REQUIRED_ARTEFACTS],
    )
    out = render_bundle(bundle)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization", "SSN", "TaxID"):
        assert token not in out


# --- E2E -------------------------------------------------


def test_e2e_full_audit_bundle_complete_and_anchored():
    """Operator hands the bundle to FINRA examiner; bundle is complete + verifiable."""
    artefacts = [_artefact(k) for k in ArtefactKind]  # all 8 types
    bundle = build_bundle(
        operator_handle="op-1",
        audit_period_start=date(2026, 1, 1),
        audit_period_end=date(2026, 12, 31),
        artefacts=artefacts,
    )
    assert bundle.is_complete
    assert len(bundle.artefacts) == 8


def test_replay_consistency():
    artefacts = [_artefact(k) for k in REQUIRED_ARTEFACTS]
    a = build_bundle(
        operator_handle="op-1",
        audit_period_start=date(2026, 1, 1),
        audit_period_end=date(2026, 12, 31),
        artefacts=artefacts,
    )
    b = build_bundle(
        operator_handle="op-1",
        audit_period_start=date(2026, 1, 1),
        audit_period_end=date(2026, 12, 31),
        artefacts=artefacts,
    )
    assert a == b
