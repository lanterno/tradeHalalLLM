"""Tests for ops/scholar_attestation.py — Round-5 Wave 19.I."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from halal_trader.ops.scholar_attestation import (
    Attestation,
    Envelope,
    EnvelopeStatus,
    ReportingPeriod,
    SignatureEvent,
    anchor_hash,
    complete,
    countersign,
    render_envelope,
    send_envelope,
    verify_envelope,
    void,
)


def _attest(
    attestation_id: str = "AT1",
    scholar_id: str = "scholar-ali",
    year: int = 2026,
    period: ReportingPeriod = ReportingPeriod.Q2,
    content: str = "I, scholar of record, attest that…",
    universe_size: int = 1500,
    products: int = 5,
    drafted_on: date = date(2026, 4, 1),
) -> Attestation:
    return Attestation(
        attestation_id=attestation_id,
        scholar_id=scholar_id,
        year=year,
        period=period,
        content=content,
        universe_size=universe_size,
        structured_products_count=products,
        drafted_on=drafted_on,
    )


def _envelope(
    envelope_id: str = "E1",
    attestation: Attestation | None = None,
    status: EnvelopeStatus = EnvelopeStatus.DRAFT,
    sent_at: datetime | None = None,
    signatures: tuple[SignatureEvent, ...] = (),
    void_reason: str = "",
) -> Envelope:
    a = attestation or _attest()
    return Envelope(
        envelope_id=envelope_id,
        attestation=a,
        anchor=anchor_hash(a),
        doc_store_uri="s3://docs/attestation-1.pdf",
        status=status,
        sent_at=sent_at,
        signatures=signatures,
        void_reason=void_reason,
    )


def _sig(
    signer_id: str = "scholar-ali",
    signer_role: str = "scholar",
    signed_at: datetime = datetime(2026, 4, 5, 12, 0),
    method: str = "docusign",
) -> SignatureEvent:
    return SignatureEvent(
        signer_id=signer_id,
        signer_role=signer_role,
        signed_at=signed_at,
        method=method,
    )


# --- Attestation validation -------------------------------------


def test_attestation_valid():
    a = _attest()
    assert a.scholar_id == "scholar-ali"


def test_attestation_empty_id_rejected():
    with pytest.raises(ValueError):
        _attest(attestation_id="")


def test_attestation_invalid_year_rejected():
    with pytest.raises(ValueError):
        _attest(year=1999)
    with pytest.raises(ValueError):
        _attest(year=2200)


def test_attestation_empty_content_rejected():
    with pytest.raises(ValueError):
        _attest(content=" ")


def test_attestation_long_content_rejected():
    with pytest.raises(ValueError):
        _attest(content="x" * 60_000)


def test_attestation_negative_universe_rejected():
    with pytest.raises(ValueError):
        _attest(universe_size=-1)


def test_attestation_content_hash_stable():
    a1 = _attest()
    a2 = _attest()
    assert a1.content_hash() == a2.content_hash()


def test_attestation_immutable():
    a = _attest()
    with pytest.raises(AttributeError):
        a.scholar_id = "x"  # type: ignore[misc]


# --- anchor_hash ------------------------------------------------


def test_anchor_hash_stable_same_inputs():
    a1 = _attest()
    a2 = _attest()
    assert anchor_hash(a1) == anchor_hash(a2)


def test_anchor_hash_changes_with_scholar():
    a1 = _attest(scholar_id="scholar-ali")
    a2 = _attest(scholar_id="scholar-yusuf")
    assert anchor_hash(a1) != anchor_hash(a2)


def test_anchor_hash_changes_with_period():
    a1 = _attest(period=ReportingPeriod.Q1)
    a2 = _attest(period=ReportingPeriod.Q2)
    assert anchor_hash(a1) != anchor_hash(a2)


def test_anchor_hash_changes_with_content():
    a1 = _attest(content="A")
    a2 = _attest(content="B")
    assert anchor_hash(a1) != anchor_hash(a2)


# --- SignatureEvent validation ---------------------------------


def test_signature_valid():
    s = _sig()
    assert s.method == "docusign"


def test_signature_empty_signer_rejected():
    with pytest.raises(ValueError):
        _sig(signer_id="")


def test_signature_empty_role_rejected():
    with pytest.raises(ValueError):
        _sig(signer_role="")


def test_signature_empty_method_rejected():
    with pytest.raises(ValueError):
        _sig(method=" ")


# --- Envelope validation -------------------------------------


def test_envelope_valid():
    e = _envelope()
    assert e.status is EnvelopeStatus.DRAFT
    assert verify_envelope(e)


def test_envelope_empty_id_rejected():
    with pytest.raises(ValueError):
        _envelope(envelope_id="")


def test_envelope_wrong_anchor_rejected():
    a = _attest()
    with pytest.raises(ValueError):
        Envelope(
            envelope_id="E1",
            attestation=a,
            anchor="wrong-anchor",
            doc_store_uri="s3://docs/x.pdf",
        )


def test_envelope_sent_without_date_rejected():
    with pytest.raises(ValueError):
        _envelope(status=EnvelopeStatus.SENT, sent_at=None)


def test_envelope_countersigned_without_scholar_sig_rejected():
    with pytest.raises(ValueError):
        _envelope(
            status=EnvelopeStatus.COUNTERSIGNED,
            sent_at=datetime(2026, 4, 1),
            signatures=(),
        )


def test_envelope_completed_without_platform_sig_rejected():
    scholar_sig = _sig()
    with pytest.raises(ValueError):
        _envelope(
            status=EnvelopeStatus.COMPLETED,
            sent_at=datetime(2026, 4, 1),
            signatures=(scholar_sig,),
        )


def test_envelope_voided_requires_reason():
    with pytest.raises(ValueError):
        _envelope(status=EnvelopeStatus.VOIDED, void_reason="")


# --- FSM transitions ----------------------------------------


def test_send_envelope_draft_to_sent():
    e = _envelope()
    e2 = send_envelope(e, at=datetime(2026, 4, 1, 9, 0))
    assert e2.status is EnvelopeStatus.SENT


def test_send_envelope_non_draft_rejected():
    e = _envelope(
        status=EnvelopeStatus.SENT,
        sent_at=datetime(2026, 4, 1),
    )
    with pytest.raises(ValueError):
        send_envelope(e, at=datetime(2026, 4, 2))


def test_countersign_sent_to_countersigned():
    e = send_envelope(_envelope(), at=datetime(2026, 4, 1))
    sig = _sig(signer_id="scholar-ali")
    e2 = countersign(e, scholar_signature=sig)
    assert e2.status is EnvelopeStatus.COUNTERSIGNED
    assert any(s.signer_role == "scholar" for s in e2.signatures)


def test_countersign_wrong_role_rejected():
    e = send_envelope(_envelope(), at=datetime(2026, 4, 1))
    sig = _sig(signer_role="platform_officer", signer_id="scholar-ali")
    with pytest.raises(ValueError):
        countersign(e, scholar_signature=sig)


def test_countersign_wrong_scholar_id_rejected():
    e = send_envelope(_envelope(), at=datetime(2026, 4, 1))
    sig = _sig(signer_id="other-scholar")
    with pytest.raises(ValueError):
        countersign(e, scholar_signature=sig)


def test_countersign_non_sent_rejected():
    e = _envelope()  # DRAFT
    sig = _sig()
    with pytest.raises(ValueError):
        countersign(e, scholar_signature=sig)


def test_complete_countersigned_to_completed():
    e = send_envelope(_envelope(), at=datetime(2026, 4, 1))
    e = countersign(e, scholar_signature=_sig())
    plat = _sig(
        signer_id="platform-officer-1",
        signer_role="platform_officer",
        signed_at=datetime(2026, 4, 6),
    )
    e2 = complete(e, platform_signature=plat)
    assert e2.status is EnvelopeStatus.COMPLETED


def test_complete_wrong_role_rejected():
    e = send_envelope(_envelope(), at=datetime(2026, 4, 1))
    e = countersign(e, scholar_signature=_sig())
    bad = _sig(signer_id="someone-else", signer_role="auditor")
    with pytest.raises(ValueError):
        complete(e, platform_signature=bad)


def test_complete_non_countersigned_rejected():
    e = _envelope()
    plat = _sig(signer_id="po-1", signer_role="platform_officer")
    with pytest.raises(ValueError):
        complete(e, platform_signature=plat)


def test_void_from_draft():
    e = _envelope()
    e2 = void(e, reason="scholar requested changes")
    assert e2.status is EnvelopeStatus.VOIDED


def test_void_from_completed_rejected():
    e = send_envelope(_envelope(), at=datetime(2026, 4, 1))
    e = countersign(e, scholar_signature=_sig())
    e = complete(
        e,
        platform_signature=_sig(signer_id="po-1", signer_role="platform_officer"),
    )
    with pytest.raises(ValueError):
        void(e, reason="too late")


def test_void_empty_reason_rejected():
    e = _envelope()
    with pytest.raises(ValueError):
        void(e, reason=" ")


# --- verify_envelope ----------------------------------------


def test_verify_clean_envelope():
    e = _envelope()
    assert verify_envelope(e)


# --- Render ------------------------------------------------


def test_render_no_secret_leak():
    a = _attest(scholar_id="scholar-ali@example.com")
    e = Envelope(
        envelope_id="E1",
        attestation=a,
        anchor=anchor_hash(a),
        doc_store_uri="s3://very-private-bucket/doc.pdf",
    )
    out = render_envelope(e)
    assert "scholar-ali@example.com" not in out
    assert "very-private-bucket" not in out


def test_render_status_emoji():
    e = _envelope()
    out = render_envelope(e)
    assert "📝" in out


def test_render_signatures_visible():
    e = send_envelope(_envelope(), at=datetime(2026, 4, 1))
    e = countersign(e, scholar_signature=_sig())
    out = render_envelope(e)
    assert "Signatures" in out
    assert "scholar" in out


def test_render_voided_shows_reason():
    e = void(_envelope(), reason="scholar requested changes")
    out = render_envelope(e)
    assert "Voided" in out
    assert "requested changes" in out
