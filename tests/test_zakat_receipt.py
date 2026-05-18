"""Tests for halal/zakat_receipt.py — Round-5 Wave 18.I."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.zakat_receipt import (
    ReceiptStatus,
    ZakatReceipt,
    canonical_digest,
    counter_sign,
    draft_receipt,
    render_receipt,
    sign_by_operator,
    void_receipt,
)


def _draft() -> ZakatReceipt:
    return draft_receipt(
        receipt_id="ZR-001",
        operator_handle="op-1",
        recipient_charity="Helping Hand",
        amount=2500.0,
        currency="SAR",
        payment_date=date(2026, 5, 1),
        zakat_year_label="1447 AH",
        nisab_basis="silver",
    )


_SIG = "0" * 60 + "abcd"


# --- Validation ----------------------------------------


def test_status_string_values():
    assert ReceiptStatus.DRAFT.value == "draft"
    assert ReceiptStatus.SIGNED_BY_OPERATOR.value == "signed_by_operator"
    assert ReceiptStatus.COUNTER_SIGNED.value == "counter_signed"
    assert ReceiptStatus.VOID.value == "void"


def test_draft_receipt_basic():
    r = _draft()
    assert r.status is ReceiptStatus.DRAFT
    assert r.operator_signature_hex == ""


def test_draft_email_handle_rejected():
    with pytest.raises(ValueError):
        draft_receipt(
            receipt_id="ZR-1",
            operator_handle="ops@example.com",
            recipient_charity="x",
            amount=100,
            currency="SAR",
            payment_date=date(2026, 5, 1),
            zakat_year_label="1447 AH",
        )


def test_draft_zero_amount_rejected():
    with pytest.raises(ValueError):
        draft_receipt(
            receipt_id="ZR-1",
            operator_handle="op-1",
            recipient_charity="x",
            amount=0,
            currency="SAR",
            payment_date=date(2026, 5, 1),
            zakat_year_label="1447 AH",
        )


def test_draft_long_currency_rejected():
    with pytest.raises(ValueError):
        draft_receipt(
            receipt_id="ZR-1",
            operator_handle="op-1",
            recipient_charity="x",
            amount=100,
            currency="LONG_CURRENCY_CODE",
            payment_date=date(2026, 5, 1),
            zakat_year_label="1447 AH",
        )


def test_draft_with_signatures_rejected():
    """A DRAFT cannot have signatures pre-filled."""
    with pytest.raises(ValueError):
        ZakatReceipt(
            receipt_id="ZR-1",
            operator_handle="op-1",
            recipient_charity="x",
            amount=100,
            currency="SAR",
            payment_date=date(2026, 5, 1),
            zakat_year_label="1447 AH",
            nisab_basis="silver",
            status=ReceiptStatus.DRAFT,
            operator_signature_hex=_SIG,
            recipient_signature_hex="",
        )


def test_signed_by_operator_must_have_op_sig_only():
    with pytest.raises(ValueError):
        ZakatReceipt(
            receipt_id="ZR-1",
            operator_handle="op-1",
            recipient_charity="x",
            amount=100,
            currency="SAR",
            payment_date=date(2026, 5, 1),
            zakat_year_label="1447 AH",
            nisab_basis="silver",
            status=ReceiptStatus.SIGNED_BY_OPERATOR,
            operator_signature_hex="",  # missing
            recipient_signature_hex="",
        )


def test_counter_signed_requires_both_sigs():
    with pytest.raises(ValueError):
        ZakatReceipt(
            receipt_id="ZR-1",
            operator_handle="op-1",
            recipient_charity="x",
            amount=100,
            currency="SAR",
            payment_date=date(2026, 5, 1),
            zakat_year_label="1447 AH",
            nisab_basis="silver",
            status=ReceiptStatus.COUNTER_SIGNED,
            operator_signature_hex=_SIG,
            recipient_signature_hex="",  # missing
        )


# --- State transitions ----------------------------------


def test_sign_by_operator_advances_status():
    r = _draft()
    signed = sign_by_operator(r, signature_hex=_SIG)
    assert signed.status is ReceiptStatus.SIGNED_BY_OPERATOR
    assert signed.operator_signature_hex == _SIG


def test_sign_by_operator_only_from_draft():
    r = _draft()
    signed = sign_by_operator(r, signature_hex=_SIG)
    with pytest.raises(ValueError):
        sign_by_operator(signed, signature_hex=_SIG)


def test_sign_invalid_signature_length_rejected():
    r = _draft()
    with pytest.raises(ValueError):
        sign_by_operator(r, signature_hex="abc")


def test_counter_sign_advances_status():
    r = _draft()
    signed = sign_by_operator(r, signature_hex=_SIG)
    counter = counter_sign(signed, signature_hex="f" * 64)
    assert counter.status is ReceiptStatus.COUNTER_SIGNED
    assert counter.recipient_signature_hex == "f" * 64


def test_counter_sign_requires_signed_by_operator():
    r = _draft()
    with pytest.raises(ValueError):
        counter_sign(r, signature_hex=_SIG)


def test_void_receipt_advances_status():
    r = _draft()
    voided = void_receipt(r)
    assert voided.status is ReceiptStatus.VOID


# --- Canonical digest ---------------------------------


def test_canonical_digest_64_hex():
    d = canonical_digest(_draft())
    assert len(d) == 64
    assert all(c in "0123456789abcdef" for c in d)


def test_canonical_digest_deterministic():
    a = canonical_digest(_draft())
    b = canonical_digest(_draft())
    assert a == b


def test_canonical_digest_changes_with_amount():
    a = canonical_digest(_draft())
    other = draft_receipt(
        receipt_id="ZR-001",
        operator_handle="op-1",
        recipient_charity="Helping Hand",
        amount=9999.0,
        currency="SAR",
        payment_date=date(2026, 5, 1),
        zakat_year_label="1447 AH",
        nisab_basis="silver",
    )
    b = canonical_digest(other)
    assert a != b


def test_canonical_digest_does_not_depend_on_signatures():
    """Signatures live OUTSIDE the canonical content (they sign it)."""
    d_draft = canonical_digest(_draft())
    signed = sign_by_operator(_draft(), signature_hex=_SIG)
    d_signed = canonical_digest(signed)
    assert d_draft == d_signed


# --- Render ------------------------------------------


def test_render_includes_summary():
    r = _draft()
    out = render_receipt(r)
    assert "ZR-001" in out
    assert "Helping Hand" in out
    assert "draft" in out


def test_render_signed_marks_signature():
    r = _draft()
    signed = sign_by_operator(r, signature_hex=_SIG)
    out = render_receipt(signed)
    assert _SIG[:8] in out


def test_render_no_secret_leak():
    r = _draft()
    out = render_receipt(r)
    for token in (
        "@",
        "zoom.us",
        "meet.google",
        "private_email",
        "+1-",
        "Authorization",
        "IBAN",
        "Bank-",
    ):
        assert token not in out


# --- E2E ----------------------------------------


def test_e2e_full_signing_workflow():
    """Operator drafts → signs → recipient counter-signs."""
    r = _draft()
    assert r.status is ReceiptStatus.DRAFT
    op_signed = sign_by_operator(r, signature_hex=_SIG)
    assert op_signed.status is ReceiptStatus.SIGNED_BY_OPERATOR
    final = counter_sign(op_signed, signature_hex="f" * 64)
    assert final.status is ReceiptStatus.COUNTER_SIGNED


def test_replay_consistency():
    a = _draft()
    b = _draft()
    assert canonical_digest(a) == canonical_digest(b)
