"""Tests for halal/charity_receipts.py — Round-5 Wave 1.K."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from halal_trader.halal.charity_receipts import (
    ChainedReceipt,
    CharityReceipt,
    append_receipt,
    chain_root,
    find_receipt,
    render_chain,
    render_receipt,
    sign_receipt,
    verify_chain,
    verify_receipt,
)

_KEY = b"unit-test-hmac-key"


def _receipt(receipt_id: str = "RCT-001", **overrides) -> CharityReceipt:
    base = {
        "receipt_id": receipt_id,
        "charity_name": "Helping Hand Foundation",
        "amount_currency": "USD",
        "amount": 123.45,
        "disbursement_date": date(2026, 4, 5),
        "purification_period_start": date(2026, 1, 1),
        "purification_period_end": date(2026, 3, 31),
        "payer_handle": "operator-a",
    }
    base.update(overrides)
    return CharityReceipt(**base)


# --- Receipt validation ------------------------------------------------------


def test_receipt_empty_id_rejected():
    with pytest.raises(ValueError):
        _receipt(receipt_id="")


def test_receipt_empty_charity_rejected():
    with pytest.raises(ValueError):
        _receipt(charity_name="")


def test_receipt_long_currency_rejected():
    with pytest.raises(ValueError):
        _receipt(amount_currency="LONG_CURRENCY_CODE")


def test_receipt_zero_amount_rejected():
    with pytest.raises(ValueError):
        _receipt(amount=0.0)


def test_receipt_negative_amount_rejected():
    with pytest.raises(ValueError):
        _receipt(amount=-1.0)


def test_receipt_period_inversion_rejected():
    with pytest.raises(ValueError):
        _receipt(
            purification_period_start=date(2026, 3, 31),
            purification_period_end=date(2026, 1, 1),
        )


def test_receipt_disbursement_before_period_start_rejected():
    with pytest.raises(ValueError):
        _receipt(disbursement_date=date(2025, 12, 1))


def test_receipt_email_handle_rejected():
    with pytest.raises(ValueError):
        _receipt(payer_handle="ops@example.com")


def test_receipt_immutable():
    r = _receipt()
    with pytest.raises(AttributeError):
        r.amount = 1.0  # type: ignore[misc]


# --- Sign + verify -----------------------------------------------------------


def test_sign_and_verify_roundtrip():
    r = _receipt()
    sig = sign_receipt(r, hmac_key=_KEY)
    assert verify_receipt(r, sig, hmac_key=_KEY)


def test_sign_deterministic():
    r = _receipt()
    a = sign_receipt(r, hmac_key=_KEY)
    b = sign_receipt(r, hmac_key=_KEY)
    assert a == b


def test_sign_with_empty_key_rejected():
    r = _receipt()
    with pytest.raises(ValueError):
        sign_receipt(r, hmac_key=b"")


def test_verify_wrong_key_fails():
    r = _receipt()
    sig = sign_receipt(r, hmac_key=_KEY)
    assert not verify_receipt(r, sig, hmac_key=b"different-key")


def test_verify_tampered_amount_fails():
    r = _receipt()
    sig = sign_receipt(r, hmac_key=_KEY)
    tampered = replace(r, amount=999.99)
    assert not verify_receipt(tampered, sig, hmac_key=_KEY)


def test_verify_tampered_charity_fails():
    r = _receipt()
    sig = sign_receipt(r, hmac_key=_KEY)
    tampered = replace(r, charity_name="Different Charity")
    assert not verify_receipt(tampered, sig, hmac_key=_KEY)


def test_signatures_differ_for_different_receipts():
    a = sign_receipt(_receipt("A"), hmac_key=_KEY)
    b = sign_receipt(_receipt("B"), hmac_key=_KEY)
    assert a != b


def test_signature_is_64_hex_chars():
    sig = sign_receipt(_receipt(), hmac_key=_KEY)
    assert len(sig) == 64
    assert all(c in "0123456789abcdef" for c in sig)


# --- Merkle chain ------------------------------------------------------------


def test_empty_chain_root_is_zero():
    assert chain_root(()) == "0" * 64


def test_append_one_receipt_extends_chain():
    chain = append_receipt((), _receipt(), hmac_key=_KEY)
    assert len(chain) == 1
    assert chain[0].prior_root == "0" * 64
    assert chain[0].current_root != "0" * 64


def test_append_two_receipts_chains():
    chain = append_receipt((), _receipt("A"), hmac_key=_KEY)
    chain = append_receipt(chain, _receipt("B"), hmac_key=_KEY)
    assert chain[1].prior_root == chain[0].current_root
    assert chain[1].current_root != chain[0].current_root


def test_chain_root_advances():
    chain = append_receipt((), _receipt("A"), hmac_key=_KEY)
    root_after_one = chain_root(chain)
    chain = append_receipt(chain, _receipt("B"), hmac_key=_KEY)
    assert chain_root(chain) != root_after_one


def test_verify_clean_chain_passes():
    chain = ()
    for i in range(5):
        chain = append_receipt(chain, _receipt(f"R{i:03d}"), hmac_key=_KEY)
    assert verify_chain(chain, hmac_key=_KEY)


def test_verify_chain_with_wrong_key_fails():
    chain = append_receipt((), _receipt("A"), hmac_key=_KEY)
    assert not verify_chain(chain, hmac_key=b"different")


def test_verify_chain_tampered_receipt_fails():
    chain = append_receipt((), _receipt("A", amount=100.0), hmac_key=_KEY)
    bad_entry = replace(chain[0], receipt=replace(chain[0].receipt, amount=999.0))
    bad_chain = (bad_entry,)
    assert not verify_chain(bad_chain, hmac_key=_KEY)


def test_verify_chain_broken_prior_root_fails():
    chain = append_receipt((), _receipt("A"), hmac_key=_KEY)
    chain = append_receipt(chain, _receipt("B"), hmac_key=_KEY)
    bad_entry = replace(chain[1], prior_root="f" * 64)
    bad_chain = (chain[0], bad_entry)
    assert not verify_chain(bad_chain, hmac_key=_KEY)


def test_verify_chain_broken_current_root_fails():
    chain = append_receipt((), _receipt("A"), hmac_key=_KEY)
    bad_entry = replace(chain[0], current_root="0" * 64)
    bad_chain = (bad_entry,)
    assert not verify_chain(bad_chain, hmac_key=_KEY)


def test_chained_receipt_validation_short_root_rejected():
    with pytest.raises(ValueError):
        ChainedReceipt(
            receipt=_receipt(),
            signature="x" * 64,
            prior_root="abc",
            current_root="0" * 64,
        )


def test_chained_receipt_validation_empty_signature_rejected():
    with pytest.raises(ValueError):
        ChainedReceipt(
            receipt=_receipt(),
            signature="",
            prior_root="0" * 64,
            current_root="0" * 64,
        )


def test_find_receipt_known():
    chain = append_receipt((), _receipt("A"), hmac_key=_KEY)
    chain = append_receipt(chain, _receipt("B"), hmac_key=_KEY)
    found = find_receipt(chain, "B")
    assert found is not None
    assert found.receipt.receipt_id == "B"


def test_find_receipt_unknown_returns_none():
    chain = append_receipt((), _receipt("A"), hmac_key=_KEY)
    assert find_receipt(chain, "MISSING") is None


# --- Render ------------------------------------------------------------------


def test_render_empty_chain():
    out = render_chain(())
    assert "empty chain" in out


def test_render_chain_lists_receipts():
    chain = append_receipt((), _receipt("RCT-A"), hmac_key=_KEY)
    chain = append_receipt(chain, _receipt("RCT-B"), hmac_key=_KEY)
    out = render_chain(chain)
    assert "RCT-A" in out
    assert "RCT-B" in out
    assert "Helping Hand Foundation" in out


def test_render_chain_signature_truncated():
    chain = append_receipt((), _receipt("RCT-A"), hmac_key=_KEY)
    out = render_chain(chain)
    # Full 64-hex signature must NOT appear; only 8-hex prefix
    full_sig = chain[0].signature
    assert full_sig not in out
    assert full_sig[:8] in out


def test_render_no_secret_leak():
    chain = append_receipt((), _receipt("RCT-A"), hmac_key=_KEY)
    out = render_chain(chain)
    for token in (
        "@",
        "zoom.us",
        "meet.google",
        "private_email",
        "+1-",
        "Authorization",
        "EIN-",
        "Bank-",
    ):
        assert token not in out


def test_render_receipt_contains_amount_and_charity():
    r = _receipt()
    out = render_receipt(r)
    assert r.charity_name in out
    assert "123.45" in out


def test_render_receipt_with_signature_truncates():
    r = _receipt()
    sig = sign_receipt(r, hmac_key=_KEY)
    out = render_receipt(r, sig)
    assert sig[:8] in out
    assert sig not in out


# --- E2E ----------------------------------------------------------------------


def test_e2e_quarterly_disbursement_chain_holds_under_audit():
    """Operator builds a four-receipt quarterly chain; auditor verifies clean."""
    chain: tuple[ChainedReceipt, ...] = ()
    for q, dt in enumerate(
        [
            date(2026, 4, 5),
            date(2026, 7, 5),
            date(2026, 10, 5),
            date(2027, 1, 5),
        ]
    ):
        r = _receipt(
            receipt_id=f"Q{q + 1}-2026",
            disbursement_date=dt,
            purification_period_start=date(2026, 1, 1),
            purification_period_end=date(2026, 12, 31),
            amount=50.0 + q,
        )
        chain = append_receipt(chain, r, hmac_key=_KEY)
    assert verify_chain(chain, hmac_key=_KEY)
    assert len(chain) == 4
    assert chain_root(chain) == chain[-1].current_root


def test_replay_consistency():
    chain_a = append_receipt((), _receipt("A"), hmac_key=_KEY)
    chain_b = append_receipt((), _receipt("A"), hmac_key=_KEY)
    assert chain_a[0].signature == chain_b[0].signature
    assert chain_a[0].current_root == chain_b[0].current_root
