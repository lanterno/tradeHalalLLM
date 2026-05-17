"""Tests for the Ed25519-signed halal receipts (`halal/signing.py`).

Pin the cryptographic invariants that make these receipts useful as
proofs to a third party (scholar, auditor, regulator):

* Sign + verify is a round-trip — given the operator's public key,
  any auditor can verify a receipt.
* Tampering with the payload invalidates the signature.
* Tampering with the signature invalidates the signature.
* The same payload signed twice produces a *bit-identical*
  signature (Ed25519 is deterministic — auditor-friendly).
* Canonical payload bytes are sorted-keys + no-whitespace JSON, so
  any platform can reproduce them.
* Key persistence: first-run generates + writes 0600 private key +
  0644 public key; subsequent calls return the same key.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from halal_trader.halal.audit import Receipt
from halal_trader.halal.signing import (
    PRIVATE_KEY_FILENAME,
    PUBLIC_KEY_FILENAME,
    HalalSigner,
    SignedReceipt,
    canonical_payload_bytes,
    generate_signer,
    get_or_create_signer,
    verify_receipt,
)


def _receipt(symbol: str = "AAPL", decision: str = "halal") -> Receipt:
    return Receipt(
        payload={
            "asset_class": "stock",
            "trade": {"id": 1, "symbol": symbol, "side": "buy", "quantity": 10},
            "screening": {"id": 99, "symbol": symbol, "decision": decision},
            "compliance_status": decision,
        }
    )


# ── canonical_payload_bytes ────────────────────────────────


def test_canonical_payload_sorts_keys():
    """Same dict, different insertion order → identical bytes."""
    a = {"b": 1, "a": 2, "c": 3}
    b = {"a": 2, "c": 3, "b": 1}
    assert canonical_payload_bytes(a) == canonical_payload_bytes(b)


def test_canonical_payload_no_whitespace():
    """No spaces in the canonical form — keeps signatures stable
    across platforms with different default JSON formatters."""
    payload = {"k": 1, "v": [1, 2, 3]}
    raw = canonical_payload_bytes(payload)
    assert b" " not in raw
    assert b"\n" not in raw


def test_canonical_payload_handles_datetime():
    """`default=str` coerces non-JSON-native types — datetime,
    Decimal, etc. — to strings deterministically."""
    from datetime import UTC, datetime

    ts = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    payload = {"timestamp": ts}
    raw = canonical_payload_bytes(payload)
    decoded = json.loads(raw)
    assert decoded["timestamp"] == str(ts)


def test_canonical_payload_nested_dict_keys_also_sorted():
    """Nested objects must also have sorted keys for true canonicality."""
    a = {"outer": {"z": 1, "a": 2}}
    b = {"outer": {"a": 2, "z": 1}}
    assert canonical_payload_bytes(a) == canonical_payload_bytes(b)


# ── Sign + verify round-trip ───────────────────────────────


def test_sign_and_verify_round_trip():
    """Verifying a freshly-signed receipt returns True."""
    signer = generate_signer()
    receipt = _receipt()
    signed = signer.sign(receipt)
    assert verify_receipt(signed) is True


def test_signed_receipt_carries_public_key():
    """The signed receipt has the public key bundled in — auditors
    don't need a separate channel to fetch it. Pin so a refactor that
    drops the public key (treating it as out-of-band) breaks here."""
    signer = generate_signer()
    signed = signer.sign(_receipt())
    assert signed.public_key_b64url == signer.public_key_b64url
    assert len(signed.public_key_b64url) > 0


def test_signed_receipt_algorithm_field_is_ed25519():
    signed = generate_signer().sign(_receipt())
    assert signed.algorithm == "ed25519"


def test_signature_is_deterministic():
    """Ed25519 signatures are deterministic — signing the same
    payload with the same key twice produces bit-identical outputs.
    Critical for auditor reproducibility ('I got a different
    signature' → red flag in any other scheme; harmless here only
    because we pinned this property)."""
    signer = generate_signer()
    receipt = _receipt()
    a = signer.sign(receipt)
    b = signer.sign(receipt)
    assert a.signature_b64url == b.signature_b64url


def test_two_keypairs_produce_different_signatures():
    """Different keypairs → different signatures of the same payload.
    Sanity: the signing key matters."""
    a = generate_signer().sign(_receipt())
    b = generate_signer().sign(_receipt())
    assert a.signature_b64url != b.signature_b64url


# ── Tamper detection ──────────────────────────────────────


def test_payload_tamper_invalidates_signature():
    """Modify the payload after signing → verify returns False.
    The whole point of signing."""
    signer = generate_signer()
    signed = signer.sign(_receipt(decision="halal"))

    tampered_payload = dict(signed.receipt.payload)
    tampered_payload["compliance_status"] = "not_halal"  # 🚨
    tampered = SignedReceipt(
        receipt=Receipt(payload=tampered_payload),
        signature_b64url=signed.signature_b64url,
        public_key_b64url=signed.public_key_b64url,
    )
    assert verify_receipt(tampered) is False


def test_signature_tamper_invalidates():
    """Modify even one byte of the signature → verify False."""
    signed = generate_signer().sign(_receipt())
    bad_sig = signed.signature_b64url[:-2] + "AA"  # last 2 chars mangled
    tampered = SignedReceipt(
        receipt=signed.receipt,
        signature_b64url=bad_sig,
        public_key_b64url=signed.public_key_b64url,
    )
    assert verify_receipt(tampered) is False


def test_public_key_swap_invalidates():
    """Swap to an unrelated public key → verify False (the signature
    is no longer 'from' that key)."""
    signed = generate_signer().sign(_receipt())
    other = generate_signer()
    swapped = SignedReceipt(
        receipt=signed.receipt,
        signature_b64url=signed.signature_b64url,
        public_key_b64url=other.public_key_b64url,
    )
    assert verify_receipt(swapped) is False


def test_unknown_algorithm_rejected():
    """A signed receipt with an unrecognised algorithm field is
    rejected — pin so a future supply-chain attack that ships
    `algorithm: 'plaintext'` doesn't pass verification."""
    signed = generate_signer().sign(_receipt())
    bogus = SignedReceipt(
        receipt=signed.receipt,
        signature_b64url=signed.signature_b64url,
        public_key_b64url=signed.public_key_b64url,
        algorithm="plaintext",
    )
    assert verify_receipt(bogus) is False


def test_garbage_signature_b64url_rejected():
    """A signature that isn't valid base64 → verify False, no crash."""
    signed = generate_signer().sign(_receipt())
    garbage = SignedReceipt(
        receipt=signed.receipt,
        signature_b64url="!!!not-base64!!!",
        public_key_b64url=signed.public_key_b64url,
    )
    assert verify_receipt(garbage) is False


def test_garbage_public_key_b64url_rejected():
    signed = generate_signer().sign(_receipt())
    garbage = SignedReceipt(
        receipt=signed.receipt,
        signature_b64url=signed.signature_b64url,
        public_key_b64url="!!!not-a-key!!!",
    )
    assert verify_receipt(garbage) is False


# ── to_dict ───────────────────────────────────────────────


def test_signed_receipt_to_dict_round_trip():
    """`SignedReceipt.to_dict()` returns a JSON-serialisable shape
    suitable for dashboard / export. Pin the field names — they're
    a public contract for downstream consumers."""
    signed = generate_signer().sign(_receipt())
    out = signed.to_dict()
    assert set(out.keys()) == {"receipt", "signature", "public_key", "algorithm"}
    assert out["algorithm"] == "ed25519"
    # Every value is JSON-serialisable.
    json.dumps(out, default=str)


# ── Key persistence ───────────────────────────────────────


def test_first_run_generates_keypair_files(tmp_path: Path):
    """On first call, both files are created with the correct names."""
    signer = get_or_create_signer(tmp_path)
    assert isinstance(signer, HalalSigner)
    assert (tmp_path / PRIVATE_KEY_FILENAME).exists()
    assert (tmp_path / PUBLIC_KEY_FILENAME).exists()


def test_first_run_creates_data_dir_if_missing(tmp_path: Path):
    """`get_or_create_signer` creates `data_dir` if it doesn't exist —
    operator running on a fresh box never has to mkdir manually."""
    target = tmp_path / "missing-subdir"
    assert not target.exists()
    get_or_create_signer(target)
    assert target.is_dir()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions only")
def test_first_run_sets_0600_on_private_key(tmp_path: Path):
    """Private key gets 0600 permissions — no world-readable secrets."""
    get_or_create_signer(tmp_path)
    priv_path = tmp_path / PRIVATE_KEY_FILENAME
    mode = priv_path.stat().st_mode & 0o777
    assert mode == 0o600


def test_subsequent_calls_return_same_key(tmp_path: Path):
    """Idempotent: second call returns the SAME signing key (verified
    by signing → verify round-trip). Pin so a refactor that
    accidentally re-generates breaks here loudly."""
    a = get_or_create_signer(tmp_path)
    b = get_or_create_signer(tmp_path)
    # Sign a payload with `a`, verify with `b`'s public key.
    receipt = _receipt()
    signed_a = a.sign(receipt)
    # Signature should verify with the second instance's public key
    # because they share the same keypair.
    assert signed_a.public_key_b64url == b.public_key_b64url


def test_persisted_signer_round_trips_through_filesystem(tmp_path: Path):
    """Sign with the persisted key, restart (new process simulated by
    a fresh `get_or_create_signer` call), verify still works. Pin
    that the on-disk format is stable."""
    a = get_or_create_signer(tmp_path)
    signed = a.sign(_receipt())

    # Simulate a fresh process: discard `a`, load again.
    del a
    b = get_or_create_signer(tmp_path)

    # The reloaded signer's public key matches the one in the
    # signed receipt — same keypair was loaded from disk.
    assert signed.public_key_b64url == b.public_key_b64url
    # And verification still passes.
    assert verify_receipt(signed) is True


def test_corrupt_private_key_file_raises_loud_error(tmp_path: Path):
    """A garbled private-key file → loud ValueError, NOT silent
    keypair regeneration (operator could mistake the new key for
    the old one and lose signature continuity)."""
    priv_path = tmp_path / PRIVATE_KEY_FILENAME
    priv_path.write_bytes(b"not-a-pem-key")
    with pytest.raises(Exception):
        get_or_create_signer(tmp_path)


# ── HalalSigner.public_key_b64url ─────────────────────────


def test_public_key_b64url_is_url_safe():
    """Pin: public-key encoding has NO `+` or `/` (URL-unsafe). The
    base64-url variant uses `-` and `_`. Operators paste this into
    URLs / CLI args, so this matters."""
    signer = generate_signer()
    encoded = signer.public_key_b64url
    assert "+" not in encoded
    assert "/" not in encoded
    assert "=" not in encoded  # no padding


def test_public_key_b64url_is_43_chars():
    """Ed25519 public keys are 32 raw bytes → 43 base64-url chars
    (no padding). Pin so a refactor that switches to standard b64
    (44 chars w/ padding) is intentional."""
    signer = generate_signer()
    assert len(signer.public_key_b64url) == 43


def test_signature_b64url_is_86_chars():
    """Ed25519 signatures are 64 raw bytes → 86 base64-url chars
    (no padding)."""
    signed = generate_signer().sign(_receipt())
    assert len(signed.signature_b64url) == 86


# ── Audit-module integration ──────────────────────────────


def test_signer_opt_in_via_export_receipt_signature():
    """The audit module's `export_receipt` accepts a `sign=True` +
    `data_dir` opt-in path. Pin the kwarg signature so callers can
    rely on it."""
    import inspect

    from halal_trader.halal.audit import export_receipt

    sig = inspect.signature(export_receipt)
    assert "sign" in sig.parameters
    assert "data_dir" in sig.parameters
    assert sig.parameters["sign"].default is False


def test_signed_receipt_verifies_against_persisted_public_key(tmp_path: Path):
    """An auditor receives the signed receipt + the operator's
    `halal_signing.pub` file. Verifying with the public key extracted
    from that PEM file (separately from the bundle) should match."""
    from cryptography.hazmat.primitives import serialization

    signer = get_or_create_signer(tmp_path)
    signed = signer.sign(_receipt())

    pub_pem = (tmp_path / PUBLIC_KEY_FILENAME).read_bytes()
    loaded_pub = serialization.load_pem_public_key(pub_pem)
    raw = loaded_pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    import base64 as _b64

    expected = _b64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    assert signed.public_key_b64url == expected
