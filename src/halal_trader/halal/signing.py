"""Ed25519 signing for halal compliance receipts.

Round-4 wave 2.A: cryptographically sign each `Receipt` so the
operator can prove to a scholar / auditor that "trade X was screened
by criteria Y at timestamp Z" without trusting our codebase.

Properties of the signing scheme:

* **Ed25519** — modern, fast, deterministic, 32-byte signatures. No
  randomness in the signing path means two signatures of the same
  payload are bit-identical (auditor-friendly).
* **Canonical JSON** — payload is serialised with sorted keys + no
  whitespace before signing, so an auditor on any platform can
  reproduce the bytes that were signed. Dates / Decimals are coerced
  via the same `default=str` already used elsewhere.
* **Base64-URL encoded outputs** — public key, signature, and
  bundled-receipt fields are all URL-safe base64 (no padding) so
  they can be pasted into URLs / CLI args / email without escaping.
* **Detached signature** — the receipt payload is unchanged; the
  signature lives in a sibling field. Existing JSON consumers keep
  working without seeing the new field.

Key management:

* The keypair lives under `data_dir/halal_signing.key` (operator
  keeps it secret). The matching public key is `halal_signing.pub`,
  shared with auditors.
* If neither file exists at first call, :func:`get_or_create_signer`
  generates a fresh keypair and writes both files. Permissions on
  the private key are set to 0600.
* For Round-4 wave 3 (multi-user), each user gets their own keypair
  rooted at `data_dir/users/<user_id>/halal_signing.key`.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from halal_trader.halal.audit import Receipt

logger = logging.getLogger(__name__)


PRIVATE_KEY_FILENAME = "halal_signing.key"
PUBLIC_KEY_FILENAME = "halal_signing.pub"


def _b64url(data: bytes) -> str:
    """URL-safe base64 (no padding) — matches RFC 7515 / JWS conventions."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * ((4 - len(text) % 4) % 4)
    return base64.urlsafe_b64decode(text + pad)


def canonical_payload_bytes(payload: dict[str, Any]) -> bytes:
    """Serialise ``payload`` to canonical bytes for signing.

    Sorted keys, no whitespace, ``default=str`` for non-JSON-native
    values (datetime, Decimal). Pin: any auditor on any platform can
    reproduce these exact bytes from the same dict.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


@dataclass(frozen=True)
class SignedReceipt:
    """A receipt + Ed25519 signature + the public key that signed it.

    Carries the public key so a scholar can verify against the
    operator's published key without separately wiring one up. The
    signature alone is meaningless; the receipt-payload bytes plus the
    public key are required to verify.
    """

    receipt: Receipt
    signature_b64url: str
    public_key_b64url: str
    algorithm: str = "ed25519"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable wrapper for the dashboard / export path."""
        return {
            "receipt": self.receipt.payload,
            "signature": self.signature_b64url,
            "public_key": self.public_key_b64url,
            "algorithm": self.algorithm,
        }


@dataclass(frozen=True)
class HalalSigner:
    """In-process signing key + the public counterpart for verifiers."""

    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey

    @property
    def public_key_b64url(self) -> str:
        raw = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return _b64url(raw)

    def sign(self, receipt: Receipt) -> SignedReceipt:
        """Sign a receipt's canonical bytes; return a `SignedReceipt`."""
        sig = self.private_key.sign(canonical_payload_bytes(receipt.payload))
        return SignedReceipt(
            receipt=receipt,
            signature_b64url=_b64url(sig),
            public_key_b64url=self.public_key_b64url,
        )


def verify_receipt(signed: SignedReceipt) -> bool:
    """Return True iff ``signed.signature_b64url`` is a valid Ed25519
    signature of ``signed.receipt.payload`` under the carried public key.

    Auditors call this. Per Ed25519, the verifier doesn't need any
    secret material — just the public key (which travels with the
    signed receipt) and the canonical payload bytes (re-derived from
    the receipt dict). Returns ``False`` on signature mismatch, key
    decode error, or any other crypto failure rather than raising —
    the caller decides whether a verification miss is fatal.
    """
    if signed.algorithm != "ed25519":
        return False
    try:
        raw_pub = _b64url_decode(signed.public_key_b64url)
        pub = Ed25519PublicKey.from_public_bytes(raw_pub)
        sig = _b64url_decode(signed.signature_b64url)
        pub.verify(sig, canonical_payload_bytes(signed.receipt.payload))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("verify_receipt failed: %s", exc)
        return False


def generate_signer() -> HalalSigner:
    """Create a fresh keypair in memory. Used by tests + first-run init."""
    priv = Ed25519PrivateKey.generate()
    return HalalSigner(private_key=priv, public_key=priv.public_key())


def _write_private(priv: Ed25519PrivateKey, path: Path) -> None:
    """Write the private key as PEM with 0600 permissions."""
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path.write_bytes(pem)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows / restricted filesystems — best-effort.
        logger.debug("could not chmod 0600 on %s; relying on filesystem ACL", path)


def _write_public(pub: Ed25519PublicKey, path: Path) -> None:
    pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    path.write_bytes(pem)


def _load_private(path: Path) -> Ed25519PrivateKey:
    raw = path.read_bytes()
    key = serialization.load_pem_private_key(raw, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"{path} does not contain an Ed25519 private key")
    return key


def get_or_create_signer(data_dir: Path) -> HalalSigner:
    """Load the operator's keypair from ``data_dir``, generating it on
    first call.

    Side effects on first call only:

    * Creates ``data_dir`` if it doesn't exist.
    * Writes ``halal_signing.key`` (private, 0600) and
      ``halal_signing.pub`` (public, 0644).

    Idempotent on subsequent calls — same keypair returned.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    priv_path = data_dir / PRIVATE_KEY_FILENAME
    pub_path = data_dir / PUBLIC_KEY_FILENAME

    if priv_path.exists():
        priv = _load_private(priv_path)
        return HalalSigner(private_key=priv, public_key=priv.public_key())

    # First run — generate + persist.
    signer = generate_signer()
    _write_private(signer.private_key, priv_path)
    _write_public(signer.public_key, pub_path)
    logger.info(
        "Generated halal-receipt signing keypair at %s (public: %s)",
        priv_path,
        pub_path,
    )
    return signer
