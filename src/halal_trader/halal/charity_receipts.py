"""Signed charitable-disbursement receipts + Merkle-anchored audit log.

Round-5 Wave 1.K primitive. The bot already (a) computes purification
amounts (`halal/purification.py`), (b) schedules quarterly disbursement
(`halal/purification_schedule.py`), and (c) reconciles paid amounts
back against owed amounts (`halal/disbursement_reconciler.py`). What's
missing is a **chain-of-custody primitive**: when the operator pays a
charity and uploads the receipt, the receipt must be signed + anchored
so an auditor (or tax authority, or future Shariah Supervisory Board)
can verify "this exact set of disbursements was made on these exact
dates" without trusting the operator's database.

This module ships the pure-Python primitives. Persistence + the upload
HTTP route live one layer up. The cryptographic primitive used is
HMAC-SHA256 (no asymmetric crypto dep), with operator-provisioned
secrets — same shape as `halal/signing.py` for trade attestations.

Pinned semantics:

- **Receipt content is canonicalised before signing.** Signing the
  string ``f"{receipt}"`` would let an attacker re-order fields and
  re-use the signature. The canonical form is a sorted-key,
  newline-separated representation defined in ``_canonicalise``.
- **Append-only Merkle log.** Inserting a receipt mid-history would
  invalidate every subsequent root. The log stores ``(receipt,
  prior_root, current_root)`` so the auditor can replay.
- **Tamper detection is deterministic.** ``verify_chain`` re-derives
  every root from the stored receipts and aborts at the first
  mismatch.
- **No-secret-leak pin.** Render outputs never include the HMAC key,
  signature bytes (only their hex prefix), or recipient EIN /
  banking info beyond a public charity name.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class CharityReceipt:
    """A single charitable-disbursement receipt the operator records."""

    receipt_id: str
    charity_name: str
    amount_currency: str
    amount: float
    disbursement_date: date
    purification_period_start: date
    purification_period_end: date
    payer_handle: str  # operator handle (NOT email / banking)

    def __post_init__(self) -> None:
        if not self.receipt_id or not self.receipt_id.strip():
            raise ValueError("receipt_id must be non-empty")
        if not self.charity_name or not self.charity_name.strip():
            raise ValueError("charity_name must be non-empty")
        if not self.amount_currency or len(self.amount_currency) > 8:
            raise ValueError("amount_currency must be a non-empty short code")
        if self.amount <= 0:
            raise ValueError("amount must be positive")
        if self.purification_period_end < self.purification_period_start:
            raise ValueError("purification_period_end before period_start")
        if self.disbursement_date < self.purification_period_start:
            raise ValueError("disbursement_date before period_start")
        if not self.payer_handle or not self.payer_handle.strip():
            raise ValueError("payer_handle must be non-empty")
        if "@" in self.payer_handle:
            raise ValueError("payer_handle must be a handle, not an email")


def _canonicalise(receipt: CharityReceipt) -> bytes:
    """Canonical byte form of a receipt for signing — sorted-key newline-joined."""
    fields = [
        ("amount", f"{receipt.amount:.8f}"),
        ("amount_currency", receipt.amount_currency),
        ("charity_name", receipt.charity_name),
        ("disbursement_date", receipt.disbursement_date.isoformat()),
        ("payer_handle", receipt.payer_handle),
        ("purification_period_end", receipt.purification_period_end.isoformat()),
        ("purification_period_start", receipt.purification_period_start.isoformat()),
        ("receipt_id", receipt.receipt_id),
    ]
    return "\n".join(f"{k}={v}" for k, v in fields).encode("utf-8")


def sign_receipt(receipt: CharityReceipt, *, hmac_key: bytes) -> str:
    """HMAC-SHA256 over the canonical form. Returns hex digest."""
    if not hmac_key:
        raise ValueError("hmac_key must be non-empty")
    return hmac.new(hmac_key, _canonicalise(receipt), hashlib.sha256).hexdigest()


def verify_receipt(receipt: CharityReceipt, signature: str, *, hmac_key: bytes) -> bool:
    """Constant-time verify a receipt signature."""
    expected = sign_receipt(receipt, hmac_key=hmac_key)
    return hmac.compare_digest(expected, signature)


# --- Merkle-anchored append-only log -----------------------------------------


_ZERO_ROOT = "0" * 64


@dataclass(frozen=True)
class ChainedReceipt:
    """A receipt + its position in the audit chain."""

    receipt: CharityReceipt
    signature: str
    prior_root: str
    current_root: str

    def __post_init__(self) -> None:
        if len(self.prior_root) != 64 or len(self.current_root) != 64:
            raise ValueError("roots must be 64-hex-char SHA-256 digests")
        if not self.signature:
            raise ValueError("signature must be non-empty")


def _next_root(prior_root: str, signature: str) -> str:
    """The Merkle-root update primitive — SHA-256 over (prior_root, signature)."""
    h = hashlib.sha256()
    h.update(prior_root.encode("ascii"))
    h.update(b"\n")
    h.update(signature.encode("ascii"))
    return h.hexdigest()


def append_receipt(
    chain: tuple[ChainedReceipt, ...],
    receipt: CharityReceipt,
    *,
    hmac_key: bytes,
) -> tuple[ChainedReceipt, ...]:
    """Append a new receipt to the chain. Returns the extended chain."""
    prior_root = chain[-1].current_root if chain else _ZERO_ROOT
    sig = sign_receipt(receipt, hmac_key=hmac_key)
    current_root = _next_root(prior_root, sig)
    chained = ChainedReceipt(
        receipt=receipt,
        signature=sig,
        prior_root=prior_root,
        current_root=current_root,
    )
    return chain + (chained,)


def chain_root(chain: tuple[ChainedReceipt, ...]) -> str:
    """The current root of the chain — what the auditor pins to."""
    return chain[-1].current_root if chain else _ZERO_ROOT


def verify_chain(chain: tuple[ChainedReceipt, ...], *, hmac_key: bytes) -> bool:
    """Verify the chain is internally consistent + every signature is valid."""
    expected_prior = _ZERO_ROOT
    for entry in chain:
        if entry.prior_root != expected_prior:
            return False
        if not verify_receipt(entry.receipt, entry.signature, hmac_key=hmac_key):
            return False
        if _next_root(entry.prior_root, entry.signature) != entry.current_root:
            return False
        expected_prior = entry.current_root
    return True


def find_receipt(chain: Iterable[ChainedReceipt], receipt_id: str) -> ChainedReceipt | None:
    for entry in chain:
        if entry.receipt.receipt_id == receipt_id:
            return entry
    return None


# --- Render ------------------------------------------------------------------


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
    "EIN-",
    "Bank-",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_chain(chain: tuple[ChainedReceipt, ...]) -> str:
    """Render the chain as a multiline auditor-facing summary.

    Signature is shown as 8-hex prefix only; full hex stays in DB.
    """
    if not chain:
        return "Charity receipts: empty chain (root: 0…0)"
    lines = [
        f"Charity receipts: {len(chain)} entries",
        f"  current root: {chain_root(chain)[:16]}…",
    ]
    for entry in chain:
        r = entry.receipt
        lines.append(
            f"  • {r.receipt_id} → {r.charity_name} "
            f"{r.amount:.2f} {r.amount_currency} on {r.disbursement_date.isoformat()} "
            f"sig={entry.signature[:8]}…"
        )
    return _scrub("\n".join(lines))


def render_receipt(receipt: CharityReceipt, signature: str | None = None) -> str:
    sig = f" sig={signature[:8]}…" if signature else ""
    return _scrub(
        f"💰 {receipt.receipt_id} → {receipt.charity_name} "
        f"{receipt.amount:.2f} {receipt.amount_currency} "
        f"({receipt.disbursement_date.isoformat()}){sig}"
    )
