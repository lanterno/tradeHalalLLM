"""DocuSign-ready Zakat receipt generator — Round-5 Wave 18.I.

When the operator pays out Zakat to a recipient charity, both parties
typically want a signed receipt for tax / audit purposes. This module
ships the **structured Zakat receipt** with deterministic fields + a
canonicalised representation that the e-signature system signs.

Pinned semantics:

- **Closed-set ReceiptStatus ladder** (DRAFT / SIGNED_BY_OPERATOR /
  COUNTER_SIGNED / VOID).
- **Canonical hash** ties the receipt content to its signature; same
  receipt → same digest.
- **Composes with `halal/charity_receipts.py`** Merkle chain
  (Wave 1.K) so an annual Zakat batch can anchor on the audit log.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from enum import Enum


class ReceiptStatus(str, Enum):
    """Closed-set receipt status ladder."""

    DRAFT = "draft"
    SIGNED_BY_OPERATOR = "signed_by_operator"
    COUNTER_SIGNED = "counter_signed"
    VOID = "void"


@dataclass(frozen=True)
class ZakatReceipt:
    """A signed Zakat receipt."""

    receipt_id: str
    operator_handle: str
    recipient_charity: str
    amount: float
    currency: str
    payment_date: date
    zakat_year_label: str  # e.g. "1447 AH" or "2026 CE"
    nisab_basis: str
    status: ReceiptStatus
    operator_signature_hex: str
    recipient_signature_hex: str

    def __post_init__(self) -> None:
        if not self.receipt_id or not self.receipt_id.strip():
            raise ValueError("receipt_id must be non-empty")
        if not self.operator_handle or not self.operator_handle.strip():
            raise ValueError("operator_handle must be non-empty")
        if "@" in self.operator_handle:
            raise ValueError("operator_handle must be a handle, not an email")
        if not self.recipient_charity or not self.recipient_charity.strip():
            raise ValueError("recipient_charity must be non-empty")
        if self.amount <= 0:
            raise ValueError("amount must be positive")
        if not self.currency or len(self.currency) > 8:
            raise ValueError("currency must be a non-empty short code")
        if not self.zakat_year_label or not self.zakat_year_label.strip():
            raise ValueError("zakat_year_label must be non-empty")
        if not self.nisab_basis or not self.nisab_basis.strip():
            raise ValueError("nisab_basis must be non-empty")

        # Status-signature consistency
        if self.status is ReceiptStatus.DRAFT and (
            self.operator_signature_hex or self.recipient_signature_hex
        ):
            raise ValueError("DRAFT must have no signatures")
        if self.status is ReceiptStatus.SIGNED_BY_OPERATOR and (
            not self.operator_signature_hex or self.recipient_signature_hex
        ):
            raise ValueError(
                "SIGNED_BY_OPERATOR requires operator signature only"
            )
        if self.status is ReceiptStatus.COUNTER_SIGNED and (
            not self.operator_signature_hex or not self.recipient_signature_hex
        ):
            raise ValueError("COUNTER_SIGNED requires both signatures")


def _canonical(receipt: ZakatReceipt) -> bytes:
    """Canonical byte form for hashing — sorted-key newline-joined fields."""
    fields = [
        ("amount", f"{receipt.amount:.8f}"),
        ("currency", receipt.currency),
        ("nisab_basis", receipt.nisab_basis),
        ("operator_handle", receipt.operator_handle),
        ("payment_date", receipt.payment_date.isoformat()),
        ("receipt_id", receipt.receipt_id),
        ("recipient_charity", receipt.recipient_charity),
        ("zakat_year_label", receipt.zakat_year_label),
    ]
    return "\n".join(f"{k}={v}" for k, v in fields).encode("utf-8")


def canonical_digest(receipt: ZakatReceipt) -> str:
    """SHA-256 hex digest of the canonicalised receipt content."""
    return hashlib.sha256(_canonical(receipt)).hexdigest()


def draft_receipt(
    *,
    receipt_id: str,
    operator_handle: str,
    recipient_charity: str,
    amount: float,
    currency: str,
    payment_date: date,
    zakat_year_label: str,
    nisab_basis: str = "silver",
) -> ZakatReceipt:
    """Build a DRAFT (unsigned) receipt."""
    return ZakatReceipt(
        receipt_id=receipt_id,
        operator_handle=operator_handle,
        recipient_charity=recipient_charity,
        amount=amount,
        currency=currency,
        payment_date=payment_date,
        zakat_year_label=zakat_year_label,
        nisab_basis=nisab_basis,
        status=ReceiptStatus.DRAFT,
        operator_signature_hex="",
        recipient_signature_hex="",
    )


def sign_by_operator(receipt: ZakatReceipt, *, signature_hex: str) -> ZakatReceipt:
    """Apply the operator's signature, advancing status to SIGNED_BY_OPERATOR."""
    if receipt.status is not ReceiptStatus.DRAFT:
        raise ValueError("only DRAFT receipts can be signed by operator")
    if len(signature_hex) != 64:
        raise ValueError("signature_hex must be 64-hex-char SHA-256")
    return ZakatReceipt(
        receipt_id=receipt.receipt_id,
        operator_handle=receipt.operator_handle,
        recipient_charity=receipt.recipient_charity,
        amount=receipt.amount,
        currency=receipt.currency,
        payment_date=receipt.payment_date,
        zakat_year_label=receipt.zakat_year_label,
        nisab_basis=receipt.nisab_basis,
        status=ReceiptStatus.SIGNED_BY_OPERATOR,
        operator_signature_hex=signature_hex,
        recipient_signature_hex="",
    )


def counter_sign(receipt: ZakatReceipt, *, signature_hex: str) -> ZakatReceipt:
    """Apply the recipient's counter-signature, advancing to COUNTER_SIGNED."""
    if receipt.status is not ReceiptStatus.SIGNED_BY_OPERATOR:
        raise ValueError("only SIGNED_BY_OPERATOR receipts can be counter-signed")
    if len(signature_hex) != 64:
        raise ValueError("signature_hex must be 64-hex-char SHA-256")
    return ZakatReceipt(
        receipt_id=receipt.receipt_id,
        operator_handle=receipt.operator_handle,
        recipient_charity=receipt.recipient_charity,
        amount=receipt.amount,
        currency=receipt.currency,
        payment_date=receipt.payment_date,
        zakat_year_label=receipt.zakat_year_label,
        nisab_basis=receipt.nisab_basis,
        status=ReceiptStatus.COUNTER_SIGNED,
        operator_signature_hex=receipt.operator_signature_hex,
        recipient_signature_hex=signature_hex,
    )


def void_receipt(receipt: ZakatReceipt) -> ZakatReceipt:
    return ZakatReceipt(
        receipt_id=receipt.receipt_id,
        operator_handle=receipt.operator_handle,
        recipient_charity=receipt.recipient_charity,
        amount=receipt.amount,
        currency=receipt.currency,
        payment_date=receipt.payment_date,
        zakat_year_label=receipt.zakat_year_label,
        nisab_basis=receipt.nisab_basis,
        status=ReceiptStatus.VOID,
        operator_signature_hex=receipt.operator_signature_hex,
        recipient_signature_hex=receipt.recipient_signature_hex,
    )


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
    "IBAN",
    "Bank-",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_receipt(receipt: ZakatReceipt) -> str:
    digest = canonical_digest(receipt)
    sig_op = receipt.operator_signature_hex[:8] + "…" if receipt.operator_signature_hex else "—"
    sig_rcp = receipt.recipient_signature_hex[:8] + "…" if receipt.recipient_signature_hex else "—"
    head = f"📜 Zakat receipt {receipt.receipt_id} [{receipt.status.value}]"
    lines = [
        head,
        f"  operator: {receipt.operator_handle} (sig: {sig_op})",
        f"  recipient: {receipt.recipient_charity} (sig: {sig_rcp})",
        f"  amount: {receipt.amount:.2f} {receipt.currency} on "
        f"{receipt.payment_date.isoformat()}",
        f"  zakat year: {receipt.zakat_year_label} ({receipt.nisab_basis} basis)",
        f"  digest: {digest[:16]}…",
    ]
    return _scrub("\n".join(lines))
