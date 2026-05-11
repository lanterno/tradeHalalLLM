"""Halal signal token — cryptographic provenance — Round-5 Wave 21.G.

Every published signal carries an HMAC-SHA256 token that lets a
subscriber verify the signal was issued by the named author and has
not been altered in transit. The platform's shared secret is namespaced
per author (an `author_id → HMAC_key` map maintained by the deployment
layer).

This module is the **token issuer + verifier**. It is pure-Python and
deterministic. The secret-store is abstracted via a callable so unit
tests can pass a fake.

Pinned semantics:

- **Closed-set SignalKind** — BUY / SELL / HOLD. SKIP intentionally
  excluded since you don't publish a do-nothing signal.
- **Nonce required** — replay protection; signals carry a unique
  nonce per (author, signal_id).
- **TTL enforced** — tokens older than `max_age_seconds` are rejected
  during verification.
- **HMAC-SHA256** — keyed cryptographic hash. The secret never appears
  on the wire.
- **Canonical JSON payload** — sorted keys + compact separators so the
  signature is reproducible across clients.
- **Pure-Python deterministic.**
- **No-secret-leak pin** — secrets never echoed in render output; HMAC
  is masked to first/last 8 chars in rendering.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SignalKind(str, Enum):
    """Closed-set signal kind."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class SignalPayload:
    """Operator-visible signal contents (not the signed bytes)."""

    signal_id: str
    author_id: str
    ticker: str
    kind: SignalKind
    issued_at: datetime
    nonce: str
    """Unique per signal; replay-prevention. Hex string."""

    def __post_init__(self) -> None:
        if not self.signal_id or not self.signal_id.strip():
            raise ValueError("signal_id must be non-empty")
        if not self.author_id or not self.author_id.strip():
            raise ValueError("author_id must be non-empty")
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        if not self.nonce or not self.nonce.strip():
            raise ValueError("nonce must be non-empty")
        if len(self.nonce) > 64:
            raise ValueError("nonce must be ≤ 64 chars")
        if self.issued_at.tzinfo is None:
            raise ValueError("issued_at must be tz-aware")


@dataclass(frozen=True)
class SignedSignal:
    """Output of `sign`."""

    payload: SignalPayload
    hmac_hex: str

    def __post_init__(self) -> None:
        if not self.hmac_hex or not self.hmac_hex.strip():
            raise ValueError("hmac_hex must be non-empty")
        if len(self.hmac_hex) != 64:
            raise ValueError("hmac_hex must be SHA256-length (64 hex)")


def _canonical_payload(payload: SignalPayload) -> bytes:
    """Canonical JSON serialisation — sorted keys + compact separators.

    Pinned: issued_at uses isoformat with explicit timezone offset; the
    nonce is preserved verbatim.
    """
    obj = {
        "author_id": payload.author_id,
        "issued_at": payload.issued_at.isoformat(),
        "kind": payload.kind.value,
        "nonce": payload.nonce,
        "signal_id": payload.signal_id,
        "ticker": payload.ticker,
    }
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


SecretLookup = Callable[[str], bytes]
"""Callable that maps author_id → HMAC secret (bytes)."""


def sign(
    payload: SignalPayload,
    *,
    secret_lookup: SecretLookup,
) -> SignedSignal:
    """Compute the HMAC for the payload and return a SignedSignal."""
    secret = secret_lookup(payload.author_id)
    if not isinstance(secret, bytes):
        raise TypeError("secret_lookup must return bytes")
    if not secret:
        raise ValueError(f"no secret registered for author {payload.author_id}")
    if len(secret) < 16:
        raise ValueError("secret must be ≥ 16 bytes")
    mac = hmac.new(secret, _canonical_payload(payload), hashlib.sha256).hexdigest()
    return SignedSignal(payload=payload, hmac_hex=mac)


class TokenError(ValueError):
    """Base class for verification errors."""


class TokenExpired(TokenError):
    """Token age exceeded `max_age_seconds`."""


class TokenInvalid(TokenError):
    """HMAC did not match — tampered or wrong secret."""


class TokenReplayed(TokenError):
    """Nonce reused — possible replay attack."""


def verify(
    signed: SignedSignal,
    *,
    secret_lookup: SecretLookup,
    now: datetime,
    max_age_seconds: int = 86_400,
    seen_nonces: frozenset[str] | None = None,
) -> bool:
    """Verify the signed signal.

    Pinned:
    - Constant-time HMAC compare.
    - Reject if `now - issued_at > max_age_seconds`.
    - If `seen_nonces` provided, reject if `payload.nonce in seen_nonces`.

    Returns True on success; raises a TokenError subclass otherwise so
    operators can route by failure kind.
    """
    if now.tzinfo is None:
        raise ValueError("now must be tz-aware")
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be positive")
    if seen_nonces is not None and signed.payload.nonce in seen_nonces:
        raise TokenReplayed(f"nonce {signed.payload.nonce} already seen")
    age = (now - signed.payload.issued_at).total_seconds()
    if age > max_age_seconds:
        raise TokenExpired(f"token age {age:.0f}s > max {max_age_seconds}s")
    if age < -max_age_seconds:
        raise TokenInvalid("token issued in the future")
    secret = secret_lookup(signed.payload.author_id)
    if not secret:
        raise TokenInvalid("no secret for author")
    expected = hmac.new(secret, _canonical_payload(signed.payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signed.hmac_hex):
        raise TokenInvalid("HMAC mismatch")
    return True


def _mask(s: str) -> str:
    if len(s) <= 16:
        return "***"
    return s[:8] + "…" + s[-8:]


def render_signed(signed: SignedSignal) -> str:
    """Operator-readable summary; HMAC truncated; secret never present."""
    payload = signed.payload
    return (
        f"📡 Signal {payload.signal_id} [{payload.kind.value} {payload.ticker}] "
        f"from {_mask(payload.author_id)} @ {payload.issued_at.isoformat()}\n"
        f"  HMAC: {_mask(signed.hmac_hex)} | nonce: {payload.nonce[:8]}…"
    )
