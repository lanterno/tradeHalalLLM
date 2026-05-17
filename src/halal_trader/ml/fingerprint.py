"""Model fingerprinting for deterministic replay.

Round-4 wave 6.E: every cycle's replay snapshot already records the
*inputs* the LLM and ML models saw (klines, indicators, sentiment).
What's missing is the *model* end of the contract — without a stable
identifier for the model that produced the decision, "rerun cycle
12345 against the exact model" can't be answered when the model
file has changed since.

This module computes a stable fingerprint from a model artefact:

* For pickled-bytes artefacts (sklearn / xgboost), SHA-256 of the
  raw bytes. The pickle protocol is deterministic-ish; small
  fluctuations from training reorder are caught downstream when
  the fingerprint changes between training runs that "should have
  been identical".
* For JSON-shaped artefacts (slippage curve, calibration
  isotonic), SHA-256 of a **canonicalised** JSON encoding —
  sorted keys, no whitespace, escape-stable. Pin so a refactor
  that changes dict insertion order or whitespace doesn't break
  fingerprint equality.

Output is a `ModelFingerprint` dataclass: the hash + algorithm +
byte count + capture timestamp + a 12-char short tag for the
dashboard ("a3b7c9d1e2f4"). Operators see the short tag in logs
and the cycle replay panel; the long hash lives in the registry.

Verification:

* `verify(actual, expected)` returns a `VerificationOutcome` with
  a clear pass/fail and a reason. Pin the conservative default —
  any mismatch (algorithm, hash, byte count) is a failure; the
  caller decides whether to halt or warn.

Halal alignment: fingerprinting is metadata only. Never opens a
position, never screens an asset. Fingerprint mismatches during
replay surface as warnings, not auto-trades — operator decides.

Pure-Python; no NumPy / DB / network. The hash digest is computed
from the bytes / JSON the caller hands in, so the module never
opens a file by itself (every IO surface in the bot is owned by
its module — fingerprinting must not introduce a new one).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# ── Algorithm vocabulary ──────────────────────────────────


class HashAlgorithm(str, Enum):
    """Hash algorithms the fingerprint helper supports.

    Pinned to two: SHA-256 (default — the cryptographic standard,
    no known collision attacks at the model-fingerprint cost
    level), and BLAKE2b-256 (faster on small inputs; useful when
    the model byte stream is short and the caller wants the
    fingerprint cheaper). Anything else raises so a caller passing
    `"md5"` for compatibility surfaces immediately rather than
    silently using a broken hash.
    """

    SHA256 = "sha256"
    BLAKE2B = "blake2b-256"


def _hasher(algorithm: HashAlgorithm) -> Any:
    if algorithm == HashAlgorithm.SHA256:
        return hashlib.sha256()
    if algorithm == HashAlgorithm.BLAKE2B:
        return hashlib.blake2b(digest_size=32)
    raise ValueError(f"unsupported hash algorithm {algorithm!r}")


# ── Output ────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelFingerprint:
    """Stable identifier for one specific model artefact.

    ``digest_hex`` is the full lowercase hex representation;
    ``short_tag`` is the first 12 hex chars — operator-readable in
    logs ("model fingerprint: a3b7c9d1e2f4"). The dashboard
    correlates the short tag against the registry to surface the
    full version.
    """

    digest_hex: str
    algorithm: HashAlgorithm
    byte_count: int
    captured_at: datetime
    short_tag: str

    def matches(self, other: "ModelFingerprint") -> bool:
        """Same hash + same algorithm + same byte count = same
        artefact. Pin all three — different algorithms produce
        different hashes for the same bytes, and a byte_count
        mismatch with matching hashes would be a freak collision
        worth surfacing."""
        return (
            self.digest_hex == other.digest_hex
            and self.algorithm == other.algorithm
            and self.byte_count == other.byte_count
        )


# ── Fingerprint construction ──────────────────────────────


def fingerprint_bytes(
    data: bytes,
    *,
    algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    captured_at: datetime | None = None,
) -> ModelFingerprint:
    """Hash a raw byte stream (pickled sklearn / xgboost / a torch
    state-dict serialised with `torch.save`)."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"expected bytes-like; got {type(data).__name__}")
    hasher = _hasher(algorithm)
    hasher.update(data)
    digest_hex = hasher.hexdigest()
    captured = captured_at or datetime.now(UTC)
    return ModelFingerprint(
        digest_hex=digest_hex,
        algorithm=algorithm,
        byte_count=len(data),
        captured_at=captured,
        short_tag=digest_hex[:12],
    )


def canonical_json(payload: Any) -> bytes:
    """Canonical JSON encoding for fingerprinting.

    Pin: ``sort_keys=True`` (dict insertion order doesn't change
    the hash), ``separators=(",", ":")`` (no whitespace —
    formatter changes don't change the hash),
    ``ensure_ascii=False`` (Unicode roundtrip preserved exactly so
    a future migration to UTF-8-everywhere doesn't shift hashes).
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def fingerprint_json(
    payload: Any,
    *,
    algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    captured_at: datetime | None = None,
) -> ModelFingerprint:
    """Hash a JSON-shaped artefact (slippage curve, calibration
    isotonic, hyperparameter blob).

    Canonicalises before hashing so dict-order / whitespace /
    encoding noise doesn't shift the fingerprint between runs
    that ought to be identical.
    """
    return fingerprint_bytes(
        canonical_json(payload),
        algorithm=algorithm,
        captured_at=captured_at,
    )


# ── Verification ──────────────────────────────────────────


class VerificationStatus(str, Enum):
    """Outcome buckets for `verify`. Pinned vocabulary so the
    dashboard can colour-code without parsing free-form strings."""

    MATCH = "match"
    HASH_MISMATCH = "hash_mismatch"
    ALGORITHM_MISMATCH = "algorithm_mismatch"
    BYTE_COUNT_MISMATCH = "byte_count_mismatch"


@dataclass(frozen=True)
class VerificationOutcome:
    """Result of comparing two fingerprints.

    ``status`` is the precise failure mode (or `MATCH`).
    ``passed`` is the convenience boolean. ``reason`` is a one-
    line operator-readable string suitable for a Telegram /
    Slack alert.
    """

    status: VerificationStatus
    passed: bool
    reason: str
    actual: ModelFingerprint
    expected: ModelFingerprint


def verify(actual: ModelFingerprint, expected: ModelFingerprint) -> VerificationOutcome:
    """Compare two fingerprints and return a structured outcome.

    Pin: failures are diagnosed by *kind* (algorithm / byte_count
    / hash) so the dashboard / log can show "the artefact is the
    right size but the bytes differ" vs "the artefact is a
    different size entirely". A blob change in the middle of the
    file produces a hash mismatch with matching size; a truncated
    file produces a byte_count mismatch.
    """
    if actual.algorithm != expected.algorithm:
        return VerificationOutcome(
            status=VerificationStatus.ALGORITHM_MISMATCH,
            passed=False,
            reason=(
                f"algorithm mismatch: actual {actual.algorithm.value} vs "
                f"expected {expected.algorithm.value}"
            ),
            actual=actual,
            expected=expected,
        )
    if actual.byte_count != expected.byte_count:
        return VerificationOutcome(
            status=VerificationStatus.BYTE_COUNT_MISMATCH,
            passed=False,
            reason=(
                f"byte count mismatch: actual {actual.byte_count} vs expected {expected.byte_count}"
            ),
            actual=actual,
            expected=expected,
        )
    if actual.digest_hex != expected.digest_hex:
        return VerificationOutcome(
            status=VerificationStatus.HASH_MISMATCH,
            passed=False,
            reason=(f"hash mismatch: actual {actual.short_tag} vs expected {expected.short_tag}"),
            actual=actual,
            expected=expected,
        )
    return VerificationOutcome(
        status=VerificationStatus.MATCH,
        passed=True,
        reason=f"fingerprint matches ({actual.short_tag})",
        actual=actual,
        expected=expected,
    )


# ── Render helper ─────────────────────────────────────────


def render_fingerprint(fp: ModelFingerprint) -> str:
    """One-line operator-readable summary for logs / replay panel."""
    return (
        f"{fp.short_tag} ({fp.algorithm.value}, "
        f"{fp.byte_count} bytes, captured {fp.captured_at:%Y-%m-%d %H:%M:%S})"
    )


__all__ = [
    "HashAlgorithm",
    "ModelFingerprint",
    "VerificationOutcome",
    "VerificationStatus",
    "canonical_json",
    "fingerprint_bytes",
    "fingerprint_json",
    "render_fingerprint",
    "verify",
]
