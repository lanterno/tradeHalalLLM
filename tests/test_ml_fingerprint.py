"""Tests for `ml/fingerprint.py` (model fingerprinting).

Pins the SHA-256 hash stability, the canonical-JSON
order-invariance, the BLAKE2b-256 alternative, the verification
outcome buckets (match / hash mismatch / byte count / algorithm),
and the bytes-only input contract.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halal_trader.ml.fingerprint import (
    HashAlgorithm,
    ModelFingerprint,
    VerificationStatus,
    canonical_json,
    fingerprint_bytes,
    fingerprint_json,
    render_fingerprint,
    verify,
)

# ── fingerprint_bytes ────────────────────────────────────


def test_fingerprint_bytes_same_input_same_output():
    """Pin the deterministic-replay invariant: identical bytes
    must produce identical fingerprints across calls."""
    a = fingerprint_bytes(b"some_pickled_model")
    b = fingerprint_bytes(b"some_pickled_model")
    assert a.matches(b)
    assert a.digest_hex == b.digest_hex


def test_fingerprint_bytes_different_input_different_output():
    a = fingerprint_bytes(b"model_v1")
    b = fingerprint_bytes(b"model_v2")
    assert not a.matches(b)


def test_fingerprint_bytes_uses_sha256_by_default():
    fp = fingerprint_bytes(b"x")
    assert fp.algorithm == HashAlgorithm.SHA256
    # SHA-256 of "x" — known constant so a refactor of the hash
    # path can't quietly switch to a different algorithm without
    # this pin failing.
    assert fp.digest_hex == ("2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881")


def test_fingerprint_bytes_blake2b_alternative():
    fp = fingerprint_bytes(b"x", algorithm=HashAlgorithm.BLAKE2B)
    assert fp.algorithm == HashAlgorithm.BLAKE2B
    # Different algorithm → different digest
    assert fp.digest_hex != fingerprint_bytes(b"x").digest_hex


def test_fingerprint_bytes_records_byte_count():
    fp = fingerprint_bytes(b"hello world")
    assert fp.byte_count == 11


def test_fingerprint_bytes_short_tag_is_first_12_hex_chars():
    """Pin: short tag is exactly 12 chars and is a prefix of the
    full hex digest. The dashboard correlates against this."""
    fp = fingerprint_bytes(b"x")
    assert len(fp.short_tag) == 12
    assert fp.digest_hex.startswith(fp.short_tag)


def test_fingerprint_bytes_accepts_bytearray_and_memoryview():
    """Pin: bytes-like inputs all work — sometimes a caller hands
    in a bytearray from a buffered reader, or a memoryview slice
    of a larger buffer. The fingerprint must not refuse them."""
    payload = b"abc"
    a = fingerprint_bytes(payload)
    b = fingerprint_bytes(bytearray(payload))
    c = fingerprint_bytes(memoryview(payload))
    assert a.digest_hex == b.digest_hex == c.digest_hex


def test_fingerprint_bytes_rejects_non_bytes_input():
    with pytest.raises(TypeError, match="bytes-like"):
        fingerprint_bytes("a string")  # type: ignore[arg-type]


def test_fingerprint_bytes_handles_empty_input():
    """Pin: zero-byte input must produce a valid fingerprint
    (the SHA-256 of empty bytes is a well-known constant)."""
    fp = fingerprint_bytes(b"")
    assert fp.byte_count == 0
    assert fp.digest_hex == ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


def test_fingerprint_bytes_uses_supplied_captured_at():
    """A replay session may supply a historical timestamp."""
    when = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)
    fp = fingerprint_bytes(b"x", captured_at=when)
    assert fp.captured_at == when


def test_unsupported_algorithm_raises():
    """Pin: an enum value the helper doesn't support must surface
    immediately rather than silently using a default."""

    class BadAlgo(str):
        pass

    with pytest.raises(ValueError, match="unsupported"):
        # Force the call into the dispatch table with a junk value.
        from halal_trader.ml.fingerprint import _hasher

        _hasher(BadAlgo("md5"))  # type: ignore[arg-type]


# ── canonical_json ───────────────────────────────────────


def test_canonical_json_sorts_keys():
    """Pin: dict insertion order doesn't change the canonical
    encoding — a refactor that changes how keys are added to a
    payload mustn't shift the fingerprint."""
    a = canonical_json({"b": 2, "a": 1})
    b = canonical_json({"a": 1, "b": 2})
    assert a == b


def test_canonical_json_strips_whitespace():
    """Pin: no whitespace in the canonical form. A pretty-printer
    in the source can't shift the hash."""
    encoded = canonical_json({"a": 1, "b": [2, 3]})
    assert b" " not in encoded


def test_canonical_json_preserves_unicode_exactly():
    """Pin: ensure_ascii=False so a Unicode payload roundtrips
    through canonical encoding without shifting the byte count."""
    encoded = canonical_json({"label": "ḥalāl"})
    assert "ḥalāl".encode("utf-8") in encoded


def test_canonical_json_handles_nested_structures():
    payload = {"x": [{"b": 2, "a": 1}, {"d": 4, "c": 3}]}
    a = canonical_json(payload)
    b = canonical_json(payload)
    assert a == b


# ── fingerprint_json ─────────────────────────────────────


def test_fingerprint_json_is_dict_order_invariant():
    """Pin the canonical-JSON guarantee end-to-end: same payload
    via different dict orders → identical fingerprint."""
    a = fingerprint_json({"a": 1, "b": 2})
    b = fingerprint_json({"b": 2, "a": 1})
    assert a.matches(b)


def test_fingerprint_json_different_payloads_diverge():
    a = fingerprint_json({"a": 1})
    b = fingerprint_json({"a": 2})
    assert not a.matches(b)


def test_fingerprint_json_handles_lists():
    fp = fingerprint_json([1, 2, 3])
    assert fp.byte_count == len(b"[1,2,3]")


def test_fingerprint_json_handles_nested():
    fp = fingerprint_json({"slippage_curve": [[0.0, 0.001], [0.5, 0.002]]})
    assert fp.algorithm == HashAlgorithm.SHA256
    assert len(fp.digest_hex) == 64


# ── verify ───────────────────────────────────────────────


def test_verify_match_when_fingerprints_identical():
    a = fingerprint_bytes(b"model")
    b = fingerprint_bytes(b"model")
    outcome = verify(a, b)
    assert outcome.passed
    assert outcome.status == VerificationStatus.MATCH
    assert "matches" in outcome.reason


def test_verify_hash_mismatch_when_bytes_differ():
    a = fingerprint_bytes(b"v1")
    b = fingerprint_bytes(b"v2_____")  # different bytes; pad to vary
    outcome = verify(a, b)
    assert not outcome.passed
    # both differ in length; expect byte_count_mismatch first
    assert outcome.status == VerificationStatus.BYTE_COUNT_MISMATCH


def test_verify_hash_mismatch_when_same_size_different_bytes():
    """Pin: same-size payloads with different bytes must surface
    as a HASH_MISMATCH, not a byte_count mismatch."""
    a = fingerprint_bytes(b"abcd")
    b = fingerprint_bytes(b"abce")
    outcome = verify(a, b)
    assert not outcome.passed
    assert outcome.status == VerificationStatus.HASH_MISMATCH
    assert a.short_tag in outcome.reason
    assert b.short_tag in outcome.reason


def test_verify_byte_count_mismatch_diagnosed_separately():
    a = fingerprint_bytes(b"abcd")
    b = fingerprint_bytes(b"abcdef")
    outcome = verify(a, b)
    assert outcome.status == VerificationStatus.BYTE_COUNT_MISMATCH


def test_verify_algorithm_mismatch_diagnosed_first():
    """Pin the precedence: algorithm mismatch is checked before
    byte count or hash. Two artefacts produced by different
    hashers can't be compared meaningfully."""
    a = fingerprint_bytes(b"x", algorithm=HashAlgorithm.SHA256)
    b = fingerprint_bytes(b"x", algorithm=HashAlgorithm.BLAKE2B)
    outcome = verify(a, b)
    assert outcome.status == VerificationStatus.ALGORITHM_MISMATCH
    assert "algorithm mismatch" in outcome.reason.lower()


def test_verify_carries_both_fingerprints_in_outcome():
    """The dashboard renders both sides; pin so a refactor doesn't
    drop one and break the UI."""
    a = fingerprint_bytes(b"x")
    b = fingerprint_bytes(b"y")
    outcome = verify(a, b)
    assert outcome.actual.matches(a)
    assert outcome.expected.matches(b)


# ── ModelFingerprint.matches ─────────────────────────────


def test_matches_compares_all_three_fields():
    """Pin: matches checks digest + algorithm + byte_count. Any
    single mismatch returns False."""
    base = fingerprint_bytes(b"abc")
    same = fingerprint_bytes(b"abc")
    assert base.matches(same)

    different_algo = fingerprint_bytes(b"abc", algorithm=HashAlgorithm.BLAKE2B)
    assert not base.matches(different_algo)


def test_matches_handles_freak_collision_via_byte_count():
    """Pin: if hash and algorithm match but byte_count differs,
    matches returns False. (A truncated file producing the same
    SHA-256 prefix is astronomically unlikely but not impossible
    for a partial-write scenario; pin so the check is robust.)"""
    a = ModelFingerprint(
        digest_hex="x" * 64,
        algorithm=HashAlgorithm.SHA256,
        byte_count=100,
        captured_at=datetime.now(UTC),
        short_tag="x" * 12,
    )
    b = ModelFingerprint(
        digest_hex="x" * 64,
        algorithm=HashAlgorithm.SHA256,
        byte_count=200,
        captured_at=datetime.now(UTC),
        short_tag="x" * 12,
    )
    assert not a.matches(b)


# ── render helper ────────────────────────────────────────


def test_render_includes_short_tag_and_size():
    fp = fingerprint_bytes(b"hello")
    text = render_fingerprint(fp)
    assert fp.short_tag in text
    assert "5 bytes" in text
    assert "sha256" in text


def test_render_includes_algorithm_label():
    fp = fingerprint_bytes(b"x", algorithm=HashAlgorithm.BLAKE2B)
    text = render_fingerprint(fp)
    assert "blake2b" in text


# ── output structure ─────────────────────────────────────


def test_fingerprint_is_immutable():
    fp = fingerprint_bytes(b"x")
    with pytest.raises(Exception):
        fp.digest_hex = "tampered"  # type: ignore[misc]


def test_outcome_is_immutable():
    a = fingerprint_bytes(b"x")
    b = fingerprint_bytes(b"x")
    out = verify(a, b)
    with pytest.raises(Exception):
        out.passed = False  # type: ignore[misc]
