"""Tests for the per-user encrypted secrets vault."""

from __future__ import annotations

import dataclasses
import os
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.web.secrets_vault import (
    DEFAULT_POLICY,
    EncryptedSecret,
    SecretIntegrityError,
    SecretKind,
    SecretVault,
    VaultPolicy,
    render_secret_metadata,
)

# A 32-byte test KEK; tests reuse this unless they specifically want a
# different KEK (e.g., for cross-KEK isolation tests).
_KEK = b"0" * 32
_OTHER_KEK = b"1" * 32

_NOW = datetime(2026, 5, 1, tzinfo=UTC)


def _vault(
    *, master_kek: bytes = _KEK, policy: VaultPolicy | None = None, now: datetime = _NOW
) -> SecretVault:
    return SecretVault(
        master_kek=master_kek,
        policy=policy or DEFAULT_POLICY,
        now_fn=lambda: now,
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_vault_rejects_short_kek() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        SecretVault(master_kek=b"too-short")


def test_vault_accepts_32_byte_kek() -> None:
    v = SecretVault(master_kek=_KEK)
    assert v is not None


def test_vault_accepts_longer_kek() -> None:
    v = SecretVault(master_kek=os.urandom(64))
    assert v is not None


# ---------------------------------------------------------------------------
# VaultPolicy validation
# ---------------------------------------------------------------------------


def test_default_policy_values() -> None:
    assert DEFAULT_POLICY.rotation_days == 90
    assert DEFAULT_POLICY.max_label_length == 64
    assert DEFAULT_POLICY.min_plaintext_bytes == 8


def test_policy_rejects_zero_rotation_days() -> None:
    with pytest.raises(ValueError, match="rotation_days"):
        VaultPolicy(rotation_days=0)


def test_policy_rejects_negative_label_length() -> None:
    with pytest.raises(ValueError, match="max_label_length"):
        VaultPolicy(max_label_length=-1)


def test_policy_rejects_zero_min_plaintext() -> None:
    with pytest.raises(ValueError, match="min_plaintext_bytes"):
        VaultPolicy(min_plaintext_bytes=0)


# ---------------------------------------------------------------------------
# Store + reveal happy path
# ---------------------------------------------------------------------------


def test_store_then_reveal_returns_plaintext() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance-prod",
        plaintext="binance-key-abcdef-1234",
    )
    plaintext, _ = v.reveal(secret, owner_id="user-1")
    assert plaintext == b"binance-key-abcdef-1234"


def test_store_returns_encrypted_secret_shape() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.LLM_API_KEY,
        label="anthropic-prod",
        plaintext="sk-anthropic-abcdef",
    )
    assert isinstance(secret, EncryptedSecret)
    assert secret.owner_id == "user-1"
    assert secret.kind is SecretKind.LLM_API_KEY
    assert secret.label == "anthropic-prod"
    assert secret.ciphertext  # non-empty
    assert secret.created_at == _NOW
    assert secret.last_rotated_at == _NOW
    assert secret.last_accessed_at is None
    assert secret.key_version == 1


def test_store_accepts_bytes_plaintext() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.SCREENER_API_KEY,
        label="zoya-prod",
        plaintext=b"zoya-key-abcdef-1234",
    )
    plaintext, _ = v.reveal(secret, owner_id="user-1")
    assert plaintext == b"zoya-key-abcdef-1234"


def test_reveal_updates_last_accessed_at() -> None:
    v = _vault(now=_NOW)
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    later = _NOW + timedelta(hours=2)
    v2 = _vault(now=later)
    _, updated = v2.reveal(secret, owner_id="user-1")
    assert updated.last_accessed_at == later
    # other fields preserved
    assert updated.created_at == secret.created_at
    assert updated.ciphertext == secret.ciphertext


def test_reveal_does_not_mutate_input_secret() -> None:
    """Pin: input secret stays frozen; caller persists the returned one."""

    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    assert secret.last_accessed_at is None
    v.reveal(secret, owner_id="user-1")
    assert secret.last_accessed_at is None  # unchanged


# ---------------------------------------------------------------------------
# Cross-user isolation
# ---------------------------------------------------------------------------


def test_reveal_with_wrong_owner_id_raises() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    with pytest.raises(SecretIntegrityError):
        v.reveal(secret, owner_id="user-2")


def test_secrets_for_different_users_have_different_ciphertext() -> None:
    """Pin: identical plaintext + different owner_id → different DEK → different ciphertext."""

    v = _vault()
    a = v.store(
        owner_id="user-A",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="identical-secret-1234",
    )
    b = v.store(
        owner_id="user-B",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="identical-secret-1234",
    )
    assert a.ciphertext != b.ciphertext


def test_user_a_cannot_decrypt_user_bs_ciphertext_via_forged_secret() -> None:
    """Pin: even if user A constructs an EncryptedSecret claiming
    ownership of user B's ciphertext, the cryptographic boundary
    rejects it (different DEK → InvalidToken)."""

    v = _vault()
    b_secret = v.store(
        owner_id="user-B",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="user-b-secret-abcdef",
    )

    forged = EncryptedSecret(
        owner_id="user-A",
        kind=b_secret.kind,
        label=b_secret.label,
        ciphertext=b_secret.ciphertext,
        created_at=b_secret.created_at,
        last_rotated_at=b_secret.last_rotated_at,
        last_accessed_at=None,
        key_version=b_secret.key_version,
    )
    with pytest.raises(SecretIntegrityError):
        v.reveal(forged, owner_id="user-A")


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------


def test_tampered_ciphertext_raises() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    # flip a single byte in the ciphertext
    bad = bytearray(secret.ciphertext)
    bad[10] = bad[10] ^ 0x01
    tampered = EncryptedSecret(
        owner_id=secret.owner_id,
        kind=secret.kind,
        label=secret.label,
        ciphertext=bytes(bad),
        created_at=secret.created_at,
        last_rotated_at=secret.last_rotated_at,
        last_accessed_at=None,
        key_version=secret.key_version,
    )
    with pytest.raises(SecretIntegrityError):
        v.reveal(tampered, owner_id="user-1")


def test_truncated_ciphertext_raises() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    truncated = EncryptedSecret(
        owner_id=secret.owner_id,
        kind=secret.kind,
        label=secret.label,
        ciphertext=secret.ciphertext[:-4],
        created_at=secret.created_at,
        last_rotated_at=secret.last_rotated_at,
        last_accessed_at=None,
        key_version=secret.key_version,
    )
    with pytest.raises(SecretIntegrityError):
        v.reveal(truncated, owner_id="user-1")


# ---------------------------------------------------------------------------
# Cross-KEK isolation
# ---------------------------------------------------------------------------


def test_reveal_under_wrong_kek_raises() -> None:
    """Pin: the same owner_id under a different master KEK derives a
    different DEK; reveal of an old-KEK ciphertext under a new-KEK
    vault fails cleanly."""

    v_old = _vault(master_kek=_KEK)
    secret = v_old.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )

    v_new = _vault(master_kek=_OTHER_KEK)
    with pytest.raises(SecretIntegrityError):
        v_new.reveal(secret, owner_id="user-1")


# ---------------------------------------------------------------------------
# Storage validation
# ---------------------------------------------------------------------------


def test_store_rejects_empty_owner_id() -> None:
    v = _vault()
    with pytest.raises(ValueError, match="owner_id"):
        v.store(
            owner_id="",
            kind=SecretKind.BROKER_API_KEY,
            label="binance",
            plaintext="binance-key-abcdef",
        )


def test_store_rejects_empty_label() -> None:
    v = _vault()
    with pytest.raises(ValueError, match="label"):
        v.store(
            owner_id="user-1",
            kind=SecretKind.BROKER_API_KEY,
            label="",
            plaintext="binance-key-abcdef",
        )


def test_store_rejects_too_short_plaintext() -> None:
    v = _vault()
    with pytest.raises(ValueError, match="too short"):
        v.store(
            owner_id="user-1",
            kind=SecretKind.BROKER_API_KEY,
            label="binance",
            plaintext="abc",  # below 8-byte minimum
        )


def test_store_rejects_too_long_label() -> None:
    v = _vault()
    long_label = "x" * 65  # 1 over the default max
    with pytest.raises(ValueError, match="longer than"):
        v.store(
            owner_id="user-1",
            kind=SecretKind.BROKER_API_KEY,
            label=long_label,
            plaintext="binance-key-abcdef",
        )


def test_custom_policy_min_plaintext_flows_through() -> None:
    strict = VaultPolicy(min_plaintext_bytes=32)
    v = _vault(policy=strict)
    with pytest.raises(ValueError, match="too short"):
        v.store(
            owner_id="user-1",
            kind=SecretKind.LLM_API_KEY,
            label="anthropic",
            plaintext="short-key-1234",
        )


# ---------------------------------------------------------------------------
# EncryptedSecret validation
# ---------------------------------------------------------------------------


def _basic_kwargs() -> dict:
    return dict(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        ciphertext=b"abcdef0123456789",
        created_at=_NOW,
        last_rotated_at=_NOW,
        last_accessed_at=None,
        key_version=1,
    )


def test_encrypted_secret_rejects_empty_owner() -> None:
    kw = _basic_kwargs()
    kw["owner_id"] = ""
    with pytest.raises(ValueError, match="owner_id"):
        EncryptedSecret(**kw)


def test_encrypted_secret_rejects_empty_label() -> None:
    kw = _basic_kwargs()
    kw["label"] = ""
    with pytest.raises(ValueError, match="label"):
        EncryptedSecret(**kw)


def test_encrypted_secret_rejects_empty_ciphertext() -> None:
    kw = _basic_kwargs()
    kw["ciphertext"] = b""
    with pytest.raises(ValueError, match="ciphertext"):
        EncryptedSecret(**kw)


def test_encrypted_secret_rejects_naive_created_at() -> None:
    kw = _basic_kwargs()
    kw["created_at"] = datetime(2026, 5, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        EncryptedSecret(**kw)


def test_encrypted_secret_rejects_naive_last_rotated_at() -> None:
    kw = _basic_kwargs()
    kw["last_rotated_at"] = datetime(2026, 5, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        EncryptedSecret(**kw)


def test_encrypted_secret_rejects_naive_last_accessed_at_when_set() -> None:
    kw = _basic_kwargs()
    kw["last_accessed_at"] = datetime(2026, 5, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        EncryptedSecret(**kw)


def test_encrypted_secret_accepts_none_last_accessed_at() -> None:
    kw = _basic_kwargs()
    secret = EncryptedSecret(**kw)
    assert secret.last_accessed_at is None


def test_encrypted_secret_rejects_zero_key_version() -> None:
    kw = _basic_kwargs()
    kw["key_version"] = 0
    with pytest.raises(ValueError, match="key_version"):
        EncryptedSecret(**kw)


def test_encrypted_secret_rejects_too_long_label() -> None:
    kw = _basic_kwargs()
    kw["label"] = "x" * 65
    with pytest.raises(ValueError, match="longer than"):
        EncryptedSecret(**kw)


# ---------------------------------------------------------------------------
# KEK rotation
# ---------------------------------------------------------------------------


def test_rotate_re_encrypts_under_new_kek() -> None:
    v_old = _vault(master_kek=_KEK)
    secret = v_old.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )

    rotated = v_old.rotate(secret, new_master_kek=_OTHER_KEK, new_key_version=2)
    assert rotated.key_version == 2
    assert rotated.ciphertext != secret.ciphertext
    assert rotated.last_rotated_at == _NOW

    # cannot decrypt under the old KEK now
    v_old_again = _vault(master_kek=_KEK)
    with pytest.raises(SecretIntegrityError):
        v_old_again.reveal(rotated, owner_id="user-1")

    # can decrypt under the new KEK
    v_new = _vault(master_kek=_OTHER_KEK)
    plaintext, _ = v_new.reveal(rotated, owner_id="user-1")
    assert plaintext == b"binance-key-abcdef"


def test_rotate_preserves_owner_kind_label_and_created_at() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.LLM_API_KEY,
        label="anthropic",
        plaintext="anthropic-key-abcdef",
    )
    rotated = v.rotate(secret, new_master_kek=_OTHER_KEK, new_key_version=2)
    assert rotated.owner_id == secret.owner_id
    assert rotated.kind is secret.kind
    assert rotated.label == secret.label
    assert rotated.created_at == secret.created_at


def test_rotate_rejects_short_new_kek() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    with pytest.raises(ValueError, match="32 bytes"):
        v.rotate(secret, new_master_kek=b"too-short", new_key_version=2)


def test_rotate_rejects_zero_new_version() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    with pytest.raises(ValueError, match="new_key_version"):
        v.rotate(secret, new_master_kek=_OTHER_KEK, new_key_version=0)


def test_rotate_fails_when_old_kek_already_changed() -> None:
    """Pin: rotation requires the *current* KEK can decrypt; if the
    operator has already lost the current KEK, rotation surfaces a
    clean integrity error rather than silently re-encrypting under
    new KEK with garbage plaintext."""

    v_old = _vault(master_kek=_KEK)
    secret = v_old.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )

    v_wrong = _vault(master_kek=_OTHER_KEK)
    with pytest.raises(SecretIntegrityError):
        v_wrong.rotate(secret, new_master_kek=os.urandom(32), new_key_version=2)


# ---------------------------------------------------------------------------
# needs_rotation
# ---------------------------------------------------------------------------


def test_fresh_secret_does_not_need_rotation() -> None:
    v = _vault(now=_NOW)
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    assert v.needs_rotation(secret) is False


def test_old_secret_needs_rotation() -> None:
    v = _vault(now=_NOW - timedelta(days=120))
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    # advance time past rotation_days
    v_now = _vault(now=_NOW)
    assert v_now.needs_rotation(secret) is True


def test_needs_rotation_at_threshold_is_inclusive() -> None:
    """Pin: at exactly rotation_days, the secret needs rotation."""

    v = _vault(now=_NOW - timedelta(days=90))
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    v_now = _vault(now=_NOW)
    assert v_now.needs_rotation(secret) is True


def test_needs_rotation_uses_last_rotated_not_last_accessed() -> None:
    """Pin: rotation flag is age-based on `last_rotated_at`, not
    `last_accessed_at`. A frequently-accessed but never-rotated
    secret still needs rotation past the cadence."""

    v_old = _vault(now=_NOW - timedelta(days=120))
    secret = v_old.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    # access yesterday
    v_recent = _vault(now=_NOW - timedelta(days=1))
    _, accessed = v_recent.reveal(secret, owner_id="user-1")
    assert accessed.last_accessed_at == _NOW - timedelta(days=1)
    assert accessed.last_rotated_at == _NOW - timedelta(days=120)

    # at NOW, the secret is still considered needing rotation
    v_now = _vault(now=_NOW)
    assert v_now.needs_rotation(accessed) is True


def test_custom_rotation_days_flow_through() -> None:
    strict = VaultPolicy(rotation_days=30)
    v_old = _vault(policy=strict, now=_NOW - timedelta(days=45))
    secret = v_old.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    v_now = _vault(policy=strict, now=_NOW)
    assert v_now.needs_rotation(secret) is True


# ---------------------------------------------------------------------------
# Render output — pinned no-secret-leak contract
# ---------------------------------------------------------------------------


def test_render_does_not_contain_plaintext() -> None:
    """Pin: the render helper must never contain the secret plaintext."""

    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="HUNTER-SECRET-2-ABCDEFGH",
    )
    text = render_secret_metadata(secret)
    assert "HUNTER-SECRET-2-ABCDEFGH" not in text


def test_render_does_not_contain_ciphertext_bytes() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    text = render_secret_metadata(secret)
    # the ciphertext is binary; we check both raw and hex/utf-8
    # representations don't leak
    assert secret.ciphertext.decode("utf-8", errors="replace") not in text
    assert secret.ciphertext.hex() not in text


def test_render_includes_owner_kind_label_dates() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.LLM_API_KEY,
        label="anthropic-prod",
        plaintext="sk-anthropic-abcdef",
    )
    text = render_secret_metadata(secret)
    assert "user-1" in text
    assert "llm_api_key" in text
    assert "anthropic-prod" in text
    assert "2026-05-01" in text
    assert "key_version: 1" in text


def test_render_shows_never_for_unaccessed_secret() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    text = render_secret_metadata(secret)
    assert "accessed: never" in text


def test_render_shows_access_time_when_set() -> None:
    v = _vault(now=_NOW)
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    later = _NOW + timedelta(hours=2)
    v2 = _vault(now=later)
    _, accessed = v2.reveal(secret, owner_id="user-1")
    text = render_secret_metadata(accessed)
    assert "accessed: 2026-05-01" in text
    assert "never" not in text


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_encrypted_secret_is_frozen() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance",
        plaintext="binance-key-abcdef",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        secret.ciphertext = b"x"  # type: ignore[misc]


def test_policy_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_POLICY.rotation_days = 30  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB serialisation
# ---------------------------------------------------------------------------


def test_secret_kind_string_values_pinned() -> None:
    assert SecretKind.BROKER_API_KEY.value == "broker_api_key"
    assert SecretKind.BROKER_API_SECRET.value == "broker_api_secret"
    assert SecretKind.LLM_API_KEY.value == "llm_api_key"
    assert SecretKind.NEWS_API_KEY.value == "news_api_key"
    assert SecretKind.CRYPTOPANIC_KEY.value == "cryptopanic_key"
    assert SecretKind.REDDIT_CLIENT_SECRET.value == "reddit_client_secret"
    assert SecretKind.SCREENER_API_KEY.value == "screener_api_key"


# ---------------------------------------------------------------------------
# Determinism + cryptographic properties
# ---------------------------------------------------------------------------


def test_two_stores_of_same_plaintext_produce_different_ciphertext() -> None:
    """Pin: Fernet uses a random IV per encrypt; two encrypts of the
    same plaintext produce different ciphertexts. A vault that
    produces the same ciphertext twice would let an attacker confirm
    "this user re-uses this secret across services" via comparison."""

    v = _vault()
    a = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance-1",
        plaintext="identical-plaintext-1234",
    )
    b = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="binance-2",
        plaintext="identical-plaintext-1234",
    )
    assert a.ciphertext != b.ciphertext

    # but both still decrypt to the same plaintext
    pa, _ = v.reveal(a, owner_id="user-1")
    pb, _ = v.reveal(b, owner_id="user-1")
    assert pa == pb == b"identical-plaintext-1234"


def test_round_trip_preserves_unicode_plaintext() -> None:
    v = _vault()
    secret = v.store(
        owner_id="user-1",
        kind=SecretKind.BROKER_API_KEY,
        label="emoji-key",
        plaintext="passphrase-👍-üñîcödé",
    )
    plaintext, _ = v.reveal(secret, owner_id="user-1")
    assert plaintext.decode("utf-8") == "passphrase-👍-üñîcödé"
