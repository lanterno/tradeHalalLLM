"""Per-user encrypted secrets vault.

Today the bot reads broker keys / LLM keys / screener keys directly
from the operator's `.env`. That's fine for single-user laptop mode
but breaks the moment the bot grows a multi-user surface — every
user's broker API keys can't share the same `.env`. This module is
the cryptographic core of the per-user vault: a server-side master
KEK derives a per-user Fernet DEK via HKDF, secrets ship encrypted
at rest, decryption requires presenting the right owner_id (so a
leaked ciphertext alone is not enough to recover plaintext), and
audit metadata tracks rotation cadence.

Picked Fernet on top of HKDF rather than AES-GCM directly because
Fernet handles the auth-tag + IV bookkeeping and the surface error
(`InvalidToken`) collapses both "wrong key" and "tampered
ciphertext" into one well-defined exception — the vault doesn't
need to leak which one happened. HKDF is the documented best-
practice for deriving per-user keys from a single high-entropy
master KEK; the alternative (a per-user random DEK stored under
the KEK as an envelope) trades a single derived key per user for a
two-step decrypt path with the same security properties — picked
HKDF for the simpler audit trail.

Pinned semantics:
- **Plaintext is never stored.** The vault holds `EncryptedSecret`
  rows only; `reveal()` returns plaintext bytes that the caller
  must scrub from memory after use. The frozen-dataclass shape
  means a future field addition can't accidentally include
  plaintext alongside the ciphertext.
- **Wrong owner_id → InvalidToken.** Decryption derives the DEK
  from `(master_kek, owner_id, key_version)`; presenting a
  different owner_id at reveal time produces a different DEK,
  Fernet rejects the ciphertext, and the operator gets a clean
  cryptographic boundary rather than a "permission denied" check
  the application layer could accidentally bypass. The pin is
  tested directly: secret stored under user A, reveal attempt by
  user B → `SecretIntegrityError`.
- **Tampered ciphertext → InvalidToken.** Fernet's auth-tag
  catches any single-byte mutation; the vault re-raises as
  `SecretIntegrityError` so the operator's exception handler
  doesn't depend on `cryptography` package internals.
- **Master KEK ≥ 32 bytes.** The HKDF input minimum is 32 bytes;
  shorter keys are rejected at construction so a misconfigured
  `SECRET_KEY` env var fails fast rather than silently weakening
  every derived DEK.
- **Render output never contains plaintext, ciphertext, or any
  derived key.** `render_secret_metadata()` shows owner_id /
  kind / label / created_at / age — the operator audit display
  can't accidentally leak the secret it's auditing.
- **Rotation flagging is age-based, not access-based.** A secret
  past `rotation_days` since `last_rotated_at` is flagged regardless
  of how recently it was accessed — operators rotate on cadence,
  not on usage. Pinned via test.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class SecretKind(str, Enum):
    """Categorical kind of secret being stored.

    Pinned string values for DB / JSON serialisation stability —
    a future schema migration that renames `broker_api_key` to
    `broker_key` would orphan every existing row, so the values
    are stable.
    """

    BROKER_API_KEY = "broker_api_key"
    BROKER_API_SECRET = "broker_api_secret"
    LLM_API_KEY = "llm_api_key"
    NEWS_API_KEY = "news_api_key"
    CRYPTOPANIC_KEY = "cryptopanic_key"
    REDDIT_CLIENT_SECRET = "reddit_client_secret"
    SCREENER_API_KEY = "screener_api_key"


_MIN_KEK_BYTES = 32
_MIN_PLAINTEXT_BYTES = 8
_MAX_LABEL_LENGTH = 64


class SecretIntegrityError(Exception):
    """Raised when decryption fails (wrong owner, tampered ciphertext, KEK rotated)."""


@dataclass(frozen=True)
class VaultPolicy:
    """Operator-tunable vault policy.

    `rotation_days` flags a secret as needing rotation past this age.
    `max_label_length` keeps the human label sized for DB columns and
    UI rendering. `min_plaintext_bytes` rejects too-short secrets at
    storage time so a placeholder like `xxx` never gets persisted.
    """

    rotation_days: int = 90
    max_label_length: int = _MAX_LABEL_LENGTH
    min_plaintext_bytes: int = _MIN_PLAINTEXT_BYTES

    def __post_init__(self) -> None:
        if self.rotation_days <= 0:
            raise ValueError("rotation_days must be positive")
        if self.max_label_length <= 0:
            raise ValueError("max_label_length must be positive")
        if self.min_plaintext_bytes <= 0:
            raise ValueError("min_plaintext_bytes must be positive")


DEFAULT_POLICY = VaultPolicy()


@dataclass(frozen=True)
class EncryptedSecret:
    """One stored secret.

    Carries the encrypted blob plus the metadata the audit trail
    surfaces. `key_version` lets the operator rotate the master KEK
    without invalidating every existing row — re-encrypt rows under
    the new KEK + version number, then sweep old-version rows
    eventually.
    """

    owner_id: str
    kind: SecretKind
    label: str
    ciphertext: bytes
    created_at: datetime
    last_rotated_at: datetime
    last_accessed_at: datetime | None
    key_version: int

    def __post_init__(self) -> None:
        if not self.owner_id or not self.owner_id.strip():
            raise ValueError("owner_id must be non-empty")
        if not self.label or not self.label.strip():
            raise ValueError("label must be non-empty")
        if len(self.label) > _MAX_LABEL_LENGTH:
            raise ValueError(f"label longer than {_MAX_LABEL_LENGTH} chars: {len(self.label)}")
        if not self.ciphertext:
            raise ValueError("ciphertext must be non-empty")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if self.last_rotated_at.tzinfo is None:
            raise ValueError("last_rotated_at must be timezone-aware")
        if self.last_accessed_at is not None and self.last_accessed_at.tzinfo is None:
            raise ValueError("last_accessed_at must be timezone-aware when set")
        if self.key_version <= 0:
            raise ValueError("key_version must be positive")


def _derive_dek(*, master_kek: bytes, owner_id: str, key_version: int) -> bytes:
    """HKDF-derive a per-(user, version) Fernet key from the master KEK.

    The Fernet key format is 32 url-safe-base64-encoded bytes (44
    chars including padding). HKDF gives us 32 raw bytes; we
    base64-urlsafe-encode them to match Fernet's expectation.

    The `info` parameter binds the derivation to this specific use
    (the per-user vault) so a future module deriving a key for a
    different purpose with the same KEK + owner_id can't recover
    these secrets. Pinned via the literal string ``b"halal-trader-vault-v1"``.
    """

    import base64

    if len(master_kek) < _MIN_KEK_BYTES:
        raise ValueError(f"master_kek must be at least {_MIN_KEK_BYTES} bytes")
    if not owner_id or not owner_id.strip():
        raise ValueError("owner_id must be non-empty")
    if key_version <= 0:
        raise ValueError("key_version must be positive")

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=owner_id.encode("utf-8"),
        info=f"halal-trader-vault-v{key_version}".encode("utf-8"),
    )
    raw = hkdf.derive(master_kek)
    return base64.urlsafe_b64encode(raw)


class SecretVault:
    """Pure-cryptography per-user secrets vault.

    Construction takes a server-side master KEK (≥ 32 bytes) and an
    optional `now_fn` for deterministic tests. The vault is
    stateless — `EncryptedSecret` rows are returned to the caller
    for persistence (DB / file / etc.); the vault has no internal
    storage. This keeps the cryptographic core independent of the
    persistence layer (Postgres, sqlite, in-memory, file).
    """

    def __init__(
        self,
        master_kek: bytes,
        *,
        policy: VaultPolicy = DEFAULT_POLICY,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if len(master_kek) < _MIN_KEK_BYTES:
            raise ValueError(f"master_kek must be at least {_MIN_KEK_BYTES} bytes")
        self._master_kek = master_kek
        self._policy = policy
        self._now_fn = now_fn

    def _fernet(self, *, owner_id: str, key_version: int) -> Fernet:
        return Fernet(
            _derive_dek(
                master_kek=self._master_kek,
                owner_id=owner_id,
                key_version=key_version,
            )
        )

    def store(
        self,
        *,
        owner_id: str,
        kind: SecretKind,
        label: str,
        plaintext: str | bytes,
        key_version: int = 1,
    ) -> EncryptedSecret:
        """Encrypt + return an `EncryptedSecret` row for persistence.

        Raises `ValueError` on missing / too-short / too-long inputs.
        """

        if not owner_id or not owner_id.strip():
            raise ValueError("owner_id must be non-empty")
        if not label or not label.strip():
            raise ValueError("label must be non-empty")
        if len(label) > self._policy.max_label_length:
            raise ValueError(f"label longer than {self._policy.max_label_length} chars")

        if isinstance(plaintext, str):
            data = plaintext.encode("utf-8")
        else:
            data = plaintext
        if len(data) < self._policy.min_plaintext_bytes:
            raise ValueError(
                f"plaintext too short: {len(data)} < {self._policy.min_plaintext_bytes} bytes"
            )

        ciphertext = self._fernet(owner_id=owner_id, key_version=key_version).encrypt(data)
        now = self._now_fn()
        return EncryptedSecret(
            owner_id=owner_id,
            kind=kind,
            label=label,
            ciphertext=ciphertext,
            created_at=now,
            last_rotated_at=now,
            last_accessed_at=None,
            key_version=key_version,
        )

    def reveal(self, secret: EncryptedSecret, *, owner_id: str) -> tuple[bytes, EncryptedSecret]:
        """Decrypt + return (plaintext, secret-with-updated-access-time).

        The caller is responsible for scrubbing the plaintext from
        memory after use. Returns a new `EncryptedSecret` because
        the dataclass is frozen — the caller should persist the
        returned row so `last_accessed_at` reflects reality.

        Raises `SecretIntegrityError` if the presented `owner_id`
        doesn't match the secret's owner OR if the ciphertext was
        tampered with — both surface as Fernet `InvalidToken`
        because the derived DEK differs OR the auth-tag fails.
        """

        if owner_id != secret.owner_id:
            # Direct guard — even before Fernet tries the wrong-DEK
            # path, an owner_id mismatch is unambiguous abuse.
            raise SecretIntegrityError("owner_id does not match secret's owner")

        try:
            plaintext = self._fernet(owner_id=owner_id, key_version=secret.key_version).decrypt(
                secret.ciphertext
            )
        except InvalidToken as exc:
            raise SecretIntegrityError(
                "decryption failed (wrong KEK, wrong owner, or tampered ciphertext)"
            ) from exc

        updated = EncryptedSecret(
            owner_id=secret.owner_id,
            kind=secret.kind,
            label=secret.label,
            ciphertext=secret.ciphertext,
            created_at=secret.created_at,
            last_rotated_at=secret.last_rotated_at,
            last_accessed_at=self._now_fn(),
            key_version=secret.key_version,
        )
        return plaintext, updated

    def rotate(
        self,
        secret: EncryptedSecret,
        *,
        new_master_kek: bytes,
        new_key_version: int,
    ) -> EncryptedSecret:
        """Re-encrypt under a new master KEK + version.

        The operator runs this when rotating the server-side KEK
        (every 90 days, after a suspected compromise, or after key
        custody changes). The function decrypts under the current
        KEK + version, then re-encrypts under the new KEK + version.
        Returns a new `EncryptedSecret` with `last_rotated_at`
        updated; persistence is the caller's responsibility.

        Raises `SecretIntegrityError` if decryption under the
        current KEK fails (KEK already changed, ciphertext
        tampered) — the operator must investigate before rotating.
        """

        if len(new_master_kek) < _MIN_KEK_BYTES:
            raise ValueError(f"new_master_kek must be at least {_MIN_KEK_BYTES} bytes")
        if new_key_version <= 0:
            raise ValueError("new_key_version must be positive")

        try:
            plaintext = self._fernet(
                owner_id=secret.owner_id, key_version=secret.key_version
            ).decrypt(secret.ciphertext)
        except InvalidToken as exc:
            raise SecretIntegrityError("current-KEK decryption failed; cannot rotate") from exc

        new_dek = _derive_dek(
            master_kek=new_master_kek,
            owner_id=secret.owner_id,
            key_version=new_key_version,
        )
        new_ciphertext = Fernet(new_dek).encrypt(plaintext)
        return EncryptedSecret(
            owner_id=secret.owner_id,
            kind=secret.kind,
            label=secret.label,
            ciphertext=new_ciphertext,
            created_at=secret.created_at,
            last_rotated_at=self._now_fn(),
            last_accessed_at=secret.last_accessed_at,
            key_version=new_key_version,
        )

    def needs_rotation(self, secret: EncryptedSecret) -> bool:
        """True if the secret is past `rotation_days` since last rotation.

        Pinned: rotation is age-based on `last_rotated_at`, not
        `last_accessed_at` — operators rotate on cadence regardless
        of usage.
        """

        age = self._now_fn() - secret.last_rotated_at
        return age >= timedelta(days=self._policy.rotation_days)


def render_secret_metadata(secret: EncryptedSecret) -> str:
    """Render-safe ops-display summary.

    Pinned no-secret-leak contract: the output never contains the
    plaintext, the ciphertext bytes, or any derived key. Operators
    can paste the output into Slack / Telegram audit channels
    without leaking the secret it summarises.
    """

    parts: list[str] = []
    parts.append(f"{secret.kind.value} · {secret.label}")
    parts.append(f"  owner: {secret.owner_id}")
    parts.append(f"  created: {secret.created_at.isoformat()}")
    parts.append(f"  rotated: {secret.last_rotated_at.isoformat()}")
    if secret.last_accessed_at is not None:
        parts.append(f"  accessed: {secret.last_accessed_at.isoformat()}")
    else:
        parts.append("  accessed: never")
    parts.append(f"  key_version: {secret.key_version}")
    return "\n".join(parts)


__all__ = [
    "DEFAULT_POLICY",
    "EncryptedSecret",
    "SecretIntegrityError",
    "SecretKind",
    "SecretVault",
    "VaultPolicy",
    "render_secret_metadata",
]
