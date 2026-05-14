"""Authentication primitives core.

The roadmap pins Wave 3.A: "Add a `User` table, OAuth2 sign-in
(Google + Apple), JWT bearer auth on the FastAPI app. Each user
has isolated bots / portfolios / purification ledgers. Existing
operator-mode keeps working when no auth is configured."

This module is the **pure-Python primitives core** that the
FastAPI route layer + OAuth callback handlers compose with:
password hashing (scrypt), constant-time verification, session
lifecycle with TTL bounds, login-rate-limit policy. The OAuth
provider integration (Google / Apple authorisation servers, JWT
issuer keys) is operator-side and uses the standard `httpx` /
`pyjwt` stacks; this module ships the deterministic primitives
those integrations compose with.

Picked stdlib `hashlib.scrypt` over bcrypt / argon2 because (a)
scrypt is available in the standard library — no extra
dependency for what's a load-bearing security boundary, (b)
RFC 7914 scrypt with `n=16384, r=8, p=1` is documented as
sufficient for password hashing, (c) the operator can tune the
cost parameters via `PasswordPolicy` if hardware advances.

Pinned semantics:
- **Password ≥12 chars + digit + symbol.** The minimum-length +
  character-class requirements catch trivially-weak passwords;
  enforced at `hash_password` not at user-input time so the
  policy is module-level not UI-level.
- **Constant-time verify via `hmac.compare_digest`.** Prevents
  timing-attack password extraction; pinned via test that
  verification of two equally-mismatched passwords doesn't
  short-circuit.
- **Session TTL bounded [5min, 24h].** Operators can tune within
  the window; below 5min is operationally awful (forces re-auth
  during normal flows); above 24h is security-debt territory
  (use refresh tokens for longer-lived auth).
- **Failed-attempt rate limiter.** N failures in window → block;
  default 5 failures in 15 minutes. Successful login resets the
  counter.
- **Render output never includes password hash bytes / salt /
  session_id.** Mirrors no-secret patterns of Wave 3.B vault +
  Wave 8.D OTLP + Wave 12.G co-pilot.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# scrypt parameters — RFC 7914 recommended for password hashing.
# n=16384 is the cost factor (work factor); r=8 is the block size;
# p=1 is the parallelisation factor. Operators tuning for harder
# hardware bump n upward (32768, 65536, ...) — never lower.
_SCRYPT_N = 16384
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32  # output bytes
_SALT_BYTES = 16

_MIN_PASSWORD_LENGTH = 12
_MIN_SESSION_TTL_MINUTES = 5
_MAX_SESSION_TTL_MINUTES = 60 * 24  # 24 hours


class AuthOutcome(str, Enum):
    """Outcome of an authentication attempt.

    Pinned string values for JSON / DB stability. The dashboard's
    auth-event audit log keys on these literals.
    """

    SUCCESS = "success"
    INVALID_CREDENTIALS = "invalid_credentials"
    RATE_LIMITED = "rate_limited"
    SESSION_EXPIRED = "session_expired"
    SESSION_NOT_FOUND = "session_not_found"


class PasswordValidationError(Exception):
    """Raised when a password fails the policy check at hash time."""


@dataclass(frozen=True)
class PasswordPolicy:
    """Operator-tunable password policy.

    Defaults satisfy NIST SP 800-63B + the Wave 3.A roadmap
    intent. Operators in stricter regimes bump min_length to 14+
    and add character-class requirements via the booleans.
    """

    min_length: int = _MIN_PASSWORD_LENGTH
    require_digit: bool = True
    require_symbol: bool = True

    def __post_init__(self) -> None:
        if self.min_length < 8:
            raise ValueError(f"min_length {self.min_length} is below 8 (NIST SP 800-63B floor)")


DEFAULT_POLICY = PasswordPolicy()


@dataclass(frozen=True)
class PasswordHash:
    """The serialised password hash + salt + algorithm parameters.

    `hash_b64` and `salt_b64` are URL-safe base64; together with
    the algorithm parameters they let `verify_password` reconstruct
    the same scrypt computation. Store the entire dataclass on the
    `users` row.
    """

    algorithm: str  # e.g., "scrypt"
    n: int
    r: int
    p: int
    salt_b64: str
    hash_b64: str

    def __post_init__(self) -> None:
        if not self.algorithm or not self.algorithm.strip():
            raise ValueError("algorithm must be non-empty")
        if self.n <= 0 or self.r <= 0 or self.p <= 0:
            raise ValueError("scrypt parameters must be positive")
        if not self.salt_b64 or not self.hash_b64:
            raise ValueError("salt_b64 and hash_b64 must be non-empty")


def _validate_password_policy(plaintext: str, policy: PasswordPolicy) -> None:
    """Apply the policy at hash time; raise on violation."""

    if len(plaintext) < policy.min_length:
        raise PasswordValidationError(f"password too short: {len(plaintext)} < {policy.min_length}")
    if policy.require_digit and not any(c.isdigit() for c in plaintext):
        raise PasswordValidationError("password must contain a digit")
    if policy.require_symbol and not any(not c.isalnum() and not c.isspace() for c in plaintext):
        raise PasswordValidationError("password must contain a symbol")


def hash_password(
    plaintext: str,
    *,
    policy: PasswordPolicy = DEFAULT_POLICY,
) -> PasswordHash:
    """Hash a password via scrypt with a fresh per-password salt.

    Raises `PasswordValidationError` if the password fails the
    policy. Returns a `PasswordHash` ready for persistence.
    """

    if not isinstance(plaintext, str):
        raise TypeError("plaintext must be a string")
    _validate_password_policy(plaintext, policy)

    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        plaintext.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return PasswordHash(
        algorithm="scrypt",
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        salt_b64=base64.urlsafe_b64encode(salt).decode("ascii"),
        hash_b64=base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(plaintext: str, hashed: PasswordHash) -> bool:
    """Verify a plaintext password against a stored hash.

    Pinned: uses `hmac.compare_digest` for constant-time comparison
    so the operator can't extract the password via timing attack.
    Returns False on any error (mismatched algorithm, malformed
    salt, etc.) — never raises into the auth path.
    """

    if not isinstance(plaintext, str):
        return False
    if hashed.algorithm != "scrypt":
        return False
    try:
        salt = base64.urlsafe_b64decode(hashed.salt_b64)
        expected = base64.urlsafe_b64decode(hashed.hash_b64)
        candidate = hashlib.scrypt(
            plaintext.encode("utf-8"),
            salt=salt,
            n=hashed.n,
            r=hashed.r,
            p=hashed.p,
            dklen=len(expected),
        )
    except ValueError, TypeError:
        return False
    return hmac.compare_digest(candidate, expected)


@dataclass(frozen=True)
class Session:
    """One issued auth session.

    `session_id` is a high-entropy random token (32 bytes URL-safe
    base64); `issued_at` and `expires_at` are timezone-aware.
    `user_id` carries the associated user's identifier so the
    persistence layer can look up the row.
    """

    session_id: str
    user_id: str
    issued_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        if not self.session_id or len(self.session_id) < 16:
            raise ValueError("session_id must be at least 16 chars (high-entropy)")
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if self.issued_at.tzinfo is None:
            raise ValueError("issued_at must be timezone-aware")
        if self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")


def issue_session(
    *,
    user_id: str,
    now: datetime,
    ttl_minutes: int = 60,
) -> Session:
    """Issue a fresh session.

    Raises ValueError on out-of-range ttl_minutes (must be in
    [5, 1440]).
    """

    if not user_id or not user_id.strip():
        raise ValueError("user_id must be non-empty")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not _MIN_SESSION_TTL_MINUTES <= ttl_minutes <= _MAX_SESSION_TTL_MINUTES:
        raise ValueError(
            f"ttl_minutes {ttl_minutes} out of range "
            f"[{_MIN_SESSION_TTL_MINUTES}, {_MAX_SESSION_TTL_MINUTES}]"
        )

    session_id = secrets.token_urlsafe(32)
    return Session(
        session_id=session_id,
        user_id=user_id,
        issued_at=now,
        expires_at=now + timedelta(minutes=ttl_minutes),
    )


def is_session_valid(session: Session, *, now: datetime) -> bool:
    """True iff the session has not expired at `now`."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return session.issued_at <= now < session.expires_at


@dataclass(frozen=True)
class LoginAttempt:
    """A login attempt record.

    The persistence layer appends one of these per attempt; the
    rate limiter reads recent attempts to decide whether to allow
    the next login.
    """

    user_id: str
    timestamp: datetime
    success: bool

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")


@dataclass(frozen=True)
class RateLimitPolicy:
    """Operator-tunable login rate-limit policy."""

    max_failures: int = 5
    window_minutes: int = 15

    def __post_init__(self) -> None:
        if self.max_failures <= 0:
            raise ValueError("max_failures must be positive")
        if self.window_minutes <= 0:
            raise ValueError("window_minutes must be positive")


DEFAULT_RATE_LIMIT = RateLimitPolicy()


def evaluate_rate_limit(
    *,
    user_id: str,
    history: tuple[LoginAttempt, ...],
    now: datetime,
    policy: RateLimitPolicy = DEFAULT_RATE_LIMIT,
) -> bool:
    """Return True iff the user is allowed to attempt login now.

    Counts consecutive failures within the rate-limit window.
    A successful attempt resets the counter (the rate limiter
    only blocks against repeated failures, not after a single
    successful login).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be non-empty")

    window_start = now - timedelta(minutes=policy.window_minutes)
    recent = [a for a in history if a.user_id == user_id and a.timestamp >= window_start]
    # Walk from newest to oldest; count failures until we hit a
    # success (which resets the counter) or run out.
    recent_sorted = sorted(recent, key=lambda a: a.timestamp, reverse=True)
    failures = 0
    for attempt in recent_sorted:
        if attempt.success:
            return True
        failures += 1
        if failures >= policy.max_failures:
            return False
    return True


@dataclass(frozen=True)
class AuthResult:
    """The decision on a single login attempt."""

    user_id: str
    outcome: AuthOutcome
    session: Session | None = None
    failures_in_window: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)


def authenticate(
    *,
    user_id: str,
    plaintext_password: str,
    stored_hash: PasswordHash,
    history: tuple[LoginAttempt, ...],
    now: datetime,
    rate_limit_policy: RateLimitPolicy = DEFAULT_RATE_LIMIT,
    ttl_minutes: int = 60,
) -> AuthResult:
    """Compose: rate-limit check → password verify → session issue.

    Returns an `AuthResult` carrying the outcome. The persistence
    layer appends a `LoginAttempt` row regardless of outcome (so
    future calls' rate-limit can see this attempt).
    """

    # Rate-limit gate.
    allowed = evaluate_rate_limit(
        user_id=user_id, history=history, now=now, policy=rate_limit_policy
    )
    if not allowed:
        return AuthResult(
            user_id=user_id,
            outcome=AuthOutcome.RATE_LIMITED,
            warnings=(
                f"user {user_id!r} exceeded {rate_limit_policy.max_failures} "
                f"failed attempts within {rate_limit_policy.window_minutes}min",
            ),
        )

    # Password check.
    if not verify_password(plaintext_password, stored_hash):
        return AuthResult(
            user_id=user_id,
            outcome=AuthOutcome.INVALID_CREDENTIALS,
        )

    # Issue session.
    session = issue_session(user_id=user_id, now=now, ttl_minutes=ttl_minutes)
    return AuthResult(
        user_id=user_id,
        outcome=AuthOutcome.SUCCESS,
        session=session,
    )


_OUTCOME_EMOJI: dict[AuthOutcome, str] = {
    AuthOutcome.SUCCESS: "✅",
    AuthOutcome.INVALID_CREDENTIALS: "🔑",
    AuthOutcome.RATE_LIMITED: "🚫",
    AuthOutcome.SESSION_EXPIRED: "⏰",
    AuthOutcome.SESSION_NOT_FOUND: "❓",
}


def render_auth_result(result: AuthResult) -> str:
    """Format an auth result for ops display.

    Pinned no-secret-leak: never includes the password, password
    hash bytes, salt, or session_id. Shows user_id + outcome +
    session expiry (if applicable). Mirrors no-secret patterns
    of Wave 3.B vault + Wave 8.D OTLP + Wave 12.G co-pilot.
    """

    emoji = _OUTCOME_EMOJI[result.outcome]
    lines = [f"{emoji} auth attempt for {result.user_id} — {result.outcome.value.upper()}"]
    if result.session is not None:
        lines.append(f"  session expires: {result.session.expires_at.isoformat()}")
    if result.warnings:
        for w in result.warnings:
            lines.append(f"  · {w}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "DEFAULT_RATE_LIMIT",
    "AuthOutcome",
    "AuthResult",
    "LoginAttempt",
    "PasswordHash",
    "PasswordPolicy",
    "PasswordValidationError",
    "RateLimitPolicy",
    "Session",
    "authenticate",
    "evaluate_rate_limit",
    "hash_password",
    "is_session_valid",
    "issue_session",
    "render_auth_result",
    "verify_password",
]
