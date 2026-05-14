"""Research-API key registry + token-bucket rate limiter.

Round-4 wave 7.D: the public research API exposes anonymised
aggregate trade history to academic / external researchers
(closed-trade summaries, halal-screening verdicts, regime
distributions). It is **read-only by construction** and **never
exposes operator-identifying data** — the anonymisation happens
upstream in the SQL layer; this module is the auth + rate-limit
gate that sits in front of the route handlers.

Three layers compose:

* **Key registry** — `ApiKey` dataclass with a stable opaque
  identifier (the "public id" the researcher quotes) + a hashed
  secret (never stored or logged in plaintext) + per-key
  permission scopes + a quota-tier that maps to rate limits.
* **Rate limiter** — token-bucket per (key_id, endpoint_class).
  Refill rate and burst cap come from the key's tier. Pin: the
  bucket is per-endpoint-class so a slow query on one endpoint
  doesn't burn the budget for fast queries on another.
* **Permission gate** — each key carries a set of `Scope`s;
  the route handler asks "is this key allowed `read:trades`?"
  before serving. Pin: scopes are additive (no implicit
  hierarchy); a key without `read:trades` cannot read trades
  even if it has every other scope.

Why a custom rate limiter rather than `slowapi` or a Redis
counter:

* The research API runs in the same process as the dashboard.
  Pure-Python in-memory keeps the auth path sub-millisecond
  without an extra Redis trip.
* Operators can run the bot without Redis (the rest of the
  codebase stores state in Postgres or in-process); adding
  Redis as a hard dep just for rate-limiting is over-engineered.
* The token-bucket math is < 100 lines and stays auditable.

Halal alignment: the key registry never authorises a trade; this
module gates *read* access to anonymised data. No operator
identity / position-level data flows through.

Pure-Python; no DB / network / async-loop ownership. Persistence
is the caller's job (a future SQL adapter loads `ApiKey`s into
the registry at startup).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable

# ── Scope vocabulary ─────────────────────────────────────


class Scope(str, Enum):
    """Permission scopes a key can carry.

    Pinned to a small set the public API actually serves. New
    scopes need an explicit code addition + a route mapping —
    pin so a corrupt registry row can't grant access to an
    unanticipated endpoint by string-matching on the field.
    """

    READ_TRADES = "read:trades"
    READ_HALAL = "read:halal"
    READ_REGIME = "read:regime"
    READ_RATIONALE = "read:rationale"
    READ_AGGREGATE = "read:aggregate"


# ── Tier vocabulary ──────────────────────────────────────


class Tier(str, Enum):
    """Quota tiers. Each tier maps to a refill_per_minute +
    burst_capacity in the rate limiter.

    * ``ANONYMOUS`` — no key required; very low limits. Useful
      for the public website to render a "try the API" demo
      without registration.
    * ``RESEARCHER`` — verified academic; standard limits.
    * ``PARTNER`` — institutional partners; higher limits.

    Pin: the Tier enum has hard-coded refill + burst values
    rather than letting the registry override them per key.
    Letting per-key overrides creep in is the path to forgotten
    edge cases (one key with a 10×-rate accidentally; nobody
    audits it).
    """

    ANONYMOUS = "anonymous"
    RESEARCHER = "researcher"
    PARTNER = "partner"


# Refill rate (tokens / minute) and burst (max tokens) per tier.
# Pin these in the source rather than `Settings` so a config
# typo can't silently 10× a tier.
_TIER_LIMITS: dict[Tier, tuple[int, int]] = {
    Tier.ANONYMOUS: (10, 20),  # 10/min, burst 20
    Tier.RESEARCHER: (60, 120),  # 60/min, burst 120
    Tier.PARTNER: (600, 1200),  # 600/min, burst 1200
}


def tier_limits(tier: Tier) -> tuple[int, int]:
    """Public accessor for the tier's (refill_per_minute,
    burst_capacity) — used by the dashboard's quota tile."""
    return _TIER_LIMITS[tier]


# ── ApiKey dataclass ─────────────────────────────────────


@dataclass(frozen=True)
class ApiKey:
    """One registered research-API key.

    ``key_id`` is the opaque public identifier the researcher
    quotes in the `X-Api-Key` header. ``secret_hash`` is the
    SHA-256 of the secret that pairs with the id; the plaintext
    secret is shown to the researcher *once* at issuance and
    never stored. ``scopes`` is the set of `Scope`s the key
    can use; ``tier`` maps to rate limits via `tier_limits`.

    ``label`` is operator-readable ("dr-jane-mit-2026"); never
    used for auth.
    """

    key_id: str
    secret_hash: str
    scopes: frozenset[Scope]
    tier: Tier
    label: str = ""

    def has_scope(self, scope: Scope) -> bool:
        """Pin: scopes are additive — a key without `read:trades`
        cannot read trades, even if it has every other scope."""
        return scope in self.scopes

    def verify_secret(self, plaintext: str) -> bool:
        """Constant-time comparison of the SHA-256 of ``plaintext``
        against the stored hash. Pin: `hmac.compare_digest` to
        avoid timing-attack secret extraction."""
        digest = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
        return hmac.compare_digest(digest, self.secret_hash)


# ── Token-bucket rate limiter ────────────────────────────


@dataclass
class _Bucket:
    """One rate-limit bucket. Stateful — operators wrap this in
    the registry; not exposed as a public class."""

    tokens: float
    last_refill: float
    refill_per_second: float
    burst_capacity: int

    def consume(self, *, cost: float = 1.0, now: float) -> tuple[bool, float]:
        """Try to spend ``cost`` tokens. Returns
        ``(allowed, retry_after_seconds)``.

        On allow, ``retry_after_seconds`` is 0; on deny, it's
        the number of seconds the caller should wait before
        retrying.
        """
        # Refill before the check.
        elapsed = max(0.0, now - self.last_refill)
        self.tokens = min(
            float(self.burst_capacity), self.tokens + elapsed * self.refill_per_second
        )
        self.last_refill = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True, 0.0
        # Compute retry delay: how long until the bucket has
        # `cost` tokens?
        deficit = cost - self.tokens
        wait = deficit / self.refill_per_second if self.refill_per_second > 0 else float("inf")
        return False, wait


# ── Outcomes ─────────────────────────────────────────────


class AuthOutcome(str, Enum):
    """Fine-grained verdict for the `authenticate` call.

    * ``ALLOWED`` — key valid, scope granted, rate-limit ok.
    * ``UNKNOWN_KEY`` — the key_id isn't in the registry.
    * ``INVALID_SECRET`` — key_id known but secret didn't match.
    * ``MISSING_SCOPE`` — auth succeeded but the key lacks the
      requested scope.
    * ``RATE_LIMITED`` — auth + scope ok but the bucket is empty.
    """

    ALLOWED = "allowed"
    UNKNOWN_KEY = "unknown_key"
    INVALID_SECRET = "invalid_secret"
    MISSING_SCOPE = "missing_scope"
    RATE_LIMITED = "rate_limited"


@dataclass(frozen=True)
class AuthResult:
    """Result of an authenticate call.

    ``allowed`` is True iff the call should proceed. ``key`` is
    set when the registry recognised the key (regardless of
    whether the rest of the auth checks passed) — useful for
    audit logs that record "key X attempted but missed scope Y".
    ``retry_after_seconds`` is non-zero only on RATE_LIMITED.
    """

    outcome: AuthOutcome
    allowed: bool
    key: ApiKey | None = None
    retry_after_seconds: float = 0.0
    detail: str = ""


# ── Registry ─────────────────────────────────────────────


class ResearchApiRegistry:
    """In-memory registry + per-key rate limiter.

    The caller (route handler) does:

        registry.authenticate(
            key_id="rk-abc123",
            secret="...",
            scope=Scope.READ_TRADES,
            endpoint_class="trades.list",
        )

    and gets back an `AuthResult`. The registry maintains one
    rate-limit bucket per (key_id, endpoint_class) so a slow
    query on `trades.list` doesn't burn budget for fast queries
    on `halal.summary`.
    """

    def __init__(self, *, now_fn: Callable[[], float] | None = None) -> None:
        self._keys: dict[str, ApiKey] = {}
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._now = now_fn or time.monotonic

    def register(self, key: ApiKey) -> None:
        """Add a key. Pin: duplicate `key_id` raises rather than
        silently overwriting — registry is the operator's source
        of truth, and silent overwrites mask key-leaks."""
        if key.key_id in self._keys:
            raise ValueError(f"duplicate key_id {key.key_id!r}; revoke and re-issue")
        self._keys[key.key_id] = key

    def revoke(self, key_id: str) -> None:
        """Remove a key. Pin: idempotent — revoking an already-
        removed key is a no-op (operator may run the same revoke
        command twice for safety)."""
        self._keys.pop(key_id, None)
        # Don't drop buckets; let them age out naturally. A re-issued
        # key gets a fresh bucket on next authenticate (different
        # key_id by construction).

    def get(self, key_id: str) -> ApiKey | None:
        return self._keys.get(key_id)

    def _bucket_for(self, key_id: str, endpoint_class: str, tier: Tier) -> _Bucket:
        bucket_key = (key_id, endpoint_class)
        if bucket_key in self._buckets:
            return self._buckets[bucket_key]
        refill_per_minute, burst = _TIER_LIMITS[tier]
        bucket = _Bucket(
            tokens=float(burst),
            last_refill=self._now(),
            refill_per_second=refill_per_minute / 60.0,
            burst_capacity=burst,
        )
        self._buckets[bucket_key] = bucket
        return bucket

    def authenticate(
        self,
        *,
        key_id: str,
        secret: str,
        scope: Scope,
        endpoint_class: str,
        cost: float = 1.0,
    ) -> AuthResult:
        """Check key validity + scope + rate limit in one call.

        Pin: the order is auth → scope → rate limit. A request
        with a bad secret never consumes a bucket token (a
        malicious caller can't rate-limit a victim's key by
        flooding requests with that key_id and a wrong secret).
        """
        key = self._keys.get(key_id)
        if key is None:
            return AuthResult(
                outcome=AuthOutcome.UNKNOWN_KEY,
                allowed=False,
                detail=f"key_id {key_id!r} not registered",
            )
        if not key.verify_secret(secret):
            return AuthResult(
                outcome=AuthOutcome.INVALID_SECRET,
                allowed=False,
                key=key,
                detail="secret did not match",
            )
        if not key.has_scope(scope):
            return AuthResult(
                outcome=AuthOutcome.MISSING_SCOPE,
                allowed=False,
                key=key,
                detail=f"key lacks scope {scope.value}",
            )
        bucket = self._bucket_for(key_id, endpoint_class, key.tier)
        allowed, retry_after = bucket.consume(cost=cost, now=self._now())
        if not allowed:
            return AuthResult(
                outcome=AuthOutcome.RATE_LIMITED,
                allowed=False,
                key=key,
                retry_after_seconds=retry_after,
                detail=(f"rate-limited; retry after {retry_after:.2f}s (tier {key.tier.value})"),
            )
        return AuthResult(
            outcome=AuthOutcome.ALLOWED,
            allowed=True,
            key=key,
            detail=f"authenticated as {key.label or key.key_id}",
        )


# ── Helpers ──────────────────────────────────────────────


def hash_secret(plaintext: str) -> str:
    """SHA-256 hex of the plaintext secret. Used at issuance time
    to populate `ApiKey.secret_hash`. Pin: never store / log the
    plaintext — it's shown to the researcher once and discarded."""
    if not plaintext:
        raise ValueError("secret must be non-empty")
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def issue_secret(*, length_bytes: int = 32) -> str:
    """Generate a fresh URL-safe secret. The caller stores
    `hash_secret(s)` in the registry and ships the plaintext
    once to the researcher.

    Pin: 32 bytes → ~256 bits of entropy. URL-safe encoding
    (`secrets.token_urlsafe`) so the researcher can paste it
    into a header without escaping."""
    if length_bytes < 16:
        raise ValueError(f"length_bytes must be >= 16 for ≥128 bits entropy; got {length_bytes}")
    return secrets.token_urlsafe(length_bytes)


def make_api_key(
    *,
    key_id: str,
    plaintext_secret: str,
    scopes: Iterable[Scope],
    tier: Tier,
    label: str = "",
) -> ApiKey:
    """Convenience constructor — hashes the secret + builds the
    `ApiKey`. Pin: callers should NEVER write to `ApiKey`
    directly with a plaintext secret stored in the wrong field;
    this helper is the single point of entry."""
    if not key_id:
        raise ValueError("key_id must be non-empty")
    return ApiKey(
        key_id=key_id,
        secret_hash=hash_secret(plaintext_secret),
        scopes=frozenset(scopes),
        tier=tier,
        label=label,
    )


__all__ = [
    "ApiKey",
    "AuthOutcome",
    "AuthResult",
    "ResearchApiRegistry",
    "Scope",
    "Tier",
    "hash_secret",
    "issue_secret",
    "make_api_key",
    "tier_limits",
]
