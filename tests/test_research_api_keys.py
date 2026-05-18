"""Tests for `web/research_api_keys.py`.

Pins the auth → scope → rate-limit order, the constant-time
secret comparison contract, the per-(key, endpoint) bucket
isolation, the four-bucket auth outcome vocabulary, and the
input validation.
"""

from __future__ import annotations

import pytest

from halal_trader.web.research_api_keys import (
    ApiKey,
    AuthOutcome,
    AuthResult,
    ResearchApiRegistry,
    Scope,
    Tier,
    hash_secret,
    issue_secret,
    make_api_key,
    tier_limits,
)


def _clock():
    """Mutable clock for deterministic rate-limit tests."""
    state = {"now": 0.0}

    def now() -> float:
        return state["now"]

    def advance(seconds: float) -> None:
        state["now"] += seconds

    return now, advance


def _key(
    *,
    key_id: str = "rk-abc123",
    secret: str = "test-secret",
    scopes: tuple[Scope, ...] = (Scope.READ_TRADES,),
    tier: Tier = Tier.RESEARCHER,
    label: str = "",
) -> ApiKey:
    return make_api_key(
        key_id=key_id,
        plaintext_secret=secret,
        scopes=scopes,
        tier=tier,
        label=label,
    )


# ── hash_secret + verify ─────────────────────────────────


def test_hash_secret_is_deterministic():
    """Pin: same plaintext → same hash. Required for the
    registry's persistence layer."""
    assert hash_secret("secret") == hash_secret("secret")


def test_hash_secret_differs_for_different_plaintext():
    assert hash_secret("a") != hash_secret("b")


def test_hash_secret_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        hash_secret("")


def test_verify_secret_constant_time_check_passes_on_match():
    key = _key(secret="test-secret")
    assert key.verify_secret("test-secret")


def test_verify_secret_rejects_wrong_secret():
    key = _key(secret="real")
    assert not key.verify_secret("wrong")


def test_verify_secret_rejects_empty_when_real_secret_set():
    key = _key(secret="real")
    assert not key.verify_secret("")


# ── issue_secret ─────────────────────────────────────────


def test_issue_secret_returns_url_safe_string():
    """URL-safe so a researcher can paste into a header without
    escaping."""
    s = issue_secret()
    # URL-safe alphabet: A-Z, a-z, 0-9, -, _
    for ch in s:
        assert ch.isalnum() or ch in "-_"


def test_issue_secret_high_entropy_unique():
    """Different calls produce different secrets — pin so a
    determinism bug never silently issues the same secret to
    multiple researchers."""
    secrets_set = {issue_secret() for _ in range(10)}
    assert len(secrets_set) == 10


def test_issue_secret_rejects_low_entropy_request():
    """Pin: minimum 16 bytes → ~128 bits entropy. Accept the
    operator's preference but block clearly-insecure values."""
    with pytest.raises(ValueError, match="length_bytes"):
        issue_secret(length_bytes=8)


# ── make_api_key ─────────────────────────────────────────


def test_make_api_key_rejects_empty_key_id():
    with pytest.raises(ValueError, match="key_id"):
        make_api_key(
            key_id="",
            plaintext_secret="x",
            scopes=(Scope.READ_TRADES,),
            tier=Tier.RESEARCHER,
        )


def test_make_api_key_stores_hash_not_plaintext():
    """Pin: `secret_hash` is the SHA-256 hex; never the plaintext."""
    key = _key(secret="real-secret")
    assert key.secret_hash != "real-secret"
    assert len(key.secret_hash) == 64  # SHA-256 hex


def test_make_api_key_uses_frozenset_for_scopes():
    """Pin: scopes are immutable. Adding to a registered key's
    scope set must require re-issuance, not a runtime mutation."""
    key = _key(scopes=(Scope.READ_TRADES, Scope.READ_HALAL))
    assert isinstance(key.scopes, frozenset)
    with pytest.raises(AttributeError):
        key.scopes.add(Scope.READ_REGIME)  # type: ignore[attr-defined]


def test_apikey_is_immutable():
    key = _key()
    with pytest.raises(Exception):
        key.tier = Tier.PARTNER  # type: ignore[misc]


# ── has_scope ────────────────────────────────────────────


def test_has_scope_returns_true_when_scope_granted():
    key = _key(scopes=(Scope.READ_TRADES, Scope.READ_HALAL))
    assert key.has_scope(Scope.READ_TRADES)
    assert key.has_scope(Scope.READ_HALAL)


def test_has_scope_returns_false_when_scope_missing():
    """Pin: scopes are additive, no implicit hierarchy. A key
    without `read:trades` cannot read trades even if it has
    every other scope."""
    key = _key(scopes=(Scope.READ_HALAL,))
    assert not key.has_scope(Scope.READ_TRADES)


# ── tier_limits ──────────────────────────────────────────


def test_tier_limits_returns_documented_values():
    """Pin the tier values so a corrupt enum addition doesn't
    silently 10× a tier."""
    assert tier_limits(Tier.ANONYMOUS) == (10, 20)
    assert tier_limits(Tier.RESEARCHER) == (60, 120)
    assert tier_limits(Tier.PARTNER) == (600, 1200)


def test_tier_limits_partner_is_strictly_higher_than_researcher():
    """Pin the relative ordering. Same for the rest."""
    p_refill, p_burst = tier_limits(Tier.PARTNER)
    r_refill, r_burst = tier_limits(Tier.RESEARCHER)
    a_refill, a_burst = tier_limits(Tier.ANONYMOUS)
    assert p_refill > r_refill > a_refill
    assert p_burst > r_burst > a_burst


# ── registry: register / revoke / get ────────────────────


def test_register_and_get():
    reg = ResearchApiRegistry()
    key = _key()
    reg.register(key)
    assert reg.get("rk-abc123") is key


def test_register_rejects_duplicate_key_id():
    """Pin: silent overwrites mask leaks (an attacker who
    registers a key with the same id as a victim's would
    take over their bucket); refuse instead."""
    reg = ResearchApiRegistry()
    reg.register(_key())
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(_key())


def test_revoke_is_idempotent():
    """Pin: revoking an already-removed key is a no-op (operator
    may run revoke twice for safety)."""
    reg = ResearchApiRegistry()
    reg.register(_key())
    reg.revoke("rk-abc123")
    reg.revoke("rk-abc123")  # second call must not raise


def test_get_returns_none_for_unknown_id():
    reg = ResearchApiRegistry()
    assert reg.get("missing") is None


# ── authenticate: outcome vocabulary ─────────────────────


def test_authenticate_unknown_key():
    reg = ResearchApiRegistry()
    res = reg.authenticate(
        key_id="not-registered",
        secret="x",
        scope=Scope.READ_TRADES,
        endpoint_class="trades.list",
    )
    assert res.outcome == AuthOutcome.UNKNOWN_KEY
    assert not res.allowed
    assert res.key is None


def test_authenticate_invalid_secret():
    reg = ResearchApiRegistry()
    reg.register(_key(secret="real"))
    res = reg.authenticate(
        key_id="rk-abc123",
        secret="wrong",
        scope=Scope.READ_TRADES,
        endpoint_class="trades.list",
    )
    assert res.outcome == AuthOutcome.INVALID_SECRET
    assert not res.allowed
    # Pin: registry surfaces *which* key was attempted so audit
    # logs can record "key X tried with bad secret".
    assert res.key is not None


def test_authenticate_missing_scope():
    reg = ResearchApiRegistry()
    reg.register(_key(scopes=(Scope.READ_HALAL,)))
    res = reg.authenticate(
        key_id="rk-abc123",
        secret="test-secret",
        scope=Scope.READ_TRADES,
        endpoint_class="trades.list",
    )
    assert res.outcome == AuthOutcome.MISSING_SCOPE
    assert not res.allowed


def test_authenticate_allowed_when_everything_passes():
    reg = ResearchApiRegistry()
    reg.register(_key())
    res = reg.authenticate(
        key_id="rk-abc123",
        secret="test-secret",
        scope=Scope.READ_TRADES,
        endpoint_class="trades.list",
    )
    assert res.outcome == AuthOutcome.ALLOWED
    assert res.allowed


# ── authenticate: check ordering ─────────────────────────


def test_invalid_secret_does_not_burn_rate_limit_bucket():
    """Pin: an attacker spamming requests with a victim's key_id
    and a wrong secret must NOT exhaust the bucket — the
    auth check happens before the bucket consume."""
    now, _advance = _clock()
    reg = ResearchApiRegistry(now_fn=now)
    reg.register(_key(secret="real", tier=Tier.ANONYMOUS))
    # Tier ANONYMOUS has burst=20; spam 100 bad-secret requests.
    for _ in range(100):
        reg.authenticate(
            key_id="rk-abc123",
            secret="wrong",
            scope=Scope.READ_TRADES,
            endpoint_class="trades.list",
        )
    # The legitimate user can still make their full burst.
    res = reg.authenticate(
        key_id="rk-abc123",
        secret="real",
        scope=Scope.READ_TRADES,
        endpoint_class="trades.list",
    )
    assert res.allowed


def test_missing_scope_does_not_burn_rate_limit_bucket():
    """Pin: same idea for scope failures. An auth-attempt that
    failed scope check shouldn't reduce the legitimate user's
    rate-limit budget."""
    now, _advance = _clock()
    reg = ResearchApiRegistry(now_fn=now)
    reg.register(_key(scopes=(Scope.READ_HALAL,), tier=Tier.ANONYMOUS))
    # Spam 100 wrong-scope requests.
    for _ in range(100):
        reg.authenticate(
            key_id="rk-abc123",
            secret="test-secret",
            scope=Scope.READ_TRADES,
            endpoint_class="trades.list",
        )
    # Legitimate read:halal request still works.
    res = reg.authenticate(
        key_id="rk-abc123",
        secret="test-secret",
        scope=Scope.READ_HALAL,
        endpoint_class="halal.summary",
    )
    assert res.allowed


# ── rate limiting ────────────────────────────────────────


def test_rate_limit_kicks_in_after_burst():
    """Pin: ANONYMOUS tier has burst=20. The 21st request in a
    burst-window should be RATE_LIMITED."""
    now, _advance = _clock()
    reg = ResearchApiRegistry(now_fn=now)
    reg.register(_key(tier=Tier.ANONYMOUS))
    for _ in range(20):
        res = reg.authenticate(
            key_id="rk-abc123",
            secret="test-secret",
            scope=Scope.READ_TRADES,
            endpoint_class="trades.list",
        )
        assert res.allowed
    # 21st call exhausts the bucket.
    res = reg.authenticate(
        key_id="rk-abc123",
        secret="test-secret",
        scope=Scope.READ_TRADES,
        endpoint_class="trades.list",
    )
    assert res.outcome == AuthOutcome.RATE_LIMITED
    assert not res.allowed
    assert res.retry_after_seconds > 0


def test_rate_limit_refills_after_time_passes():
    """ANONYMOUS = 10/min refill; after 6 seconds, 1 token has
    refilled."""
    now, advance = _clock()
    reg = ResearchApiRegistry(now_fn=now)
    reg.register(_key(tier=Tier.ANONYMOUS))
    # Burn the bucket.
    for _ in range(20):
        reg.authenticate(
            key_id="rk-abc123",
            secret="test-secret",
            scope=Scope.READ_TRADES,
            endpoint_class="trades.list",
        )
    # 6s → 1 token refilled (10/min × 6/60s).
    advance(6.0)
    res = reg.authenticate(
        key_id="rk-abc123",
        secret="test-secret",
        scope=Scope.READ_TRADES,
        endpoint_class="trades.list",
    )
    assert res.allowed


def test_rate_limit_per_endpoint_class():
    """Pin: a slow query on `trades.list` doesn't burn budget
    for fast queries on `halal.summary`. Per-(key, endpoint)
    bucket isolation."""
    now, _advance = _clock()
    reg = ResearchApiRegistry(now_fn=now)
    reg.register(
        _key(
            scopes=(Scope.READ_TRADES, Scope.READ_HALAL),
            tier=Tier.ANONYMOUS,
        )
    )
    # Burn the trades.list bucket.
    for _ in range(20):
        reg.authenticate(
            key_id="rk-abc123",
            secret="test-secret",
            scope=Scope.READ_TRADES,
            endpoint_class="trades.list",
        )
    # halal.summary bucket is untouched.
    res = reg.authenticate(
        key_id="rk-abc123",
        secret="test-secret",
        scope=Scope.READ_HALAL,
        endpoint_class="halal.summary",
    )
    assert res.allowed


def test_rate_limit_partner_tier_handles_higher_burst():
    """Pin: PARTNER tier (burst 1200) doesn't get rate-limited
    on a normal-burst workload."""
    now, _advance = _clock()
    reg = ResearchApiRegistry(now_fn=now)
    reg.register(_key(tier=Tier.PARTNER))
    for _ in range(500):
        res = reg.authenticate(
            key_id="rk-abc123",
            secret="test-secret",
            scope=Scope.READ_TRADES,
            endpoint_class="trades.list",
        )
        assert res.allowed


def test_rate_limit_retry_after_makes_sense():
    """Pin: retry_after_seconds is the time until the bucket has
    enough tokens. After the burst is exhausted, retry_after
    should equal `1 / refill_per_second` for cost=1 (one
    token's worth of wait)."""
    now, _advance = _clock()
    reg = ResearchApiRegistry(now_fn=now)
    reg.register(_key(tier=Tier.ANONYMOUS))  # 10/min = ~0.167/s
    for _ in range(20):
        reg.authenticate(
            key_id="rk-abc123",
            secret="test-secret",
            scope=Scope.READ_TRADES,
            endpoint_class="trades.list",
        )
    res = reg.authenticate(
        key_id="rk-abc123",
        secret="test-secret",
        scope=Scope.READ_TRADES,
        endpoint_class="trades.list",
    )
    # 1 / (10/60) = 6.0 seconds
    assert res.retry_after_seconds == pytest.approx(6.0, rel=0.05)


def test_revoked_key_cannot_authenticate():
    reg = ResearchApiRegistry()
    reg.register(_key())
    reg.revoke("rk-abc123")
    res = reg.authenticate(
        key_id="rk-abc123",
        secret="test-secret",
        scope=Scope.READ_TRADES,
        endpoint_class="trades.list",
    )
    assert res.outcome == AuthOutcome.UNKNOWN_KEY


# ── output structure ─────────────────────────────────────


def test_auth_result_immutable():
    res = AuthResult(outcome=AuthOutcome.ALLOWED, allowed=True)
    with pytest.raises(Exception):
        res.allowed = False  # type: ignore[misc]


def test_auth_outcome_is_string_enum():
    """Pin: string-valued enum so the values JSON-serialise
    cleanly for the dashboard."""
    assert AuthOutcome.ALLOWED.value == "allowed"
    assert isinstance(AuthOutcome.ALLOWED.value, str)


def test_scope_is_string_enum():
    assert Scope.READ_TRADES.value == "read:trades"


def test_tier_is_string_enum():
    assert Tier.RESEARCHER.value == "researcher"
