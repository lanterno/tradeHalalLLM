"""Tests for the auth primitives core."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.web.auth import (
    DEFAULT_POLICY,
    DEFAULT_RATE_LIMIT,
    AuthOutcome,
    LoginAttempt,
    PasswordHash,
    PasswordPolicy,
    PasswordValidationError,
    RateLimitPolicy,
    Session,
    authenticate,
    evaluate_rate_limit,
    hash_password,
    is_session_valid,
    issue_session,
    render_auth_result,
    verify_password,
)

_NOW = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
_VALID_PW = "Password123!secure"  # 19 chars, has digit + symbol


# ---------------------------------------------------------------------------
# PasswordPolicy validation
# ---------------------------------------------------------------------------


def test_default_policy() -> None:
    assert DEFAULT_POLICY.min_length == 12
    assert DEFAULT_POLICY.require_digit is True
    assert DEFAULT_POLICY.require_symbol is True


def test_policy_rejects_below_8() -> None:
    with pytest.raises(ValueError, match="NIST"):
        PasswordPolicy(min_length=7)


def test_policy_accepts_strict_settings() -> None:
    p = PasswordPolicy(min_length=20, require_digit=True, require_symbol=True)
    assert p.min_length == 20


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def test_hash_password_returns_password_hash() -> None:
    h = hash_password(_VALID_PW)
    assert isinstance(h, PasswordHash)
    assert h.algorithm == "scrypt"
    assert h.salt_b64
    assert h.hash_b64


def test_hash_password_uses_unique_salt() -> None:
    """Pin: each hash uses a fresh random salt."""

    a = hash_password(_VALID_PW)
    b = hash_password(_VALID_PW)
    assert a.salt_b64 != b.salt_b64
    assert a.hash_b64 != b.hash_b64


def test_hash_password_rejects_short() -> None:
    """Pin: <12 char password rejected by default policy."""

    with pytest.raises(PasswordValidationError, match="too short"):
        hash_password("short1!")


def test_hash_password_rejects_no_digit() -> None:
    with pytest.raises(PasswordValidationError, match="digit"):
        hash_password("Password!secure!")


def test_hash_password_rejects_no_symbol() -> None:
    with pytest.raises(PasswordValidationError, match="symbol"):
        hash_password("Password123secure")


def test_hash_password_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="string"):
        hash_password(12345)  # type: ignore[arg-type]


def test_hash_password_with_custom_policy() -> None:
    """Pin: stricter policy enforced."""

    strict = PasswordPolicy(min_length=20)
    with pytest.raises(PasswordValidationError, match="too short"):
        hash_password(_VALID_PW, policy=strict)


def test_hash_password_relaxed_policy() -> None:
    """Operator can disable digit/symbol requirements."""

    relaxed = PasswordPolicy(min_length=12, require_digit=False, require_symbol=False)
    h = hash_password("just-a-long-passphrase", policy=relaxed)
    assert h.algorithm == "scrypt"


# ---------------------------------------------------------------------------
# Password verification
# ---------------------------------------------------------------------------


def test_verify_password_correct() -> None:
    h = hash_password(_VALID_PW)
    assert verify_password(_VALID_PW, h) is True


def test_verify_password_wrong() -> None:
    h = hash_password(_VALID_PW)
    assert verify_password("WrongPassword123!", h) is False


def test_verify_password_empty() -> None:
    h = hash_password(_VALID_PW)
    assert verify_password("", h) is False


def test_verify_password_non_string_returns_false() -> None:
    h = hash_password(_VALID_PW)
    assert verify_password(12345, h) is False  # type: ignore[arg-type]


def test_verify_password_unknown_algorithm_returns_false() -> None:
    """Pin: unknown algorithm → False, never raises."""

    bad = PasswordHash(
        algorithm="md5",
        n=1,
        r=1,
        p=1,
        salt_b64="aaaa",
        hash_b64="bbbb",
    )
    assert verify_password(_VALID_PW, bad) is False


def test_verify_password_malformed_salt_returns_false() -> None:
    """Pin: malformed base64 → False, never raises."""

    h = PasswordHash(
        algorithm="scrypt",
        n=16384,
        r=8,
        p=1,
        salt_b64="!!!not-base64!!!",
        hash_b64="!!!also-bad!!!",
    )
    assert verify_password(_VALID_PW, h) is False


# ---------------------------------------------------------------------------
# PasswordHash validation
# ---------------------------------------------------------------------------


def test_password_hash_rejects_empty_algorithm() -> None:
    with pytest.raises(ValueError, match="algorithm"):
        PasswordHash(algorithm="", n=1, r=1, p=1, salt_b64="x", hash_b64="y")


def test_password_hash_rejects_zero_n() -> None:
    with pytest.raises(ValueError, match="parameters"):
        PasswordHash(algorithm="scrypt", n=0, r=1, p=1, salt_b64="x", hash_b64="y")


def test_password_hash_rejects_empty_salt() -> None:
    with pytest.raises(ValueError, match="salt"):
        PasswordHash(algorithm="scrypt", n=1, r=1, p=1, salt_b64="", hash_b64="y")


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_issue_session_returns_session() -> None:
    s = issue_session(user_id="user-1", now=_NOW)
    assert isinstance(s, Session)
    assert s.user_id == "user-1"
    assert s.issued_at == _NOW
    assert s.expires_at > _NOW


def test_issue_session_default_ttl_60min() -> None:
    s = issue_session(user_id="user-1", now=_NOW)
    assert s.expires_at - s.issued_at == timedelta(minutes=60)


def test_issue_session_custom_ttl() -> None:
    s = issue_session(user_id="user-1", now=_NOW, ttl_minutes=120)
    assert s.expires_at - s.issued_at == timedelta(minutes=120)


def test_issue_session_high_entropy_id() -> None:
    """Pin: session_id is high-entropy (≥16 chars after token_urlsafe)."""

    s = issue_session(user_id="user-1", now=_NOW)
    assert len(s.session_id) >= 16


def test_issue_session_unique_ids() -> None:
    """Pin: each session has a unique ID."""

    s1 = issue_session(user_id="user-1", now=_NOW)
    s2 = issue_session(user_id="user-1", now=_NOW)
    assert s1.session_id != s2.session_id


def test_issue_session_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        issue_session(user_id="", now=_NOW)


def test_issue_session_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        issue_session(user_id="user-1", now=datetime(2026, 5, 1))


def test_issue_session_rejects_below_5_min_ttl() -> None:
    with pytest.raises(ValueError, match="ttl_minutes"):
        issue_session(user_id="user-1", now=_NOW, ttl_minutes=4)


def test_issue_session_rejects_above_24h_ttl() -> None:
    with pytest.raises(ValueError, match="ttl_minutes"):
        issue_session(user_id="user-1", now=_NOW, ttl_minutes=60 * 24 + 1)


def test_issue_session_accepts_5_min_minimum() -> None:
    s = issue_session(user_id="user-1", now=_NOW, ttl_minutes=5)
    assert s.expires_at - s.issued_at == timedelta(minutes=5)


def test_issue_session_accepts_24h_maximum() -> None:
    s = issue_session(user_id="user-1", now=_NOW, ttl_minutes=60 * 24)
    assert s.expires_at - s.issued_at == timedelta(hours=24)


def test_session_validation_rejects_empty_session_id() -> None:
    with pytest.raises(ValueError, match="session_id"):
        Session(
            session_id="",
            user_id="user-1",
            issued_at=_NOW,
            expires_at=_NOW + timedelta(minutes=60),
        )


def test_session_validation_rejects_short_session_id() -> None:
    """Pin: session_id must be at least 16 chars (high-entropy)."""

    with pytest.raises(ValueError, match="16 chars"):
        Session(
            session_id="short",
            user_id="user-1",
            issued_at=_NOW,
            expires_at=_NOW + timedelta(minutes=60),
        )


def test_session_validation_rejects_naive_issued_at() -> None:
    with pytest.raises(ValueError, match="issued_at"):
        Session(
            session_id="x" * 16,
            user_id="user-1",
            issued_at=datetime(2026, 5, 1),
            expires_at=_NOW + timedelta(minutes=60),
        )


def test_session_validation_rejects_expires_before_issued() -> None:
    with pytest.raises(ValueError, match="expires_at"):
        Session(
            session_id="x" * 16,
            user_id="user-1",
            issued_at=_NOW,
            expires_at=_NOW - timedelta(minutes=1),
        )


# ---------------------------------------------------------------------------
# is_session_valid
# ---------------------------------------------------------------------------


def test_session_valid_within_window() -> None:
    s = issue_session(user_id="user-1", now=_NOW, ttl_minutes=60)
    assert is_session_valid(s, now=_NOW + timedelta(minutes=30)) is True


def test_session_invalid_after_expiry() -> None:
    s = issue_session(user_id="user-1", now=_NOW, ttl_minutes=60)
    assert is_session_valid(s, now=_NOW + timedelta(minutes=61)) is False


def test_session_valid_at_exactly_issued_at() -> None:
    """Pin: at issued_at, session is valid (boundary inclusive)."""

    s = issue_session(user_id="user-1", now=_NOW, ttl_minutes=60)
    assert is_session_valid(s, now=_NOW) is True


def test_session_invalid_at_exactly_expires_at() -> None:
    """Pin: at expires_at, session is invalid (exclusive on the upper bound)."""

    s = issue_session(user_id="user-1", now=_NOW, ttl_minutes=60)
    assert is_session_valid(s, now=s.expires_at) is False


def test_session_valid_rejects_naive_now() -> None:
    s = issue_session(user_id="user-1", now=_NOW)
    with pytest.raises(ValueError, match="timezone-aware"):
        is_session_valid(s, now=datetime(2026, 5, 1))


# ---------------------------------------------------------------------------
# RateLimitPolicy validation
# ---------------------------------------------------------------------------


def test_default_rate_limit() -> None:
    assert DEFAULT_RATE_LIMIT.max_failures == 5
    assert DEFAULT_RATE_LIMIT.window_minutes == 15


def test_rate_limit_rejects_zero_failures() -> None:
    with pytest.raises(ValueError, match="max_failures"):
        RateLimitPolicy(max_failures=0)


def test_rate_limit_rejects_zero_window() -> None:
    with pytest.raises(ValueError, match="window_minutes"):
        RateLimitPolicy(window_minutes=0)


# ---------------------------------------------------------------------------
# LoginAttempt validation
# ---------------------------------------------------------------------------


def test_login_attempt_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        LoginAttempt(user_id="", timestamp=_NOW, success=True)


def test_login_attempt_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        LoginAttempt(user_id="user-1", timestamp=datetime(2026, 5, 1), success=True)


# ---------------------------------------------------------------------------
# evaluate_rate_limit
# ---------------------------------------------------------------------------


def test_rate_limit_no_history_allowed() -> None:
    assert evaluate_rate_limit(user_id="user-1", history=(), now=_NOW) is True


def test_rate_limit_under_threshold_allowed() -> None:
    history = tuple(
        LoginAttempt(
            user_id="user-1",
            timestamp=_NOW - timedelta(minutes=i + 1),
            success=False,
        )
        for i in range(3)
    )
    assert evaluate_rate_limit(user_id="user-1", history=history, now=_NOW) is True


def test_rate_limit_at_threshold_blocked() -> None:
    """Pin: 5 failures within 15min → blocked."""

    history = tuple(
        LoginAttempt(
            user_id="user-1",
            timestamp=_NOW - timedelta(minutes=i + 1),
            success=False,
        )
        for i in range(5)
    )
    assert evaluate_rate_limit(user_id="user-1", history=history, now=_NOW) is False


def test_rate_limit_success_resets_counter() -> None:
    """Pin: a successful login resets the failure counter."""

    history = (
        LoginAttempt(
            user_id="user-1",
            timestamp=_NOW - timedelta(minutes=10),
            success=False,
        ),
        LoginAttempt(
            user_id="user-1",
            timestamp=_NOW - timedelta(minutes=8),
            success=False,
        ),
        LoginAttempt(
            user_id="user-1",
            timestamp=_NOW - timedelta(minutes=5),
            success=True,
        ),
        LoginAttempt(
            user_id="user-1",
            timestamp=_NOW - timedelta(minutes=2),
            success=False,
        ),
    )
    # 1 failure since the success — under the 5-failure threshold
    assert evaluate_rate_limit(user_id="user-1", history=history, now=_NOW) is True


def test_rate_limit_old_failures_outside_window_dont_count() -> None:
    """Pin: failures outside the 15min window don't count toward the limit."""

    history = tuple(
        LoginAttempt(
            user_id="user-1",
            timestamp=_NOW - timedelta(hours=i + 1),
            success=False,
        )
        for i in range(10)
    )
    assert evaluate_rate_limit(user_id="user-1", history=history, now=_NOW) is True


def test_rate_limit_other_users_dont_count() -> None:
    """Pin: only user-1's attempts count toward user-1's limit."""

    history = tuple(
        LoginAttempt(
            user_id="user-2",
            timestamp=_NOW - timedelta(minutes=i + 1),
            success=False,
        )
        for i in range(10)
    )
    assert evaluate_rate_limit(user_id="user-1", history=history, now=_NOW) is True


def test_rate_limit_custom_policy() -> None:
    """Stricter 3-failures-per-5min policy blocks at 3."""

    strict = RateLimitPolicy(max_failures=3, window_minutes=5)
    history = tuple(
        LoginAttempt(
            user_id="user-1",
            timestamp=_NOW - timedelta(minutes=i + 1),
            success=False,
        )
        for i in range(3)
    )
    assert evaluate_rate_limit(user_id="user-1", history=history, now=_NOW, policy=strict) is False


def test_rate_limit_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate_rate_limit(user_id="user-1", history=(), now=datetime(2026, 5, 1))


def test_rate_limit_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        evaluate_rate_limit(user_id="", history=(), now=_NOW)


# ---------------------------------------------------------------------------
# authenticate composition
# ---------------------------------------------------------------------------


def test_authenticate_success() -> None:
    """Correct password + clean rate-limit → SUCCESS with session."""

    stored = hash_password(_VALID_PW)
    result = authenticate(
        user_id="user-1",
        plaintext_password=_VALID_PW,
        stored_hash=stored,
        history=(),
        now=_NOW,
    )
    assert result.outcome is AuthOutcome.SUCCESS
    assert result.session is not None
    assert result.session.user_id == "user-1"


def test_authenticate_invalid_credentials() -> None:
    stored = hash_password(_VALID_PW)
    result = authenticate(
        user_id="user-1",
        plaintext_password="WrongPassword123!",
        stored_hash=stored,
        history=(),
        now=_NOW,
    )
    assert result.outcome is AuthOutcome.INVALID_CREDENTIALS
    assert result.session is None


def test_authenticate_rate_limited() -> None:
    """Pin: rate-limited user is blocked even with correct password."""

    stored = hash_password(_VALID_PW)
    history = tuple(
        LoginAttempt(
            user_id="user-1",
            timestamp=_NOW - timedelta(minutes=i + 1),
            success=False,
        )
        for i in range(5)
    )
    result = authenticate(
        user_id="user-1",
        plaintext_password=_VALID_PW,  # correct password!
        stored_hash=stored,
        history=history,
        now=_NOW,
    )
    assert result.outcome is AuthOutcome.RATE_LIMITED
    assert result.session is None
    assert any("attempts" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_password_hash_is_frozen() -> None:
    h = hash_password(_VALID_PW)
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.algorithm = "md5"  # type: ignore[misc]


def test_session_is_frozen() -> None:
    s = issue_session(user_id="user-1", now=_NOW)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.user_id = "other"  # type: ignore[misc]


def test_login_attempt_is_frozen() -> None:
    a = LoginAttempt(user_id="user-1", timestamp=_NOW, success=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.success = False  # type: ignore[misc]


def test_policy_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_POLICY.min_length = 8  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB stability
# ---------------------------------------------------------------------------


def test_auth_outcome_string_values() -> None:
    assert AuthOutcome.SUCCESS.value == "success"
    assert AuthOutcome.INVALID_CREDENTIALS.value == "invalid_credentials"
    assert AuthOutcome.RATE_LIMITED.value == "rate_limited"
    assert AuthOutcome.SESSION_EXPIRED.value == "session_expired"
    assert AuthOutcome.SESSION_NOT_FOUND.value == "session_not_found"


# ---------------------------------------------------------------------------
# Render output — pinned no-secret-leak contract
# ---------------------------------------------------------------------------


def test_render_success_includes_user_and_outcome() -> None:
    stored = hash_password(_VALID_PW)
    result = authenticate(
        user_id="user-1",
        plaintext_password=_VALID_PW,
        stored_hash=stored,
        history=(),
        now=_NOW,
    )
    text = render_auth_result(result)
    assert "✅" in text
    assert "user-1" in text
    assert "SUCCESS" in text


def test_render_invalid_credentials() -> None:
    stored = hash_password(_VALID_PW)
    result = authenticate(
        user_id="user-1",
        plaintext_password="WrongPassword123!",
        stored_hash=stored,
        history=(),
        now=_NOW,
    )
    text = render_auth_result(result)
    assert "🔑" in text
    assert "INVALID_CREDENTIALS" in text


def test_render_rate_limited() -> None:
    stored = hash_password(_VALID_PW)
    history = tuple(
        LoginAttempt(
            user_id="user-1",
            timestamp=_NOW - timedelta(minutes=i + 1),
            success=False,
        )
        for i in range(5)
    )
    result = authenticate(
        user_id="user-1",
        plaintext_password=_VALID_PW,
        stored_hash=stored,
        history=history,
        now=_NOW,
    )
    text = render_auth_result(result)
    assert "🚫" in text
    assert "RATE_LIMITED" in text


def test_render_does_not_include_session_id() -> None:
    """Pin: session_id is a secret — never in render."""

    stored = hash_password(_VALID_PW)
    result = authenticate(
        user_id="user-1",
        plaintext_password=_VALID_PW,
        stored_hash=stored,
        history=(),
        now=_NOW,
    )
    text = render_auth_result(result)
    assert result.session is not None
    assert result.session.session_id not in text


def test_render_does_not_include_password() -> None:
    """Pin: render never includes the plaintext password."""

    stored = hash_password("Secret-Password-456!")
    result = authenticate(
        user_id="user-1",
        plaintext_password="Secret-Password-456!",
        stored_hash=stored,
        history=(),
        now=_NOW,
    )
    text = render_auth_result(result)
    assert "Secret-Password-456!" not in text


def test_render_does_not_include_password_hash_bytes() -> None:
    """Pin: render never includes the password hash / salt bytes."""

    stored = hash_password(_VALID_PW)
    result = authenticate(
        user_id="user-1",
        plaintext_password=_VALID_PW,
        stored_hash=stored,
        history=(),
        now=_NOW,
    )
    text = render_auth_result(result)
    assert stored.salt_b64 not in text
    assert stored.hash_b64 not in text


def test_render_includes_session_expiry_when_present() -> None:
    """Session expiry timestamp is rendered (it's not a secret)."""

    stored = hash_password(_VALID_PW)
    result = authenticate(
        user_id="user-1",
        plaintext_password=_VALID_PW,
        stored_hash=stored,
        history=(),
        now=_NOW,
    )
    text = render_auth_result(result)
    assert result.session is not None
    # The expiry datetime appears in ISO format
    assert "2026-05-01" in text


# ---------------------------------------------------------------------------
# Determinism / cryptographic properties
# ---------------------------------------------------------------------------


def test_two_hashes_of_same_password_differ() -> None:
    """Pin: random salt → identical passwords produce different hashes.

    Defends against rainbow-table attacks on a leaked database.
    """

    a = hash_password(_VALID_PW)
    b = hash_password(_VALID_PW)
    assert a.hash_b64 != b.hash_b64


def test_round_trip_hash_then_verify_works() -> None:
    """Pin: hash → verify round-trip is the canonical flow."""

    h = hash_password(_VALID_PW)
    assert verify_password(_VALID_PW, h) is True


# ---------------------------------------------------------------------------
# End-to-end realistic scenarios
# ---------------------------------------------------------------------------


def test_full_login_lifecycle() -> None:
    """Operator creates a user, hashes password, later authenticates."""

    stored = hash_password(_VALID_PW)
    # ... time passes, persistence layer stores the hash ...
    later = _NOW + timedelta(hours=2)
    result = authenticate(
        user_id="user-1",
        plaintext_password=_VALID_PW,
        stored_hash=stored,
        history=(),
        now=later,
    )
    assert result.outcome is AuthOutcome.SUCCESS
    assert result.session is not None
    assert is_session_valid(result.session, now=later) is True
    # 30 minutes later, still valid
    assert is_session_valid(result.session, now=later + timedelta(minutes=30)) is True
    # 61 minutes later, expired
    assert is_session_valid(result.session, now=later + timedelta(minutes=61)) is False


def test_brute_force_attack_blocked() -> None:
    """Operator brute-forces 5 wrong passwords → 6th attempt blocked."""

    stored = hash_password(_VALID_PW)
    history: list[LoginAttempt] = []
    # 5 failed attempts
    for i in range(5):
        attempt_time = _NOW + timedelta(minutes=i)
        result = authenticate(
            user_id="user-1",
            plaintext_password="WrongGuess123!",
            stored_hash=stored,
            history=tuple(history),
            now=attempt_time,
        )
        history.append(
            LoginAttempt(
                user_id="user-1",
                timestamp=attempt_time,
                success=result.outcome is AuthOutcome.SUCCESS,
            )
        )

    # 6th attempt with correct password — should be rate-limited
    result = authenticate(
        user_id="user-1",
        plaintext_password=_VALID_PW,  # correct!
        stored_hash=stored,
        history=tuple(history),
        now=_NOW + timedelta(minutes=6),
    )
    assert result.outcome is AuthOutcome.RATE_LIMITED


def test_attacker_targets_user_a_does_not_block_user_b() -> None:
    """Pin: attack on user-a doesn't block user-b's logins."""

    stored = hash_password(_VALID_PW)
    history = tuple(
        LoginAttempt(
            user_id="user-a",
            timestamp=_NOW - timedelta(minutes=i + 1),
            success=False,
        )
        for i in range(10)
    )
    # User-a is rate-limited
    result_a = authenticate(
        user_id="user-a",
        plaintext_password=_VALID_PW,
        stored_hash=stored,
        history=history,
        now=_NOW,
    )
    assert result_a.outcome is AuthOutcome.RATE_LIMITED

    # User-b can still log in
    result_b = authenticate(
        user_id="user-b",
        plaintext_password=_VALID_PW,
        stored_hash=stored,
        history=history,
        now=_NOW,
    )
    assert result_b.outcome is AuthOutcome.SUCCESS
