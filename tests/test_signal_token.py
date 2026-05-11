"""Tests for marketplace/signal_token.py — Round-5 Wave 21.G."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from halal_trader.marketplace.signal_token import (
    SignalKind,
    SignalPayload,
    SignedSignal,
    TokenExpired,
    TokenInvalid,
    TokenReplayed,
    render_signed,
    sign,
    verify,
)

_ALICE_SECRET = b"alice-secret-bytes-long-enough"
_BOB_SECRET = b"bob-secret-bytes-different-key"


def _secret_lookup(author_id: str) -> bytes:
    if author_id == "alice":
        return _ALICE_SECRET
    if author_id == "bob":
        return _BOB_SECRET
    return b""


def _payload(
    signal_id: str = "S1",
    author_id: str = "alice",
    ticker: str = "AAPL",
    kind: SignalKind = SignalKind.BUY,
    issued_at: datetime | None = None,
    nonce: str = "n-abc-001",
) -> SignalPayload:
    if issued_at is None:
        issued_at = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)
    return SignalPayload(
        signal_id=signal_id,
        author_id=author_id,
        ticker=ticker,
        kind=kind,
        issued_at=issued_at,
        nonce=nonce,
    )


# --- Payload validation ----------------------------------------------


def test_payload_valid():
    p = _payload()
    assert p.kind is SignalKind.BUY


def test_payload_empty_id_rejected():
    with pytest.raises(ValueError):
        _payload(signal_id="")


def test_payload_empty_nonce_rejected():
    with pytest.raises(ValueError):
        _payload(nonce="")


def test_payload_long_nonce_rejected():
    with pytest.raises(ValueError):
        _payload(nonce="x" * 100)


def test_payload_naive_datetime_rejected():
    """Pin: issued_at must be tz-aware."""
    with pytest.raises(ValueError):
        SignalPayload(
            signal_id="S1",
            author_id="alice",
            ticker="AAPL",
            kind=SignalKind.BUY,
            issued_at=datetime(2026, 5, 11, 12, 0),
            nonce="n1",
        )


def test_payload_immutable():
    p = _payload()
    with pytest.raises(AttributeError):
        p.ticker = "X"  # type: ignore[misc]


# --- sign ---------------------------------------------------------


def test_sign_basic():
    p = _payload()
    signed = sign(p, secret_lookup=_secret_lookup)
    assert len(signed.hmac_hex) == 64
    assert signed.payload is p


def test_sign_deterministic():
    p = _payload()
    s1 = sign(p, secret_lookup=_secret_lookup)
    s2 = sign(p, secret_lookup=_secret_lookup)
    assert s1.hmac_hex == s2.hmac_hex


def test_sign_different_secret_different_hmac():
    p_alice = _payload(author_id="alice")
    p_bob = _payload(author_id="bob")
    s_alice = sign(p_alice, secret_lookup=_secret_lookup)
    s_bob = sign(p_bob, secret_lookup=_secret_lookup)
    assert s_alice.hmac_hex != s_bob.hmac_hex


def test_sign_secret_not_registered_rejected():
    p = _payload(author_id="charlie")
    with pytest.raises(ValueError):
        sign(p, secret_lookup=_secret_lookup)


def test_sign_short_secret_rejected():
    p = _payload()
    with pytest.raises(ValueError):
        sign(p, secret_lookup=lambda _: b"short")


def test_sign_wrong_type_rejected():
    p = _payload()
    with pytest.raises(TypeError):
        sign(p, secret_lookup=lambda _: "not-bytes")  # type: ignore[arg-type,return-value]


# --- SignedSignal validation -------------------------------------


def test_signed_invalid_hmac_length_rejected():
    p = _payload()
    with pytest.raises(ValueError):
        SignedSignal(payload=p, hmac_hex="short")


def test_signed_empty_hmac_rejected():
    p = _payload()
    with pytest.raises(ValueError):
        SignedSignal(payload=p, hmac_hex="")


# --- verify — happy path -----------------------------------------


def test_verify_clean_signature():
    p = _payload()
    signed = sign(p, secret_lookup=_secret_lookup)
    assert verify(
        signed,
        secret_lookup=_secret_lookup,
        now=datetime(2026, 5, 11, 12, 1, tzinfo=timezone.utc),
    )


# --- verify — tamper detection -----------------------------------


def test_verify_tampered_payload_invalid():
    p = _payload()
    signed = sign(p, secret_lookup=_secret_lookup)
    # Mutate the payload to a different ticker; HMAC won't match.
    tampered = SignedSignal(
        payload=_payload(ticker="MSFT"),
        hmac_hex=signed.hmac_hex,
    )
    with pytest.raises(TokenInvalid):
        verify(
            tampered,
            secret_lookup=_secret_lookup,
            now=datetime(2026, 5, 11, 12, 1, tzinfo=timezone.utc),
        )


def test_verify_wrong_secret_invalid():
    p = _payload()
    signed = sign(p, secret_lookup=_secret_lookup)

    # Use a different secret store on verify.
    def other_lookup(_: str) -> bytes:
        return b"different-secret-bytes-of-min-len"

    with pytest.raises(TokenInvalid):
        verify(
            signed,
            secret_lookup=other_lookup,
            now=datetime(2026, 5, 11, 12, 1, tzinfo=timezone.utc),
        )


def test_verify_no_secret_for_author_invalid():
    p = _payload()
    signed = sign(p, secret_lookup=_secret_lookup)
    with pytest.raises(TokenInvalid):
        verify(
            signed,
            secret_lookup=lambda _: b"",
            now=datetime(2026, 5, 11, 12, 1, tzinfo=timezone.utc),
        )


# --- verify — TTL ----------------------------------------------


def test_verify_expired_token():
    p = _payload()
    signed = sign(p, secret_lookup=_secret_lookup)
    # 1 day + 1 second later.
    with pytest.raises(TokenExpired):
        verify(
            signed,
            secret_lookup=_secret_lookup,
            now=datetime(2026, 5, 12, 12, 0, 1, tzinfo=timezone.utc),
            max_age_seconds=86_400,
        )


def test_verify_future_token_invalid():
    p = _payload(issued_at=datetime(2027, 5, 11, 12, 0, tzinfo=timezone.utc))
    signed = sign(p, secret_lookup=_secret_lookup)
    with pytest.raises(TokenInvalid):
        verify(
            signed,
            secret_lookup=_secret_lookup,
            now=datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc),
        )


def test_verify_invalid_max_age_rejected():
    p = _payload()
    signed = sign(p, secret_lookup=_secret_lookup)
    with pytest.raises(ValueError):
        verify(
            signed,
            secret_lookup=_secret_lookup,
            now=datetime(2026, 5, 11, 12, 1, tzinfo=timezone.utc),
            max_age_seconds=0,
        )


def test_verify_naive_now_rejected():
    p = _payload()
    signed = sign(p, secret_lookup=_secret_lookup)
    with pytest.raises(ValueError):
        verify(
            signed,
            secret_lookup=_secret_lookup,
            now=datetime(2026, 5, 11, 12, 1),
        )


# --- verify — replay protection -------------------------------


def test_verify_replayed_nonce_rejected():
    p = _payload(nonce="seen-already")
    signed = sign(p, secret_lookup=_secret_lookup)
    with pytest.raises(TokenReplayed):
        verify(
            signed,
            secret_lookup=_secret_lookup,
            now=datetime(2026, 5, 11, 12, 1, tzinfo=timezone.utc),
            seen_nonces=frozenset({"seen-already"}),
        )


def test_verify_unseen_nonce_passes():
    p = _payload(nonce="fresh-nonce")
    signed = sign(p, secret_lookup=_secret_lookup)
    assert verify(
        signed,
        secret_lookup=_secret_lookup,
        now=datetime(2026, 5, 11, 12, 1, tzinfo=timezone.utc),
        seen_nonces=frozenset({"other-nonce"}),
    )


# --- Render --------------------------------------------------


def test_render_no_secret_leak():
    p = _payload(author_id="alice@example.com")
    signed = sign(p, secret_lookup=lambda _: _ALICE_SECRET)
    out = render_signed(signed)
    assert "alice@example.com" not in out
    # HMAC is masked.
    assert signed.hmac_hex not in out
    # And the secret bytes are not in render.
    assert _ALICE_SECRET.decode() not in out


def test_render_includes_signal_id():
    p = _payload(signal_id="S-42")
    signed = sign(p, secret_lookup=_secret_lookup)
    out = render_signed(signed)
    assert "S-42" in out
