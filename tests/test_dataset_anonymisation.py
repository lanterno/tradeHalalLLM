"""Tests for the open-dataset anonymisation engine."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from halal_trader.web.dataset_anonymisation import (
    DEFAULT_POLICY,
    AnonymisationPolicy,
    AnonymisedDecision,
    RawDecision,
    USDBucket,
    anonymise_dataset,
    anonymise_decision,
    render_summary,
)

_NOW = datetime(2026, 5, 1, 14, 35, 12, tzinfo=UTC)
_SALT = b"abcdef0123456789-test-salt-32-bytes"


def _decision(
    *,
    decision_id: str = "dec-001",
    user_id: str = "user-1",
    timestamp: datetime = _NOW,
    symbol: str = "BTCUSDT",
    sector: str = "crypto",
    regime: str = "risk_on",
    action: str = "buy",
    notional_usd: float = 500.0,
    rationale: str = "Strong momentum + halal screen pass.",
) -> RawDecision:
    return RawDecision(
        decision_id=decision_id,
        user_id=user_id,
        timestamp=timestamp,
        symbol=symbol,
        sector=sector,
        regime=regime,
        action=action,
        notional_usd=notional_usd,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_default_policy() -> None:
    p = DEFAULT_POLICY
    assert p.k_anonymity_floor == 5
    assert p.enable_pii_redaction is True
    assert p.bucket_usd_amounts is True


def test_policy_rejects_zero_k() -> None:
    with pytest.raises(ValueError, match="k_anonymity_floor"):
        AnonymisationPolicy(k_anonymity_floor=0)


def test_policy_accepts_k_one_for_opt_out() -> None:
    """Pin: k=1 disables the check (operator opt-out for internal exports)."""

    p = AnonymisationPolicy(k_anonymity_floor=1)
    assert p.k_anonymity_floor == 1


# ---------------------------------------------------------------------------
# RawDecision validation
# ---------------------------------------------------------------------------


def test_raw_rejects_empty_decision_id() -> None:
    with pytest.raises(ValueError, match="decision_id"):
        _decision(decision_id="")


def test_raw_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        _decision(user_id="")


def test_raw_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        _decision(symbol="")


def test_raw_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _decision(timestamp=datetime(2026, 5, 1))


def test_raw_rejects_negative_notional() -> None:
    with pytest.raises(ValueError, match="notional_usd"):
        _decision(notional_usd=-1.0)


# ---------------------------------------------------------------------------
# Salt requirement
# ---------------------------------------------------------------------------


def test_anonymise_rejects_short_salt() -> None:
    with pytest.raises(ValueError, match="salt"):
        anonymise_decision(_decision(), salt=b"short")


def test_anonymise_dataset_rejects_short_salt() -> None:
    with pytest.raises(ValueError, match="salt"):
        anonymise_dataset((_decision(),), salt=b"short")


def test_anonymise_accepts_16_byte_salt() -> None:
    salt = b"a" * 16
    result = anonymise_decision(_decision(), salt=salt)
    assert isinstance(result, AnonymisedDecision)


# ---------------------------------------------------------------------------
# User-id hashing
# ---------------------------------------------------------------------------


def test_user_id_hashed_to_anonymous_token() -> None:
    """Pin: user_id is replaced by a deterministic salt-hashed token."""

    result = anonymise_decision(_decision(user_id="alice"), salt=_SALT)
    assert result.anonymous_user.startswith("anon-")
    assert "alice" not in result.anonymous_user


def test_same_user_id_produces_same_token() -> None:
    """Pin: deterministic — same user_id maps to same token within a salt."""

    a = anonymise_decision(_decision(user_id="alice"), salt=_SALT)
    b = anonymise_decision(_decision(user_id="alice"), salt=_SALT)
    assert a.anonymous_user == b.anonymous_user


def test_different_users_get_different_tokens() -> None:
    a = anonymise_decision(_decision(user_id="alice"), salt=_SALT)
    b = anonymise_decision(_decision(user_id="bob"), salt=_SALT)
    assert a.anonymous_user != b.anonymous_user


def test_different_salts_produce_different_tokens() -> None:
    """Pin: across exports with different salts, tokens differ.

    Re-exports cannot be cross-referenced without the same salt.
    """

    salt_a = b"a" * 16
    salt_b = b"b" * 16
    a = anonymise_decision(_decision(user_id="alice"), salt=salt_a)
    b = anonymise_decision(_decision(user_id="alice"), salt=salt_b)
    assert a.anonymous_user != b.anonymous_user


# ---------------------------------------------------------------------------
# Timestamp rounding
# ---------------------------------------------------------------------------


def test_timestamp_rounded_to_hour() -> None:
    """Pin: timestamp at 14:35:12 rounds to 14:00:00."""

    result = anonymise_decision(_decision(timestamp=_NOW), salt=_SALT)
    assert result.timestamp_hour.hour == 14
    assert result.timestamp_hour.minute == 0
    assert result.timestamp_hour.second == 0


def test_timestamp_at_top_of_hour_unchanged() -> None:
    top = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
    result = anonymise_decision(_decision(timestamp=top), salt=_SALT)
    assert result.timestamp_hour == top


# ---------------------------------------------------------------------------
# USD bucketing
# ---------------------------------------------------------------------------


def test_micro_amount_bucketed() -> None:
    result = anonymise_decision(_decision(notional_usd=50.0), salt=_SALT)
    assert result.usd_bucket is USDBucket.MICRO


def test_small_amount_bucketed() -> None:
    result = anonymise_decision(_decision(notional_usd=500.0), salt=_SALT)
    assert result.usd_bucket is USDBucket.SMALL


def test_medium_amount_bucketed() -> None:
    result = anonymise_decision(_decision(notional_usd=5_000.0), salt=_SALT)
    assert result.usd_bucket is USDBucket.MEDIUM


def test_large_amount_bucketed() -> None:
    result = anonymise_decision(_decision(notional_usd=50_000.0), salt=_SALT)
    assert result.usd_bucket is USDBucket.LARGE


def test_whale_amount_bucketed() -> None:
    result = anonymise_decision(_decision(notional_usd=500_000.0), salt=_SALT)
    assert result.usd_bucket is USDBucket.WHALE


def test_exactly_100_is_small_not_micro() -> None:
    """Pin: exactly $100 boundary → SMALL (≥100)."""

    result = anonymise_decision(_decision(notional_usd=100.0), salt=_SALT)
    assert result.usd_bucket is USDBucket.SMALL


def test_just_below_100_is_micro() -> None:
    result = anonymise_decision(_decision(notional_usd=99.99), salt=_SALT)
    assert result.usd_bucket is USDBucket.MICRO


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------


def test_email_redacted_in_rationale() -> None:
    raw = _decision(rationale="Operator at alice@example.com made this call.")
    result = anonymise_decision(raw, salt=_SALT)
    assert "alice@example.com" not in result.rationale_redacted
    assert "<redacted-email>" in result.rationale_redacted


def test_ssn_redacted_in_rationale() -> None:
    raw = _decision(rationale="Customer SSN 123-45-6789 flagged for KYC.")
    result = anonymise_decision(raw, salt=_SALT)
    assert "123-45-6789" not in result.rationale_redacted
    assert "<redacted-ssn>" in result.rationale_redacted


def test_ip_redacted_in_rationale() -> None:
    raw = _decision(rationale="Source IP 192.168.1.42 flagged.")
    result = anonymise_decision(raw, salt=_SALT)
    assert "192.168.1.42" not in result.rationale_redacted
    assert "<redacted-ip>" in result.rationale_redacted


def test_eth_address_redacted_in_rationale() -> None:
    raw = _decision(rationale="Tx hash to 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb6 detected.")
    result = anonymise_decision(raw, salt=_SALT)
    assert "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb6" not in result.rationale_redacted
    assert "<redacted-eth-address>" in result.rationale_redacted


def test_clean_rationale_unchanged() -> None:
    raw = _decision(rationale="Strong momentum + halal screen pass.")
    result = anonymise_decision(raw, salt=_SALT)
    assert result.rationale_redacted == "Strong momentum + halal screen pass."


def test_pii_redaction_can_be_disabled_via_policy() -> None:
    """Pin: operators with already-clean data can disable redaction for performance."""

    no_pii_redact = AnonymisationPolicy(enable_pii_redaction=False)
    raw = _decision(rationale="Email alice@example.com here.")
    result = anonymise_decision(raw, salt=_SALT, policy=no_pii_redact)
    # Disabled: email passes through
    assert "alice@example.com" in result.rationale_redacted


# ---------------------------------------------------------------------------
# Symbol + sector + regime preserved
# ---------------------------------------------------------------------------


def test_symbol_preserved() -> None:
    result = anonymise_decision(_decision(symbol="ETHUSDT"), salt=_SALT)
    assert result.symbol == "ETHUSDT"


def test_sector_preserved() -> None:
    result = anonymise_decision(_decision(sector="tech"), salt=_SALT)
    assert result.sector == "tech"


def test_regime_preserved() -> None:
    result = anonymise_decision(_decision(regime="risk_off"), salt=_SALT)
    assert result.regime == "risk_off"


def test_action_preserved() -> None:
    result = anonymise_decision(_decision(action="sell"), salt=_SALT)
    assert result.action == "sell"


# ---------------------------------------------------------------------------
# k-anonymity filter
# ---------------------------------------------------------------------------


def test_dataset_keeps_rows_meeting_k_anonymity() -> None:
    """5 rows with same QID → all kept under default k=5."""

    decisions = tuple(_decision(decision_id=f"dec-{i}", user_id="alice") for i in range(5))
    result = anonymise_dataset(decisions, salt=_SALT)
    assert len(result.decisions) == 5
    assert result.dropped_for_k_anonymity == 0


def test_dataset_drops_rows_below_k_anonymity() -> None:
    """4 rows with same QID under k=5 → all dropped."""

    decisions = tuple(_decision(decision_id=f"dec-{i}", user_id="alice") for i in range(4))
    result = anonymise_dataset(decisions, salt=_SALT)
    assert len(result.decisions) == 0
    assert result.dropped_for_k_anonymity == 4
    assert any("anonymity threshold" in w for w in result.warnings)


def test_k_one_disables_check() -> None:
    """Pin: k=1 → no rows dropped."""

    decisions = (_decision(),)  # single row, would fail k=5
    result = anonymise_dataset(
        decisions, salt=_SALT, policy=AnonymisationPolicy(k_anonymity_floor=1)
    )
    assert len(result.decisions) == 1
    assert result.dropped_for_k_anonymity == 0


def test_k_anonymity_groups_by_qid_tuple() -> None:
    """Pin: QID = (anonymous_user, sector, regime).

    Different sectors → different QID groups.
    """

    decisions = (
        _decision(decision_id="d1", user_id="alice", sector="crypto"),
        _decision(decision_id="d2", user_id="alice", sector="crypto"),
        _decision(decision_id="d3", user_id="alice", sector="crypto"),
        _decision(decision_id="d4", user_id="alice", sector="crypto"),
        _decision(decision_id="d5", user_id="alice", sector="crypto"),
        # different sector → different QID; only 1 row → drops under k=5
        _decision(decision_id="d6", user_id="alice", sector="tech"),
    )
    result = anonymise_dataset(decisions, salt=_SALT)
    # 5 crypto rows kept; 1 tech row dropped
    assert len(result.decisions) == 5
    assert result.dropped_for_k_anonymity == 1


def test_strict_k_10() -> None:
    """Pin: stricter k=10 drops rows that would survive k=5."""

    decisions = tuple(_decision(decision_id=f"dec-{i}", user_id="alice") for i in range(7))
    strict = AnonymisationPolicy(k_anonymity_floor=10)
    result = anonymise_dataset(decisions, salt=_SALT, policy=strict)
    assert len(result.decisions) == 0
    assert result.dropped_for_k_anonymity == 7


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------


def test_result_carries_raw_input_count() -> None:
    decisions = tuple(_decision(decision_id=f"dec-{i}", user_id="alice") for i in range(7))
    result = anonymise_dataset(decisions, salt=_SALT)
    assert result.raw_input_count == 7


def test_empty_dataset() -> None:
    result = anonymise_dataset((), salt=_SALT)
    assert result.decisions == ()
    assert result.raw_input_count == 0
    assert result.dropped_for_k_anonymity == 0


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_raw_decision_is_frozen() -> None:
    raw = _decision()
    with pytest.raises(dataclasses.FrozenInstanceError):
        raw.user_id = "other"  # type: ignore[misc]


def test_anonymised_decision_is_frozen() -> None:
    result = anonymise_decision(_decision(), salt=_SALT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.anonymous_user = "other"  # type: ignore[misc]


def test_policy_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_POLICY.k_anonymity_floor = 1  # type: ignore[misc]


def test_result_is_frozen() -> None:
    result = anonymise_dataset((), salt=_SALT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.decisions = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB stability
# ---------------------------------------------------------------------------


def test_usd_bucket_string_values() -> None:
    assert USDBucket.MICRO.value == "micro"
    assert USDBucket.SMALL.value == "small"
    assert USDBucket.MEDIUM.value == "medium"
    assert USDBucket.LARGE.value == "large"
    assert USDBucket.WHALE.value == "whale"


# ---------------------------------------------------------------------------
# Render output — pinned no-raw-data contract
# ---------------------------------------------------------------------------


def test_render_summary_includes_counts() -> None:
    decisions = tuple(_decision(decision_id=f"dec-{i}", user_id="alice") for i in range(5))
    result = anonymise_dataset(decisions, salt=_SALT)
    text = render_summary(result)
    assert "5 rows out of 5" in text


def test_render_summary_omits_user_ids() -> None:
    """Pin: render never includes raw user_ids."""

    decisions = tuple(
        _decision(decision_id=f"dec-{i}", user_id="alice-secret-id") for i in range(5)
    )
    result = anonymise_dataset(decisions, salt=_SALT)
    text = render_summary(result)
    assert "alice-secret-id" not in text


def test_render_summary_omits_raw_rationale() -> None:
    """Pin: render never includes raw rationale text."""

    decisions = tuple(
        _decision(
            decision_id=f"dec-{i}",
            user_id="alice",
            rationale="SECRET-OPERATOR-TEXT-XYZ",
        )
        for i in range(5)
    )
    result = anonymise_dataset(decisions, salt=_SALT)
    text = render_summary(result)
    assert "SECRET-OPERATOR-TEXT-XYZ" not in text


def test_render_summary_shows_bucket_distribution() -> None:
    decisions = tuple(
        _decision(decision_id=f"dec-{i}", user_id="alice", notional_usd=500.0) for i in range(5)
    )
    result = anonymise_dataset(decisions, salt=_SALT)
    text = render_summary(result)
    assert "USD bucket distribution" in text
    assert "small" in text


def test_render_summary_shows_drops_when_present() -> None:
    decisions = tuple(_decision(decision_id=f"dec-{i}", user_id="alice") for i in range(3))
    result = anonymise_dataset(decisions, salt=_SALT)
    text = render_summary(result)
    assert "dropped: 3" in text
    assert "k-anonymity" in text


# ---------------------------------------------------------------------------
# End-to-end realistic flows
# ---------------------------------------------------------------------------


def test_full_dataset_anonymisation_flow() -> None:
    """100 decisions across 3 users in 2 sectors → realistic dataset.

    Each user has multiple decisions per sector; under k=5 most
    rows survive.
    """

    decisions = []
    for user in ("alice", "bob", "carol"):
        for i in range(10):
            decisions.append(
                _decision(
                    decision_id=f"{user}-crypto-{i}",
                    user_id=user,
                    sector="crypto",
                    regime="risk_on",
                )
            )
    result = anonymise_dataset(tuple(decisions), salt=_SALT)
    # Each (user, crypto, risk_on) tuple has 10 rows >= k=5 → all kept
    assert len(result.decisions) == 30


def test_realistic_pii_scrubbing_pipeline() -> None:
    """A realistic decision with multiple PII vectors → all scrubbed."""

    raw = _decision(
        rationale=(
            "Operator alice@example.com from IP 10.0.0.42 cited regulatory "
            "decision for SSN 999-00-1234 holder. ETH transfer to "
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb6 noted."
        ),
    )
    result = anonymise_decision(raw, salt=_SALT)
    assert "alice@example.com" not in result.rationale_redacted
    assert "10.0.0.42" not in result.rationale_redacted
    assert "999-00-1234" not in result.rationale_redacted
    assert "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb6" not in result.rationale_redacted
    # Multiple redaction markers present
    assert "<redacted-email>" in result.rationale_redacted
    assert "<redacted-ip>" in result.rationale_redacted
    assert "<redacted-ssn>" in result.rationale_redacted
    assert "<redacted-eth-address>" in result.rationale_redacted
