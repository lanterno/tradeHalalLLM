"""Tests for the halal compliance attestation generator."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.halal.compliance_attestation import (
    AttestationOutcome,
    AttestationPeriod,
    AttestationStatement,
    ComplianceAttestation,
    DeploymentEvidence,
    generate_attestation,
    render_attestation,
)

_NOW = datetime(2026, 5, 1, tzinfo=UTC)
_START = _NOW - timedelta(days=180)
_END = _NOW


def _period(*, start_at: datetime = _START, end_at: datetime = _END) -> AttestationPeriod:
    return AttestationPeriod(start_at=start_at, end_at=end_at)


def _evidence(
    *,
    signed_receipts_count: int = 1500,
    purification_disbursements_usd: float = 12_500.50,
    purification_outstanding_usd: float = 0.0,
    ssb_ruling_ids: tuple[str, ...] = ("SSB-2026-Q1-001", "SSB-2026-Q1-002"),
    screener_decisions_count: int = 8000,
    exception_queue_resolutions: int = 12,
    halt_events_count: int = 0,
) -> DeploymentEvidence:
    return DeploymentEvidence(
        signed_receipts_count=signed_receipts_count,
        purification_disbursements_usd=purification_disbursements_usd,
        purification_outstanding_usd=purification_outstanding_usd,
        ssb_ruling_ids=ssb_ruling_ids,
        screener_decisions_count=screener_decisions_count,
        exception_queue_resolutions=exception_queue_resolutions,
        halt_events_count=halt_events_count,
    )


# ---------------------------------------------------------------------------
# AttestationPeriod validation
# ---------------------------------------------------------------------------


def test_period_rejects_naive_start() -> None:
    with pytest.raises(ValueError, match="start_at"):
        AttestationPeriod(start_at=datetime(2026, 1, 1), end_at=_END)


def test_period_rejects_naive_end() -> None:
    with pytest.raises(ValueError, match="end_at"):
        AttestationPeriod(start_at=_START, end_at=datetime(2026, 6, 1))


def test_period_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match="must be after"):
        AttestationPeriod(start_at=_END, end_at=_START)


def test_period_rejects_below_30_days() -> None:
    """Pin: < 30 days → quarterly review territory, not annual."""

    short_start = _NOW - timedelta(days=20)
    with pytest.raises(ValueError, match="30d minimum"):
        AttestationPeriod(start_at=short_start, end_at=_NOW)


def test_period_rejects_above_366_days() -> None:
    """Pin: > 366 days → not annual, operator should split."""

    long_start = _NOW - timedelta(days=400)
    with pytest.raises(ValueError, match="366d maximum"):
        AttestationPeriod(start_at=long_start, end_at=_NOW)


def test_period_accepts_30_day_minimum() -> None:
    p = AttestationPeriod(start_at=_NOW - timedelta(days=30), end_at=_NOW)
    assert p.length_days == 30


def test_period_accepts_366_day_maximum() -> None:
    p = AttestationPeriod(start_at=_NOW - timedelta(days=366), end_at=_NOW)
    assert p.length_days == 366


def test_period_length_days() -> None:
    p = _period()
    assert p.length_days == 180


# ---------------------------------------------------------------------------
# DeploymentEvidence validation
# ---------------------------------------------------------------------------


def test_evidence_rejects_negative_signed_receipts() -> None:
    with pytest.raises(ValueError, match="signed_receipts_count"):
        _evidence(signed_receipts_count=-1)


def test_evidence_rejects_negative_purification_disbursements() -> None:
    with pytest.raises(ValueError, match="purification_disbursements_usd"):
        _evidence(purification_disbursements_usd=-1.0)


def test_evidence_rejects_negative_outstanding() -> None:
    with pytest.raises(ValueError, match="purification_outstanding_usd"):
        _evidence(purification_outstanding_usd=-1.0)


def test_evidence_rejects_negative_screener_decisions() -> None:
    with pytest.raises(ValueError, match="screener_decisions_count"):
        _evidence(screener_decisions_count=-1)


def test_evidence_rejects_negative_resolutions() -> None:
    with pytest.raises(ValueError, match="exception_queue_resolutions"):
        _evidence(exception_queue_resolutions=-1)


def test_evidence_rejects_negative_halt_events() -> None:
    with pytest.raises(ValueError, match="halt_events_count"):
        _evidence(halt_events_count=-1)


def test_evidence_accepts_zero_outstanding() -> None:
    e = _evidence(purification_outstanding_usd=0.0)
    assert e.purification_outstanding_usd == 0.0


# ---------------------------------------------------------------------------
# AttestationStatement validation
# ---------------------------------------------------------------------------


def test_statement_rejects_empty_section() -> None:
    with pytest.raises(ValueError, match="section"):
        AttestationStatement(section="", text="x")


def test_statement_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="text"):
        AttestationStatement(section="x", text="")


# ---------------------------------------------------------------------------
# ComplianceAttestation validation
# ---------------------------------------------------------------------------


def test_attestation_rejects_empty_firm_name() -> None:
    with pytest.raises(ValueError, match="firm_name"):
        ComplianceAttestation(
            firm_name="",
            period=_period(),
            evidence=_evidence(),
            outcome=AttestationOutcome.SUFFICIENT,
            statements=(),
        )


# ---------------------------------------------------------------------------
# generate_attestation — outcomes
# ---------------------------------------------------------------------------


def test_clean_evidence_is_sufficient() -> None:
    """All required evidence + zero outstanding → SUFFICIENT."""

    attestation = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=_period(),
        evidence=_evidence(),
    )
    assert attestation.outcome is AttestationOutcome.SUFFICIENT
    assert attestation.failures == ()
    assert attestation.warnings == ()


def test_no_ssb_rulings_is_insufficient() -> None:
    """Pin: zero SSB rulings → INSUFFICIENT (cannot attest to governance)."""

    attestation = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=_period(),
        evidence=_evidence(ssb_ruling_ids=()),
    )
    assert attestation.outcome is AttestationOutcome.INSUFFICIENT
    assert any("SSB rulings" in f for f in attestation.failures)


def test_zero_signed_receipts_is_insufficient() -> None:
    """Pin: zero signed receipts → platform isn't operating, can't attest."""

    attestation = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=_period(),
        evidence=_evidence(signed_receipts_count=0),
    )
    assert attestation.outcome is AttestationOutcome.INSUFFICIENT
    assert any("signed-receipt" in f for f in attestation.failures)


def test_outstanding_purification_is_provisional() -> None:
    """Pin: outstanding balance → PROVISIONAL with disbursement warning."""

    attestation = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=_period(),
        evidence=_evidence(purification_outstanding_usd=2_500.0),
    )
    assert attestation.outcome is AttestationOutcome.PROVISIONAL
    assert any("outstanding" in w for w in attestation.warnings)


def test_zero_resolutions_with_screening_warns() -> None:
    """Pin: active screening but no exception resolutions → warning."""

    attestation = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=_period(),
        evidence=_evidence(exception_queue_resolutions=0),
    )
    assert attestation.outcome is AttestationOutcome.PROVISIONAL
    assert any("exception-queue" in w for w in attestation.warnings)


def test_no_screening_no_resolutions_does_not_warn() -> None:
    """Pin: zero screening + zero resolutions doesn't trigger warning
    (a fresh deployment has nothing to resolve yet).

    But zero screening also means zero signed receipts in a real flow,
    which IS a failure — so this is a synthetic edge case.
    """

    attestation = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=_period(),
        evidence=_evidence(
            screener_decisions_count=0,
            exception_queue_resolutions=0,
        ),
    )
    # Should not have the "no resolutions despite screening" warning
    assert not any("exception-queue" in w for w in attestation.warnings)


def test_halt_events_recorded_in_statement() -> None:
    """Halt events surface in the halt_events statement."""

    attestation = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=_period(),
        evidence=_evidence(halt_events_count=3),
    )
    halt_stmt = next(s for s in attestation.statements if s.section == "halt_events")
    assert "3 times" in halt_stmt.text


def test_zero_halt_events_recorded_in_statement() -> None:
    """Zero halt events also surfaces — the auditor wants to see this explicitly."""

    attestation = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=_period(),
        evidence=_evidence(halt_events_count=0),
    )
    halt_stmt = next(s for s in attestation.statements if s.section == "halt_events")
    assert "No platform-halt events" in halt_stmt.text


def test_generate_rejects_empty_firm_name() -> None:
    with pytest.raises(ValueError, match="firm_name"):
        generate_attestation(
            firm_name="",
            period=_period(),
            evidence=_evidence(),
        )


# ---------------------------------------------------------------------------
# Statements covered
# ---------------------------------------------------------------------------


def test_attestation_includes_six_statement_sections() -> None:
    """Pin: every attestation has the standard six sections."""

    attestation = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=_period(),
        evidence=_evidence(),
    )
    sections = {s.section for s in attestation.statements}
    assert "screening" in sections
    assert "audit_trail" in sections
    assert "purification" in sections
    assert "ssb" in sections
    assert "exception_queue" in sections
    assert "halt_events" in sections


def test_screening_statement_includes_count() -> None:
    attestation = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(screener_decisions_count=12345),
    )
    screening = next(s for s in attestation.statements if s.section == "screening")
    assert "12345" in screening.text


def test_audit_trail_statement_includes_signed_receipts_count() -> None:
    attestation = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(signed_receipts_count=2500),
    )
    audit = next(s for s in attestation.statements if s.section == "audit_trail")
    assert "2500" in audit.text


def test_purification_statement_includes_disbursed_amount() -> None:
    attestation = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(purification_disbursements_usd=500_000.0),
    )
    pur = next(s for s in attestation.statements if s.section == "purification")
    # Should be formatted as $500.00k
    assert "$500.00k" in pur.text


def test_purification_statement_formats_million() -> None:
    attestation = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(purification_disbursements_usd=2_500_000.0),
    )
    pur = next(s for s in attestation.statements if s.section == "purification")
    # Should be formatted as $2.50M
    assert "$2.50M" in pur.text


def test_purification_statement_formats_small_amount() -> None:
    attestation = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(purification_disbursements_usd=42.50),
    )
    pur = next(s for s in attestation.statements if s.section == "purification")
    assert "$42.50" in pur.text


def test_ssb_statement_lists_single_ruling() -> None:
    attestation = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(ssb_ruling_ids=("SSB-2026-Q1-001",)),
    )
    ssb = next(s for s in attestation.statements if s.section == "ssb")
    assert "1 ruling" in ssb.text
    assert "SSB-2026-Q1-001" in ssb.text


def test_ssb_statement_lists_multiple_rulings() -> None:
    attestation = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(
            ssb_ruling_ids=("SSB-2026-Q1-001", "SSB-2026-Q1-002", "SSB-2026-Q2-001"),
        ),
    )
    ssb = next(s for s in attestation.statements if s.section == "ssb")
    assert "3 rulings" in ssb.text
    assert "SSB-2026-Q1-001" in ssb.text


def test_exception_queue_statement_includes_resolution_count() -> None:
    attestation = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(exception_queue_resolutions=7),
    )
    eq = next(s for s in attestation.statements if s.section == "exception_queue")
    assert "7 items" in eq.text


# ---------------------------------------------------------------------------
# Determinism — same input → same output
# ---------------------------------------------------------------------------


def test_attestation_deterministic_for_same_input() -> None:
    """Pin: same inputs produce byte-identical output."""

    a = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(),
    )
    b = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(),
    )
    assert render_attestation(a) == render_attestation(b)


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_period_is_frozen() -> None:
    p = _period()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.start_at = _NOW  # type: ignore[misc]


def test_evidence_is_frozen() -> None:
    e = _evidence()
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.signed_receipts_count = 0  # type: ignore[misc]


def test_statement_is_frozen() -> None:
    s = AttestationStatement(section="x", text="y")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.section = "z"  # type: ignore[misc]


def test_attestation_is_frozen() -> None:
    a = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.outcome = AttestationOutcome.INSUFFICIENT  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB stability
# ---------------------------------------------------------------------------


def test_outcome_string_values() -> None:
    assert AttestationOutcome.SUFFICIENT.value == "sufficient"
    assert AttestationOutcome.INSUFFICIENT.value == "insufficient"
    assert AttestationOutcome.PROVISIONAL.value == "provisional"


# ---------------------------------------------------------------------------
# Render output — pinned no-user-PII contract
# ---------------------------------------------------------------------------


def test_render_includes_firm_name() -> None:
    """Pin: firm_name preserved (the auditor needs it)."""

    a = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=_period(),
        evidence=_evidence(),
    )
    text = render_attestation(a)
    assert "Halal Trader Inc." in text


def test_render_includes_period_dates() -> None:
    a = generate_attestation(
        firm_name="X",
        period=AttestationPeriod(
            start_at=datetime(2026, 1, 1, tzinfo=UTC),
            end_at=datetime(2026, 6, 30, tzinfo=UTC),
        ),
        evidence=_evidence(),
    )
    text = render_attestation(a)
    assert "2026-01-01" in text
    assert "2026-06-30" in text


def test_render_includes_outcome() -> None:
    a = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(),
    )
    text = render_attestation(a)
    assert "SUFFICIENT" in text


def test_render_outcome_emoji() -> None:
    sufficient = generate_attestation(firm_name="X", period=_period(), evidence=_evidence())
    provisional = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(purification_outstanding_usd=100.0),
    )
    insufficient = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(ssb_ruling_ids=()),
    )
    assert "✅" in render_attestation(sufficient)
    assert "⚠️" in render_attestation(provisional)
    assert "❌" in render_attestation(insufficient)


def test_render_does_not_leak_user_id() -> None:
    """Pin: render preserves firm_name but never user IDs / customer data."""

    a = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=_period(),
        evidence=_evidence(),
    )
    text = render_attestation(a)
    assert "user_id" not in text
    assert "@" not in text  # no email-shaped strings


def test_render_includes_all_statements() -> None:
    a = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(),
    )
    text = render_attestation(a)
    assert "[screening]" in text
    assert "[audit_trail]" in text
    assert "[purification]" in text
    assert "[ssb]" in text
    assert "[exception_queue]" in text
    assert "[halt_events]" in text


def test_render_shows_warnings() -> None:
    a = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(purification_outstanding_usd=500.0),
    )
    text = render_attestation(a)
    assert "Warnings:" in text
    assert "outstanding" in text


def test_render_shows_failures() -> None:
    a = generate_attestation(
        firm_name="X",
        period=_period(),
        evidence=_evidence(ssb_ruling_ids=()),
    )
    text = render_attestation(a)
    assert "Failures:" in text
    assert "SSB" in text


# ---------------------------------------------------------------------------
# End-to-end realistic scenarios
# ---------------------------------------------------------------------------


def test_realistic_annual_attestation() -> None:
    """Operator's full-year attestation: clean evidence → SUFFICIENT."""

    attestation = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=AttestationPeriod(
            start_at=datetime(2025, 5, 1, tzinfo=UTC),
            end_at=datetime(2026, 5, 1, tzinfo=UTC),
        ),
        evidence=DeploymentEvidence(
            signed_receipts_count=5_847,
            purification_disbursements_usd=18_750.50,
            purification_outstanding_usd=0.0,
            ssb_ruling_ids=(
                "SSB-2025-Q3-001",
                "SSB-2025-Q4-001",
                "SSB-2026-Q1-001",
                "SSB-2026-Q1-002",
            ),
            screener_decisions_count=42_100,
            exception_queue_resolutions=78,
            halt_events_count=2,
        ),
    )
    assert attestation.outcome is AttestationOutcome.SUFFICIENT
    text = render_attestation(attestation)
    assert "Halal Trader Inc." in text
    assert "365d" in text  # full-year period (365 days)


def test_realistic_failed_attestation_no_ssb_activity() -> None:
    """Operator forgot to convene SSB → INSUFFICIENT."""

    attestation = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=_period(),
        evidence=_evidence(ssb_ruling_ids=()),
    )
    assert attestation.outcome is AttestationOutcome.INSUFFICIENT
    text = render_attestation(attestation)
    assert "INSUFFICIENT" in text
    assert "SSB" in text


def test_realistic_provisional_attestation_outstanding_purification() -> None:
    """Operator has 5k outstanding purification at year end → PROVISIONAL.

    The operator can submit but the auditor will note the unfinished
    disbursements; the next period's attestation should show them resolved.
    """

    attestation = generate_attestation(
        firm_name="Halal Trader Inc.",
        period=_period(),
        evidence=_evidence(
            purification_disbursements_usd=10_000.0,
            purification_outstanding_usd=5_000.0,
        ),
    )
    assert attestation.outcome is AttestationOutcome.PROVISIONAL
    assert any("outstanding" in w for w in attestation.warnings)
