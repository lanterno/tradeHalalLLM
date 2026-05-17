"""Tests for `halal/scholar_review.py`.

Pins the review-packet rendering, the verdict-validation rules,
the apply-verdict status guard, and the audit-trail rendering.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halal_trader.halal.scholar_review import (
    RecordedVerdict,
    ReviewContext,
    ReviewPacket,
    ScholarVerdict,
    VerdictKind,
    apply_verdict,
    render_recorded_verdict,
    render_review_packet,
)

# ── render_review_packet ─────────────────────────────────


def test_packet_includes_instrument_and_entry_id():
    packet = render_review_packet(
        entry_id="AAPL:equity",
        instrument="AAPL",
        kind="equity",
        reasoning="Insufficient screening data — IPO 2026-04-15.",
    )
    assert isinstance(packet, ReviewPacket)
    assert "AAPL" in packet.subject
    assert "AAPL" in packet.markdown
    assert "AAPL:equity" in packet.markdown


def test_packet_subject_uses_canonical_format():
    """Pin: subject `[Halal Review] <symbol> (<kind>) — pending verdict`
    so scholars can filter their inbox."""
    packet = render_review_packet(entry_id="x", instrument="BTCUSDT", kind="crypto", reasoning="x")
    assert packet.subject.startswith("[Halal Review]")
    assert "BTCUSDT" in packet.subject
    assert "crypto" in packet.subject
    assert "pending verdict" in packet.subject


def test_packet_includes_reasoning_section():
    packet = render_review_packet(
        entry_id="x",
        instrument="X",
        kind="equity",
        reasoning="Newly listed; Zoya hasn't classified yet.",
    )
    assert "Why this review is needed" in packet.markdown
    assert "Newly listed" in packet.markdown


def test_packet_handles_empty_reasoning():
    """Pin: missing reasoning surfaces as `_(no reasoning recorded)_`
    rather than crashing."""
    packet = render_review_packet(entry_id="x", instrument="X", kind="equity", reasoning="")
    assert "no reasoning recorded" in packet.markdown


def test_packet_includes_profile_name():
    """Pin: profile name in the brief so the scholar knows
    which threshold set the operator is using."""
    packet = render_review_packet(
        entry_id="x",
        instrument="X",
        kind="equity",
        reasoning="x",
        profile_name="taqi_usmani",
    )
    assert "taqi_usmani" in packet.markdown


def test_packet_omits_context_sections_when_no_context():
    """Pin: empty context produces a minimal packet — sections
    appear only when the operator filled them in."""
    packet = render_review_packet(entry_id="x", instrument="X", kind="equity", reasoning="x")
    assert "Symbol context" not in packet.markdown
    assert "Financial screen inputs" not in packet.markdown
    assert "Notes" not in packet.markdown
    assert "References" not in packet.markdown


def test_packet_renders_sector_when_provided():
    ctx = ReviewContext(sector="Information Technology")
    packet = render_review_packet(
        entry_id="x",
        instrument="X",
        kind="equity",
        reasoning="x",
        context=ctx,
    )
    assert "Information Technology" in packet.markdown
    assert "Symbol context" in packet.markdown


def test_packet_renders_market_cap_in_billions():
    """Pin: large numbers display as `$NB` not raw cents."""
    ctx = ReviewContext(market_cap_usd=2_500_000_000)
    packet = render_review_packet(
        entry_id="x", instrument="X", kind="equity", reasoning="x", context=ctx
    )
    assert "$2.50B" in packet.markdown


def test_packet_renders_market_cap_in_millions():
    ctx = ReviewContext(market_cap_usd=500_000_000)
    packet = render_review_packet(
        entry_id="x", instrument="X", kind="equity", reasoning="x", context=ctx
    )
    assert "$500.00M" in packet.markdown


def test_packet_renders_financial_ratios_as_percent():
    ctx = ReviewContext(
        debt_to_marketcap_pct=0.28,
        non_permissible_income_pct=0.04,
    )
    packet = render_review_packet(
        entry_id="x", instrument="X", kind="equity", reasoning="x", context=ctx
    )
    assert "28.00%" in packet.markdown
    assert "4.00%" in packet.markdown
    assert "Financial screen inputs" in packet.markdown


def test_packet_renders_revenue_breakdown_as_blockquote():
    ctx = ReviewContext(
        recent_revenue_breakdown="92% iPhone, 5% services, 3% other",
    )
    packet = render_review_packet(
        entry_id="x", instrument="X", kind="equity", reasoning="x", context=ctx
    )
    assert "iPhone" in packet.markdown
    assert "> " in packet.markdown


def test_packet_renders_notes_section():
    ctx = ReviewContext(notes="The board approved the spin-off last week.")
    packet = render_review_packet(
        entry_id="x", instrument="X", kind="equity", reasoning="x", context=ctx
    )
    assert "Notes" in packet.markdown
    assert "spin-off" in packet.markdown


def test_packet_renders_references_as_bullet_list():
    ctx = ReviewContext(
        references=(
            "https://sec.gov/abc",
            "Mufti Faraz Adam framework s.3.2",
        )
    )
    packet = render_review_packet(
        entry_id="x", instrument="X", kind="equity", reasoning="x", context=ctx
    )
    assert "References" in packet.markdown
    assert "sec.gov/abc" in packet.markdown
    assert "Mufti Faraz Adam" in packet.markdown


def test_packet_includes_response_instructions():
    """Pin: the brief must end with clear instructions on how
    to respond — APPROVED / REJECTED / DEFERRED + one-line
    rationale."""
    packet = render_review_packet(entry_id="x", instrument="X", kind="equity", reasoning="x")
    assert "How to respond" in packet.markdown
    assert "APPROVED" in packet.markdown
    assert "REJECTED" in packet.markdown
    assert "DEFERRED" in packet.markdown


def test_packet_does_not_leak_operator_pii():
    """Pin: ReviewContext has no operator-identifying fields by
    design. A regression test that constructs ReviewContext with
    everything filled in must NOT contain "account" / "operator
    id" / "balance" in the output."""
    ctx = ReviewContext(
        sector="Tech",
        market_cap_usd=1_000_000_000,
        recent_revenue_breakdown="x",
        debt_to_marketcap_pct=0.2,
        non_permissible_income_pct=0.02,
        notes="x",
        references=("ref1",),
    )
    packet = render_review_packet(
        entry_id="x", instrument="X", kind="equity", reasoning="x", context=ctx
    )
    lower = packet.markdown.lower()
    assert "account" not in lower
    assert "balance" not in lower
    assert "operator id" not in lower


# ── ScholarVerdict validation ────────────────────────────


def test_verdict_rejects_empty_entry_id():
    with pytest.raises(ValueError, match="entry_id"):
        ScholarVerdict(
            entry_id="",
            kind=VerdictKind.APPROVED,
            decided_by="scholar@example.com",
            rationale="ok",
        )


def test_verdict_rejects_empty_decided_by():
    """Pin: audit trail needs the scholar's identifier — refuse
    construction without one."""
    with pytest.raises(ValueError, match="decided_by"):
        ScholarVerdict(
            entry_id="x",
            kind=VerdictKind.APPROVED,
            decided_by="",
            rationale="ok",
        )


def test_approved_verdict_requires_rationale():
    """Pin: approval without justification is exactly the kind
    of audit-trail gap the bot can't tolerate."""
    with pytest.raises(ValueError, match="rationale"):
        ScholarVerdict(
            entry_id="x",
            kind=VerdictKind.APPROVED,
            decided_by="scholar@example.com",
            rationale="",
        )


def test_rejected_verdict_requires_rationale():
    """Symmetric: rejection also needs a rationale so future
    reviewers know why."""
    with pytest.raises(ValueError, match="rationale"):
        ScholarVerdict(
            entry_id="x",
            kind=VerdictKind.REJECTED,
            decided_by="scholar@example.com",
            rationale="",
        )


def test_deferred_verdict_does_not_require_rationale():
    """Pin: deferral is "I need more time", which is itself a
    valid response without an explanation."""
    verdict = ScholarVerdict(
        entry_id="x",
        kind=VerdictKind.DEFERRED,
        decided_by="scholar@example.com",
        rationale="",
    )
    assert verdict.kind == VerdictKind.DEFERRED


def test_withdrawn_verdict_does_not_require_rationale():
    """An operator withdraws when the symbol no longer needs
    review (de-listed, position closed); no scholar reasoning
    needed."""
    verdict = ScholarVerdict(
        entry_id="x",
        kind=VerdictKind.WITHDRAWN,
        decided_by="operator@example.com",
        rationale="",
    )
    assert verdict.kind == VerdictKind.WITHDRAWN


def test_verdict_rationale_whitespace_only_rejected():
    """Pin: a single-space rationale is the same as empty."""
    with pytest.raises(ValueError, match="rationale"):
        ScholarVerdict(
            entry_id="x",
            kind=VerdictKind.APPROVED,
            decided_by="scholar@example.com",
            rationale="   ",
        )


def test_verdict_immutable():
    verdict = ScholarVerdict(
        entry_id="x",
        kind=VerdictKind.DEFERRED,
        decided_by="scholar@example.com",
    )
    with pytest.raises(Exception):
        verdict.kind = VerdictKind.APPROVED  # type: ignore[misc]


# ── apply_verdict ────────────────────────────────────────


def test_apply_verdict_returns_recorded_verdict():
    verdict = ScholarVerdict(
        entry_id="AAPL:equity",
        kind=VerdictKind.APPROVED,
        decided_by="scholar@example.com",
        rationale="Tech sector + 28% debt within AAOIFI threshold.",
    )
    recorded = apply_verdict(
        verdict,
        pending_entry_status="pending",
        pending_entry_instrument="AAPL",
    )
    assert isinstance(recorded, RecordedVerdict)
    assert recorded.entry_id == "AAPL:equity"
    assert recorded.kind == VerdictKind.APPROVED
    assert recorded.previous_status == "pending"


def test_apply_verdict_uses_supplied_decided_at():
    when = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)
    verdict = ScholarVerdict(
        entry_id="x",
        kind=VerdictKind.DEFERRED,
        decided_by="scholar",
        decided_at=when,
    )
    recorded = apply_verdict(verdict, pending_entry_status="pending", pending_entry_instrument="X")
    assert recorded.decided_at == when


def test_apply_verdict_uses_now_when_decided_at_unset():
    verdict = ScholarVerdict(
        entry_id="x",
        kind=VerdictKind.DEFERRED,
        decided_by="scholar",
    )
    fixed_now = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)
    recorded = apply_verdict(
        verdict,
        pending_entry_status="pending",
        pending_entry_instrument="X",
        now=fixed_now,
    )
    assert recorded.decided_at == fixed_now


def test_apply_verdict_rejects_already_decided_entry():
    """Pin: a verdict on a non-pending entry surfaces as a
    clear error rather than silently overwriting an earlier
    decision."""
    verdict = ScholarVerdict(
        entry_id="x",
        kind=VerdictKind.APPROVED,
        decided_by="scholar",
        rationale="ok",
    )
    with pytest.raises(ValueError, match="not 'pending'"):
        apply_verdict(
            verdict,
            pending_entry_status="approved",
            pending_entry_instrument="X",
        )


def test_apply_verdict_rejects_rejected_entry():
    verdict = ScholarVerdict(
        entry_id="x",
        kind=VerdictKind.APPROVED,
        decided_by="scholar",
        rationale="ok",
    )
    with pytest.raises(ValueError, match="not 'pending'"):
        apply_verdict(
            verdict,
            pending_entry_status="rejected",
            pending_entry_instrument="X",
        )


def test_apply_verdict_carries_instrument_through():
    verdict = ScholarVerdict(
        entry_id="x",
        kind=VerdictKind.DEFERRED,
        decided_by="scholar",
    )
    recorded = apply_verdict(
        verdict,
        pending_entry_status="pending",
        pending_entry_instrument="BTCUSDT",
    )
    assert recorded.instrument == "BTCUSDT"


def test_recorded_verdict_immutable():
    verdict = ScholarVerdict(entry_id="x", kind=VerdictKind.DEFERRED, decided_by="s")
    recorded = apply_verdict(verdict, pending_entry_status="pending", pending_entry_instrument="X")
    with pytest.raises(Exception):
        recorded.entry_id = "y"  # type: ignore[misc]


# ── render_recorded_verdict ──────────────────────────────


def test_render_uses_emoji_per_kind():
    base = dict(
        entry_id="x",
        instrument="X",
        decided_by="s",
        rationale="r",
        decided_at=datetime(2026, 5, 1, tzinfo=UTC),
        previous_status="pending",
    )
    approved = RecordedVerdict(**base, kind=VerdictKind.APPROVED)
    rejected = RecordedVerdict(**base, kind=VerdictKind.REJECTED)
    deferred = RecordedVerdict(**base, kind=VerdictKind.DEFERRED)
    withdrawn = RecordedVerdict(**base, kind=VerdictKind.WITHDRAWN)
    assert "✅" in render_recorded_verdict(approved)
    assert "❌" in render_recorded_verdict(rejected)
    assert "⏳" in render_recorded_verdict(deferred)
    assert "↩️" in render_recorded_verdict(withdrawn)


def test_render_includes_instrument_and_decider():
    recorded = RecordedVerdict(
        entry_id="AAPL:equity",
        instrument="AAPL",
        kind=VerdictKind.APPROVED,
        decided_by="scholar@university.edu",
        rationale="ok",
        decided_at=datetime(2026, 5, 1, 14, 30, tzinfo=UTC),
        previous_status="pending",
    )
    text = render_recorded_verdict(recorded)
    assert "AAPL" in text
    assert "scholar@university.edu" in text
    assert "approved" in text


def test_render_includes_rationale_when_set():
    recorded = RecordedVerdict(
        entry_id="x",
        instrument="X",
        kind=VerdictKind.REJECTED,
        decided_by="s",
        rationale="Pork-related revenue exceeds 5%.",
        decided_at=datetime(2026, 5, 1, tzinfo=UTC),
        previous_status="pending",
    )
    text = render_recorded_verdict(recorded)
    assert "Pork-related" in text
    assert "→" in text


def test_render_omits_rationale_line_when_empty():
    recorded = RecordedVerdict(
        entry_id="x",
        instrument="X",
        kind=VerdictKind.DEFERRED,
        decided_by="s",
        rationale="",
        decided_at=datetime(2026, 5, 1, tzinfo=UTC),
        previous_status="pending",
    )
    text = render_recorded_verdict(recorded)
    assert "→" not in text
