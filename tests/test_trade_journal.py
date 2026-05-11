"""Tests for education/trade_journal.py — Round-5 Wave 20.D."""

from __future__ import annotations

from datetime import datetime

import pytest

from halal_trader.education.trade_journal import (
    CoachFlagKind,
    CoachPolicy,
    EmotionalState,
    EntryStatus,
    JournalEntry,
    Severity,
    Side,
    coach,
    finalise,
    render_entry,
    render_report,
    supersede,
)


def _entry(
    entry_id: str = "J1",
    trade_id: str = "T1",
    author_id: str = "alice",
    ticker: str = "AAPL",
    side: Side = Side.LONG,
    entry_price: float = 200.0,
    quantity: float = 5.0,
    account_equity: float = 100_000.0,
    rationale: str = (
        "Strong earnings beat with raised guidance; halal-compliant; support at 195 holding."
    ),
    thesis_tags: tuple[str, ...] = ("earnings_beat",),
    stop_price: float | None = 195.0,
    target_price: float | None = 220.0,
    emotional_state: EmotionalState = EmotionalState.CALM,
    rsi: float | None = 55.0,
) -> JournalEntry:
    return JournalEntry(
        entry_id=entry_id,
        trade_id=trade_id,
        author_id=author_id,
        ticker=ticker,
        side=side,
        entry_price=entry_price,
        quantity=quantity,
        account_equity_at_entry=account_equity,
        rationale=rationale,
        thesis_tags=thesis_tags,
        stop_price=stop_price,
        target_price=target_price,
        emotional_state=emotional_state,
        rsi_at_entry=rsi,
    )


# --- JournalEntry validation -------------------------------------------


def test_entry_valid():
    e = _entry()
    assert e.position_notional() == pytest.approx(1000.0)
    assert e.position_pct() == pytest.approx(0.01)


def test_entry_empty_id_rejected():
    with pytest.raises(ValueError):
        _entry(entry_id="")


def test_entry_zero_entry_price_rejected():
    with pytest.raises(ValueError):
        _entry(entry_price=0)


def test_entry_long_stop_geometry_rejected():
    with pytest.raises(ValueError):
        _entry(entry_price=200.0, stop_price=210.0)


def test_entry_long_target_geometry_rejected():
    with pytest.raises(ValueError):
        _entry(entry_price=200.0, target_price=190.0)


def test_entry_short_stop_geometry_rejected():
    with pytest.raises(ValueError):
        _entry(
            side=Side.SHORT,
            entry_price=200.0,
            stop_price=195.0,
            target_price=180.0,
        )


def test_entry_short_target_geometry_rejected():
    with pytest.raises(ValueError):
        _entry(
            side=Side.SHORT,
            entry_price=200.0,
            stop_price=210.0,
            target_price=215.0,
        )


def test_entry_long_rationale_rejected():
    with pytest.raises(ValueError):
        _entry(rationale="x" * 2500)


def test_entry_invalid_rsi_rejected():
    with pytest.raises(ValueError):
        _entry(rsi=120.0)


def test_entry_immutable():
    e = _entry()
    with pytest.raises(AttributeError):
        e.entry_price = 0  # type: ignore[misc]


# --- reward_to_risk ----------------------------------------------------


def test_rr_long():
    e = _entry(entry_price=200.0, stop_price=195.0, target_price=220.0)
    # reward = 20, risk = 5 → RR = 4.
    assert e.reward_to_risk() == pytest.approx(4.0)


def test_rr_short():
    e = _entry(
        side=Side.SHORT,
        entry_price=200.0,
        stop_price=210.0,
        target_price=180.0,
    )
    # reward = 20, risk = 10 → RR = 2.
    assert e.reward_to_risk() == pytest.approx(2.0)


def test_rr_missing_when_no_stop_or_target():
    e = _entry(stop_price=None, target_price=220.0)
    assert e.reward_to_risk() is None
    e2 = _entry(stop_price=195.0, target_price=None)
    assert e2.reward_to_risk() is None


# --- CoachPolicy validation -------------------------------------------


def test_policy_default_valid():
    p = CoachPolicy()
    assert p.max_position_pct == 0.05


def test_policy_invalid_max_position_rejected():
    with pytest.raises(ValueError):
        CoachPolicy(max_position_pct=0.0)
    with pytest.raises(ValueError):
        CoachPolicy(max_position_pct=1.5)


def test_policy_invalid_rsi_band_rejected():
    with pytest.raises(ValueError):
        CoachPolicy(extreme_rsi_low=70, extreme_rsi_high=30)


# --- coach — clean path ----------------------------------------------


def test_coach_clean_entry():
    e = _entry()
    report = coach(e)
    assert not report.flags
    assert report.process_score == 1.0


# --- coach — NO_STOP --------------------------------------------------


def test_coach_no_stop_blocks():
    e = _entry(stop_price=None)
    report = coach(e)
    kinds = {f.kind for f in report.flags}
    assert CoachFlagKind.NO_STOP in kinds
    blocks = report.by_severity(Severity.BLOCK)
    assert any(f.kind is CoachFlagKind.NO_STOP for f in blocks)


# --- coach — NO_TARGET ----------------------------------------------


def test_coach_no_target_warns():
    e = _entry(target_price=None)
    report = coach(e)
    kinds = {f.kind for f in report.flags}
    assert CoachFlagKind.NO_TARGET in kinds


# --- coach — VAGUE_RATIONALE ----------------------------------------


def test_coach_short_rationale_flagged():
    e = _entry(rationale="lgtm")
    report = coach(e)
    assert any(f.kind is CoachFlagKind.VAGUE_RATIONALE for f in report.flags)


def test_coach_yolo_without_concrete_flagged():
    e = _entry(rationale="YOLO gut feel, going long on a hunch.")
    report = coach(e)
    assert any(f.kind is CoachFlagKind.VAGUE_RATIONALE for f in report.flags)


def test_coach_vague_word_with_concrete_passes():
    e = _entry(
        rationale=(
            "Feel strongly about the earnings breakout above resistance with RSI confirmation."
        )
    )
    report = coach(e)
    assert not any(f.kind is CoachFlagKind.VAGUE_RATIONALE for f in report.flags)


# --- coach — SIZING_BREACH -----------------------------------------


def test_coach_sizing_breach_blocks():
    # 10% position on $100k equity → above 5% cap.
    e = _entry(entry_price=200.0, quantity=50.0, account_equity=100_000.0)
    report = coach(e)
    assert any(
        f.kind is CoachFlagKind.SIZING_BREACH and f.severity is Severity.BLOCK for f in report.flags
    )


def test_coach_sizing_at_cap_passes():
    e = _entry(entry_price=200.0, quantity=25.0, account_equity=100_000.0)  # exactly 5%
    report = coach(e)
    assert not any(f.kind is CoachFlagKind.SIZING_BREACH for f in report.flags)


# --- coach — NO_THESIS_TAG -----------------------------------------


def test_coach_no_thesis_tags_note():
    e = _entry(thesis_tags=())
    report = coach(e)
    assert any(
        f.kind is CoachFlagKind.NO_THESIS_TAG and f.severity is Severity.NOTE for f in report.flags
    )


# --- coach — EMOTIONAL_RISK ----------------------------------------


def test_coach_revenge_state_blocks():
    e = _entry(emotional_state=EmotionalState.REVENGE)
    report = coach(e)
    assert any(
        f.kind is CoachFlagKind.EMOTIONAL_RISK and f.severity is Severity.BLOCK
        for f in report.flags
    )


def test_coach_fomo_state_blocks():
    e = _entry(emotional_state=EmotionalState.FOMO)
    report = coach(e)
    assert any(
        f.kind is CoachFlagKind.EMOTIONAL_RISK and f.severity is Severity.BLOCK
        for f in report.flags
    )


def test_coach_excited_warns_not_blocks():
    e = _entry(emotional_state=EmotionalState.EXCITED)
    report = coach(e)
    em_flags = [f for f in report.flags if f.kind is CoachFlagKind.EMOTIONAL_RISK]
    assert em_flags
    assert all(f.severity is Severity.WARN for f in em_flags)


def test_coach_confident_state_no_flag():
    e = _entry(emotional_state=EmotionalState.CONFIDENT)
    report = coach(e)
    assert not any(f.kind is CoachFlagKind.EMOTIONAL_RISK for f in report.flags)


# --- coach — R_R_TOO_LOW ------------------------------------------


def test_coach_low_rr_warns():
    # Long: stop=199, target=201; RR = 1.
    e = _entry(entry_price=200.0, stop_price=199.0, target_price=201.0)
    report = coach(e)
    assert any(f.kind is CoachFlagKind.R_R_TOO_LOW for f in report.flags)


def test_coach_rr_at_threshold_passes():
    # RR = 1.5 exactly.
    e = _entry(entry_price=200.0, stop_price=190.0, target_price=215.0)
    report = coach(e)
    assert not any(f.kind is CoachFlagKind.R_R_TOO_LOW for f in report.flags)


def test_coach_missing_rr_no_flag():
    """No stop/target → no R_R_TOO_LOW flag."""
    e = _entry(stop_price=None, target_price=None)
    report = coach(e)
    assert not any(f.kind is CoachFlagKind.R_R_TOO_LOW for f in report.flags)


# --- coach — ENTRY_AT_EXTREME --------------------------------------


def test_coach_long_overbought_warns():
    e = _entry(side=Side.LONG, rsi=85.0)
    report = coach(e)
    assert any(f.kind is CoachFlagKind.ENTRY_AT_EXTREME for f in report.flags)


def test_coach_short_oversold_warns():
    e = _entry(
        side=Side.SHORT,
        entry_price=200.0,
        stop_price=210.0,
        target_price=180.0,
        rsi=15.0,
    )
    report = coach(e)
    assert any(f.kind is CoachFlagKind.ENTRY_AT_EXTREME for f in report.flags)


def test_coach_no_rsi_no_extreme_flag():
    e = _entry(rsi=None)
    report = coach(e)
    assert not any(f.kind is CoachFlagKind.ENTRY_AT_EXTREME for f in report.flags)


# --- process_score ------------------------------------------------


def test_score_one_warn_drops_to_0_90():
    e = _entry(target_price=None)
    report = coach(e)
    # Just the WARN flag → 1.0 - 0.10 = 0.90.
    assert report.process_score == pytest.approx(0.90)


def test_score_one_block_drops_to_0_80():
    e = _entry(stop_price=None)
    report = coach(e)
    # NO_STOP (BLOCK) + NO_TARGET? no target is still set. So just BLOCK.
    # 1.0 - 0.20 = 0.80.
    n_warn = sum(1 for f in report.flags if f.severity is Severity.WARN)
    n_block = sum(1 for f in report.flags if f.severity is Severity.BLOCK)
    expected = max(0.0, 1.0 - 0.10 * n_warn - 0.20 * n_block)
    assert report.process_score == pytest.approx(expected)


def test_score_floored_at_zero():
    e = _entry(
        rationale="lgtm",
        stop_price=None,
        target_price=None,
        thesis_tags=(),
        emotional_state=EmotionalState.REVENGE,
        entry_price=200.0,
        quantity=50.0,  # also breaches sizing
    )
    report = coach(e)
    assert report.process_score >= 0.0


def test_report_helpers():
    e = _entry(stop_price=None, emotional_state=EmotionalState.FOMO)
    report = coach(e)
    assert report.has_block()
    blocks = report.by_severity(Severity.BLOCK)
    assert len(blocks) >= 2


# --- finalise + supersede -----------------------------------------


def test_finalise_promotes_draft():
    e = _entry()
    e2 = finalise(e, at=datetime(2026, 5, 11))
    assert e2.status is EntryStatus.FINALISED
    assert e2.created_at == datetime(2026, 5, 11)


def test_finalise_idempotent_on_finalised():
    e = finalise(_entry(), at=datetime(2026, 5, 11))
    e2 = finalise(e, at=datetime(2026, 5, 12))
    assert e2.status is EntryStatus.FINALISED


def test_finalise_superseded_rejected():
    e = supersede(_entry())
    with pytest.raises(ValueError):
        finalise(e, at=datetime(2026, 5, 11))


def test_supersede_double_rejected():
    e = supersede(_entry())
    with pytest.raises(ValueError):
        supersede(e)


# --- Render -------------------------------------------------------


def test_render_report_clean():
    e = _entry()
    report = coach(e)
    out = render_report(report)
    assert "✅" in out
    assert "clean" in out


def test_render_report_with_flags():
    e = _entry(stop_price=None, emotional_state=EmotionalState.FOMO)
    report = coach(e)
    out = render_report(report)
    assert "🧭" in out
    assert "🛑" in out
    assert "no_stop" in out


def test_render_entry_no_secret_leak():
    e = _entry(author_id="alice@example.com")
    out = render_entry(e)
    assert "alice@example.com" not in out


def test_render_entry_rr_present():
    e = _entry()
    out = render_entry(e)
    assert "R:R" in out


def test_render_entry_rr_missing():
    e = _entry(stop_price=None, target_price=None)
    out = render_entry(e)
    assert "R:R=—" in out
