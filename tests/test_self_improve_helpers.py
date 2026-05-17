"""Tests for the pure helpers in :class:`TradeSelfReview`.

The DB-backed review path (`load_from_db`, `should_trigger_review`,
the actual LLM call) is integration-heavy. The bits that don't need
DB or LLM are the in-memory state mutations: tracking execution
failures, formatting active adjustments for the prompt, and the
``active_adjustments`` / ``pairs_to_avoid`` snapshot properties.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from halal_trader.crypto.self_improve import TradeSelfReview


def _review() -> TradeSelfReview:
    return TradeSelfReview(
        llm=MagicMock(),
        strategy_adjustments=MagicMock(),
        crypto_trades=MagicMock(),
    )


# ── format_adjustments_for_prompt ─────────────────────────────


def test_format_adjustments_empty_returns_empty_string():
    """No adjustments + no avoided pairs → no section in the prompt."""
    r = _review()
    assert r.format_adjustments_for_prompt() == ""


def test_format_adjustments_renders_each_param():
    r = _review()
    r._active_adjustments = {
        "max_position_pct": 0.18,
        "stop_loss_pct": 0.008,
    }
    out = r.format_adjustments_for_prompt()
    assert "max_position_pct: 0.18" in out
    assert "stop_loss_pct: 0.008" in out


def test_format_adjustments_appends_avoid_pairs_line():
    r = _review()
    r._pairs_to_avoid = ["DOGEUSDT", "SHIBUSDT"]
    out = r.format_adjustments_for_prompt()
    assert "Avoid these pairs" in out
    assert "DOGEUSDT" in out
    assert "SHIBUSDT" in out


def test_format_adjustments_combines_both_sections():
    r = _review()
    r._active_adjustments = {"max_position_pct": 0.20}
    r._pairs_to_avoid = ["XRPUSDT"]
    out = r.format_adjustments_for_prompt()
    assert "max_position_pct" in out
    assert "Avoid these pairs" in out


# ── snapshot properties ───────────────────────────────────────


def test_active_adjustments_returns_a_copy():
    """Mutating the returned dict must not affect internal state."""
    r = _review()
    r._active_adjustments = {"max_position_pct": 0.18}
    snapshot = r.active_adjustments
    snapshot["max_position_pct"] = 99.0
    assert r._active_adjustments["max_position_pct"] == 0.18


def test_pairs_to_avoid_returns_a_copy():
    r = _review()
    r._pairs_to_avoid = ["DOGEUSDT"]
    snapshot = r.pairs_to_avoid
    snapshot.append("SHIBUSDT")
    assert r._pairs_to_avoid == ["DOGEUSDT"]


# ── record_execution_failure / _get_failure_summary ───────────


def test_record_failure_appends_to_per_pair_list():
    r = _review()
    r.record_execution_failure("BTCUSDT", "MIN_NOTIONAL")
    r.record_execution_failure("BTCUSDT", "INSUFFICIENT_BALANCE")
    assert r._exec_failures["BTCUSDT"] == ["MIN_NOTIONAL", "INSUFFICIENT_BALANCE"]


def test_record_failure_caps_at_50_per_pair():
    """Failure list trims to the most recent 50 entries — keeps the
    review prompt manageable on a misbehaving pair."""
    r = _review()
    for i in range(60):
        r.record_execution_failure("BTCUSDT", f"err-{i}")
    assert len(r._exec_failures["BTCUSDT"]) == 50
    # Only the trailing window survives.
    assert r._exec_failures["BTCUSDT"][0] == "err-10"
    assert r._exec_failures["BTCUSDT"][-1] == "err-59"


def test_failure_summary_empty_when_no_failures():
    r = _review()
    assert r._get_failure_summary() == ""


def test_failure_summary_renders_per_pair_top_5_errors():
    r = _review()
    for _ in range(3):
        r.record_execution_failure("BTCUSDT", "MIN_NOTIONAL")
    r.record_execution_failure("BTCUSDT", "INSUFFICIENT_BALANCE")
    out = r._get_failure_summary()
    assert "EXECUTION FAILURES" in out
    assert "BTCUSDT" in out
    assert "4 failures" in out
    assert "MIN_NOTIONAL: 3" in out
