"""Tests for `TradeSelfReview._parse_review`, `_apply_adjustments`,
and `_format_trades_for_review`.

`test_self_improve_helpers.py` covers the in-memory state primitives
(format_adjustments, record_failure, etc.). This file pins the LLM-
response → ReviewResult parsing pipeline:

* parameter clamping to `_SAFE_BOUNDS` (a hallucinated value can
  never push the bot past the safety rails — pin the clamp);
* unknown parameter name skip (`max_drawdown_xyz` is silently
  ignored, not crashed on);
* no-op epsilon (already-set values within tolerance produce no
  StrategyAdjustment row — keeps the audit table clean);
* `_apply_adjustments` mutates `_active_adjustments` per row;
* `_format_trades_for_review` rendering: WIN/LOSS labels, m vs h
  duration formatting, exit_reason fallback.
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


# ── _parse_review ──────────────────────────────────────────


def test_parse_review_extracts_observations_and_pairs_to_avoid():
    """Top-level fields flow through verbatim."""
    raw = {
        "observations": ["BTC overtraded", "ETH underperformed"],
        "pairs_to_avoid": ["DOGEUSDT", "SHIBUSDT"],
        "strategy_notes": "tighten stops",
        "parameter_adjustments": {},
    }
    result = _review()._parse_review(raw)
    assert result.observations == ["BTC overtraded", "ETH underperformed"]
    assert result.pairs_to_avoid == ["DOGEUSDT", "SHIBUSDT"]
    assert result.strategy_notes == "tighten stops"


def test_parse_review_missing_top_level_keys_are_optional():
    """Defensive: an LLM response that drops a section → empty list /
    string fallback rather than KeyError."""
    result = _review()._parse_review({})
    assert result.observations == []
    assert result.pairs_to_avoid == []
    assert result.strategy_notes == ""
    assert result.adjustments == []


def test_parse_review_ignores_unknown_parameter_names():
    """A param name not in `_SAFE_BOUNDS` (typo, future param) is
    silently skipped — never lands as an adjustment."""
    raw = {
        "parameter_adjustments": {
            "rsi_buy_threshold": 35.0,  # known
            "max_drawdown_xyz": 0.99,  # unknown — must skip
            "frontier_param": 1.0,
        }
    }
    result = _review()._parse_review(raw)
    assert len(result.adjustments) == 1
    assert result.adjustments[0].parameter == "rsi_buy_threshold"


def test_parse_review_skips_none_values():
    """LLM may emit `null` for a param it doesn't want to change —
    treated as no-op rather than coerced to 0."""
    raw = {
        "parameter_adjustments": {
            "rsi_buy_threshold": None,
            "stop_loss_pct": 0.01,
        }
    }
    result = _review()._parse_review(raw)
    assert [a.parameter for a in result.adjustments] == ["stop_loss_pct"]


def test_parse_review_clamps_above_upper_bound():
    """A hallucinated overshoot (`max_position_pct=0.95`) is clamped
    to the safe upper bound (0.30) — pin so a runaway LLM can't put
    the bot past the operator's safety rail."""
    raw = {"parameter_adjustments": {"max_position_pct": 0.95}}
    result = _review()._parse_review(raw)
    assert len(result.adjustments) == 1
    adj = result.adjustments[0]
    assert adj.new_value == 0.30  # clamped to _SAFE_BOUNDS upper
    assert "0.95" in adj.reasoning  # original value visible in audit


def test_parse_review_clamps_below_lower_bound():
    """Mirror clamp on the lower side: `stop_loss_pct=0.0001` is
    too tight — clamped to 0.003."""
    raw = {"parameter_adjustments": {"stop_loss_pct": 0.0001}}
    result = _review()._parse_review(raw)
    assert result.adjustments[0].new_value == 0.003


def test_parse_review_within_bounds_passes_through():
    """A reasonable value lands as-is."""
    raw = {"parameter_adjustments": {"rsi_buy_threshold": 35.0}}
    result = _review()._parse_review(raw)
    assert result.adjustments[0].new_value == 35.0


def test_parse_review_skips_no_op_within_epsilon():
    """If the new value is within `_NOOP_EPSILON` of the already-set
    one, no StrategyAdjustment row is produced — keeps the audit
    table from filling with rounding-noise rows on every cycle."""
    r = _review()
    r._active_adjustments["rsi_buy_threshold"] = 35.0

    raw = {"parameter_adjustments": {"rsi_buy_threshold": 35.0000000001}}
    result = r._parse_review(raw)
    assert result.adjustments == []  # within epsilon → no row


def test_parse_review_records_change_above_epsilon():
    """A real change (≥ epsilon) DOES produce a row."""
    r = _review()
    r._active_adjustments["rsi_buy_threshold"] = 35.0
    raw = {"parameter_adjustments": {"rsi_buy_threshold": 36.0}}
    result = r._parse_review(raw)
    assert len(result.adjustments) == 1
    assert result.adjustments[0].old_value == 35.0
    assert result.adjustments[0].new_value == 36.0


def test_parse_review_first_time_value_has_old_value_none():
    """The first adjustment for a parameter has `old_value=None` —
    important so the audit row and `_apply_adjustments` can branch
    on first-time vs update."""
    raw = {"parameter_adjustments": {"take_profit_pct": 0.015}}
    result = _review()._parse_review(raw)
    assert result.adjustments[0].old_value is None


def test_parse_review_string_value_coerced_to_float():
    """Defensive: an LLM returning a string number should still parse —
    the helper does `float(value)`. Pin so a future tightening to an
    isinstance check is intentional."""
    raw = {"parameter_adjustments": {"rsi_buy_threshold": "35"}}
    result = _review()._parse_review(raw)
    assert len(result.adjustments) == 1
    assert result.adjustments[0].new_value == 35.0


def test_parse_review_adjustment_reasoning_includes_clamp_bounds():
    """The reasoning string captures both the suggested value and the
    clamp range — operators see exactly what the LLM said and where
    the rail intervened."""
    raw = {"parameter_adjustments": {"max_position_pct": 0.50}}
    result = _review()._parse_review(raw)
    reasoning = result.adjustments[0].reasoning
    assert "0.5" in reasoning  # original suggested
    assert "0.1" in reasoning  # lower bound
    assert "0.3" in reasoning  # upper bound


# ── _apply_adjustments ─────────────────────────────────────


def test_apply_adjustments_mutates_active_adjustments():
    """Each adjustment writes its `new_value` into the active dict.
    Subsequent reads reflect the change."""
    from halal_trader.crypto.self_improve import ReviewResult, StrategyAdjustment

    r = _review()
    result = ReviewResult()
    result.adjustments = [
        StrategyAdjustment(
            parameter="rsi_buy_threshold",
            old_value=None,
            new_value=35.0,
            reasoning="x",
        ),
        StrategyAdjustment(
            parameter="stop_loss_pct",
            old_value=0.01,
            new_value=0.005,
            reasoning="y",
        ),
    ]
    r._apply_adjustments(result)
    assert r._active_adjustments == {"rsi_buy_threshold": 35.0, "stop_loss_pct": 0.005}


def test_apply_adjustments_overwrites_existing():
    """Re-applying with a different value updates in place."""
    from halal_trader.crypto.self_improve import ReviewResult, StrategyAdjustment

    r = _review()
    r._active_adjustments["rsi_buy_threshold"] = 30.0
    result = ReviewResult()
    result.adjustments = [
        StrategyAdjustment(
            parameter="rsi_buy_threshold", old_value=30.0, new_value=40.0, reasoning="x"
        ),
    ]
    r._apply_adjustments(result)
    assert r._active_adjustments["rsi_buy_threshold"] == 40.0


def test_apply_adjustments_empty_is_noop():
    from halal_trader.crypto.self_improve import ReviewResult

    r = _review()
    r._active_adjustments["x"] = 1.0
    r._apply_adjustments(ReviewResult())
    assert r._active_adjustments == {"x": 1.0}


# ── _format_trades_for_review ──────────────────────────────


def _trade(
    *,
    pair: str = "BTCUSDT",
    pnl: float = 5.0,
    pnl_pct: float = 0.025,
    duration: float = 30.0,
    exit_reason: str = "take_profit",
) -> dict:
    return {
        "pair": pair,
        "buy_price": 100.0,
        "sell_price": 100.0 + pnl,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "duration_minutes": duration,
        "exit_reason": exit_reason,
    }


def test_format_trades_renders_win_label_for_positive_pnl():
    out = _review()._format_trades_for_review([_trade(pnl=5.0)])
    assert "[WIN]" in out


def test_format_trades_renders_loss_label_for_negative_pnl():
    out = _review()._format_trades_for_review([_trade(pnl=-5.0)])
    assert "[LOSS]" in out


def test_format_trades_renders_loss_label_for_zero_pnl():
    """Edge: exactly zero P&L counts as a LOSS in the prompt — pin so
    a refactor doesn't accidentally relabel break-even trades as wins."""
    out = _review()._format_trades_for_review([_trade(pnl=0.0)])
    assert "[LOSS]" in out


def test_format_trades_short_duration_in_minutes():
    """< 60 min → "Nm" format."""
    out = _review()._format_trades_for_review([_trade(duration=45)])
    assert "Duration: 45m" in out


def test_format_trades_long_duration_in_hours():
    """≥ 60 min → "N.Nh" format with one decimal."""
    out = _review()._format_trades_for_review([_trade(duration=90)])
    assert "Duration: 1.5h" in out


def test_format_trades_includes_exit_reason():
    out = _review()._format_trades_for_review([_trade(exit_reason="stop_loss")])
    assert "Reason: stop_loss" in out


def test_format_trades_unknown_exit_reason_default():
    """Defensive: a trade missing `exit_reason` → "unknown" fallback."""
    trade = _trade()
    del trade["exit_reason"]
    out = _review()._format_trades_for_review([trade])
    assert "Reason: unknown" in out


def test_format_trades_includes_signed_pnl_with_pct():
    """Both dollar P&L and pct rendered with sign — operator sees
    direction at a glance."""
    out = _review()._format_trades_for_review([_trade(pnl=5.0, pnl_pct=0.025)])
    assert "+$5.00" in out or "$+5.00" in out
    assert "+2.50%" in out


def test_format_trades_numbered_sequentially():
    """Trades are 1-indexed for human-friendly review (`Trade #1`,
    `#2`, …)."""
    out = _review()._format_trades_for_review([_trade(), _trade(), _trade()])
    assert "Trade #1" in out
    assert "Trade #2" in out
    assert "Trade #3" in out


def test_format_trades_empty_returns_empty_string():
    """No round trips → empty rendered block."""
    assert _review()._format_trades_for_review([]) == ""
