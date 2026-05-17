"""Tests for the pure helpers in :mod:`core.llm.adversarial`.

`test_adversarial.py` covers `critique_plan` end-to-end against the
LLM. This file pins the small surface underneath:

* `AdversarialReview.sizing_multiplier` — how the verdict translates
  into a buy-quantity scale (used by both crypto and stocks).
* `_action_str` — enum/string normalisation (decisions can carry
  either form depending on whether the plan came from the LLM-validated
  pydantic model or a raw dict).
* `_summarize_plan` — the prompt-block builder fed to the attacker;
  includes char-bounded outlook + truncated reasoning so the attacker
  call stays cheap.
* `_classify` — threshold-driven severity → recommendation mapping
  (proceed / downsize / skip).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from halal_trader.core.llm.adversarial import (
    AdversarialReview,
    _action_str,
    _classify,
    _summarize_plan,
)

# ── AdversarialReview.sizing_multiplier ────────────────────


def test_sizing_multiplier_skip_returns_zero():
    """`skip` recommendation → multiplier 0.0 → buy quantities zeroed
    out. Critical for the `apply_review_to_buys` path."""
    r = AdversarialReview(severity=0.9, counter_thesis="x", recommendation="skip")
    assert r.sizing_multiplier == 0.0


def test_sizing_multiplier_downsize_returns_half():
    """`downsize` → 0.5 → buy quantities halved (the attacker is
    confident enough to flag but not skip — caller still trades but
    smaller)."""
    r = AdversarialReview(severity=0.5, counter_thesis="x", recommendation="downsize")
    assert r.sizing_multiplier == 0.5


def test_sizing_multiplier_proceed_returns_one():
    """`proceed` → 1.0 → no scaling; buys are unmodified."""
    r = AdversarialReview(severity=0.1, counter_thesis="x", recommendation="proceed")
    assert r.sizing_multiplier == 1.0


def test_sizing_multiplier_unknown_recommendation_treated_as_proceed():
    """Defensive: an unrecognised string defaults to 1.0 (no scaling).
    Pin so a typo on the LLM-emitted recommendation falls through
    safely rather than zeroing the plan accidentally."""
    r = AdversarialReview(severity=0.5, counter_thesis="x", recommendation="weird")
    assert r.sizing_multiplier == 1.0


# ── _action_str ────────────────────────────────────────────


def test_action_str_with_enum_value():
    """A pydantic-validated TradeDecision carries an Enum action — the
    helper unwraps `.value` to a string."""

    class _Action(Enum):
        BUY = "buy"

    @dataclass
    class _D:
        action: _Action

    assert _action_str(_D(action=_Action.BUY)) == "buy"


def test_action_str_with_plain_string():
    """A raw dict-shaped decision carries a string — passes through."""

    @dataclass
    class _D:
        action: str

    assert _action_str(_D(action="sell")) == "sell"


def test_action_str_missing_attr_returns_empty_string():
    """Defensive: a decision without an `action` attr → empty string,
    not an AttributeError."""

    @dataclass
    class _Decision:
        symbol: str = "X"

    assert _action_str(_Decision()) == ""


# ── _summarize_plan ────────────────────────────────────────


@dataclass
class _Decision:
    """Minimal shape mirroring TradingDecision / TradeDecision."""

    action: str = "buy"
    symbol: str = "BTCUSDT"
    quantity: float = 0.1
    confidence: float = 0.7
    reasoning: str = ""


def test_summarize_plan_renders_each_decision_uppercase_action():
    """Each decision becomes one line; action is upper-cased so the
    attacker sees BUY / SELL / HOLD distinctly."""
    out = _summarize_plan([_Decision(action="buy", symbol="BTC")], market_outlook="")
    assert "BUY BTC" in out


def test_summarize_plan_includes_quantity_confidence_and_reasoning():
    out = _summarize_plan(
        [_Decision(quantity=0.5, confidence=0.9, reasoning="trend follow")],
        market_outlook="",
    )
    assert "qty=0.5" in out
    assert "conf=0.9" in out
    assert "trend follow" in out


def test_summarize_plan_truncates_reasoning_at_80_chars():
    """Long reasoning is capped at 80 chars to keep the attacker
    prompt cheap — the attacker only needs the gist."""
    long_reason = "x" * 200
    out = _summarize_plan([_Decision(reasoning=long_reason)], market_outlook="")
    # Find the line with the decision and assert the reasoning suffix
    # is bounded to 80 chars max.
    lines = out.splitlines()
    decision_line = lines[0]
    # The reasoning is at the tail after `:: `.
    suffix = decision_line.split(":: ", 1)[1] if ":: " in decision_line else ""
    assert len(suffix) <= 80


def test_summarize_plan_truncates_outlook_at_160_chars():
    """Outlook is also capped — same cost-reduction rationale."""
    long_outlook = "y" * 500
    out = _summarize_plan([_Decision()], market_outlook=long_outlook)
    # Find the outlook: line.
    outlook_line = next((line for line in out.splitlines() if line.startswith("outlook:")), "")
    # `outlook: ` prefix is 9 chars; then ≤ 160 of the value.
    payload = outlook_line[len("outlook: ") :]
    assert len(payload) <= 160


def test_summarize_plan_empty_returns_sentinel():
    """No decisions + no outlook → the `(empty plan)` sentinel.
    Otherwise the attacker would see an empty string and waste a turn
    asking what to attack."""
    assert _summarize_plan([], market_outlook="") == "(empty plan)"


def test_summarize_plan_outlook_only_does_not_yield_sentinel():
    """If there's an outlook but no decisions, render the outlook —
    don't fall through to '(empty plan)'."""
    out = _summarize_plan([], market_outlook="bullish")
    assert "(empty plan)" not in out
    assert "outlook: bullish" in out


def test_summarize_plan_handles_missing_attrs_with_question_marks():
    """Defensive: a decision missing some attrs gets `?` placeholders
    rather than raising — the attacker still sees a partial summary."""

    @dataclass
    class _Bare:
        action: str = "buy"
        # No symbol, quantity, confidence, reasoning.

    out = _summarize_plan([_Bare()], market_outlook="")
    assert "BUY" in out
    assert "?" in out  # at least one placeholder for missing fields


def test_summarize_plan_none_reasoning_treated_as_empty():
    """`reasoning=None` (some pydantic-Optional fields default this) →
    empty truncation, not a "None" string."""

    @dataclass
    class _D:
        action: str = "buy"
        symbol: str = "X"
        quantity: float = 1
        confidence: float = 0.5
        reasoning: None = None

    out = _summarize_plan([_D()], market_outlook="")
    assert "None" not in out


# ── _classify ──────────────────────────────────────────────


def test_classify_below_downsize_returns_proceed():
    """Severity below the downsize threshold → no action."""
    assert _classify(0.30, downsize_at=0.45, skip_at=0.75) == "proceed"


def test_classify_at_downsize_threshold_returns_downsize():
    """Inclusive at the threshold (>=)."""
    assert _classify(0.45, downsize_at=0.45, skip_at=0.75) == "downsize"


def test_classify_above_downsize_below_skip_returns_downsize():
    assert _classify(0.60, downsize_at=0.45, skip_at=0.75) == "downsize"


def test_classify_at_skip_threshold_returns_skip():
    """Skip threshold is also inclusive — exactly at `skip_at` → skip."""
    assert _classify(0.75, downsize_at=0.45, skip_at=0.75) == "skip"


def test_classify_high_severity_returns_skip():
    assert _classify(0.95, downsize_at=0.45, skip_at=0.75) == "skip"


def test_classify_zero_severity_returns_proceed():
    assert _classify(0.0, downsize_at=0.45, skip_at=0.75) == "proceed"


def test_classify_one_severity_returns_skip():
    assert _classify(1.0, downsize_at=0.45, skip_at=0.75) == "skip"


def test_classify_custom_thresholds_respected():
    """Operator-tunable thresholds — pin that the helper doesn't
    hard-code 0.45/0.75 anywhere."""
    # Tighter: even mild concern downsizes.
    assert _classify(0.25, downsize_at=0.20, skip_at=0.50) == "downsize"
    # Looser: only severe concerns skip.
    assert _classify(0.85, downsize_at=0.50, skip_at=0.90) == "downsize"
