"""Pin the position-cap awareness wording in the stocks user prompt.

On 2026-05-21 12:00 ET we observed a fully wasted cycle: bot held
5/5 positions, LLM proposed 2 buys + 0 sells, both rejected by the
executor's position cap. Root cause: the prompt's `Maximum
simultaneous open positions: 5` rule was static and easy to ignore.
The fix injects a dynamic CAPACITY STATUS block that calls out the
cap explicitly when reached.
"""

from __future__ import annotations

from halal_trader.trading.strategy import _format_capacity


def test_capacity_at_cap_warns_explicitly():
    out = _format_capacity(5, 5)
    assert "AT POSITION CAP" in out
    assert "5/5" in out
    # The LLM must be told that adding a buy without a sell will fail.
    assert "REJECTED" in out
    assert "SELL" in out


def test_capacity_below_cap_is_permissive():
    out = _format_capacity(3, 5)
    assert "3/5" in out
    assert "2 slot(s) free" in out
    assert "may add new BUYs" in out
    # No alarm wording when below the cap.
    assert "AT POSITION CAP" not in out


def test_capacity_at_zero_open():
    out = _format_capacity(0, 5)
    assert "0/5" in out
    assert "5 slot(s) free" in out


def test_capacity_over_cap_defensive():
    """Should not crash if open_count somehow exceeds the cap."""
    out = _format_capacity(7, 5)
    assert "AT POSITION CAP" in out
    assert "7/5" in out


def test_system_prompt_has_transaction_cost_rule():
    """Pin the rule that discourages whipsaw round-trips on noise.

    Observed 2026-05-21: QCOM bought 11:30 / sold 11:45 (15 min hold),
    GOOG bought 11:45 / sold 12:30 (45 min hold). Without explicit cost
    awareness, the LLM keeps flipping fresh positions on similar macro
    reasoning. The rule tells it to leave <30-min positions alone unless
    SL is breached, capacity demands the swap, or a hard catalyst hits.
    """
    from halal_trader.trading.strategy import SYSTEM_PROMPT

    assert "TRANSACTION COST AWARENESS" in SYSTEM_PROMPT
    assert "round-trip" in SYSTEM_PROMPT
    assert "<30-min" in SYSTEM_PROMPT or "30-min" in SYSTEM_PROMPT
    # The three valid exit reasons must each be mentioned so the LLM
    # has a concrete checklist instead of "use judgment".
    assert "stop-loss" in SYSTEM_PROMPT
    assert "capacity" in SYSTEM_PROMPT
    assert "catalyst" in SYSTEM_PROMPT
