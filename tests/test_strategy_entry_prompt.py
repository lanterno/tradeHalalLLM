"""The day-trader's prompt does not push it to trade.

Pinned after the pre-registered test of its old setup (quant_trials 31,
research/intraday_entry.py): buying intraday momentum trailed SPY to the
close on 2016-2024, and any entry without a reason loses its round trip.
"""

from __future__ import annotations

from halal_trader.trading.strategy import SYSTEM_PROMPT

PROMPT = SYSTEM_PROMPT.format(max_position_pct=0.2, daily_loss_limit=0.02, max_positions=5)


def test_the_prompt_sets_no_entry_quota() -> None:
    for pressure in (
        "1-4 entries",
        "1–4 setups",
        "zero-trade day is not",
        "sitting on cash all day is a failure",
        "Aim for 2–4 open positions",
    ):
        assert pressure not in PROMPT


def test_the_prompt_does_not_name_momentum_as_a_setup() -> None:
    assert "intraday momentum (>0.5%" not in PROMPT
    assert "Being up today is not a reason to buy" in PROMPT


def test_an_empty_plan_is_a_valid_answer() -> None:
    assert "An empty decisions list is the right answer" in PROMPT
    assert "Cash is a position" in PROMPT


def test_the_entry_setup_is_a_catalyst_with_confirming_price() -> None:
    assert "fresh catalyst with a confirming price move" in PROMPT
