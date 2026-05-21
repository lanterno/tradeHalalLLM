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


def test_capacity_below_cap_hard_caps_proposal_count():
    """Permissive wording ("you may add") was read by the LLM as
    "propose freely" — observed 2026-05-21 13:00 ET, 4/5 positions,
    LLM proposed 2 buys and the 2nd hit the slot cap. Tighten to
    "PROPOSE AT MOST N" so the count is unambiguous."""
    out = _format_capacity(3, 5)
    assert "3/5" in out
    assert "2 slots free" in out
    assert "PROPOSE AT MOST 2" in out
    # No "AT POSITION CAP" alarm when not at cap.
    assert "AT POSITION CAP" not in out


def test_capacity_one_slot_free_uses_singular_grammar():
    out = _format_capacity(4, 5)
    assert "4/5" in out
    assert "1 slot free" in out  # singular
    assert "PROPOSE AT MOST 1 new BUY" in out  # singular BUY
    assert "AT POSITION CAP" not in out


def test_capacity_at_zero_open():
    out = _format_capacity(0, 5)
    assert "0/5" in out
    assert "5 slots free" in out
    assert "PROPOSE AT MOST 5" in out


def test_capacity_over_cap_defensive():
    """Should not crash if open_count somehow exceeds the cap."""
    out = _format_capacity(7, 5)
    assert "AT POSITION CAP" in out
    assert "7/5" in out


# ── Sector exposure ──────────────────────────────────────────


def _pos(symbol: str, qty: float, price: float):
    """Build a Position with current_price set so values are computable."""
    from halal_trader.domain.models import Position
    return Position(symbol=symbol, qty=qty, avg_entry_price=price, current_price=price)


def test_sector_exposure_empty_when_no_positions():
    from halal_trader.trading.strategy import _format_sector_exposure
    out = _format_sector_exposure([], equity=100_000, max_sector_pct=0.40)
    assert "all-cash" in out
    assert "40%" in out


def test_sector_exposure_below_cap_no_warning():
    from halal_trader.trading.strategy import _format_sector_exposure
    # MSFT 10% of equity — well below 40% cap
    out = _format_sector_exposure(
        [_pos("MSFT", 50, 200)], equity=100_000, max_sector_pct=0.40
    )
    assert "Technology" in out
    assert "10%" in out
    # No alarm wording when comfortably below the cap.
    assert "AT CAP" not in out
    assert "near cap" not in out
    assert "WILL BE REJECTED" not in out


def test_sector_exposure_at_cap_emits_warning():
    from halal_trader.trading.strategy import _format_sector_exposure
    # MSFT $40k of $100k equity = exactly 40% — at cap
    out = _format_sector_exposure(
        [_pos("MSFT", 200, 200)], equity=100_000, max_sector_pct=0.40
    )
    assert "AT CAP" in out
    assert "WILL BE REJECTED" in out
    assert "Technology" in out


def test_sector_exposure_near_cap_emits_warning():
    from halal_trader.trading.strategy import _format_sector_exposure
    # MSFT $33k of $100k = 33% — within 80% of 40% cap (>=32%)
    out = _format_sector_exposure(
        [_pos("MSFT", 165, 200)], equity=100_000, max_sector_pct=0.40
    )
    assert "near cap" in out
    assert "WILL BE REJECTED" in out


def test_sector_exposure_aggregates_multiple_symbols_in_same_sector():
    from halal_trader.trading.strategy import _format_sector_exposure
    # MSFT 20% + NVDA 25% = 45% Tech — over cap
    out = _format_sector_exposure(
        [_pos("MSFT", 100, 200), _pos("NVDA", 250, 100)],
        equity=100_000,
        max_sector_pct=0.40,
    )
    assert "AT CAP" in out
    assert "Technology" in out
    # 45% Tech total
    assert "45%" in out


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
