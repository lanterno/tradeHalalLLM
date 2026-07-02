"""Pin the RECENTLY CLOSED block in the stocks user prompt.

Observed 2026-05-21 13:00 → 13:30 (cycles -d15e30ce, -272f537c, -30226c12):
the bot bought AMZN, sold it 15 min later, then bought it BACK 15 min
after that. Same ticker, same FOMC-volatility thesis, ~$50 round-trip
slippage. The LLM had no visibility into recent exits — the prompt
only showed CURRENT POSITIONS, so each cycle looked like a fresh
context to it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from halal_trader.trading.strategy import _format_recent_closed


def _row(symbol: str, mins_ago: float, *, qty: float = 50, entry: float = 200, exit: float = 202):
    return {
        "symbol": symbol,
        "side": "buy",
        "quantity": qty,
        "filled_quantity": qty,
        "price": entry,
        "filled_price": entry,
        "exit_price": exit,
        "exit_reason": "rotate",
        "closed_at": datetime.now(UTC) - timedelta(minutes=mins_ago),
    }


def test_empty_rows_returns_default():
    out = _format_recent_closed([])
    assert "No closed trades" in out


def test_single_recent_close_rendered():
    out = _format_recent_closed([_row("AMZN", 15)])
    # Header pins the hard-gate wording (third escalation: "Avoid
    # re-entering" → "DO NOT re-buy" → mechanical AUTO-REJECTED framing,
    # after GLM argued past the soft version on 2026-07-02 and burned
    # consecutive cycles on gated ADBE/INTU buys).
    assert "AUTO-REJECTED" in out
    assert "AMZN" in out
    assert "15 min ago" in out


def test_header_offers_alternatives_not_just_prohibition():
    """The header must direct the LLM toward viable actions (other
    halal symbols / hold) so the block reads as guidance, not noise."""
    out = _format_recent_closed([_row("AMZN", 15)])
    assert "halal symbol" in out
    assert "FRESH" in out


def test_gate_flag_inside_close_cooldown():
    """An exit 15 min ago with a 30-min cooldown shows ~15 min remaining."""
    out = _format_recent_closed([_row("INTU", 15)], close_cooldown_min=30)
    assert "⛔ BUY BLOCKED ~15 more min" in out


def test_gate_flag_stop_loss_uses_longer_reentry_window():
    """A stop-out 60 min ago is PAST the 30-min close cooldown but still
    inside the 120-min re-entry gate — the flag must reflect the gate
    the executor will actually apply."""
    row = _row("ADBE", 60)
    row["exit_reason"] = "stop_loss"
    out = _format_recent_closed([row], close_cooldown_min=30, reentry_cooldown_min=120)
    assert "⛔ BUY BLOCKED ~60 more min" in out


def test_no_gate_flag_once_window_elapsed():
    out = _format_recent_closed([_row("MSFT", 45)], close_cooldown_min=30)
    assert "BUY BLOCKED" not in out.split("\n")[1]  # row line carries no flag
    # Stop-out past its full window is also clean.
    row = _row("ADBE", 130)
    row["exit_reason"] = "stop_loss"
    out = _format_recent_closed([row], close_cooldown_min=30, reentry_cooldown_min=120)
    assert "BUY BLOCKED" not in out.split("\n")[1]


def test_pnl_pct_included_when_prices_present():
    # entry 100, exit 105 → +5%
    out = _format_recent_closed([_row("MSFT", 20, qty=10, entry=100, exit=105)])
    assert "+5.00%" in out


def test_iso_string_closed_at_parsed():
    """`closed_at` may arrive as an ISO string from `model_dump()`."""
    iso_ts = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    out = _format_recent_closed(
        [
            {
                "symbol": "NVDA",
                "side": "buy",
                "quantity": 10,
                "filled_quantity": 10,
                "filled_price": 220,
                "exit_price": 218,
                "exit_reason": "stop_loss",
                "closed_at": iso_ts,
            }
        ]
    )
    assert "NVDA" in out
    assert "stop_loss" in out
    # 30 min ago (give or take a second of skew)
    assert "30 min ago" in out or "29 min ago" in out


def test_capped_at_eight_rows():
    """The prompt only shows up to 8 to keep the context compact."""
    rows = [_row(f"SYM{i}", i) for i in range(20)]
    out = _format_recent_closed(rows)
    # The warning header is line 1, so 1 + 8 = 9 lines max.
    assert out.count("\n") <= 8


def test_gated_buy_symbols_mirrors_gate_windows():
    from halal_trader.trading.strategy import gated_buy_symbols

    rows = [
        _row("INTU", 15),  # inside 30-min close cooldown
        _row("MSFT", 45),  # past it
    ]
    sl = _row("ADBE", 60)
    sl["exit_reason"] = "stop_loss"  # past close cooldown, inside re-entry gate
    rows.append(sl)
    old_sl = _row("NVDA", 130)
    old_sl["exit_reason"] = "stop_loss"  # past both
    rows.append(old_sl)

    gated = gated_buy_symbols(rows, close_cooldown_min=30, reentry_cooldown_min=120)
    assert gated == {"INTU", "ADBE"}


def test_gated_buy_symbols_handles_bad_rows():
    from halal_trader.trading.strategy import gated_buy_symbols

    assert gated_buy_symbols([]) == set()
    assert gated_buy_symbols([{"symbol": "", "closed_at": None}]) == set()
    assert gated_buy_symbols([{"symbol": "X", "closed_at": "not-a-date"}]) == set()
