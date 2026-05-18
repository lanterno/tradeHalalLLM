"""Tests for the prompt-block formatters in :mod:`trading.strategy`.

These three functions (``_format_positions``, ``_format_snapshots``,
``_format_bars``) shape what the LLM sees each cycle. They have no
side effects and are pure dict-walkers — easy to lock in.
"""

from unittest.mock import MagicMock

from halal_trader.trading.strategy import (
    _format_bars,
    _format_positions,
    _format_snapshots,
)

# ── _format_positions ──────────────────────────────────────────


def test_format_positions_empty_returns_sentinel():
    assert _format_positions([]) == "No open positions."


def test_format_positions_renders_per_position_line():
    p = MagicMock(
        symbol="AAPL",
        qty=10,
        avg_entry_price=180.0,
        current_price=182.5,
        unrealized_pl=25.0,
        unrealized_plpc=0.0139,
    )
    out = _format_positions([p])
    assert "AAPL" in out
    assert "180.00" in out
    assert "182.50" in out
    assert "+$25.00" in out or "$+25.00" in out


def test_format_positions_negative_plpc_renders_minus_sign():
    """Sign is meaningful — the prompt uses it to nudge the LLM."""
    p = MagicMock(
        symbol="AAPL",
        qty=10,
        avg_entry_price=180.0,
        current_price=170.0,
        unrealized_pl=-100.0,
        unrealized_plpc=-0.0556,
    )
    out = _format_positions([p])
    assert "-$100.00" in out or "$-100.00" in out
    assert "-5" in out  # -5.56%


# ── _format_snapshots ──────────────────────────────────────────


def test_format_snapshots_empty_returns_sentinel():
    assert _format_snapshots({}) == "No snapshot data available."


def test_format_snapshots_walks_alpaca_shape():
    """Alpaca's `latest_trade` / `latest_quote` / `daily_bar` keys."""
    snaps = {
        "AAPL": {
            "latest_trade": {"price": 182.5},
            "latest_quote": {"bid_price": 182.0, "ask_price": 183.0},
            "daily_bar": {"volume": 12_000_000},
        }
    }
    out = _format_snapshots(snaps)
    assert "AAPL" in out
    assert "182.5" in out
    assert "12000000" in out


def test_format_snapshots_handles_missing_subkeys_with_na():
    snaps = {"AAPL": {}}  # no latest_trade / latest_quote / daily_bar
    out = _format_snapshots(snaps)
    assert "AAPL" in out
    assert "N/A" in out


def test_format_snapshots_falls_back_for_non_dict_value():
    snaps = {"AAPL": "raw string"}
    out = _format_snapshots(snaps)
    assert "AAPL" in out
    assert "raw string" in out


# ── _format_bars ──────────────────────────────────────────────


def test_format_bars_empty_returns_sentinel():
    assert _format_bars({}) == "No bar data available."


def test_format_bars_lists_last_five_bars_per_symbol():
    """Even with > 5 bars, only the last 5 are emitted (LLM-friendly)."""
    bars = {
        "AAPL": [
            {
                "timestamp": f"t{i}",
                "open": 100 + i,
                "high": 101 + i,
                "low": 99 + i,
                "close": 100 + i,
                "volume": 1000,
            }
            for i in range(8)
        ]
    }
    out = _format_bars(bars)
    # Should contain t3..t7 (the last 5), not t0..t2.
    assert "t7" in out
    assert "t3" in out
    assert "t2" not in out


def test_format_bars_falls_back_for_non_list_value():
    bars = {"AAPL": "raw string"}
    out = _format_bars(bars)
    assert "AAPL" in out
    assert "raw string" in out


def test_format_bars_handles_missing_bar_fields_with_zero_default():
    bars = {"AAPL": [{"timestamp": "t0"}]}  # no open/high/low/close/volume
    out = _format_bars(bars)
    assert "t0" in out
    assert "0.00" in out


# ── Round-7 follow-up: performance_text + active_adjustments ──────


def test_user_prompt_template_renders_performance_block():
    """The Wave-equivalent ``=== RECENT PERFORMANCE ===`` block lands in
    the rendered prompt. Pinning so a future refactor doesn't silently
    drop the new prompt-context block crypto already has."""
    from halal_trader.trading.strategy import USER_PROMPT_TEMPLATE

    assert "=== RECENT PERFORMANCE" in USER_PROMPT_TEMPLATE
    assert "{performance_text}" in USER_PROMPT_TEMPLATE


def test_user_prompt_template_renders_active_adjustments_block():
    from halal_trader.trading.strategy import USER_PROMPT_TEMPLATE

    assert "=== ACTIVE STRATEGY ADJUSTMENTS" in USER_PROMPT_TEMPLATE
    assert "{active_adjustments}" in USER_PROMPT_TEMPLATE


def test_user_prompt_template_renders_news_block():
    """Stocks-side equities news block — populated by FetchStockNewsStage
    from Yahoo Finance per cycle."""
    from halal_trader.trading.strategy import USER_PROMPT_TEMPLATE

    assert "=== RECENT NEWS HEADLINES" in USER_PROMPT_TEMPLATE
    assert "{news_text}" in USER_PROMPT_TEMPLATE


def test_user_prompt_template_falls_back_to_friendly_defaults():
    """The strategy passes ``performance_text or "No completed trades yet."``
    so a fresh bot with no analytics wired still renders a clean prompt."""
    from halal_trader.trading.strategy import USER_PROMPT_TEMPLATE

    rendered = USER_PROMPT_TEMPLATE.format(
        buying_power=100000.0,
        portfolio_value=100000.0,
        cash=100000.0,
        today_pnl=0.0,
        today_pnl_pct=0.0,
        positions_text="No open positions.",
        halal_symbols="AAPL, MSFT",
        snapshots_text="(none)",
        bars_text="(none)",
        sentiment_text="(none)",
        risk_text="(none)",
        regime_text="(none)",
        ml_signals_text="(none)",
        timeframe_text="(none)",
        catalysts_text="(none)",
        performance_text="No completed trades yet.",
        active_adjustments="None.",
        news_text="No recent news.",
    )
    # The blocks render in the canonical order — bars before performance
    # before active_adjustments before news before sentiment — so the
    # LLM sees context in a stable layout.
    bars_pos = rendered.index("RECENT PRICE BARS")
    perf_pos = rendered.index("RECENT PERFORMANCE")
    adj_pos = rendered.index("ACTIVE STRATEGY ADJUSTMENTS")
    news_pos = rendered.index("RECENT NEWS HEADLINES")
    sentiment_pos = rendered.index("SENTIMENT ANALYSIS")
    assert bars_pos < perf_pos < adj_pos < news_pos < sentiment_pos
