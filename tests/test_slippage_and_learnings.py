"""Round B telemetry: slippage tracking + self-review feed-forward.

Built 2026-05-22 night after observing $607 gross closed-trip P&L vs
$42 net EOD equity change — implying ~$565 lost to slippage and MTM
drag on the 2026-05-21 session. We now (a) compute and persist
``paper_slippage_pct`` per BUY fill, (b) surface a rolling-window
summary in the user prompt, and (c) persist self-review observations
so they re-appear in tomorrow's system prompt.
"""

from __future__ import annotations

from halal_trader.trading.executor import _compute_slippage_pct
from halal_trader.trading.strategy import _format_learnings, _format_slippage


# ── _compute_slippage_pct ───────────────────────────────────────


def test_buy_filled_higher_is_positive_slippage():
    """BUY at 102 when estimate was 100 → +2% adverse."""
    assert _compute_slippage_pct(side="buy", estimated_price=100.0, filled_price=102.0) == 0.02


def test_buy_filled_lower_is_negative_slippage():
    """BUY at 99 when estimate was 100 → -1% (favorable)."""
    assert _compute_slippage_pct(side="buy", estimated_price=100.0, filled_price=99.0) == -0.01


def test_sell_filled_lower_is_positive_slippage():
    """SELL at 99 when estimate was 100 → +1% adverse (got less)."""
    assert _compute_slippage_pct(side="sell", estimated_price=100.0, filled_price=99.0) == 0.01


def test_sell_filled_higher_is_negative_slippage():
    """SELL at 101 when estimate was 100 → -1% favorable (got more)."""
    val = _compute_slippage_pct(side="sell", estimated_price=100.0, filled_price=101.0)
    assert val is not None
    assert abs(val - (-0.01)) < 1e-9


def test_slippage_none_when_either_price_missing():
    assert _compute_slippage_pct(side="buy", estimated_price=None, filled_price=100) is None
    assert _compute_slippage_pct(side="buy", estimated_price=100, filled_price=None) is None
    assert _compute_slippage_pct(side="buy", estimated_price=0, filled_price=100) is None


# ── _format_slippage ────────────────────────────────────────────


def test_slippage_empty_rows_default_message():
    assert "No slippage data yet." in _format_slippage([])


def test_slippage_skips_rows_without_paper_slippage_pct():
    out = _format_slippage([{"side": "buy"}, {"side": "buy", "paper_slippage_pct": None}])
    assert "No slippage data" in out


def test_slippage_summary_renders_avg_and_worst():
    rows = [
        {"paper_slippage_pct": 0.001},
        {"paper_slippage_pct": 0.003},
        {"paper_slippage_pct": -0.001},
        {"paper_slippage_pct": 0.005},  # worst adverse
    ]
    out = _format_slippage(rows)
    assert "4 buy fills" in out
    # Avg = (0.001 + 0.003 - 0.001 + 0.005) / 4 = 0.002 = +0.200%
    assert "+0.200%" in out
    # 3/4 were adverse (positive)
    assert "3/4" in out
    # Worst was +0.005 = +0.500%
    assert "+0.500%" in out


def test_slippage_caps_at_limit():
    rows = [{"paper_slippage_pct": 0.001}] * 50
    out = _format_slippage(rows, limit=20)
    assert "20 buy fills" in out


# ── _format_learnings ───────────────────────────────────────────


def test_learnings_empty_default():
    assert "No prior observations yet." in _format_learnings([])


def test_learnings_rendered_as_bullets():
    obs = ["pattern A: NVDA exits at a loss", "pattern B: too much Tech"]
    out = _format_learnings(obs)
    assert "factor them into today" in out
    assert "pattern A" in out
    assert "pattern B" in out
    # Bulleted shape
    assert out.count("•") == 2


def test_learnings_caps_at_six():
    obs = [f"obs {i}" for i in range(20)]
    out = _format_learnings(obs)
    assert out.count("•") == 6


def test_learnings_skips_empty_strings():
    obs = ["real observation", "", "   ", "another real one"]
    out = _format_learnings(obs)
    assert "real observation" in out
    assert "another real one" in out
    assert out.count("•") == 2


def test_learnings_truncates_long_text():
    long = "x" * 500
    out = _format_learnings([long])
    # Each obs is capped at 240 chars in the rendered output.
    assert "x" * 240 in out
    assert "x" * 250 not in out
