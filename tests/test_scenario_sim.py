"""Tests for `core/scenario_sim.py` (scenario simulator).

Pins the per-bar SL/TP fill semantic, the SL-wins-on-tie invariant,
the trailing-stop ratchet, the portfolio aggregation, and the
zero-everywhere edges (empty positions / empty klines).
"""

from __future__ import annotations

import pytest

from halal_trader.core.scenario_sim import (
    PositionProjection,
    ScenarioReport,
    SimulatedPosition,
    render_report,
    simulate,
)
from halal_trader.crypto.stress import flash_crash_klines, gap_down_klines
from halal_trader.domain.models import Kline


def _bar(open_: float, high: float, low: float, close: float, *, t: int = 0) -> Kline:
    return Kline(
        open_time=t,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100.0,
        close_time=t + 60_000,
    )


# ── trivial / empty edges ─────────────────────────────────


def test_empty_positions_produces_zero_report():
    rep = simulate([], [_bar(100, 100, 100, 100)])
    assert isinstance(rep, ScenarioReport)
    assert rep.total_starting_equity == 0.0
    assert rep.total_end_equity == 0.0
    assert rep.portfolio_pnl == 0.0
    assert rep.portfolio_pnl_pct == 0.0
    assert rep.projections == []


def test_empty_klines_leaves_position_open_at_entry_value():
    pos = SimulatedPosition(pair="BTCUSDT", quantity=0.5, entry_price=60_000)
    rep = simulate([pos], [])
    assert len(rep.projections) == 1
    proj = rep.projections[0]
    assert not proj.filled
    assert proj.end_equity == proj.starting_equity == 30_000.0


# ── SL fills ──────────────────────────────────────────────


def test_sl_fires_when_bar_low_pierces_stop():
    """A wick that pierces SL fires the order even if close
    recovers — pin the high/low semantic, not close-only."""
    pos = SimulatedPosition(
        pair="BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
    )
    # Bar dips to 94, recovers to close at 99.
    klines = [_bar(100, 100, 94, 99)]
    rep = simulate([pos], klines)
    proj = rep.projections[0]
    assert proj.filled
    assert proj.fill_reason == "stop_loss"
    assert proj.fill_price == 95.0
    assert proj.fill_bar_index == 0


def test_sl_fires_on_subsequent_bar_when_first_bar_safe():
    pos = SimulatedPosition(pair="BTCUSDT", quantity=1.0, entry_price=100.0, stop_loss=95.0)
    klines = [
        _bar(100, 102, 99, 101),  # safe
        _bar(101, 101, 90, 91),  # SL hit
    ]
    rep = simulate([pos], klines)
    proj = rep.projections[0]
    assert proj.filled
    assert proj.fill_bar_index == 1


# ── TP fills ──────────────────────────────────────────────


def test_tp_fires_when_bar_high_pierces_target():
    pos = SimulatedPosition(pair="BTCUSDT", quantity=1.0, entry_price=100.0, take_profit=110.0)
    klines = [_bar(100, 112, 99, 105)]  # wick to 112
    rep = simulate([pos], klines)
    proj = rep.projections[0]
    assert proj.filled
    assert proj.fill_reason == "take_profit"
    assert proj.fill_price == 110.0


# ── SL vs TP tiebreak ─────────────────────────────────────


def test_sl_wins_when_both_sl_and_tp_fill_in_same_bar():
    """A volatile gap bar can have both bands inside its H-L range.
    Pin: SL wins — worst-case execution is the safer projection."""
    pos = SimulatedPosition(
        pair="BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
    )
    # Bar: low 94, high 112 — both bands hit.
    klines = [_bar(100, 112, 94, 100)]
    rep = simulate([pos], klines)
    proj = rep.projections[0]
    assert proj.filled
    assert proj.fill_reason == "stop_loss"


# ── trailing stop ─────────────────────────────────────────


def test_trailing_stop_ratchets_up_on_new_high():
    """As the bar high makes new highs, the SL should track up
    by `trailing_stop_pct`. Pin: SL never moves down."""
    pos = SimulatedPosition(
        pair="BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=95.0,
        trailing_stop_pct=0.05,  # 5% trail
    )
    klines = [
        _bar(100, 110, 99, 108),  # high=110 → trail SL = 110*0.95 = 104.5
        _bar(108, 109, 104, 105),  # low 104 < new SL 104.5 → fill at 104.5
    ]
    rep = simulate([pos], klines)
    proj = rep.projections[0]
    assert proj.filled
    assert proj.fill_reason == "trailing_stop"
    assert proj.fill_price == pytest.approx(104.5)


def test_trailing_stop_does_not_fire_if_bar_holds_above_trail():
    pos = SimulatedPosition(
        pair="BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
        trailing_stop_pct=0.05,
    )
    klines = [
        _bar(100, 110, 99, 108),  # trail = 104.5
        _bar(108, 110, 105, 109),  # low 105 > trail 104.5 → safe
    ]
    rep = simulate([pos], klines)
    proj = rep.projections[0]
    assert not proj.filled


def test_trailing_stop_can_be_combined_with_take_profit():
    pos = SimulatedPosition(
        pair="BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
        take_profit=120.0,
        trailing_stop_pct=0.05,
    )
    klines = [
        _bar(100, 121, 99, 115),  # TP hit at 120 (before trail kicks in)
    ]
    rep = simulate([pos], klines)
    proj = rep.projections[0]
    assert proj.filled
    assert proj.fill_reason == "take_profit"
    assert proj.fill_price == 120.0


# ── min/max equity tracking ───────────────────────────────


def test_min_equity_reflects_lowest_bar_low_until_fill():
    pos = SimulatedPosition(pair="BTCUSDT", quantity=2.0, entry_price=100.0, stop_loss=80.0)
    klines = [
        _bar(100, 105, 90, 95),  # low 90 → trough = 180
        _bar(95, 99, 92, 96),  # low 92 → trough still 180
    ]
    rep = simulate([pos], klines)
    proj = rep.projections[0]
    # 90 × 2 = 180
    assert proj.min_equity == pytest.approx(180.0)


def test_max_equity_reflects_highest_bar_high_until_fill():
    pos = SimulatedPosition(pair="BTCUSDT", quantity=2.0, entry_price=100.0)
    klines = [
        _bar(100, 110, 99, 105),
        _bar(105, 115, 104, 110),
    ]
    rep = simulate([pos], klines)
    proj = rep.projections[0]
    assert proj.max_equity == pytest.approx(230.0)  # 115 × 2


# ── portfolio aggregation ─────────────────────────────────


def test_portfolio_aggregates_pnl_across_positions():
    """Two positions in different scenarios — totals must equal the
    sum of per-position outcomes."""
    pos_a = SimulatedPosition(pair="A", quantity=1.0, entry_price=100.0, take_profit=110.0)
    pos_b = SimulatedPosition(pair="B", quantity=1.0, entry_price=100.0, stop_loss=95.0)
    # A hits TP at 110; B hits SL at 95.
    klines = [_bar(100, 112, 94, 100)]
    rep = simulate([pos_a, pos_b], klines)
    assert rep.total_starting_equity == 200.0
    assert rep.total_end_equity == 205.0
    assert rep.portfolio_pnl == pytest.approx(5.0)
    assert rep.portfolio_pnl_pct == pytest.approx(0.025)


def test_portfolio_drawdown_sums_position_troughs():
    """Worst-case projection sums each position's trough — pin the
    conservative semantic so a refactor can't silently soften it."""
    pos_a = SimulatedPosition(pair="A", quantity=1.0, entry_price=100.0)
    pos_b = SimulatedPosition(pair="B", quantity=1.0, entry_price=100.0)
    klines = [_bar(100, 110, 90, 95)]  # trough -10 each
    rep = simulate([pos_a, pos_b], klines)
    assert rep.portfolio_drawdown == pytest.approx(-20.0)


# ── integration with stress generators ───────────────────


def test_simulator_runs_against_flash_crash_scenario():
    """End-to-end: a tight SL on a flash-crash-shaped scenario must
    fire the stop. Uses the round-3 generator already in the
    stress module — verifies the simulator and stress modules are
    interoperable."""
    pos = SimulatedPosition(
        pair="BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=92.0,  # 8% stop
    )
    rep = simulate([pos], flash_crash_klines())
    proj = rep.projections[0]
    assert proj.filled
    assert proj.fill_reason == "stop_loss"


def test_simulator_runs_against_gap_down_scenario():
    pos = SimulatedPosition(
        pair="BTCUSDT",
        quantity=1.0,
        entry_price=100.0,
        stop_loss=95.0,  # 5% stop
    )
    rep = simulate([pos], gap_down_klines())
    proj = rep.projections[0]
    assert proj.filled
    assert proj.fill_reason == "stop_loss"


# ── render_report ─────────────────────────────────────────


def test_render_report_includes_portfolio_summary_line():
    pos = SimulatedPosition(pair="BTCUSDT", quantity=1.0, entry_price=100.0)
    rep = simulate([pos], [_bar(100, 105, 99, 103)])
    text = render_report(rep)
    assert "Portfolio" in text
    assert "drawdown" in text.lower()


def test_render_report_marks_filled_positions_with_close_label():
    pos = SimulatedPosition(pair="BTCUSDT", quantity=1.0, entry_price=100.0, stop_loss=95.0)
    rep = simulate([pos], [_bar(100, 100, 94, 95)])
    text = render_report(rep)
    assert "CLOSED" in text


def test_render_report_marks_open_positions():
    pos = SimulatedPosition(pair="BTCUSDT", quantity=1.0, entry_price=100.0)
    rep = simulate([pos], [_bar(100, 105, 99, 103)])
    text = render_report(rep)
    assert "OPEN" in text


# ── output structure ─────────────────────────────────────


def test_projection_carries_fill_bar_index():
    pos = SimulatedPosition(pair="BTCUSDT", quantity=1.0, entry_price=100.0, stop_loss=95.0)
    klines = [
        _bar(100, 102, 99, 101),
        _bar(101, 101, 99, 100),
        _bar(100, 100, 90, 91),  # SL fills here
    ]
    rep = simulate([pos], klines)
    assert rep.projections[0].fill_bar_index == 2


def test_projection_dataclass_is_frozen():
    """Pin immutability so a downstream consumer can safely cache
    a projection without worrying about mutation."""
    pos = SimulatedPosition(pair="X", quantity=1.0, entry_price=100.0)
    rep = simulate([pos], [_bar(100, 101, 99, 100)])
    proj = rep.projections[0]
    assert isinstance(proj, PositionProjection)
    with pytest.raises(Exception):  # FrozenInstanceError
        proj.pair = "Y"  # type: ignore[misc]
