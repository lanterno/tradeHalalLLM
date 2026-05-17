"""Tests for `core/counterfactual.py` (counterfactual trade analyzer).

Covers the analyzer mechanics (skip-flatlines-equity, kept-trade
stats are computed without the held-flat zeros, return_uplift sign
matches direction) plus the three convenience predicate factories
and the input-tolerance edges (dict vs attribute rows, missing
fields, exception in predicate).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from halal_trader.core.counterfactual import (
    CounterfactualReport,
    _build_curve,
    _row_return,
    analyze_counterfactual,
    by_loss_streak,
    by_regime,
    by_symbol,
)


@dataclass
class _Trade:
    """Minimal attribute-shaped row for the predicate / extractor
    tolerance tests. Mirrors the Trade / CryptoTrade fields the
    analyzer touches."""

    return_pct: float | None = None
    symbol: str | None = None
    pair: str | None = None
    regime: str | None = None


# ── _row_return tolerance ─────────────────────────────────


def test_row_return_handles_dict():
    assert _row_return({"return_pct": 0.05}) == 0.05


def test_row_return_handles_attribute_object():
    assert _row_return(_Trade(return_pct=-0.02)) == -0.02


def test_row_return_returns_none_for_missing():
    assert _row_return({}) is None
    assert _row_return(_Trade()) is None


def test_row_return_returns_none_for_non_numeric():
    """A garbage value mustn't crash the analyzer; pin the soft
    failure path."""
    assert _row_return({"return_pct": "not-a-number"}) is None


def test_row_return_returns_none_for_non_finite():
    assert _row_return({"return_pct": float("inf")}) is None
    assert _row_return({"return_pct": float("nan")}) is None


# ── _build_curve ──────────────────────────────────────────


def test_build_curve_compounds_correctly():
    curve = _build_curve([0.10, -0.05, 0.02])
    # 1.0 → 1.10 → 1.045 → 1.0659
    assert curve[0] == pytest.approx(1.10)
    assert curve[1] == pytest.approx(1.045)
    assert curve[2] == pytest.approx(1.0659)


def test_build_curve_handles_empty():
    assert _build_curve([]) == []


def test_build_curve_respects_starting_equity():
    curve = _build_curve([0.10], starting=1000.0)
    assert curve == [pytest.approx(1100.0)]


# ── analyze_counterfactual mechanics ──────────────────────


def test_analyzer_no_skips_returns_identical_curves():
    """If the predicate matches nothing, both curves must be
    bit-identical and uplift must be zero."""
    trades = [{"return_pct": 0.05}, {"return_pct": -0.02}]
    rep = analyze_counterfactual(trades, lambda _: False)
    assert rep.actual_curve == rep.counterfactual_curve
    assert rep.skipped_count == 0
    assert rep.return_uplift == pytest.approx(0.0)
    assert rep.actual.n_trades == 2
    assert rep.counterfactual.n_trades == 2


def test_analyzer_skip_flatlines_counterfactual_curve_at_skip():
    """The counterfactual curve must hold equity *constant* across
    a skipped trade — that's the convention the dashboard plots."""
    trades = [
        {"return_pct": 0.10, "symbol": "BTC"},
        {"return_pct": -0.50, "symbol": "DOGE"},  # ← skipped
        {"return_pct": 0.05, "symbol": "BTC"},
    ]
    rep = analyze_counterfactual(trades, by_symbol("DOGE"))
    # Actual: 1.0 → 1.10 → 0.55 → 0.5775
    # Counterfactual: 1.0 → 1.10 → 1.10 (flat) → 1.155
    assert rep.actual_curve[1] == pytest.approx(0.55)
    assert rep.counterfactual_curve[1] == pytest.approx(1.10)
    assert rep.counterfactual_curve[2] == pytest.approx(1.155)
    assert rep.skipped_count == 1


def test_analyzer_skip_improves_total_return_when_skip_was_loss():
    """If we drop a losing trade, counterfactual total_return must
    exceed actual total_return → uplift > 0."""
    trades = [
        {"return_pct": 0.10},
        {"return_pct": -0.20},  # skip
        {"return_pct": 0.05},
    ]
    rep = analyze_counterfactual(trades, lambda r: r["return_pct"] < 0)
    assert rep.return_uplift > 0
    assert rep.counterfactual.total_return > rep.actual.total_return


def test_analyzer_skip_hurts_total_return_when_skip_was_winner():
    """If we drop a winning trade, uplift must be negative."""
    trades = [
        {"return_pct": -0.05},
        {"return_pct": 0.30},  # skip the big winner
        {"return_pct": -0.02},
    ]
    rep = analyze_counterfactual(trades, lambda r: r["return_pct"] > 0.20)
    assert rep.return_uplift < 0


def test_analyzer_counterfactual_stats_exclude_held_flat_zeros():
    """The counterfactual cohort's n_trades must reflect *kept*
    trades only — counting the held-flat zeros would distort
    win-rate / Sharpe / drawdown."""
    trades = [
        {"return_pct": 0.10},
        {"return_pct": -0.05},  # skip
        {"return_pct": -0.05},  # skip
        {"return_pct": 0.20},
    ]
    rep = analyze_counterfactual(trades, lambda r: r["return_pct"] < 0)
    assert rep.actual.n_trades == 4
    assert rep.counterfactual.n_trades == 2
    # Win rate of kept-only is 100% (both kept trades positive).
    assert rep.counterfactual.win_rate == 1.0


def test_analyzer_drops_rows_with_missing_return():
    """A legacy row missing ``return_pct`` shouldn't be counted
    in either cohort — we want to surface what's analyzable."""
    trades = [
        {"return_pct": 0.05},
        {"symbol": "AAPL"},  # no return_pct
        {"return_pct": -0.02},
    ]
    rep = analyze_counterfactual(trades, lambda _: False)
    assert rep.actual.n_trades == 2


def test_analyzer_handles_attribute_shaped_rows():
    trades = [
        _Trade(return_pct=0.05, symbol="BTC"),
        _Trade(return_pct=-0.10, symbol="ETH"),
    ]
    rep = analyze_counterfactual(trades, by_symbol("ETH"))
    assert rep.skipped_count == 1
    assert rep.return_uplift > 0


def test_analyzer_predicate_exception_is_treated_as_no_skip():
    """A buggy predicate mustn't crash the operator's research
    session; pin the soft failure path."""

    def bad(row):
        raise RuntimeError("predicate bug")

    trades = [{"return_pct": 0.05}, {"return_pct": -0.02}]
    rep = analyze_counterfactual(trades, bad)
    assert rep.skipped_count == 0
    assert rep.actual.n_trades == 2


def test_analyzer_rejects_non_positive_starting_equity():
    with pytest.raises(ValueError, match="starting_equity"):
        analyze_counterfactual([], lambda _: False, starting_equity=0.0)


def test_analyzer_empty_trades_returns_empty_report():
    rep = analyze_counterfactual([], lambda _: False)
    assert isinstance(rep, CounterfactualReport)
    assert rep.actual.n_trades == 0
    assert rep.counterfactual.n_trades == 0
    assert rep.skipped_count == 0
    assert rep.actual_curve == []
    assert rep.counterfactual_curve == []
    assert rep.return_uplift == 0.0


def test_analyzer_curves_use_custom_starting_equity():
    trades = [{"return_pct": 0.10}, {"return_pct": -0.05}]
    rep = analyze_counterfactual(trades, lambda _: False, starting_equity=10_000.0)
    assert rep.actual_curve[0] == pytest.approx(11_000.0)
    assert rep.actual_curve[1] == pytest.approx(10_450.0)


# ── by_symbol predicate ──────────────────────────────────


def test_by_symbol_matches_dict_symbol_field():
    pred = by_symbol("AAPL")
    assert pred({"symbol": "AAPL"}) is True
    assert pred({"symbol": "MSFT"}) is False


def test_by_symbol_matches_dict_pair_field():
    """Crypto rows use ``pair`` instead of ``symbol`` — predicate
    must handle both."""
    pred = by_symbol("BTCUSDT")
    assert pred({"pair": "BTCUSDT"}) is True


def test_by_symbol_is_case_insensitive():
    pred = by_symbol("btc")
    assert pred({"symbol": "BTC"}) is True
    assert pred({"symbol": "btc"}) is True


def test_by_symbol_handles_missing_field_as_false():
    pred = by_symbol("AAPL")
    assert pred({}) is False


def test_by_symbol_matches_attribute_objects():
    pred = by_symbol("BTC")
    assert pred(_Trade(symbol="BTC")) is True
    assert pred(_Trade(pair="BTC")) is True
    assert pred(_Trade(symbol="ETH")) is False


# ── by_regime predicate ──────────────────────────────────


def test_by_regime_matches_dict_regime_field():
    pred = by_regime("downtrend")
    assert pred({"regime": "downtrend"}) is True
    assert pred({"regime": "uptrend"}) is False


def test_by_regime_is_case_insensitive():
    pred = by_regime("DOWNTREND")
    assert pred({"regime": "downtrend"}) is True


def test_by_regime_handles_missing_field():
    pred = by_regime("downtrend")
    assert pred({}) is False


# ── by_loss_streak predicate ─────────────────────────────


def test_by_loss_streak_skips_after_threshold():
    """After 3 consecutive losses, the predicate must start
    skipping. Pin the count-then-act semantic."""
    pred = by_loss_streak(3)
    seq = [
        {"return_pct": -0.01},  # streak=1, no skip yet
        {"return_pct": -0.01},  # streak=2, no skip
        {"return_pct": -0.01},  # streak=3, no skip yet
        {"return_pct": -0.01},  # streak=4 — SHOULD skip (>=3 already)
        {"return_pct": -0.01},  # SHOULD skip
    ]
    results = [pred(r) for r in seq]
    # The first three losses build the streak but aren't skipped
    # themselves — the contract is "after N consecutive losses",
    # so the (N+1)th onwards is the skipped territory.
    assert results == [False, False, False, True, True]


def test_by_loss_streak_resets_on_a_win():
    pred = by_loss_streak(2)
    seq = [
        {"return_pct": -0.01},  # streak=1
        {"return_pct": -0.01},  # streak=2
        {"return_pct": -0.01},  # >=2 → skip (this and beyond)
        {"return_pct": 0.05},  # WIN — but at this point streak was already >=2
        {"return_pct": -0.01},  # streak should have reset; back to streak=1
        {"return_pct": -0.01},  # streak=2; not skipped yet
        {"return_pct": -0.01},  # >=2 → skip
    ]
    results = [pred(r) for r in seq]
    assert results == [False, False, True, True, False, False, True]


def test_by_loss_streak_predicate_independent_per_factory_call():
    """Each call to ``by_loss_streak`` must produce a fresh closure
    — otherwise concurrent analyses would share state."""
    pred1 = by_loss_streak(2)
    pred2 = by_loss_streak(2)
    pred1({"return_pct": -0.01})
    pred1({"return_pct": -0.01})
    # pred1 has a streak of 2 internally; pred2 must be at 0.
    assert pred2({"return_pct": -0.01}) is False


# ── end-to-end with realistic shapes ─────────────────────


def test_end_to_end_drop_doge_makes_portfolio_better():
    """Realistic scenario: portfolio holds BTC, ETH, and DOGE.
    DOGE was a net loser; the counterfactual that drops it must
    show a positive uplift and a smaller drawdown."""
    trades = [
        _Trade(return_pct=0.05, symbol="BTC"),
        _Trade(return_pct=-0.30, symbol="DOGE"),
        _Trade(return_pct=0.02, symbol="ETH"),
        _Trade(return_pct=-0.20, symbol="DOGE"),
        _Trade(return_pct=0.08, symbol="BTC"),
        _Trade(return_pct=0.03, symbol="ETH"),
    ]
    rep = analyze_counterfactual(trades, by_symbol("DOGE"))
    assert rep.skipped_count == 2
    assert rep.return_uplift > 0
    # Counterfactual drawdown must be no worse than actual.
    assert rep.counterfactual.max_drawdown >= rep.actual.max_drawdown
