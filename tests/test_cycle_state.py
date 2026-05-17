"""Tests for the Wave B ``CycleState`` scaffolding.

The dataclass declares the per-cycle data shape that future stage
classes will mutate. Nothing reads from it yet — these tests just
lock in the field defaults so a partial state is always valid
mid-pipeline (a downstream stage early-returns when its prerequisites
aren't populated rather than raising).
"""

from __future__ import annotations

from halal_trader.core.cycle_pipeline import CycleState, StageOutcome


def test_cycle_state_defaults_construct_without_args():
    state = CycleState()
    # Every field defaults — partial states are valid mid-pipeline.
    assert state.cycle_id == ""
    assert state.halal_pairs == []
    assert state.klines_by_symbol == {}
    assert state.indicators_cache == {}
    assert state.snapshots == {}
    assert state.bars == {}
    assert state.risk_text == ""
    assert state.regime_text == ""
    assert state.ml_signals_text == ""
    assert state.forecasts_text == ""  # Chronos forecaster seed for the ML stage
    assert state.timeframe_text == ""
    assert state.plan is None
    assert state.halt is False
    assert state.risk_state is None  # structured PortfolioRiskState slot
    assert state.stage_outcomes == []
    assert state.today_pnl == 0.0
    assert state.current_prices == {}


def test_cycle_state_carries_forecasts_text_field():
    """`BuildForecastsStage` populates this; `BuildMlSignalsStage` reads it.
    The field's existence is load-bearing for the stage chain."""
    state = CycleState(forecasts_text="ML predicts UP 2%")
    assert state.forecasts_text == "ML predicts UP 2%"


def test_cycle_state_collections_are_per_instance():
    """Field defaults that are mutable containers must use ``default_factory``."""
    a = CycleState()
    b = CycleState()
    a.halal_pairs.append("BTCUSDT")
    a.indicators_cache["BTCUSDT"] = {"rsi_14": 55}
    a.stage_outcomes.append(StageOutcome(name="test", elapsed_ms=1.0))
    # Sibling state must be untouched.
    assert b.halal_pairs == []
    assert b.indicators_cache == {}
    assert b.stage_outcomes == []


def test_cycle_state_carries_text_block_per_prompt_source():
    """One prompt-context source = one field on the state."""
    fields_for_prompt = {
        "risk_text",
        "regime_text",
        "sentiment_text",
        "timeframe_text",
        "ml_signals_text",
        "microstructure_text",
        "news_text",
        "catalysts_text",
        "performance_text",
        "exchange_rules_text",
        "active_adjustments",
    }
    actual = {f.name for f in CycleState.__dataclass_fields__.values()}
    assert fields_for_prompt.issubset(actual)
