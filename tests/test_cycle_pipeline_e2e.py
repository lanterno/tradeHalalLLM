"""End-to-end stage-pipeline smoke test.

Each stage so far has been tested in isolation. This file proves the
stages compose: a single :class:`CycleState` flowing through a real
stage list produces every prompt-context field, which is the shape
the eventual ``_run_cycle_impl`` refactor will land on.

We don't drive a real cycle here — just the stage list.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.core.cycle_pipeline import CycleState
from halal_trader.core.cycle_stages import (
    BuildActiveAdjustmentsStage,
    BuildCatalystsStage,
    BuildCryptoRiskStage,
    BuildExchangeRulesStage,
    BuildForecastsStage,
    BuildMicrostructureStage,
    BuildMlSignalsStage,
    BuildNewsStage,
    BuildPerformanceStage,
    BuildRegimeStage,
    BuildSentimentStage,
    BuildTimeframeStage,
)
from halal_trader.crypto.regime import MarketRegime


@pytest.mark.asyncio
async def test_full_stage_pipeline_populates_every_text_field():
    """Drive 11 stages over one state — including the crypto risk path."""
    # Stage deps — minimal happy-path mocks.
    detector = MagicMock()
    detector.detect.return_value = (MarketRegime.TRENDING_UP, 0.85, "trade with the trend")

    anomaly = MagicMock()
    anomaly.detect.return_value = (False, 0.1)
    signal = MagicMock()
    signal.predict_confidence.return_value = 0.55

    timeframe = MagicMock()
    timeframe.analyze = AsyncMock(
        return_value={
            "BTCUSDT": {
                "alignment_score": 0.7,
                "per_tf": {"1Day": "RSI=58, MACD=bullish"},
                "support_resistance": [],
            }
        }
    )

    analytics = MagicMock()
    analytics.compute_stats = AsyncMock(return_value="<stats>")
    analytics.format_for_prompt = MagicMock(return_value="Win rate: 60%")

    self_review = MagicMock()
    self_review.format_adjustments_for_prompt.return_value = "- max_position_pct: 0.10"

    broker = MagicMock()
    broker.format_filters_for_prompt.return_value = "BTCUSDT: min_notional=10"

    feed = MagicMock()
    feed.fetch_all = AsyncMock(
        return_value=[
            # Use dataclass-style import so format_catalysts_for_prompt is happy.
        ]
    )
    # Use a real Catalyst so the formatter accepts it.
    from halal_trader.trading.catalysts import Catalyst

    feed.fetch_all = AsyncMock(
        return_value=[
            Catalyst(
                symbol="BTCUSDT",
                kind="news",
                title="BTC ETF approved",
                timestamp=datetime.now(timezone.utc),
                sentiment="positive",
                source="Bloomberg",
            )
        ]
    )

    news_feed = MagicMock()
    news_feed.snapshot.return_value = []  # exercises the empty path

    sentiment_mgr = MagicMock()
    sentiment_mgr.enabled = False  # skip composite signals path; keep test simple

    # ── Crypto risk dependencies ───────────────────────────────
    risk_engine = MagicMock()
    risk_state = MagicMock()
    risk_state.is_halted = False
    risk_engine.evaluate.return_value = risk_state
    risk_engine.format_for_prompt.return_value = "Heat 4.2%, drawdown 1.1%"

    account = MagicMock()
    account.total_balance_usdt = 10_000.0

    # Seed state with the indicators + orderbooks + halal_pairs +
    # account + current_prices that the various stages consume.
    state = CycleState(
        account=account,
        halal_pairs=["BTCUSDT"],
        klines_by_symbol={"BTCUSDT": [MagicMock(close=42_000.0)]},
        indicators_cache={"BTCUSDT": {"rsi_14": 60, "ema_9": 100, "ema_21": 99}},
        orderbooks={
            "BTCUSDT": {
                "bids": [[42000.0, 1.0], [41999.0, 2.0]],
                "asks": [[42001.0, 1.5], [42002.0, 0.5]],
            }
        },
        current_prices={"BTCUSDT": 42_000.0},
    )

    # ── Drive the stages in the order the cycle will land on ──
    stages = [
        BuildRegimeStage(detector=detector),
        BuildForecastsStage(forecaster=None),  # no-op when no forecaster wired
        BuildMlSignalsStage(anomaly_detector=anomaly, signal_classifier=signal),
        BuildTimeframeStage(analyzer=timeframe),
        BuildPerformanceStage(analytics=analytics),
        BuildActiveAdjustmentsStage(self_review=self_review),
        BuildExchangeRulesStage(broker=broker),
        BuildCatalystsStage(feed=feed),
        BuildMicrostructureStage(),
        BuildNewsStage(news_feed=news_feed),
        BuildSentimentStage(sentiment_manager=sentiment_mgr),
        BuildCryptoRiskStage(risk_engine=risk_engine),
    ]
    for stage in stages:
        out = await stage.run(state)
        # Every stage returns the state in place (the contract).
        assert out is state

    # Assertions on every prompt-context field the cycle would feed
    # into the LLM.
    assert "Heat" in state.risk_text  # populated by the risk stage
    assert state.risk_state is risk_state  # structured state threaded through
    assert state.halt is False
    assert "BTCUSDT" in state.regime_text
    assert "TRENDING_UP" in state.regime_text
    # ML stage emits a confidence section even with no anomalies.
    assert "ML confidence" in state.ml_signals_text
    assert "BTCUSDT" in state.timeframe_text
    assert "BULLISH" in state.timeframe_text  # alignment 0.7 → BULLISH bucket
    assert state.performance_text == "Win rate: 60%"
    assert state.active_adjustments == "- max_position_pct: 0.10"
    assert state.exchange_rules_text == "BTCUSDT: min_notional=10"
    assert "BTCUSDT" in state.catalysts_text
    assert "ETF" in state.catalysts_text
    assert isinstance(state.microstructure_text, str)  # specific content is helper's job
    # News stage with empty snapshot can produce empty or sentinel text.
    assert isinstance(state.news_text, str)
    # Disabled sentiment manager → empty text (no Reddit fetcher wired).
    assert state.sentiment_text == ""


@pytest.mark.asyncio
async def test_pipeline_runs_with_no_deps_wired():
    """Every stage's no-op path: empty state in, empty state out."""
    state = CycleState()
    stages = [
        BuildRegimeStage(detector=None),
        BuildForecastsStage(forecaster=None),
        BuildMlSignalsStage(),
        BuildTimeframeStage(analyzer=None),
        BuildPerformanceStage(analytics=None),
        BuildActiveAdjustmentsStage(self_review=None),
        BuildExchangeRulesStage(broker=None),
        BuildCatalystsStage(feed=None),
        BuildMicrostructureStage(),
        BuildNewsStage(news_feed=None),
        BuildSentimentStage(),
        BuildCryptoRiskStage(risk_engine=None),
    ]
    for stage in stages:
        await stage.run(state)
    # Every text field is still the empty default.
    assert state.regime_text == ""
    assert state.ml_signals_text == ""
    assert state.forecasts_text == ""
    assert state.timeframe_text == ""
    assert state.performance_text == ""
    assert state.active_adjustments == ""
    assert state.exchange_rules_text == ""
    assert state.catalysts_text == ""
    assert state.microstructure_text == ""
    assert state.news_text == ""
    assert state.sentiment_text == ""
    assert state.risk_text == ""
    assert state.halt is False


@pytest.mark.asyncio
async def test_run_stages_drives_list_and_returns_state():
    """``run_stages`` runs each stage in order and returns the mutated state."""
    from halal_trader.core.cycle_pipeline import run_stages

    detector = MagicMock()
    detector.detect.return_value = (MarketRegime.RANGING, 0.6, "use mean reversion")
    state = CycleState(indicators_cache={"AAPL": {"rsi_14": 50}})
    out = await run_stages(state, [BuildRegimeStage(detector=detector)])
    assert out is state
    assert "RANGING" in out.regime_text


@pytest.mark.asyncio
async def test_run_stages_short_circuits_on_halt():
    """``stop_on_halt=True`` stops the chain after a stage sets ``state.halt``."""
    from halal_trader.core.cycle_pipeline import run_stages

    class _Halt:
        name = "halt_setter"

        async def run(self, state):  # noqa: ANN001
            state.halt = True
            return state

    class _After:
        name = "after_halt"

        def __init__(self):
            self.ran = False

        async def run(self, state):  # noqa: ANN001
            self.ran = True
            return state

    after = _After()
    state = CycleState()
    await run_stages(state, [_Halt(), after], stop_on_halt=True)
    assert state.halt is True
    assert after.ran is False  # didn't run because halt short-circuited


@pytest.mark.asyncio
async def test_run_stages_runs_all_when_stop_on_halt_default():
    """Default behavior (``stop_on_halt=False``) runs every stage even if halt is set."""
    from halal_trader.core.cycle_pipeline import run_stages

    class _Halt:
        name = "halt_setter"

        async def run(self, state):  # noqa: ANN001
            state.halt = True
            return state

    class _After:
        name = "after_halt"

        def __init__(self):
            self.ran = False

        async def run(self, state):  # noqa: ANN001
            self.ran = True
            return state

    after = _After()
    state = CycleState()
    await run_stages(state, [_Halt(), after])  # default stop_on_halt=False
    assert state.halt is True
    assert after.ran is True  # ran despite halt — opt-in semantics


@pytest.mark.asyncio
async def test_run_stages_publishes_per_stage_events():
    """Each stage runs inside the instrumentation context — bus sees stage.start/end."""
    from halal_trader.core.cycle_pipeline import run_stages

    bus = MagicMock()
    bus.publish = AsyncMock()
    state = CycleState()
    await run_stages(
        state,
        [BuildRegimeStage(detector=None), BuildMlSignalsStage()],
        bus=bus,
    )
    # 2 stages × (start + end) = 4 publishes.
    assert bus.publish.await_count == 4
    topics = [call.args[0] for call in bus.publish.await_args_list]
    assert topics == [
        "cycle.stage.start",
        "cycle.stage.end",
        "cycle.stage.start",
        "cycle.stage.end",
    ]
