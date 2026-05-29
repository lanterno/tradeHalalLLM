"""Regime classifier + cheap interpreters."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.belief.schema import EvidenceItem, Regime
from halabot.cognition.bars import Bar, BarBuffer
from halabot.cognition.interpreters import (
    AnomalyInterpreter,
    DriftInterpreter,
    ForecasterInterpreter,
    IndicatorInterpreter,
    MultiFrameInterpreter,
    NewsLexiconInterpreter,
    NewsLlmInterpreter,
    RsiInterpreter,
    SupportResistanceInterpreter,
    TrendAlignmentInterpreter,
    VolumeConfirmationInterpreter,
)
from halabot.cognition.regime import EvidenceRegimeClassifier
from halabot.platform.clock import FakeClock
from halabot.platform.events import EventType, new_event

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)
CLOCK = FakeClock(T0)


def _ev(direction, weight=1.0, *, source="x", directional=True):
    return EvidenceItem(
        source=source, direction=direction, weight=weight, ts=T0, directional=directional
    )


# ── regime ──
def test_regime_empty_is_ranging():
    assert EvidenceRegimeClassifier().classify([])[0] == Regime.RANGING


def test_regime_strong_bullish_is_trending_up():
    regime, conf = EvidenceRegimeClassifier().classify([_ev(1.0), _ev(0.9)])
    assert regime == Regime.TRENDING_UP
    assert conf > 0


def test_regime_strong_bearish_is_trending_down():
    regime, _ = EvidenceRegimeClassifier().classify([_ev(-1.0), _ev(-0.8)])
    assert regime == Regime.TRENDING_DOWN


def test_regime_weak_signal_is_ranging():
    regime, _ = EvidenceRegimeClassifier().classify([_ev(0.1, 0.2)])
    assert regime == Regime.RANGING


def test_regime_anomaly_flag_is_volatile():
    regime, _ = EvidenceRegimeClassifier().classify(
        [_ev(1.0), _ev(0.0, source="anomaly", directional=False)]
    )
    assert regime == Regime.VOLATILE


# ── IndicatorInterpreter ──
@pytest.mark.asyncio
async def test_indicator_interpreter_emits_on_uptrend():
    buf = BarBuffer()
    for i in range(30):
        c = 100 + i
        buf.append("NVDA", Bar(o=c, h=c + 1, low=c - 1, c=c, v=1.0, ts=T0))
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await IndicatorInterpreter(buf).interpret(obs)
    assert len(out) == 1
    assert out[0].source == "indicator.momentum"
    assert out[0].direction > 0
    assert out[0].event_id == obs.id  # provenance


@pytest.mark.asyncio
async def test_indicator_interpreter_silent_without_history():
    buf = BarBuffer()
    buf.append("NVDA", Bar(o=1, h=1, low=1, c=1, v=1.0, ts=T0))
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    assert await IndicatorInterpreter(buf).interpret(obs) == []


# ── RsiInterpreter ──
def _fill(buf, asset, closes):
    for i, c in enumerate(closes):
        buf.append(asset, Bar(o=c, h=c + 1, low=c - 1, c=c, v=1.0, ts=T0))


@pytest.mark.asyncio
async def test_rsi_interpreter_bullish_on_rising():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100 + i for i in range(20)])  # all gains → RSI ~100
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await RsiInterpreter(buf).interpret(obs)
    assert len(out) == 1 and out[0].source == "indicator.rsi" and out[0].direction > 0


@pytest.mark.asyncio
async def test_rsi_interpreter_silent_without_history():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100, 101, 102])
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    assert await RsiInterpreter(buf).interpret(obs) == []


# ── TrendAlignmentInterpreter ──
@pytest.mark.asyncio
async def test_alignment_bullish_when_both_horizons_up():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100 + i for i in range(50)])  # steadily up
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await TrendAlignmentInterpreter(buf).interpret(obs)
    assert len(out) == 1 and out[0].source == "indicator.alignment" and out[0].direction > 0


@pytest.mark.asyncio
async def test_alignment_silent_when_horizons_disagree():
    buf = BarBuffer()
    # long-window up overall, but last 10 bars down → mixed → no signal
    closes = [100 + i for i in range(40)] + [139 - i for i in range(10)]
    _fill(buf, "NVDA", closes)
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    assert await TrendAlignmentInterpreter(buf).interpret(obs) == []


# ── AnomalyInterpreter ──
@pytest.mark.asyncio
async def test_anomaly_flag_on_volatility_spike():
    buf = BarBuffer()
    calm = [100 + 0.1 * i for i in range(40)]          # low-vol drift
    spike = [104, 96, 108, 92, 110]                     # sudden chaos
    _fill(buf, "NVDA", calm + spike)
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await AnomalyInterpreter(buf).interpret(obs)
    assert len(out) == 1
    assert out[0].source == "anomaly"
    assert out[0].directional is False  # a flag, not a directional vote


@pytest.mark.asyncio
async def test_anomaly_silent_on_calm_tape():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100 + 0.1 * i for i in range(45)])  # steady, no spike
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    assert await AnomalyInterpreter(buf).interpret(obs) == []


# ── DriftInterpreter ──
@pytest.mark.asyncio
async def test_drift_flag_on_distribution_shift():
    buf = BarBuffer()
    # 50 bars of tiny calm drift, then a persistent strong-up regime (mean shift).
    calm = [100 + 0.05 * i for i in range(50)]
    shifted = [calm[-1] + 3.0 * (i + 1) for i in range(12)]
    _fill(buf, "NVDA", calm + shifted)
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await DriftInterpreter(buf).interpret(obs)
    assert len(out) == 1
    assert out[0].source == "drift"
    assert out[0].directional is False  # widens uncertainty, not a directional vote


@pytest.mark.asyncio
async def test_drift_silent_on_stable_distribution():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100 + i for i in range(70)])  # steady, constant-slope trend
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    assert await DriftInterpreter(buf).interpret(obs) == []


@pytest.mark.asyncio
async def test_drift_silent_without_history():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100, 101, 102])
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    assert await DriftInterpreter(buf).interpret(obs) == []


# ── MultiFrameInterpreter ──
@pytest.mark.asyncio
async def test_multiframe_bullish_when_emas_stacked_up():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100 + i for i in range(70)])  # clean uptrend → ef>em>es
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await MultiFrameInterpreter(buf).interpret(obs)
    assert len(out) == 1
    assert out[0].source == "indicator.multiframe"
    assert out[0].direction > 0


@pytest.mark.asyncio
async def test_multiframe_bearish_when_emas_stacked_down():
    buf = BarBuffer()
    _fill(buf, "NVDA", [170 - i for i in range(70)])  # clean downtrend
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await MultiFrameInterpreter(buf).interpret(obs)
    assert len(out) == 1 and out[0].direction < 0


@pytest.mark.asyncio
async def test_multiframe_silent_without_enough_history():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100 + i for i in range(20)])  # < slow(55) EMA window
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    assert await MultiFrameInterpreter(buf).interpret(obs) == []


# ── ForecasterInterpreter ──
@pytest.mark.asyncio
async def test_forecaster_projects_uptrend():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100 + 2 * i for i in range(30)])  # clean linear up → high R²
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await ForecasterInterpreter(buf).interpret(obs)
    assert len(out) == 1
    assert out[0].source == "forecaster"
    assert out[0].direction > 0
    assert 0.0 < out[0].weight <= 0.6


@pytest.mark.asyncio
async def test_forecaster_silent_on_noisy_series():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100, 90, 110, 88, 112, 91, 109, 95, 105, 92,
                        108, 94, 106, 93, 107, 96, 104, 97, 103, 99])  # low R²
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    assert await ForecasterInterpreter(buf).interpret(obs) == []


# ── VolumeConfirmationInterpreter ──
def _fill_vol(buf, asset, closes, vols):
    for c, v in zip(closes, vols):
        buf.append(asset, Bar(o=c, h=c + 1, low=c - 1, c=c, v=v, ts=T0))


@pytest.mark.asyncio
async def test_volume_confirms_a_move_on_elevated_volume():
    buf = BarBuffer()
    closes = [100 + i for i in range(25)]
    vols = [1000.0] * 22 + [3000.0, 3000.0, 3000.0]  # last bars on 3x volume
    _fill_vol(buf, "NVDA", closes, vols)
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await VolumeConfirmationInterpreter(buf).interpret(obs)
    assert len(out) == 1 and out[0].source == "indicator.volume" and out[0].direction > 0


@pytest.mark.asyncio
async def test_volume_silent_on_thin_volume():
    buf = BarBuffer()
    closes = [100 + i for i in range(25)]
    _fill_vol(buf, "NVDA", closes, [1000.0] * 25)  # flat volume → no confirmation
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    assert await VolumeConfirmationInterpreter(buf).interpret(obs) == []


# ── SupportResistanceInterpreter ──
@pytest.mark.asyncio
async def test_structure_bullish_near_support():
    buf = BarBuffer()
    # A dip to ~100 (swing low) then recovery; price ends just above that support.
    closes = [120, 115, 110, 105, 100, 105, 110, 108, 106, 104, 102, 101]
    _fill(buf, "NVDA", closes)
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    out = await SupportResistanceInterpreter(buf, lookback=1, proximity=0.03).interpret(obs)
    assert out and out[0].source == "indicator.structure" and out[0].direction > 0


@pytest.mark.asyncio
async def test_structure_silent_mid_range():
    buf = BarBuffer()
    _fill(buf, "NVDA", [100 + i for i in range(40)])  # steady climb, no nearby swing
    obs = new_event(CLOCK, EventType.OBSERVATION_BAR, source="alpaca", asset="NVDA")
    # Far from any swing low/high → no structural signal.
    out = await SupportResistanceInterpreter(buf, lookback=2, proximity=0.005).interpret(obs)
    assert out == []


# ── NewsLexiconInterpreter ──
@pytest.mark.asyncio
async def test_news_interpreter_emits_polarity_evidence():
    obs = new_event(
        CLOCK, EventType.OBSERVATION_NEWS, source="finnhub", asset="NVDA",
        payload={"lexicon_polarity": 0.8, "headline": "blowout earnings"},
    )
    out = await NewsLexiconInterpreter().interpret(obs)
    assert len(out) == 1
    assert out[0].source == "news"
    assert out[0].direction == 0.8


@pytest.mark.asyncio
async def test_news_interpreter_silent_when_lexicon_abstains():
    obs = new_event(
        CLOCK, EventType.OBSERVATION_NEWS, source="finnhub", asset="NVDA",
        payload={"lexicon_polarity": None, "headline": "ambiguous"},
    )
    assert await NewsLexiconInterpreter().interpret(obs) == []


# ── NewsLlmInterpreter (sparse: only when the lexicon abstained) ──
class _Scorer:
    def __init__(self, polarity, *, raises=False):
        self.polarity, self.raises, self.calls = polarity, raises, 0

    async def score(self, headline):
        self.calls += 1
        if self.raises:
            raise RuntimeError("llm down")
        return self.polarity


@pytest.mark.asyncio
async def test_news_llm_scores_when_lexicon_abstained():
    obs = new_event(
        CLOCK, EventType.OBSERVATION_NEWS, source="finnhub", asset="NVDA",
        payload={"lexicon_polarity": None, "headline": "surprise FDA approval"},
    )
    out = await NewsLlmInterpreter(_Scorer(0.7)).interpret(obs)
    assert len(out) == 1 and out[0].source == "news" and out[0].direction == 0.7


@pytest.mark.asyncio
async def test_news_llm_skips_when_lexicon_already_scored():
    scorer = _Scorer(0.7)
    obs = new_event(
        CLOCK, EventType.OBSERVATION_NEWS, source="finnhub", asset="NVDA",
        payload={"lexicon_polarity": 0.5, "headline": "x"},  # lexicon scored it
    )
    assert await NewsLlmInterpreter(scorer).interpret(obs) == []
    assert scorer.calls == 0  # LLM never spent (sparse)


@pytest.mark.asyncio
async def test_news_llm_silent_on_scorer_error():
    obs = new_event(
        CLOCK, EventType.OBSERVATION_NEWS, source="finnhub", asset="NVDA",
        payload={"lexicon_polarity": None, "headline": "x"},
    )
    assert await NewsLlmInterpreter(_Scorer(None, raises=True)).interpret(obs) == []  # INV-1
