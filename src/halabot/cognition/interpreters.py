"""Cheap, continuous interpreters (REARCHITECTURE L2) — all LLM-free.

Bar-driven (momentum, RSI, multi-horizon trend alignment) read the rolling
buffer the router fills; the news interpreter uses the cheap lexicon polarity
perception attached. Multiple bar interpreters from the same series are
correlated, so they don't add *independent* information — but they raise
conviction mass when they agree and, via the agreement/dispersion penalty,
*dampen* it when they disagree (e.g. price momentum up but RSI rolling over).
The fitted calibrator (L8) learns each source's true weight once outcomes exist.
"""

from __future__ import annotations

import logging
import statistics
from typing import Protocol

from halabot.belief.schema import EvidenceItem
from halabot.cognition.bars import BarBuffer, ema, momentum_signal, returns, rsi, swing_points
from halabot.platform.events import Event, EventType

logger = logging.getLogger(__name__)

# News evidence decays fast; one headline shouldn't dominate for long. The
# decay half-life is global, so we encode "freshness" via a modest base weight.
_NEWS_WEIGHT = 0.8
# Secondary confirmations carry less weight than the primary trend signal.
_RSI_WEIGHT = 0.5
_ALIGN_WEIGHT = 0.6
# The multi-EMA stack is a strong, low-noise structural confirmation.
_MULTIFRAME_WEIGHT = 0.7
# The forecaster's vote is sized by its own fit confidence (R²); this caps it.
_FORECASTER_MAX_WEIGHT = 0.6
# A volume-confirmed move and a structural (support/resistance) read.
_VOLUME_WEIGHT = 0.5
_STRUCTURE_WEIGHT = 0.45


class IndicatorInterpreter:
    """Emits a trend-momentum :class:`EvidenceItem` from the bar buffer."""

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(self, buffer: BarBuffer) -> None:
        self._buffer = buffer

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        direction, weight = momentum_signal(self._buffer.closes(asset))
        if weight <= 0.0:
            return []  # insufficient history — neutral, weightless
        return [
            EvidenceItem(
                source="indicator.momentum",
                direction=direction,
                weight=weight,
                detail=f"fast/slow EMA momentum {direction:+.2f}",
                ts=observation.ts,
                event_id=observation.id,
            )
        ]


class RsiInterpreter:
    """Emits an RSI momentum-confirmation signal (RSI>50 bullish, <50 bearish)."""

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(self, buffer: BarBuffer, *, period: int = 14) -> None:
        self._buffer = buffer
        self._period = period

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        r = rsi(self._buffer.closes(asset), period=self._period)
        if r is None:
            return []
        direction = max(-1.0, min(1.0, (r - 50.0) / 25.0))  # 50→0, 75→+1, 25→-1
        if abs(direction) < 0.04:
            return []  # ~neutral — no signal
        return [
            EvidenceItem(
                source="indicator.rsi",
                direction=direction,
                weight=_RSI_WEIGHT,
                detail=f"RSI {r:.0f}",
                ts=observation.ts,
                event_id=observation.id,
            )
        ]


class TrendAlignmentInterpreter:
    """Multi-horizon alignment: short- AND long-window returns agreeing is a
    stronger directional signal than either alone; mixed → no signal."""

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(self, buffer: BarBuffer, *, short: int = 10, long: int = 40) -> None:
        self._buffer = buffer
        self._short = short
        self._long = long

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        closes = self._buffer.closes(asset)
        if len(closes) < self._long or closes[-self._short] <= 0 or closes[-self._long] <= 0:
            return []
        short_ret = (closes[-1] - closes[-self._short]) / closes[-self._short]
        long_ret = (closes[-1] - closes[-self._long]) / closes[-self._long]
        if short_ret > 0 and long_ret > 0:
            direction = 0.8
        elif short_ret < 0 and long_ret < 0:
            direction = -0.8
        else:
            return []  # horizons disagree → no alignment signal
        return [
            EvidenceItem(
                source="indicator.alignment",
                direction=direction,
                weight=_ALIGN_WEIGHT,
                detail=f"short {short_ret:+.2%} / long {long_ret:+.2%}",
                ts=observation.ts,
                event_id=observation.id,
            )
        ]


class AnomalyInterpreter:
    """Emits a non-directional ``anomaly`` flag when short-window return
    volatility spikes above its baseline — wiring up conviction_raw's anomaly
    down-weight (×0.6), so the engine trusts a signal less in chaotic tape."""

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(
        self, buffer: BarBuffer, *, short: int = 5, baseline: int = 30, mult: float = 2.0
    ) -> None:
        self._buffer = buffer
        self._short = short
        self._baseline = baseline
        self._mult = mult

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        closes = self._buffer.closes(asset)
        if len(closes) < self._baseline + 1:
            return []
        rets = returns(closes)
        if len(rets) < self._baseline:
            return []
        short_vol = statistics.pstdev(rets[-self._short :])
        base_vol = statistics.pstdev(rets[-self._baseline :])
        if base_vol <= 0 or short_vol <= self._mult * base_vol:
            return []
        return [
            EvidenceItem(
                source="anomaly",
                direction=0.0,
                weight=1.0,
                detail=f"vol spike {short_vol / base_vol:.1f}x baseline",
                ts=observation.ts,
                event_id=observation.id,
                directional=False,  # a flag, not a directional vote
            )
        ]


class DriftInterpreter:
    """Concept-drift detector: a non-directional ``drift`` flag when the recent
    return *distribution* shifts away from its baseline (a persistent mean shift,
    not just a one-off vol spike — that's :class:`AnomalyInterpreter`).

    Wires up ``conviction_raw``'s drift down-weight (×0.7): when the process the
    indicators were trained on has changed, the engine widens its uncertainty
    and trusts the directional signal less until the new regime is established.
    """

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(
        self, buffer: BarBuffer, *, recent: int = 10, baseline: int = 50, z_threshold: float = 2.0
    ) -> None:
        self._buffer = buffer
        self._recent = recent
        self._baseline = baseline
        self._z = z_threshold

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        closes = self._buffer.closes(asset)
        if len(closes) < self._baseline + 1:
            return []
        rets = returns(closes)
        if len(rets) < self._baseline:
            return []
        base = rets[-self._baseline : -self._recent]
        recent = rets[-self._recent :]
        if len(base) < 2 or len(recent) < 2:
            return []
        base_mean = statistics.fmean(base)
        base_std = statistics.pstdev(base)
        if base_std <= 0:
            return []
        z = abs(statistics.fmean(recent) - base_mean) / base_std
        if z <= self._z:
            return []  # distribution stable → no drift
        return [
            EvidenceItem(
                source="drift",
                direction=0.0,
                weight=1.0,
                detail=f"return-distribution shift z={z:.1f}",
                ts=observation.ts,
                event_id=observation.id,
                directional=False,  # a flag, not a directional vote
            )
        ]


class MultiFrameInterpreter:
    """Multi-timeframe EMA-stack alignment: short > medium > long EMAs (all
    rising in order) is a clean, low-noise uptrend confirmation across horizons;
    the inverse is a downtrend. A non-stacked (tangled) EMA set → no signal.

    Stronger and less whippy than the two-window :class:`TrendAlignmentInterpreter`
    because it requires *ordered* separation across three horizons at once."""

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(
        self, buffer: BarBuffer, *, fast: int = 8, medium: int = 21, slow: int = 55
    ) -> None:
        self._fast = fast
        self._medium = medium
        self._slow = slow
        self._buffer = buffer

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        closes = self._buffer.closes(asset)
        ef, em, es = (ema(closes, self._fast), ema(closes, self._medium), ema(closes, self._slow))
        if ef is None or em is None or es is None or es == 0:
            return []
        if ef > em > es:
            direction = 0.85
        elif ef < em < es:
            direction = -0.85
        else:
            return []  # tangled stack → no clean multi-frame trend
        # Scale weight by how separated the stack is (relative fast–slow gap),
        # so a barely-ordered stack votes less than a strongly fanned one.
        sep = min(1.0, abs(ef - es) / abs(es) * 20.0)
        weight = _MULTIFRAME_WEIGHT * max(0.3, sep)
        return [
            EvidenceItem(
                source="indicator.multiframe",
                direction=direction,
                weight=weight,
                detail=f"EMA stack {self._fast}/{self._medium}/{self._slow} sep={sep:.2f}",
                ts=observation.ts,
                event_id=observation.id,
            )
        ]


class ForecasterInterpreter:
    """Cheap, deterministic forward projection: least-squares trend slope over a
    recent window → expected next-bar return, voted with weight = fit quality
    (R²). LLM- and ML-extra-free (INV-1) — a low-cost structural forecaster that
    a richer model (Chronos, the ``[ml]`` extra) can later replace behind the
    same interpreter seam. A flat/noisy series (low R²) abstains."""

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(
        self, buffer: BarBuffer, *, window: int = 20, min_r2: float = 0.3
    ) -> None:
        self._buffer = buffer
        self._window = window
        self._min_r2 = min_r2

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        closes = self._buffer.closes(asset)
        if len(closes) < self._window:
            return []
        ys = closes[-self._window :]
        slope, r2 = _ols_slope_r2(ys)
        mean_y = statistics.fmean(ys)
        if slope is None or mean_y <= 0 or r2 < self._min_r2:
            return []  # no reliable trend to project
        # Projected one-bar return as a fraction of price; scale to a [-1, 1] vote.
        projected_ret = slope / mean_y
        direction = max(-1.0, min(1.0, projected_ret * 100.0))
        if abs(direction) < 0.05:
            return []
        return [
            EvidenceItem(
                source="forecaster",
                direction=direction,
                weight=_FORECASTER_MAX_WEIGHT * r2,
                detail=f"OLS slope proj {projected_ret:+.3%}/bar R²={r2:.2f}",
                ts=observation.ts,
                event_id=observation.id,
            )
        ]


def _ols_slope_r2(ys: list[float]) -> tuple[float | None, float]:
    """Least-squares slope of ``ys`` over x=0..n-1, plus the fit R² ∈ [0, 1].

    Returns ``(None, 0.0)`` on a degenerate (zero-variance) series."""
    n = len(ys)
    if n < 2:
        return None, 0.0
    xs = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None, 0.0
    slope = sxy / sxx
    r2 = (sxy * sxy) / (sxx * syy)  # coefficient of determination for a line
    return slope, max(0.0, min(1.0, r2))


class VolumeConfirmationInterpreter:
    """Volume-confirmed move (B3): a recent price move BACKED by above-average
    volume is stronger evidence than the same move on thin volume — participation
    confirms conviction. The engine otherwise ignores volume entirely. Abstains
    when volume isn't elevated or there's no directional move (adds no noise)."""

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(
        self, buffer: BarBuffer, *, short: int = 3, baseline: int = 20, min_ratio: float = 1.3
    ) -> None:
        self._buffer = buffer
        self._short = short
        self._baseline = baseline
        self._min_ratio = min_ratio

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        bars = self._buffer.bars(asset)
        if len(bars) < self._baseline + 1:
            return []
        closes = [b.c for b in bars]
        vols = [b.v for b in bars]
        recent_vol = statistics.fmean(vols[-self._short :])
        base_vol = statistics.fmean(vols[-self._baseline :])
        if base_vol <= 0 or closes[-self._short] <= 0:
            return []
        ratio = recent_vol / base_vol
        short_ret = (closes[-1] - closes[-self._short]) / closes[-self._short]
        if ratio < self._min_ratio or abs(short_ret) < 0.001:
            return []  # no volume confirmation, or no move to confirm
        direction = max(-1.0, min(1.0, short_ret * 50.0))
        # More excess volume → more weight (ratio 1.3→~0.15, 2.0→~0.5, capped).
        weight = _VOLUME_WEIGHT * min(1.0, ratio - 1.0)
        if weight <= 0.0:
            return []
        return [
            EvidenceItem(
                source="indicator.volume",
                direction=direction,
                weight=weight,
                detail=f"vol {ratio:.1f}x conf move {short_ret:+.2%}",
                ts=observation.ts,
                event_id=observation.id,
            )
        ]


class SupportResistanceInterpreter:
    """Structural read (B3): where price sits vs recent swing support/resistance.
    Near a recent swing LOW (a support shelf) is a favorable long entry zone →
    mild bullish; pressing into a recent swing HIGH (resistance) risks rejection →
    mild bearish. Uses the same swing detection as the level engine. The engine
    otherwise has no notion of structure in its conviction."""

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(
        self, buffer: BarBuffer, *, lookback: int = 2, proximity: float = 0.02, window: int = 60
    ) -> None:
        self._buffer = buffer
        self._lookback = lookback
        self._proximity = proximity
        self._window = window

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        highs = self._buffer.highs(asset)[-self._window :]
        lows = self._buffer.lows(asset)[-self._window :]
        closes = self._buffer.closes(asset)
        if len(highs) < 2 * self._lookback + 1 or not closes:
            return []
        swing_highs, swing_lows = swing_points(highs, lows, self._lookback)
        price = closes[-1]
        if price <= 0:
            return []
        # Nearest support below price, nearest resistance above price.
        support = max((s for s in swing_lows if s <= price), default=None)
        resistance = min((r for r in swing_highs if r >= price), default=None)
        near_support = support is not None and (price - support) / price <= self._proximity
        near_resistance = (
            resistance is not None and (resistance - price) / price <= self._proximity
        )
        if near_support and not near_resistance:
            direction = 0.6  # bounce zone — favorable long entry
            detail = f"near support {support:.2f}"
        elif near_resistance and not near_support:
            direction = -0.6  # pressing into resistance — rejection risk
            detail = f"near resistance {resistance:.2f}"
        else:
            return []  # mid-range or squeezed between both → no structural signal
        return [
            EvidenceItem(
                source="indicator.structure",
                direction=direction,
                weight=_STRUCTURE_WEIGHT,
                detail=detail,
                ts=observation.ts,
                event_id=observation.id,
            )
        ]


class NewsLexiconInterpreter:
    """Emits polarity evidence from the cheap lexicon score on a news observation."""

    consumes = frozenset({EventType.OBSERVATION_NEWS})

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        polarity = observation.payload.get("lexicon_polarity")
        if asset is None or polarity is None:
            return []  # lexicon abstained — no evidence (LLM scoring is the sparse path)
        direction = max(-1.0, min(1.0, float(polarity)))
        if direction == 0.0:
            return []
        headline = str(observation.payload.get("headline", ""))[:80]
        return [
            EvidenceItem(
                source="news",
                direction=direction,
                weight=_NEWS_WEIGHT,
                detail=f"news polarity {direction:+.2f}: {headline}",
                ts=observation.ts,
                event_id=observation.id,
            )
        ]


class HeadlineScorer(Protocol):
    """Scores a headline to a directional polarity in [-1, 1], or None to abstain."""

    async def score(self, headline: str) -> float | None: ...


class NewsLlmInterpreter:
    """Sparse LLM news scoring — fires ONLY when the cheap lexicon abstained
    (``lexicon_polarity`` is None), so the LLM is spent only on the headlines the
    free path couldn't read. LLM-down or a scorer error yields no evidence (INV-1):
    perception already recorded "we saw news"; the directional read is best-effort."""

    consumes = frozenset({EventType.OBSERVATION_NEWS})

    def __init__(self, scorer: HeadlineScorer) -> None:
        self._scorer = scorer

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None or observation.payload.get("lexicon_polarity") is not None:
            return []  # lexicon already scored it (or no asset) → don't spend the LLM
        headline = str(observation.payload.get("headline", "")).strip()
        if not headline:
            return []
        try:
            polarity = await self._scorer.score(headline)
        except Exception as exc:  # noqa: BLE001 — an LLM hiccup yields no evidence (INV-1)
            logger.warning("news LLM scoring failed: %r", exc)
            return []
        if polarity is None:
            return []
        direction = max(-1.0, min(1.0, float(polarity)))
        if direction == 0.0:
            return []
        return [
            EvidenceItem(
                source="news",
                direction=direction,
                weight=_NEWS_WEIGHT,
                detail=f"news(llm) {direction:+.2f}: {headline[:80]}",
                ts=observation.ts,
                event_id=observation.id,
            )
        ]
