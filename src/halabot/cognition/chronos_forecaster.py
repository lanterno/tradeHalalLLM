"""Chronos foundation-model forecaster (REARCHITECTURE L2, B1).

A drop-in upgrade for the cheap OLS ``ForecasterInterpreter``, behind the SAME
``source="forecaster"`` evidence seam: Amazon's Chronos (a pretrained
time-series transformer, HuggingFace) forecasts the next ``horizon`` bars'
prices as quantiles, and we vote the median's projected return with a weight
sized by the forecast's signal-to-noise (median move vs. the 10–90% predictive
band width). A wide/uncertain band → low weight; a tight, decisive forecast →
near-full weight.

Why a separate module: the ``torch``/``chronos`` imports (the ``[ml]`` extra)
are heavy and optional, so they live behind :func:`load_chronos_pipeline` and
the ``_ChronosAdapter`` — :class:`ChronosForecasterInterpreter` itself is
torch-free (it talks to a small :class:`ChronosForecaster` protocol), so it is
unit-testable without the extra and degrades gracefully when it is absent
(INV-1). Because it emits ``source="forecaster"``, it is mutually exclusive with
the OLS forecaster (one forecaster per engine; the config selects which).
"""

from __future__ import annotations

import logging
from typing import Protocol

from halabot.belief.schema import EvidenceItem
from halabot.cognition.bars import BarBuffer
from halabot.platform.events import Event, EventType

logger = logging.getLogger(__name__)

# Shared cap with the OLS forecaster so the two are interchangeable in weight.
_FORECASTER_MAX_WEIGHT = 0.6
_EPS = 1e-9
_ABSTAIN: tuple[float, float, float] = (0.0, 0.0, 0.0)  # hi<=lo → interpreter abstains


class ChronosForecaster(Protocol):
    """Minimal forecasting interface the interpreter depends on (torch-free seam).

    Returns the (10%, 50%, 90%) quantile PRICES at the forecast horizon's end."""

    def forecast(self, closes: list[float], *, horizon: int) -> tuple[float, float, float]: ...


class ChronosForecasterInterpreter:
    """Forecaster interpreter backed by a Chronos model. Emits a single
    ``source="forecaster"`` vote: direction = projected median return (scaled),
    weight = max_weight × signal-to-noise of the forecast. Abstains on too little
    history, a degenerate band, or a near-zero projected move."""

    consumes = frozenset({EventType.OBSERVATION_BAR})

    def __init__(
        self,
        buffer: BarBuffer,
        forecaster: ChronosForecaster,
        *,
        window: int = 64,
        horizon: int = 5,
        scale: float = 50.0,
        min_direction: float = 0.05,
    ) -> None:
        self._buffer = buffer
        self._forecaster = forecaster
        self._window = window
        self._horizon = horizon
        self._scale = scale
        self._min_direction = min_direction

    async def interpret(self, observation: Event) -> list[EvidenceItem]:
        asset = observation.asset
        if asset is None:
            return []
        closes = self._buffer.closes(asset)
        if len(closes) < self._window:
            return []
        last = closes[-1]
        if last <= 0:
            return []
        lo, mid, hi = self._forecaster.forecast(closes[-self._window :], horizon=self._horizon)
        if hi <= lo:  # degenerate / collapsed predictive band → no usable signal
            return []
        projected_ret = (mid - last) / last
        direction = max(-1.0, min(1.0, projected_ret * self._scale))
        if abs(direction) < self._min_direction:
            return []
        # Signal-to-noise: median move vs. the half-width of the 10–90% band, both
        # as fractions of price. A confident forecast (move >> band) → weight≈max.
        band = (hi - lo) / last
        snr = abs(projected_ret) / (0.5 * band + _EPS)
        weight = _FORECASTER_MAX_WEIGHT * max(0.0, min(1.0, snr))
        if weight <= 0.0:
            return []
        return [
            EvidenceItem(
                source="forecaster",
                direction=direction,
                weight=weight,
                detail=f"chronos {self._horizon}-bar proj {projected_ret:+.3%} snr={snr:.2f}",
                ts=observation.ts,
                event_id=observation.id,
            )
        ]


class _ChronosAdapter:
    """Wraps a loaded Chronos pipeline as a torch-free :class:`ChronosForecaster`.
    All torch lives here; the interpreter never sees it."""

    def __init__(self, pipeline: object, quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)):
        self._pipeline = pipeline
        self._quantiles = list(quantiles)

    def forecast(self, closes: list[float], *, horizon: int) -> tuple[float, float, float]:
        import torch

        ctx = torch.tensor(closes, dtype=torch.float32)
        q, _mean = self._pipeline.predict_quantiles(  # type: ignore[attr-defined]
            ctx, prediction_length=horizon, quantile_levels=self._quantiles
        )
        end = q[0, -1, :]  # quantiles at the horizon's last step
        return float(end[0]), float(end[1]), float(end[2])


class LazyChronosForecaster:
    """A :class:`ChronosForecaster` that loads the (heavy) torch/chronos model on
    the FIRST forecast, not at construction — so wiring it into ``build_engine``
    costs nothing until a real forecast is needed (keeps tests + the no-history
    warmup cheap, and never imports torch at module/app import time, INV-1). If
    the model can't load (``[ml]`` missing, no weights), it logs ONCE and abstains
    forever — the engine runs fine without the forecaster (graceful degradation),
    rather than crashing."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._impl: ChronosForecaster | None = None
        self._failed = False

    def forecast(self, closes: list[float], *, horizon: int) -> tuple[float, float, float]:
        if self._failed:
            return _ABSTAIN
        if self._impl is None:
            try:
                import torch
                from chronos import BaseChronosPipeline

                pipeline = BaseChronosPipeline.from_pretrained(
                    self._model_name, device_map=self._device, dtype=torch.float32
                )
                self._impl = _ChronosAdapter(pipeline)
                logger.info("Chronos forecaster loaded: %s", self._model_name)
            except Exception as exc:  # noqa: BLE001 — degrade, never crash the engine
                self._failed = True
                logger.warning(
                    "Chronos load failed (%s) — forecaster disabled, engine continues: %r",
                    self._model_name, exc,
                )
                return _ABSTAIN
        return self._impl.forecast(closes, horizon=horizon)


def load_chronos_pipeline(
    model_name: str = "amazon/chronos-bolt-small", device: str = "cpu"
) -> ChronosForecaster:
    """Return a lazily-loaded Chronos forecaster (model loads on first forecast,
    downloads + caches weights then). Never imports torch eagerly; degrades to a
    no-op forecaster if the [ml] extra/weights are unavailable."""
    return LazyChronosForecaster(model_name, device)
