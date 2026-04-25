"""Price forecaster — wraps HuggingFace time-series models (Chronos-T5)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from halal_trader.ml.hub import ModelHub

logger = logging.getLogger(__name__)

_CHRONOS_MODEL = "amazon/chronos-t5-small"


@dataclass
class PriceForecast:
    """Probabilistic price forecast for the next N candles."""

    pair: str
    current_price: float
    predicted_prices: list[float] = field(default_factory=list)
    upper_bound: list[float] = field(default_factory=list)
    lower_bound: list[float] = field(default_factory=list)
    confidence: float = 0.0
    horizon: int = 5


class PriceForecaster:
    """Uses Chronos-T5 to forecast future price movements."""

    def __init__(self, hub: ModelHub) -> None:
        self._hub = hub
        self._pipeline = None
        self._load_attempted = False

    def _ensure_loaded(self) -> bool:
        """Lazily load the Chronos model."""
        if self._pipeline is not None:
            return True
        if self._load_attempted:
            return False

        self._load_attempted = True
        try:
            import torch
            from chronos import ChronosPipeline

            self._pipeline = ChronosPipeline.from_pretrained(
                _CHRONOS_MODEL,
                device_map=self._hub.device,
                torch_dtype=torch.float32,
            )
            self._hub.register("chronos", self._pipeline)
            logger.info("Chronos-T5 price forecaster loaded")
            return True
        except ImportError:
            logger.info("chronos-forecasting not installed — price forecasting disabled")
            return False
        except Exception as e:
            logger.warning("Failed to load Chronos model: %s", e)
            return False

    def forecast(self, pair: str, closes: list[float], horizon: int = 5) -> PriceForecast | None:
        """Forecast next N candle prices from historical closes."""
        if not self._ensure_loaded() or len(closes) < 20:
            return None

        try:
            import torch

            context = torch.tensor(closes[-100:], dtype=torch.float32)
            forecast = self._pipeline.predict(context.unsqueeze(0), horizon)

            # forecast shape: (1, num_samples, horizon)
            samples = forecast.numpy()[0]
            median = np.median(samples, axis=0).tolist()
            upper = np.percentile(samples, 90, axis=0).tolist()
            lower = np.percentile(samples, 10, axis=0).tolist()

            med_mean = np.mean(median)
            spread = (np.mean(upper) - np.mean(lower)) / med_mean if med_mean > 0 else 1.0
            confidence = max(0.0, min(1.0, 1.0 - spread * 5))

            return PriceForecast(
                pair=pair,
                current_price=closes[-1],
                predicted_prices=median,
                upper_bound=upper,
                lower_bound=lower,
                confidence=confidence,
                horizon=horizon,
            )
        except Exception as e:
            logger.warning("Price forecast failed for %s: %s", pair, e)
            return None


def format_forecasts_for_prompt(forecasts: dict[str, PriceForecast]) -> str:
    """Format price forecasts into a text block for the LLM prompt."""
    if not forecasts:
        return "No ML price forecasts available."

    lines = []
    for pair, fc in sorted(forecasts.items()):
        if not fc.predicted_prices:
            continue

        direction = "UP" if fc.predicted_prices[-1] > fc.current_price else "DOWN"
        change_pct = (
            (fc.predicted_prices[-1] - fc.current_price) / fc.current_price * 100
            if fc.current_price > 0
            else 0
        )

        lines.append(
            f"  {pair}: ML predicts {direction} {abs(change_pct):.2f}% in {fc.horizon} candles "
            f"(confidence: {fc.confidence:.0%}), "
            f"range: ${fc.lower_bound[-1]:,.2f} - ${fc.upper_bound[-1]:,.2f}"
        )

    return "\n".join(lines) if lines else "No ML price forecasts available."
