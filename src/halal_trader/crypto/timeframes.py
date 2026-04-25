"""Multi-timeframe analysis — fetches higher-TF klines and computes trend alignment."""

from __future__ import annotations

import logging
import time
from typing import Any

from halal_trader.crypto.exchange import BinanceClient
from halal_trader.crypto.indicators import compute_all

logger = logging.getLogger(__name__)

_TIMEFRAMES = [
    ("5m", 300),
    ("15m", 900),
    ("1h", 3600),
    ("4h", 14400),
    ("1d", 86400),
]


class TimeframeAnalyzer:
    """Fetches and analyzes multiple timeframes for trend alignment."""

    def __init__(self, broker: BinanceClient) -> None:
        self._broker = broker
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    async def analyze(self, pairs: list[str]) -> dict[str, dict[str, Any]]:
        """Compute multi-timeframe indicators for all pairs.

        Returns dict[pair, {alignment_score, per_tf_summary, support_resistance}].
        """
        results: dict[str, dict[str, Any]] = {}

        for pair in pairs:
            try:
                tf_data = await self._fetch_all_timeframes(pair)
                alignment = self._compute_alignment(tf_data)
                sr_levels = self._compute_support_resistance(tf_data)

                results[pair] = {
                    "alignment_score": alignment,
                    "per_tf": {tf: self._summarize_tf(ind) for tf, ind in tf_data.items()},
                    "support_resistance": sr_levels,
                }
            except Exception as e:
                logger.debug("Multi-timeframe analysis failed for %s: %s", pair, e)

        return results

    async def _fetch_all_timeframes(self, pair: str) -> dict[str, dict[str, Any]]:
        """Fetch klines and compute indicators for each higher timeframe."""
        tf_indicators: dict[str, dict[str, Any]] = {}

        for interval, ttl in _TIMEFRAMES:
            cache_key = f"{pair}:{interval}"
            now = time.monotonic()

            cached = self._cache.get(cache_key)
            if cached and (now - cached[0]) < ttl:
                tf_indicators[interval] = cached[1]
                continue

            try:
                klines = await self._broker.get_klines(pair, interval=interval, limit=100)
                if len(klines) >= 20:
                    indicators = compute_all(klines)
                    tf_indicators[interval] = indicators
                    self._cache[cache_key] = (now, indicators)
            except Exception as e:
                logger.debug("Failed to get %s klines for %s: %s", interval, pair, e)

        return tf_indicators

    def _compute_alignment(self, tf_data: dict[str, dict[str, Any]]) -> float:
        """Compute trend alignment score from -1 (all bearish) to +1 (all bullish)."""
        if not tf_data:
            return 0.0

        scores = []
        for _tf, ind in tf_data.items():
            if "error" in ind:
                continue
            tf_score = 0.0
            count = 0

            # EMA alignment: EMA9 > EMA21 = bullish
            ema9 = ind.get("ema_9")
            ema21 = ind.get("ema_21")
            if ema9 is not None and ema21 is not None:
                tf_score += 1.0 if ema9 > ema21 else -1.0
                count += 1

            # MACD: positive histogram = bullish
            macd_hist = ind.get("macd_histogram")
            if macd_hist is not None:
                tf_score += 1.0 if macd_hist > 0 else -1.0
                count += 1

            # Price vs EMA50: above = bullish
            price = ind.get("current_price")
            ema50 = ind.get("ema_50")
            if price is not None and ema50 is not None:
                tf_score += 1.0 if price > ema50 else -1.0
                count += 1

            if count > 0:
                scores.append(tf_score / count)

        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _compute_support_resistance(
        self, tf_data: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract key support/resistance levels from higher timeframes."""
        levels: list[dict[str, Any]] = []

        for tf, ind in tf_data.items():
            if "error" in ind:
                continue

            if "bb_upper" in ind:
                levels.append(
                    {
                        "level": ind["bb_upper"],
                        "type": "resistance",
                        "source": f"BB upper ({tf})",
                    }
                )
                levels.append(
                    {
                        "level": ind["bb_lower"],
                        "type": "support",
                        "source": f"BB lower ({tf})",
                    }
                )

            if "vwap" in ind:
                levels.append({"level": ind["vwap"], "type": "pivot", "source": f"VWAP ({tf})"})

            if "ema_50" in ind:
                levels.append(
                    {
                        "level": ind["ema_50"],
                        "type": "dynamic",
                        "source": f"EMA50 ({tf})",
                    }
                )

        levels.sort(key=lambda x: x["level"])
        return levels

    def _summarize_tf(self, indicators: dict[str, Any]) -> str:
        """One-line summary of a timeframe's indicators."""
        if "error" in indicators:
            return "insufficient data"

        parts = []
        rsi = indicators.get("rsi_14")
        if rsi is not None:
            parts.append(f"RSI={rsi:.0f}")

        macd_hist = indicators.get("macd_histogram")
        if macd_hist is not None:
            parts.append("MACD=" + ("bullish" if macd_hist > 0 else "bearish"))

        ema9 = indicators.get("ema_9")
        ema21 = indicators.get("ema_21")
        if ema9 is not None and ema21 is not None:
            parts.append("EMA=" + ("bull cross" if ema9 > ema21 else "bear cross"))

        return ", ".join(parts) if parts else "N/A"


def format_timeframes_for_prompt(tf_results: dict[str, dict[str, Any]]) -> str:
    """Format multi-timeframe analysis into a text block for the LLM prompt."""
    if not tf_results:
        return "No multi-timeframe data available."

    lines = []
    for pair, data in sorted(tf_results.items()):
        alignment = data.get("alignment_score", 0)
        direction = "BULLISH" if alignment > 0.3 else ("BEARISH" if alignment < -0.3 else "MIXED")
        lines.append(f"  {pair}: Trend Alignment={alignment:+.2f} ({direction})")

        per_tf = data.get("per_tf", {})
        for tf, summary in per_tf.items():
            lines.append(f"    {tf}: {summary}")

        sr = data.get("support_resistance", [])
        if sr:
            nearest = sr[:3] + sr[-3:] if len(sr) > 6 else sr
            sr_text = ", ".join(f"${lv['level']:,.2f} ({lv['source']})" for lv in nearest[:4])
            lines.append(f"    Key levels: {sr_text}")

    return "\n".join(lines)
