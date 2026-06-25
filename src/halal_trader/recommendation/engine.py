"""Daily halal "stock of the day" recommendation engine.

ADVISORY ONLY. The engine assembles a compact technical snapshot of the
curated AAOIFI halal universe, asks the LLM to pick the single most
promising buy for the day, and persists the pick. It NEVER places an
order or touches the execution path — it is purely a research surface
for the dashboard / CLI / API.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from halal_trader.config import Settings
from halal_trader.core.llm.base import BaseLLM
from halal_trader.core.llm.factory import create_llm
from halal_trader.core.llm.prompts import register as _register_prompt
from halal_trader.crypto.indicators import compute_all
from halal_trader.db.repository import Repository
from halal_trader.domain.ports import Broker
from halal_trader.halal.cache import DEFAULT_HALAL_SYMBOLS
from halal_trader.trading.bars import bars_to_klines

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")

SYSTEM_PROMPT = """\
You are a disciplined halal (Shariah-compliant) equity analyst for a
single-user paper-trading research tool. Each day you pick the SINGLE most
promising stock to BUY from a pre-screened, AAOIFI Shariah-compliant universe.

Rules:
- Every candidate is ALREADY halal-screened — do not reject on compliance.
  In `halal_note`, briefly affirm why a long equity position in your pick is
  Shariah-permissible (a real productive business, not interest/gambling/
  prohibited sectors, within AAOIFI financial-ratio limits).
- Pick exactly ONE symbol FROM THE PROVIDED UNIVERSE. Never invent a symbol.
- "Most promising" = best risk/reward for a swing buy given the momentum,
  trend and technical posture in the data. Favour constructive structure
  (above key EMAs, healthy RSI, positive/improving MACD, not over-extended at
  the top of its Bollinger band) over chasing a parabolic move.
- Be decisive but honest: give the thesis, the key drivers/catalysts if
  evident, and the main risks.
- suggested_stop MUST be strictly BELOW suggested_entry, and suggested_target
  strictly ABOVE it. Base the levels on the price and ATR shown.
- conviction is a float 0..1.

Return ONLY a JSON object with EXACTLY this shape (no prose, no markdown):
{"symbol": "TICKER", "conviction": 0.0, "thesis": "...", "halal_note": "...",
 "suggested_entry": 0.0, "suggested_target": 0.0, "suggested_stop": 0.0,
 "catalysts": "...", "risks": "..."}
"""

PROMPT_VERSION = _register_prompt("recommendation.daily.system", SYSTEM_PROMPT)

USER_PROMPT_TEMPLATE = """\
Date: {date} (US/Eastern).

Candidate universe — {n} AAOIFI Shariah-compliant large-caps. Metrics are from
daily bars (≈60 trading days): chg5d/chg20d = % change over the last 5/20 daily
bars; rsi = RSI-14; macd_h = MACD histogram; bb = position in the Bollinger band
(0 low .. 1 high); vol = volume vs 20-day average; atr = ATR-14; adx = ADX-14;
from_hi = % below the 60-day high.

{table}

Pick the single most promising BUY for today and return the JSON object."""


class DailyRecommendationEngine:
    """Builds the daily halal recommendation. Advisory — never trades."""

    def __init__(
        self,
        *,
        broker: Broker,
        repo: Repository,
        settings: Settings,
        llm: BaseLLM | None = None,
        universe: list[str] | None = None,
    ) -> None:
        self._broker = broker
        self._repo = repo
        self._settings = settings
        self._llm = llm or create_llm(settings)
        # Curated AAOIFI list, NOT the (randomised in sandbox) Zoya screener —
        # so "most promising" spans a real, stable opportunity set.
        self._universe = universe or list(DEFAULT_HALAL_SYMBOLS)

    async def generate(self) -> dict[str, Any]:
        """Assemble candidate data, pick the best, persist and return it."""
        candidates = await self._build_candidates()
        if not candidates:
            raise RuntimeError("no candidate market data available")
        pick = await self._pick(candidates)
        date = datetime.now(_ET).strftime("%Y-%m-%d")
        rec: dict[str, Any] = {
            "date": date,
            "symbol": pick["symbol"],
            "conviction": pick["conviction"],
            "thesis": pick["thesis"],
            "halal_note": pick["halal_note"],
            "suggested_entry": pick.get("suggested_entry"),
            "suggested_target": pick.get("suggested_target"),
            "suggested_stop": pick.get("suggested_stop"),
            "catalysts": pick.get("catalysts"),
            "risks": pick.get("risks"),
            "universe_size": len(candidates),
            "model": getattr(self._llm, "model", None),
            "prompt_version": PROMPT_VERSION.short,
            "candidates": candidates,
        }
        rec["id"] = await self._repo.save_recommendation(rec)
        logger.info(
            "Daily halal recommendation: %s (conviction %.2f) from %d candidates — %s",
            rec["symbol"],
            rec["conviction"],
            rec["universe_size"],
            rec["thesis"][:120],
        )
        return rec

    async def _build_candidates(self) -> dict[str, dict[str, Any]]:
        """Per-symbol compact technical summary for the LLM context."""
        out: dict[str, dict[str, Any]] = {}
        for sym in self._universe:
            try:
                bars = await self._broker.get_stock_bars(
                    sym, days=60, timeframe="1Day"
                )
            except Exception as exc:  # noqa: BLE001 — skip a flaky symbol, keep the rest
                logger.debug("recommendation: bars fetch failed for %s: %s", sym, exc)
                continue
            klines = bars_to_klines(bars)
            if len(klines) < 20:
                continue
            ind = compute_all(klines)
            if "error" in ind:
                continue
            closes = [k.close for k in klines]
            highs = [k.high for k in klines]
            last = closes[-1]
            summary = {
                "price": round(last, 2),
                "chg5d": round((last / closes[-6] - 1) * 100, 2)
                if len(closes) >= 6
                else None,
                "chg20d": round((last / closes[-21] - 1) * 100, 2)
                if len(closes) >= 21
                else None,
                "rsi": ind.get("rsi_14"),
                "macd_h": ind.get("macd_histogram"),
                "bb": ind.get("bb_position"),
                "vol": ind.get("volume_ratio"),
                "atr": ind.get("atr_14"),
                "adx": ind.get("adx_14"),
                "from_hi": round((last / max(highs) - 1) * 100, 2) if highs else None,
            }
            out[sym] = summary
        return out

    def _format_table(self, candidates: dict[str, dict[str, Any]]) -> str:
        def _f(v: Any) -> str:
            return "n/a" if v is None else (f"{v:g}" if isinstance(v, int | float) else str(v))

        lines = []
        for sym, c in candidates.items():
            lines.append(
                f"{sym:6} ${_f(c['price'])}  chg5d={_f(c['chg5d'])}% "
                f"chg20d={_f(c['chg20d'])}% rsi={_f(c['rsi'])} "
                f"macd_h={_f(c['macd_h'])} bb={_f(c['bb'])} vol={_f(c['vol'])}x "
                f"atr={_f(c['atr'])} adx={_f(c['adx'])} from_hi={_f(c['from_hi'])}%"
            )
        return "\n".join(lines)

    def _apply_factors(self, candidates: dict[str, dict[str, Any]]) -> str:
        """Cross-sectional factor rank: merge each composite into the candidate
        (stored + shown) and return a "factor leaders" block for the prompt."""
        from halal_trader.core.factors import rank_factors

        ranked = rank_factors(candidates)
        for fs in ranked:
            candidates[fs.symbol]["factor_score"] = fs.composite
        top = ranked[:5]
        lines = [
            f"  {fs.symbol}: composite={fs.composite:+.2f} "
            f"(mom {fs.momentum:+.2f}, lowvol {fs.low_vol:+.2f}, "
            f"trend {fs.trend_quality:+.2f})"
            for fs in top
        ]
        return (
            "Cross-sectional factor leaders (z-scored momentum + low-vol + "
            "trend-quality across the universe; higher = stronger long tilt):\n"
            + "\n".join(lines)
        )

    async def _pick(self, candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
        date = datetime.now(_ET).strftime("%Y-%m-%d")
        factor_block = self._apply_factors(candidates)
        user = USER_PROMPT_TEMPLATE.format(
            date=date, n=len(candidates), table=self._format_table(candidates)
        )
        user += "\n\n" + factor_block
        raw = await self._llm.generate_json(user, system=SYSTEM_PROMPT)
        return self._validate(raw, candidates)

    def _validate(
        self, raw: dict[str, Any], candidates: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        symbol = str(raw.get("symbol", "")).upper().strip()
        if symbol not in candidates:
            raise ValueError(
                f"LLM picked {symbol!r} which is not in the candidate universe"
            )
        price = candidates[symbol].get("price")
        entry = _as_float(raw.get("suggested_entry")) or price
        target = _as_float(raw.get("suggested_target"))
        stop = _as_float(raw.get("suggested_stop"))
        # Enforce the long invariant: stop < entry < target (drop a bad level).
        if entry and stop is not None and stop >= entry:
            stop = round(entry * 0.95, 2)
        if entry and target is not None and target <= entry:
            target = round(entry * 1.08, 2)
        conviction = _as_float(raw.get("conviction")) or 0.0
        return {
            "symbol": symbol,
            "conviction": max(0.0, min(1.0, conviction)),
            "thesis": str(raw.get("thesis", "")).strip(),
            "halal_note": str(raw.get("halal_note", "")).strip(),
            "suggested_entry": entry,
            "suggested_target": target,
            "suggested_stop": stop,
            "catalysts": (str(raw["catalysts"]).strip() if raw.get("catalysts") else None),
            "risks": (str(raw["risks"]).strip() if raw.get("risks") else None),
        }


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
