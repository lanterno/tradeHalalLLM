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
daily bars: chg5d/chg20d = % change over the last 5/20 daily bars; rsi =
RSI-14; macd_h = MACD histogram; bb = position in the Bollinger band (0 low ..
1 high); vol = volume vs 20-day average; atr = ATR-14; adx = ADX-14; from_hi =
% below the recent high. Quantitative range model (Yang-Zhang/HAR volatility):
band5d = statistical 5-day low..high price band ({band_semantics}); rng1d =
expected 1-day trading range as % of price; vpct = current volatility
percentile vs the symbol's own history (0 calm .. 1 extreme). impl (when
present) = the OPTIONS-implied expected move ±%/DTE-days to the nearest
weekly expiry with the implied low..high band — the market's own range
forecast. When impl is materially WIDER than the statistical band, the
options market is pricing an event (often earnings) inside that window:
treat the extra width as event risk, not opportunity.

{table}

Ground suggested_target and suggested_stop in the range model: a
suggested_target beyond band5d's high needs an explicit catalyst in the
thesis, and the stop must sit wide enough that ordinary 1-day noise (rng1d)
cannot hit it.

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
                # 200 calendar days ≈ 138 trading bars — enough for the HAR
                # vol forecaster (needs ~110+); indicators use the tail.
                bars = await self._broker.get_stock_bars(sym, days=200, timeframe="1Day")
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
                "chg5d": round((last / closes[-6] - 1) * 100, 2) if len(closes) >= 6 else None,
                "chg20d": round((last / closes[-21] - 1) * 100, 2) if len(closes) >= 21 else None,
                "rsi": ind.get("rsi_14"),
                "macd_h": ind.get("macd_histogram"),
                "bb": ind.get("bb_position"),
                "vol": ind.get("volume_ratio"),
                "atr": ind.get("atr_14"),
                "adx": ind.get("adx_14"),
                "from_hi": round((last / max(highs) - 1) * 100, 2) if highs else None,
            }
            summary.update(self._quant_fields(klines, ind))
            if self._settings.stocks.recommendation_expected_move:
                summary.update(await self._implied_fields(sym, last))
            out[sym] = summary
        return out

    async def _implied_fields(self, symbol: str, spot: float) -> dict[str, Any]:
        """Options-implied expected move for one candidate (advisory).

        Fully defensive: an options-feed outage returns ``{}`` and the pick
        proceeds on the statistical band alone. The implied band's horizon
        (nearest weekly expiry) differs from the statistical 5-day band, so
        both the % move and its DTE are surfaced — disagreement between the
        implied and statistical ranges flags an event premium.
        """
        from datetime import datetime as _dt

        from halal_trader.quant.expected_move import fetch_expected_move

        today = _dt.now(_ET).date()
        em = await fetch_expected_move(self._broker, symbol, spot, today=today)
        if em is None:
            return {}
        return {
            "impl_move_pct": em.move_pct,
            "impl_dte": em.dte,
            "impl_low": em.low,
            "impl_high": em.high,
        }

    @staticmethod
    def _quant_fields(klines: list[Any], ind: dict[str, Any]) -> dict[str, Any]:
        """Quantitative range-model fields for one candidate (advisory).

        Prompt-facing scalars (band5d_lo/hi, rng1d_pct, vol_pctl) plus the
        full per-horizon band dict under ``quant_bands`` — persisted in the
        ``candidates`` JSONB so the scorecard can label band coverage later.
        Degrades to ``{}`` when the series is too thin for any estimator.
        """
        from halal_trader.quant.calibration import load_default_artifact
        from halal_trader.quant.outlook import build_outlook

        try:
            outlook = build_outlook(
                [k.open for k in klines],
                [k.high for k in klines],
                [k.low for k in klines],
                [k.close for k in klines],
                atr=ind.get("atr_14"),
                calibration=load_default_artifact(),
            )
        except ValueError as exc:
            logger.debug("recommendation: outlook failed: %s", exc)
            return {}
        if outlook is None:
            return {}
        fields: dict[str, Any] = {
            "vol_pctl": round(outlook.vol_percentile, 2)
            if outlook.vol_percentile is not None
            else None,
            "quant_bands": {
                str(h): {
                    "low": round(hb.band.low, 4),
                    "high": round(hb.band.high, 4),
                    "expected_range": round(hb.band.expected_range, 4),
                    "sigma_daily": round(hb.band.sigma_daily, 6),
                    "z": round(hb.band.z, 3),
                    "source": hb.sigma_source,
                }
                for h, hb in outlook.bands.items()
            }
            | {
                "calibrated": outlook.calibrated,
                "calibration_version": outlook.calibration_version,
            },
        }
        five = outlook.bands.get(5)
        if five is not None:
            fields["band5d_lo"] = round(five.band.low, 2)
            fields["band5d_hi"] = round(five.band.high, 2)
        one = outlook.bands.get(1)
        if one is not None and outlook.close > 0:
            fields["rng1d_pct"] = round(one.band.expected_range / outlook.close * 100, 2)
        return fields

    def _format_table(self, candidates: dict[str, dict[str, Any]]) -> str:
        def _f(v: Any) -> str:
            return "n/a" if v is None else (f"{v:g}" if isinstance(v, int | float) else str(v))

        lines = []
        for sym, c in candidates.items():
            line = (
                f"{sym:6} ${_f(c['price'])}  chg5d={_f(c['chg5d'])}% "
                f"chg20d={_f(c['chg20d'])}% rsi={_f(c['rsi'])} "
                f"macd_h={_f(c['macd_h'])} bb={_f(c['bb'])} vol={_f(c['vol'])}x "
                f"atr={_f(c['atr'])} adx={_f(c['adx'])} from_hi={_f(c['from_hi'])}%"
            )
            if c.get("band5d_lo") is not None:
                line += (
                    f" band5d={_f(c['band5d_lo'])}..{_f(c['band5d_hi'])}"
                    f" rng1d={_f(c.get('rng1d_pct'))}%"
                    f" vpct={_f(c.get('vol_pctl'))}"
                )
            if c.get("impl_move_pct") is not None:
                line += (
                    f" impl=±{_f(c['impl_move_pct'])}%/{_f(c.get('impl_dte'))}d"
                    f"({_f(c.get('impl_low'))}..{_f(c.get('impl_high'))})"
                )
            lines.append(line)
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
            "trend-quality across the universe; higher = stronger long tilt):\n" + "\n".join(lines)
        )

    async def _pick(self, candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
        from halal_trader.quant.calibration import load_default_artifact

        date = datetime.now(_ET).strftime("%Y-%m-%d")
        factor_block = self._apply_factors(candidates)
        artifact = load_default_artifact()
        if artifact is not None:
            band_semantics = (
                f"coverage-calibrated to contain the 5-day price path "
                f"~{artifact.target_coverage:.0%} of the time on walk-forward "
                f"history; {artifact.version}"
            )
        else:
            band_semantics = (
                "±1.28σ√h — UNCALIBRATED, an approximation, not a measured-coverage interval"
            )
        user = USER_PROMPT_TEMPLATE.format(
            date=date,
            n=len(candidates),
            table=self._format_table(candidates),
            band_semantics=band_semantics,
        )
        user += "\n\n" + factor_block
        raw = await self._llm.generate_json(user, system=SYSTEM_PROMPT)
        return self._validate(raw, candidates)

    def _validate(
        self, raw: dict[str, Any], candidates: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        symbol = str(raw.get("symbol", "")).upper().strip()
        if symbol not in candidates:
            raise ValueError(f"LLM picked {symbol!r} which is not in the candidate universe")
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
    except TypeError, ValueError:
        return None
