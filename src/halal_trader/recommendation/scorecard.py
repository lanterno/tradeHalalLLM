"""Forward-return + price-path labeling and scorecard for the daily pick.

This is the "build-once" outcome-labeling foundation (Phase 0 of
``docs/QUANT_PREDICTION_ROADMAP.md``): given a recommendation and the
symbol's daily bars, it computes leakage-safe N-trading-day forward returns
from the close on/after the recommendation date, the realized price *path*
over the 5-day anchor window (max high / min low, MFE/MAE), whether the
LLM's suggested_target/suggested_stop were actually touched (and which
first), plus a halal-benchmark (SPUS) comparison — and aggregates an honest
track record. Advisory only — measurement, never trading.

Labeling honesty rules:

- A pick whose rec date predates the fetched bar window is **unlabelable**
  with that window: anchoring on the window's first bar would silently score
  it against the wrong entry (the pre-fix behavior). Such picks are marked
  ``skipped`` — never guessed.
- Path stats use bars strictly AFTER the entry bar (the plan is "buy at the
  entry close"; the forward path starts the next session).
- Daily bars cannot sequence target-vs-stop touches within one session:
  when both levels are touched first on the same bar, ``first_hit`` is
  ``both_same_bar`` — honest ignorance, never a coin flip.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from halal_trader.core.sample_guard import SampleGate
from halal_trader.core.signal_eval import information_coefficient

logger = logging.getLogger(__name__)

# Halal benchmark: SP Funds S&P 500 Sharia Industry Exclusions ETF.
DEFAULT_BENCHMARK = "SPUS"
HORIZONS = (1, 5, 20)  # trading days
PATH_HORIZON = 5  # anchor window for realized high/low, MFE/MAE, level hits
# Enough lookback to cover the oldest pending pick (~40 trading days) plus its
# 20-day forward window, in calendar days.
_BARS_LOOKBACK_DAYS = 90
# The audit refuses absurd per-row fetches (rows older than ~2 years are
# marked skipped rather than re-labeled).
_AUDIT_MAX_LOOKBACK_DAYS = 730


def _ohlc_by_date(bars: Any) -> list[tuple[str, float, float, float]]:
    """Extract ascending ``(YYYY-MM-DD, high, low, close)`` from a bars payload.

    Unlike ``bars_to_klines`` (which synthesises monotonic timestamps and so
    loses real dates), this preserves the real session date — required to
    align forward outcomes to the recommendation date. Missing high/low fall
    back to the close (degraded but defined); bad prints are sanitized so
    ``low <= close <= high`` always holds.
    """
    raw = bars
    if isinstance(bars, dict):
        raw = bars.get("bars") or bars.get("data") or []
        if isinstance(raw, dict):  # symbol-keyed {"bars": {"NVDA": [...]}}
            flat: list[Any] = []
            for v in raw.values():
                if isinstance(v, list):
                    flat.extend(v)
            raw = flat
    if not isinstance(raw, list):
        return []
    out: list[tuple[str, float, float, float]] = []
    for bar in raw:
        if not isinstance(bar, dict):
            continue
        t = bar.get("t") or bar.get("timestamp") or bar.get("time")
        c = bar.get("c", bar.get("close"))
        if t is None or c is None:
            continue
        try:
            close = float(c)
            high = float(bar.get("h", bar.get("high", close)))
            low = float(bar.get("l", bar.get("low", close)))
        except TypeError, ValueError:
            continue
        if close <= 0:
            continue
        high = max(high, low, close)
        low = min(high, low, close)
        out.append((str(t)[:10], high, low, close))
    out.sort(key=lambda x: x[0])
    return out


def _forward_outcomes(
    ohlc_by_date: list[tuple[str, float, float, float]],
    rec_date: str,
    horizons: tuple[int, ...] = HORIZONS,
    *,
    target: float | None = None,
    stop: float | None = None,
) -> dict[Any, Any] | None:
    """Forward returns + realized 5d path from the close on/after ``rec_date``.

    Returns one of:

    - ``None`` — no bar at/after the recommendation date exists yet (pending).
    - ``{"window_missed": True}`` — the bar window starts AFTER the rec date,
      so the true entry bar may lie outside it; labeling would be a guess.
    - ``{"entry_close": float, 1: pct|None, ..., "realized_high_5d": ...}``.
    """
    if not ohlc_by_date:
        return None
    dates = [d for d, _, _, _ in ohlc_by_date]
    highs = [h for _, h, _, _ in ohlc_by_date]
    lows = [lo for _, _, lo, _ in ohlc_by_date]
    closes = [c for _, _, _, c in ohlc_by_date]
    entry_idx = next((i for i, d in enumerate(dates) if d >= rec_date), None)
    if entry_idx is None:
        return None
    # Mislabeling guard: if the window's first bar is already past the rec
    # date, we cannot verify it is the true first session on/after the pick —
    # earlier bars may simply have fallen outside the fetch.
    if dates[0] > rec_date:
        return {"window_missed": True}
    entry_close = closes[entry_idx]
    if entry_close <= 0:
        return None
    # Mixed keys by design: str metadata + integer horizons.
    res: dict[Any, Any] = {"entry_close": entry_close, "window_missed": False}
    for h in horizons:
        j = entry_idx + h
        res[h] = round((closes[j] / entry_close - 1) * 100, 4) if j < len(closes) else None
    # Realized path over the anchor window: bars AFTER the entry bar.
    path_end = entry_idx + PATH_HORIZON
    if path_end < len(closes):
        p_highs = highs[entry_idx + 1 : path_end + 1]
        p_lows = lows[entry_idx + 1 : path_end + 1]
        realized_high = max(p_highs)
        realized_low = min(p_lows)
        res["realized_high_5d"] = round(realized_high, 4)
        res["realized_low_5d"] = round(realized_low, 4)
        res["mfe_pct"] = round((realized_high / entry_close - 1) * 100, 4)
        res["mae_pct"] = round((realized_low / entry_close - 1) * 100, 4)
        t_bar = (
            next((i for i, h in enumerate(p_highs) if h >= target), None)
            if target is not None
            else None
        )
        s_bar = (
            next((i for i, lo in enumerate(p_lows) if lo <= stop), None)
            if stop is not None
            else None
        )
        res["target_hit"] = (t_bar is not None) if target is not None else None
        res["stop_hit"] = (s_bar is not None) if stop is not None else None
        if target is None and stop is None:
            res["first_hit"] = None
        elif t_bar is None and s_bar is None:
            res["first_hit"] = "none"
        elif s_bar is None or (t_bar is not None and t_bar < s_bar):
            res["first_hit"] = "target"
        elif t_bar is None or s_bar < t_bar:
            res["first_hit"] = "stop"
        else:  # same bar — daily data can't order intraday touches
            res["first_hit"] = "both_same_bar"
    return res


def _outcome_fields(fr: dict[Any, Any]) -> dict[str, Any]:
    """Map a ``_forward_outcomes`` result onto DailyRecommendation columns."""
    fields: dict[str, Any] = {
        "entry_close": round(fr["entry_close"], 4),
        "fwd_return_1d": fr.get(1),
        "fwd_return_5d": fr.get(5),
        "fwd_return_20d": fr.get(20),
    }
    for key in (
        "realized_high_5d",
        "realized_low_5d",
        "mfe_pct",
        "mae_pct",
        "target_hit",
        "stop_hit",
        "first_hit",
    ):
        if key in fr:
            fields[key] = fr[key]
    return fields


async def backfill_outcomes(
    broker: Any, repo: Any, *, benchmark: str = DEFAULT_BENCHMARK
) -> dict[str, int]:
    """Label every not-yet-fully-scored pick with forward returns + path stats.

    Idempotent + progressive: each run fills whatever horizons have matured
    and flips a pick to ``scored`` once its 20-day return is available. Picks
    whose rec date has fallen out of the bar window are marked ``skipped``
    (unlabelable — see module docstring) and never revisited.
    """
    pending = await repo.get_recommendations_to_score()
    if not pending:
        return {"updated": 0, "scored": 0, "skipped": 0}

    bench_ohlc: list[tuple[str, float, float, float]] = []
    try:
        bench_bars = await broker.get_stock_bars(
            benchmark, days=_BARS_LOOKBACK_DAYS, timeframe="1Day"
        )
        bench_ohlc = _ohlc_by_date(bench_bars)
    except Exception as exc:  # noqa: BLE001 — benchmark is optional
        logger.debug("scorecard: benchmark %s unavailable: %s", benchmark, exc)

    updated = 0
    scored = 0
    skipped = 0
    for rec in pending:
        try:
            bars = await broker.get_stock_bars(
                rec["symbol"], days=_BARS_LOOKBACK_DAYS, timeframe="1Day"
            )
        except Exception as exc:  # noqa: BLE001 — skip a flaky symbol
            logger.debug("scorecard: bars for %s failed: %s", rec.get("symbol"), exc)
            continue
        fr = _forward_outcomes(
            _ohlc_by_date(bars),
            rec["date"],
            target=rec.get("suggested_target"),
            stop=rec.get("suggested_stop"),
        )
        if fr is None:
            continue
        if fr.get("window_missed"):
            await repo.update_recommendation_outcome(
                rec["id"],
                outcome_status="skipped",
                scored_at=datetime.now(UTC),
            )
            skipped += 1
            logger.warning(
                "scorecard: pick %s (%s %s) predates the bar window — skipped",
                rec["id"],
                rec.get("symbol"),
                rec.get("date"),
            )
            continue
        fields = _outcome_fields(fr)
        if bench_ohlc:
            bfr = _forward_outcomes(bench_ohlc, rec["date"], horizons=(5,))
            if bfr is not None and not bfr.get("window_missed"):
                fields["benchmark_return_5d"] = bfr.get(5)
        if fr.get(20) is not None:
            fields["outcome_status"] = "scored"
            fields["scored_at"] = datetime.now(UTC)
            scored += 1
        await repo.update_recommendation_outcome(rec["id"], **fields)
        updated += 1

    logger.info(
        "scorecard backfill: %d updated, %d fully scored, %d skipped",
        updated,
        scored,
        skipped,
    )
    return {"updated": updated, "scored": scored, "skipped": skipped}


async def audit_scored_outcomes(broker: Any, repo: Any, *, limit: int = 500) -> dict[str, int]:
    """One-time re-label of already-``scored`` rows against a covering window.

    The pre-guard backfill could silently score a stale pick against the bar
    window's first bar (wrong entry baseline). This audit re-fetches bars
    with a window that actually covers each row's rec date, recomputes every
    outcome field, and repairs rows whose stored ``entry_close`` disagrees.
    Idempotent and non-destructive: values are recomputed from broker truth.
    """
    rows = await repo.get_recent_recommendations(limit=limit)
    scored_rows = [r for r in rows if r.get("outcome_status") == "scored"]
    audited = 0
    repaired = 0
    skipped = 0
    today = datetime.now(UTC).date()
    for rec in scored_rows:
        try:
            rec_day = datetime.strptime(rec["date"], "%Y-%m-%d").date()
        except KeyError, ValueError:
            continue
        # Cover the rec date plus the 20-trading-day forward window (+buffer).
        days_needed = (today - rec_day).days + 45
        if days_needed > _AUDIT_MAX_LOOKBACK_DAYS:
            await repo.update_recommendation_outcome(rec["id"], outcome_status="skipped")
            skipped += 1
            continue
        try:
            bars = await broker.get_stock_bars(
                rec["symbol"], days=max(days_needed, _BARS_LOOKBACK_DAYS), timeframe="1Day"
            )
        except Exception as exc:  # noqa: BLE001 — leave the row for a retry
            logger.debug("audit: bars for %s failed: %s", rec.get("symbol"), exc)
            continue
        fr = _forward_outcomes(
            _ohlc_by_date(bars),
            rec["date"],
            target=rec.get("suggested_target"),
            stop=rec.get("suggested_stop"),
        )
        audited += 1
        if fr is None or fr.get("window_missed"):
            await repo.update_recommendation_outcome(rec["id"], outcome_status="skipped")
            skipped += 1
            continue
        fields = _outcome_fields(fr)
        # A "scored" row whose 20d return no longer computes was mis-scored;
        # send it back through the normal progressive backfill.
        fields["outcome_status"] = "scored" if fr.get(20) is not None else "pending"
        old_entry = rec.get("entry_close")
        if old_entry is None or abs(float(old_entry) - fields["entry_close"]) > 1e-4:
            repaired += 1
            logger.warning(
                "audit: pick %s (%s %s) entry_close %s -> %s — repaired",
                rec["id"],
                rec.get("symbol"),
                rec.get("date"),
                old_entry,
                fields["entry_close"],
            )
        await repo.update_recommendation_outcome(rec["id"], **fields)

    logger.info(
        "scorecard audit: %d audited, %d repaired, %d skipped",
        audited,
        repaired,
        skipped,
    )
    return {"audited": audited, "repaired": repaired, "skipped": skipped}


def _band_covered_5d(rec: dict[str, Any]) -> bool | None:
    """Did the pick's stored 5d quant band contain the realized path?

    Reads the band the engine persisted in ``candidates`` JSONB at
    generation time (never recomputed — recomputing would leak today's
    calibration into yesterday's prediction). ``None`` when the pick has
    no stored band or its path outcomes haven't matured.
    """
    hi = rec.get("realized_high_5d")
    lo = rec.get("realized_low_5d")
    if hi is None or lo is None:
        return None
    cand = (rec.get("candidates") or {}).get(rec.get("symbol") or "") or {}
    b5 = (cand.get("quant_bands") or {}).get("5")
    if not isinstance(b5, dict):
        return None
    try:
        return bool(float(lo) >= float(b5["low"]) and float(hi) <= float(b5["high"]))
    except KeyError, TypeError, ValueError:
        return None


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    """Fraction of rows where boolean ``key`` is True, among rows where set."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None


async def compute_scorecard(repo: Any, *, limit: int = 500) -> dict[str, Any]:
    """Aggregate track record over labeled picks (5-day horizon as the anchor)."""
    rows = await repo.get_recent_recommendations(limit=limit)
    labeled = [r for r in rows if r.get("fwd_return_5d") is not None]
    n = len(labeled)
    if n == 0:
        return {"available": False, "n_total": len(rows), "n_scored": 0}

    fwd5 = [r["fwd_return_5d"] for r in labeled]
    hit = sum(1 for x in fwd5 if x > 0) / n
    excess = [
        r["fwd_return_5d"] - r["benchmark_return_5d"]
        for r in labeled
        if r.get("benchmark_return_5d") is not None
    ]
    best = max(labeled, key=lambda r: r["fwd_return_5d"])
    worst = min(labeled, key=lambda r: r["fwd_return_5d"])
    gate = SampleGate(n)  # is the track record long enough to trust the rates?
    # Does the model's conviction actually rank-correlate with outcomes? Only
    # report it once the sample is big enough to mean anything.
    conviction_ic: float | None = None
    if gate.sufficient:
        convictions = [r.get("conviction") or 0.0 for r in labeled]
        conviction_ic = round(information_coefficient(convictions, fwd5), 4)
    # Plan quality: did the LLM's own suggested levels survive the price path?
    # This is the baseline any quantitative level/band must beat.
    with_levels = [
        r for r in labeled if r.get("target_hit") is not None or r.get("stop_hit") is not None
    ]
    first_hits = [r["first_hit"] for r in labeled if r.get("first_hit")]
    first_hit_counts = {k: first_hits.count(k) for k in sorted(set(first_hits))}
    # Band coverage: did the pick's own stored 5d quant band (written into
    # candidates JSONB at generation time — leakage-safe by construction)
    # contain the realized path? The live coverage feed for quant/calibration.
    band_covered = [_band_covered_5d(r) for r in labeled]
    band_results = [b for b in band_covered if b is not None]
    return {
        "available": True,
        "n_total": len(rows),
        "n_scored": n,
        # Honest caveat: below ~20 scored picks the hit-rate/averages are noise.
        "sufficient": gate.sufficient,
        "min_samples": gate.min_n,
        "conviction_ic": conviction_ic,
        "hit_rate_5d": round(hit, 4),
        "avg_fwd_1d": _avg(labeled, "fwd_return_1d"),
        "avg_fwd_5d": _avg(labeled, "fwd_return_5d"),
        "avg_fwd_20d": _avg(labeled, "fwd_return_20d"),
        "avg_excess_5d": round(sum(excess) / len(excess), 4) if excess else None,
        "benchmark": DEFAULT_BENCHMARK,
        # ── Plan quality (LLM suggested_target/stop vs the realized path) ──
        "n_with_levels": len(with_levels),
        "levels_sufficient": SampleGate(len(with_levels)).sufficient,
        "target_hit_rate": _rate(labeled, "target_hit"),
        "stop_hit_rate": _rate(labeled, "stop_hit"),
        "first_hit_counts": first_hit_counts,
        "avg_mfe_5d": _avg(labeled, "mfe_pct"),
        "avg_mae_5d": _avg(labeled, "mae_pct"),
        # ── Quant band coverage (stored band vs realized 5d path) ──
        "band_n": len(band_results),
        "band_coverage_5d": (
            round(sum(1 for b in band_results if b) / len(band_results), 4)
            if band_results
            else None
        ),
        "best": {
            "symbol": best["symbol"],
            "date": best["date"],
            "fwd_5d": best["fwd_return_5d"],
        },
        "worst": {
            "symbol": worst["symbol"],
            "date": worst["date"],
            "fwd_5d": worst["fwd_return_5d"],
        },
    }


async def whatif_equity_curve(
    repo: Any, *, limit: int = 500, start: float = 100.0
) -> dict[str, Any]:
    """Equity curve of *taking every scored stock-of-the-day pick*.

    Honest "would this have made money?" — compounds each scored pick's 5-day
    forward return in date order (buy the pick, hold 5 trading days, repeat),
    against the same-pick SPUS benchmark. Sequential/non-overlapping is an
    approximation (real picks overlap), but it's a fair directional read.
    """
    rows = await repo.get_recent_recommendations(limit=limit)
    scored = sorted(
        (r for r in rows if r.get("fwd_return_5d") is not None),
        key=lambda r: (r.get("date", ""), r.get("id", 0)),
    )
    if not scored:
        return {"available": False, "n": 0, "points": []}

    equity = start
    bench_equity = start
    points: list[dict[str, Any]] = []
    for r in scored:
        equity *= 1.0 + r["fwd_return_5d"] / 100.0
        b = r.get("benchmark_return_5d")
        if b is not None:
            bench_equity *= 1.0 + b / 100.0
        points.append(
            {
                "date": r["date"],
                "symbol": r["symbol"],
                "equity": round(equity, 2),
                "benchmark": round(bench_equity, 2),
            }
        )
    return {
        "available": True,
        "n": len(scored),
        "start": start,
        "final_equity": round(equity, 2),
        "total_return_pct": round((equity / start - 1.0) * 100, 2),
        "benchmark_return_pct": round((bench_equity / start - 1.0) * 100, 2),
        "benchmark": DEFAULT_BENCHMARK,
        "points": points,
    }
