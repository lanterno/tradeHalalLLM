"""Forward-return labeling + scorecard for the daily halal recommendation.

This is the "build-once" forward-return labeling foundation (the roadmap's
Phase-0 cornerstone): given a recommendation and the symbol's daily bars, it
computes leakage-safe N-trading-day forward returns from the close on/after
the recommendation date, plus a halal-benchmark (SPUS) comparison, and
aggregates an honest track record. Advisory only — measurement, never trading.
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
# Enough lookback to cover the oldest pending pick (~40 trading days) plus its
# 20-day forward window, in calendar days.
_BARS_LOOKBACK_DAYS = 90


def _closes_by_date(bars: Any) -> list[tuple[str, float]]:
    """Extract ascending ``(YYYY-MM-DD, close)`` from a get_stock_bars payload.

    Unlike ``bars_to_klines`` (which synthesises monotonic timestamps and so
    loses real dates), this preserves the real session date — required to
    align forward returns to the recommendation date.
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
    out: list[tuple[str, float]] = []
    for bar in raw:
        if not isinstance(bar, dict):
            continue
        t = bar.get("t") or bar.get("timestamp") or bar.get("time")
        c = bar.get("c", bar.get("close"))
        if t is None or c is None:
            continue
        try:
            close = float(c)
        except (TypeError, ValueError):
            continue
        if close <= 0:
            continue
        out.append((str(t)[:10], close))
    out.sort(key=lambda x: x[0])
    return out


def _forward_returns(
    closes_by_date: list[tuple[str, float]],
    rec_date: str,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[Any, Any] | None:
    """Forward % returns from the close on/after ``rec_date``.

    Returns ``{"entry_close": float, 1: pct|None, 5: pct|None, ...}`` or None
    if no bar at/after the recommendation date exists yet.
    """
    if not closes_by_date:
        return None
    dates = [d for d, _ in closes_by_date]
    closes = [c for _, c in closes_by_date]
    entry_idx = next((i for i, d in enumerate(dates) if d >= rec_date), None)
    if entry_idx is None:
        return None
    entry_close = closes[entry_idx]
    if entry_close <= 0:
        return None
    # Mixed keys by design: "entry_close" (str) + integer horizons.
    res: dict[Any, Any] = {"entry_close": entry_close}
    for h in horizons:
        j = entry_idx + h
        res[h] = round((closes[j] / entry_close - 1) * 100, 4) if j < len(closes) else None
    return res


async def backfill_outcomes(
    broker: Any, repo: Any, *, benchmark: str = DEFAULT_BENCHMARK
) -> dict[str, int]:
    """Label every not-yet-fully-scored pick with forward returns.

    Idempotent + progressive: each run fills whatever horizons have matured
    and flips a pick to ``scored`` once its 20-day return is available.
    """
    pending = await repo.get_recommendations_to_score()
    if not pending:
        return {"updated": 0, "scored": 0}

    bench_cbd: list[tuple[str, float]] = []
    try:
        bench_bars = await broker.get_stock_bars(
            benchmark, days=_BARS_LOOKBACK_DAYS, timeframe="1Day"
        )
        bench_cbd = _closes_by_date(bench_bars)
    except Exception as exc:  # noqa: BLE001 — benchmark is optional
        logger.debug("scorecard: benchmark %s unavailable: %s", benchmark, exc)

    updated = 0
    scored = 0
    for rec in pending:
        try:
            bars = await broker.get_stock_bars(
                rec["symbol"], days=_BARS_LOOKBACK_DAYS, timeframe="1Day"
            )
        except Exception as exc:  # noqa: BLE001 — skip a flaky symbol
            logger.debug("scorecard: bars for %s failed: %s", rec.get("symbol"), exc)
            continue
        fr = _forward_returns(_closes_by_date(bars), rec["date"])
        if fr is None:
            continue
        fields: dict[str, Any] = {
            "entry_close": round(fr["entry_close"], 4),
            "fwd_return_1d": fr.get(1),
            "fwd_return_5d": fr.get(5),
            "fwd_return_20d": fr.get(20),
        }
        if bench_cbd:
            bfr = _forward_returns(bench_cbd, rec["date"], horizons=(5,))
            if bfr is not None:
                fields["benchmark_return_5d"] = bfr.get(5)
        if fr.get(20) is not None:
            fields["outcome_status"] = "scored"
            fields["scored_at"] = datetime.now(UTC)
            scored += 1
        await repo.update_recommendation_outcome(rec["id"], **fields)
        updated += 1

    logger.info("scorecard backfill: %d updated, %d fully scored", updated, scored)
    return {"updated": updated, "scored": scored}


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


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
