"""The belief board's server side: the shadow's beliefs in the context a reader needs.

The halabot queries read the engine's own ``hb_`` tables; this module adds
what lives in the bot's tables (the strict halal screen, the core account's
snapshot, heartbeats, LLM spend) and turns the lot into the two payloads the
Belief Board page polls:

* :func:`board`: one row per belief, with its price, strict verdict, the
  core's and the shadow's weight in it, and a server-computed stance;
* :func:`overview`: the trust strip and side column (the track record and
  its baseline, calibration, the shadow book, engine health, LLM cost, the
  macro calendar).

Everything is read from the database: the web process never calls the broker
or the LLM. Each section degrades on its own, so a missing table (the
``hb_outcome.cohort`` column before the shadow's first start on new code, say)
blanks one tile rather than the page.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# The verdict waits for both a sample and a stretch of calendar: 30 closed
# trades in a fortnight of one market mood prove nothing.
MIN_CLOSED = 30
MIN_COHORT_DAYS = 28

# What random entries won in the 2026-10-06 review, used until the current
# cohort has closed trades to measure the baseline against.
REVIEW_BASELINE = 0.35
REVIEW_BASELINE_NOTE = (
    "From the 2026-10-06 review: random 1-3 bar entries in the same names since "
    "28 Sep won 35%. Measured here once the cohort has closed trades."
)
# Fewer random entries than this and the measured baseline is noise.
MIN_BASELINE_TRADES = 30
_BASELINE_TTL_S = 900.0
_baseline_cache: dict[tuple[Any, ...], tuple[float, tuple[float | None, int]]] = {}

SHADOW_CONSUMER = "shadow"  # halabot/cli.py's SpendMeter consumer
LLM_DAYS = 7


async def _soft[T](what: str, coro: Awaitable[T], fallback: T) -> T:
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001 -- one missing table must not 500 the page
        logger.debug("belief board %s degraded: %r", what, exc)
        return fallback


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts is not None else None


# ── the bot's tables ──
async def strict_verdicts(
    engine: AsyncEngine, assets: Sequence[str], *, today: date
) -> tuple[date | None, bool, dict[str, str]]:
    """(screen date, stale?, {asset: verdict}) from the newest strict screen.

    A name absent from the newest screen is ``unscreened``; several methods on
    one day resolve towards the stricter verdict.
    """
    from halal_trader.halal.strict import MAX_SCREEN_AGE

    async with engine.connect() as conn:
        as_of = (
            await conn.execute(
                text("SELECT max(as_of) FROM halal_screen_results WHERE as_of <= :d"),
                {"d": today},
            )
        ).scalar()
        if not isinstance(as_of, date):
            return None, True, dict.fromkeys(assets, "unscreened")
        rows = await conn.execute(
            text(
                "SELECT symbol, CASE WHEN bool_or(verdict = 'not_halal') THEN 'not_halal' "
                "WHEN bool_and(verdict = 'halal') THEN 'halal' ELSE 'doubtful' END AS verdict "
                "FROM halal_screen_current WHERE as_of = :a AND symbol = ANY(:s) "
                "GROUP BY symbol"
            ),
            {"a": as_of, "s": list(assets)},
        )
        found = {r.symbol: str(r.verdict) for r in rows}
    stale = today - as_of > MAX_SCREEN_AGE
    return as_of, stale, {a: found.get(a, "unscreened") for a in assets}


async def core_weights(
    engine: AsyncEngine, account: str
) -> tuple[datetime | None, dict[str, float]]:
    """The core account's weight in each holding, from its newest snapshot."""
    async with engine.connect() as conn:
        snap = (
            await conn.execute(
                text(
                    "SELECT taken_at, equity, positions FROM account_snapshots WHERE account = :a"
                ),
                {"a": account},
            )
        ).first()
    if snap is None or not snap.equity or float(snap.equity) <= 0:
        return None, {}
    equity = float(snap.equity)
    weights: dict[str, float] = {}
    for p in snap.positions or []:
        symbol, value = p.get("symbol"), p.get("market_value")
        if symbol and isinstance(value, int | float):
            weights[str(symbol)] = weights.get(str(symbol), 0.0) + float(value) / equity
    return snap.taken_at, {s: round(w, 5) for s, w in weights.items()}


async def shadow_heartbeat(engine: AsyncEngine) -> datetime | None:
    from halal_trader.core.heartbeat import SHADOW_PROCESS

    async with engine.connect() as conn:
        value = (
            await conn.execute(
                text("SELECT beat_at FROM heartbeats WHERE component = :c"),
                {"c": SHADOW_PROCESS},
            )
        ).scalar()
    return value if isinstance(value, datetime) else None


async def shadow_llm_spend(engine: AsyncEngine, *, today: date) -> dict[str, Any]:
    """The shadow's LLM cost: the average complete UTC day of the last week."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT day, spent_usd FROM llm_spend WHERE consumer = :c "
                    "AND day >= :d0 AND day <= :d1"
                ),
                {"c": SHADOW_CONSUMER, "d0": today - timedelta(days=LLM_DAYS), "d1": today},
            )
        ).all()
    past = [float(r.spent_usd) for r in rows if r.day < today]
    today_usd = sum(float(r.spent_usd) for r in rows if r.day == today)
    return {
        "per_day_usd": round(sum(past) / len(past), 4) if past else None,
        "days": len(past),
        "today_usd": round(today_usd, 4),
    }


# ── the board ──
def _bands() -> tuple[float, float, str]:
    from halabot.platform.config import get_settings

    hb = get_settings()
    return (
        hb.policy.conviction_entry_band,
        hb.policy.conviction_exit_band,
        hb.cognition.benchmark_symbol,
    )


async def board(engine: AsyncEngine, *, core_account: str, now: datetime) -> dict[str, Any]:
    from halabot.api import plain, queries
    from halabot.platform.db import OUTCOME_COHORT

    rows = await _soft("beliefs", queries.list_beliefs(engine), [])
    entry, exit_, benchmark = _bands()
    today = now.astimezone(_et()).date()
    assets = [b["asset"] for b in rows]
    prices = await _soft("prices", queries.latest_prices(engine, assets, now=now), {})
    screen_as_of, stale, verdicts = await _soft(
        "strict",
        strict_verdicts(engine, assets, today=today),
        (None, True, dict.fromkeys(assets, "unscreened")),
    )
    core_at, core = await _soft("core", core_weights(engine, core_account), (None, {}))
    book = {
        p["asset"]: p
        for p in await _soft(
            "shadow book", queries.open_positions(engine, cohort=OUTCOME_COHORT), []
        )
    }
    for b in rows:
        asset = b["asset"]
        price = prices.get(asset)
        b["price"] = round(price[0], 4) if price else None
        b["price_at"] = _iso(price[1]) if price else None
        b["strict"] = verdicts.get(asset, "unscreened")
        b["core_weight"] = core.get(asset) if core_at is not None else None
        pos = book.get(asset)
        b["shadow"] = (
            {k: pos[k] for k in ("weight", "entry_price", "last_price", "return_pct", "opened_at")}
            if pos
            else None
        )
        b["stance"] = plain.stance(
            asset=asset,
            direction=b["direction"],
            conviction=b["conviction"],
            strict="unscreened" if stale else b["strict"],
            benchmark=benchmark,
            entry_band=entry,
            exit_band=exit_,
        )
    return {
        "available": bool(rows),
        "beliefs": rows,
        "entry_band": entry,
        "exit_band": exit_,
        "benchmark": benchmark,
        "screen_as_of": screen_as_of.isoformat() if screen_as_of else None,
        "screen_stale": stale,
        "core_as_of": _iso(core_at),
    }


def _et() -> Any:
    from halal_trader.market_hours import MARKET_TZ

    return MARKET_TZ


# ── decisions ──
def decorate_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each proposal with a plain reason, flagged when made outside the session."""
    from halabot.api import plain
    from halabot.platform.session import is_regular_session

    entry, exit_, _ = _bands()
    for d in rows:
        payload = d.get("payload") or {}
        d["plain"] = plain.decision_reason(payload, entry_band=entry, exit_band=exit_)
        d["outside_session"] = not is_regular_session(datetime.fromisoformat(d["ts"]))
    return rows


# ── the overview ──
async def _baseline(
    engine: AsyncEngine,
    assets: Sequence[str],
    *,
    since: datetime | None,
    hold_s: float | None,
    threshold: float,
) -> tuple[float | None, int]:
    """Random-entry win rate over the cohort's names and period, cached: the
    bars only grow by a few an hour and the page polls every half minute."""
    from halabot.api import queries
    from halabot.platform.session import bar_closes_in_session

    if since is None or hold_s is None or not assets:
        return None, 0
    key = (since, round(hold_s), tuple(sorted(assets)), threshold)
    hit = _baseline_cache.get(key)
    if hit is not None and hit[0] > time.monotonic():
        return hit[1]
    closes = await queries.bar_closes(engine, assets, since=since)
    result = queries.random_entry_win_rate(
        closes,
        hold=timedelta(seconds=hold_s),
        threshold=threshold,
        # Hourly bars, stamped at their start: an entry at a bar's close is
        # possible when that close lands in the regular session.
        in_session=lambda t: bar_closes_in_session(t, timedelta(hours=1)),
    )
    _baseline_cache.clear()  # one live key at a time
    _baseline_cache[key] = (time.monotonic() + _BASELINE_TTL_S, result)
    return result


def _verdict(stats: dict[str, Any], baseline: float, *, now: datetime) -> dict[str, Any]:
    closed = int(stats["current"]["closed"])
    started = stats.get("started")
    age_days = (now - datetime.fromisoformat(started)).days if started else 0
    if closed < MIN_CLOSED or age_days < MIN_COHORT_DAYS:
        status, label = "unproven", "Unproven"
    elif (stats["current"]["win_rate"] or 0.0) > baseline:
        status, label = "beating", "Beating random entry"
    else:
        status, label = "not_beating", "Not beating random entry"
    return {
        "status": status,
        "label": label,
        "age_days": age_days,
        "min_closed": MIN_CLOSED,
        "min_days": MIN_COHORT_DAYS,
    }


def _calendar(beliefs: list[dict[str, Any]], *, now: datetime) -> list[dict[str, Any]]:
    """The macro releases every belief carries, once each, in time order."""
    from halabot.api.plain import MACRO_PLAIN

    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for b in beliefs:
        for c in b.get("catalysts_pending") or []:
            when = datetime.fromisoformat(c["scheduled_for"])
            if when < now - timedelta(hours=1):
                continue
            key = (c["kind"], c["scheduled_for"])
            item = seen.setdefault(
                key,
                {
                    "kind": c["kind"],
                    "plain": MACRO_PLAIN.get(str(c["kind"]).upper(), c.get("detail") or ""),
                    "scheduled_for": c["scheduled_for"],
                    "expected_impact": c["expected_impact"],
                    "detail": c.get("detail") or "",
                    "names": 0,
                },
            )
            item["expected_impact"] = max(item["expected_impact"], c["expected_impact"])
            item["names"] += 1
    return sorted(seen.values(), key=lambda c: c["scheduled_for"])


def _book(positions: list[dict[str, Any]], verdicts: dict[str, str]) -> dict[str, Any]:
    invested = sum(p["weight"] for p in positions)
    contribution = sum(p["weight"] * p["return_pct"] for p in positions)
    return {
        "positions": [{**p, "strict": verdicts.get(p["asset"], "unscreened")} for p in positions],
        "invested": round(invested, 4),
        "cash": round(max(0.0, 1.0 - invested), 4),
        # The book's return since its entries, weight-averaged, and what that
        # adds to the whole book (cash earning nothing).
        "return_pct": round(contribution / invested, 5) if invested > 0 else None,
        "contribution_pct": round(contribution, 5),
    }


async def overview(engine: AsyncEngine, *, now: datetime) -> dict[str, Any]:
    from halabot.api import queries
    from halabot.platform.config import get_settings
    from halabot.platform.db import OUTCOME_COHORT
    from halal_trader.core.heartbeat import SHADOW_PROCESS, STALE_AFTER

    hb = get_settings()
    entry, exit_, benchmark = _bands()
    today = now.astimezone(_et()).date()
    beliefs = await _soft("beliefs", queries.list_beliefs(engine), [])
    assets = [b["asset"] for b in beliefs]
    traded = [a for a in assets if a != benchmark]

    empty = {"closed": 0, "wins": 0, "win_rate": None, "mean_return_pct": None}
    stats = await _soft(
        "outcomes",
        queries.outcome_stats(engine, cohort=OUTCOME_COHORT),
        {
            "cohort": OUTCOME_COHORT,
            "started": None,
            "current": dict(empty),
            "earlier": dict(empty),
            "median_hold_s": None,
        },
    )
    started = datetime.fromisoformat(stats["started"]) if stats["started"] else None
    measured, trades = await _soft(
        "baseline",
        _baseline(
            engine,
            traded,
            since=started,
            hold_s=stats["median_hold_s"],
            threshold=hb.conviction.win_threshold_pct,
        ),
        (None, 0),
    )
    if measured is not None and trades >= MIN_BASELINE_TRADES:
        baseline, is_measured = measured, True
        hold_h = (stats["median_hold_s"] or 0) / 3600
        note = (
            f"Random entries in the same {len(traded)} names since the cohort began, "
            f"held {hold_h:.1f} h (its median hold): {trades} trades."
        )
    else:
        baseline, note, is_measured = REVIEW_BASELINE, REVIEW_BASELINE_NOTE, False

    scored, moved = await _soft(
        "calibration",
        queries.calibration_probe(engine, assets, since=now - timedelta(hours=24)),
        (0, 0),
    )
    calibration = {
        "status": "fitted" if moved else ("identity" if scored else "unknown"),
        "scored_24h": scored,
        "moved_24h": moved,
        "samples": stats["current"]["closed"],
        "min_samples": hb.conviction.min_samples_to_calibrate,
    }

    positions = await _soft("book", queries.open_positions(engine, cohort=OUTCOME_COHORT), [])
    _, _, verdicts = await _soft(
        "strict",
        strict_verdicts(engine, [p["asset"] for p in positions], today=today),
        (None, True, {}),
    )

    beat_at = await _soft("heartbeat", shadow_heartbeat(engine), None)
    age = (now - beat_at).total_seconds() if beat_at else None
    limit = STALE_AFTER[SHADOW_PROCESS].total_seconds()
    engine_state = {
        "status": "missing" if age is None else ("stale" if age > limit else "live"),
        "heartbeat_at": _iso(beat_at),
        "heartbeat_age_s": round(age) if age is not None else None,
        "last_bar_at": _iso(await _soft("last bar", queries.last_bar_at(engine), None)),
        "last_event_at": _iso(await _soft("last event", queries.last_event_at(engine), None)),
        "names": len(beliefs),
        "refreshed_at": max(
            (b["last_updated"] for b in beliefs if b["last_updated"]), default=None
        ),
    }

    return {
        "available": bool(beliefs or stats["current"]["closed"] or stats["earlier"]["closed"]),
        "cohort": stats,
        "verdict": _verdict(stats, baseline, now=now),
        "baseline_win_rate": round(baseline, 4),
        "baseline_note": note,
        "baseline_measured": is_measured,
        "baseline_trades": trades,
        "calibration": calibration,
        "book": _book(positions, verdicts),
        "engine": engine_state,
        "llm": await _soft(
            "llm spend",
            shadow_llm_spend(engine, today=now.astimezone(UTC).date()),
            {"per_day_usd": None, "days": 0, "today_usd": 0.0},
        ),
        "calendar": _calendar(beliefs, now=now),
        "entry_band": entry,
        "exit_band": exit_,
    }
