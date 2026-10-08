"""The Core page: the strict-halal core portfolio's own account, in one payload.

Read-only, from the database the core's jobs fill: equity and holdings
(the bot's newest minute snapshot of the account, else the ledger at the
last close), each holding against the forward book's target and its band,
the screen's view of it, the benchmarks, the live-money gate with the days
it is counted over, and every run with its orders and how they filled. The
account is the core's in the configured environment ("core" on paper,
"core-live" live).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.num import money, ratio, rounded
from halal_trader.market_hours import MARKET_TZ, is_trading_day, next_trading_day

TOP = 10
# Fills against the close, bucketed for the histogram: 10 bps wide.
_FILL_BIN_BPS = 10
_FILL_BINS = 4  # each side of zero; the outer bins take the tails


def _fill_histogram(fills: list[Any]) -> list[dict[str, Any]]:
    """Filled value per 10 bps bin of the fill against the close (a cost is positive)."""
    edges = range(-_FILL_BINS * _FILL_BIN_BPS, (_FILL_BINS + 1) * _FILL_BIN_BPS, _FILL_BIN_BPS)
    bins = dict.fromkeys(list(edges)[:-1], 0.0)
    for o in fills:
        bps = o.vs_close_bps
        if bps is None or o.filled_notional <= 0:
            continue
        lo = int(bps // _FILL_BIN_BPS) * _FILL_BIN_BPS
        lo = max(min(lo, max(bins)), min(bins))
        bins[lo] += o.filled_notional
    return [
        {"from_bps": lo, "to_bps": lo + _FILL_BIN_BPS, "value": round(v, 2)}
        for lo, v in sorted(bins.items())
    ]


def _weighted(fills: list[Any], attr: str) -> float | None:
    pairs = [(getattr(o, attr), o.filled_notional) for o in fills if getattr(o, attr) is not None]
    total = sum(w for _, w in pairs if w > 0)
    return sum(v * w for v, w in pairs if w > 0) / total if total else None


def _earliest_pass(today: date, *, days: int, min_days: int, run_today_pending: bool) -> date:
    """The first day the paper record can be long enough: one run a trading day."""
    needed = max(min_days - days, 0)
    d = today if run_today_pending else next_trading_day(today)
    if needed == 0:
        return today
    while True:
        if is_trading_day(d):
            needed -= 1
            if needed == 0:
                return d
        d += timedelta(days=1)


async def build(
    engine: AsyncEngine, settings: Any, *, now: datetime | None = None
) -> dict[str, Any]:
    from halal_trader.compliance.purification import paper_positions
    from halal_trader.compliance.sectors import sector_of
    from halal_trader.core.heartbeat import CORE_TRADE, DAILY_JOBS, core_running, scheduled_at
    from halal_trader.data.store import last_closes
    from halal_trader.execution.ledger import equity_history
    from halal_trader.halal import strict
    from halal_trader.market_hours import trading_days_back
    from halal_trader.portfolio import readiness as gate
    from halal_trader.portfolio.core_account import core_account
    from halal_trader.portfolio.core_executor import monthly_due
    from halal_trader.portfolio.execution_quality import report as fill_report
    from halal_trader.portfolio.holdings import broker_position
    from halal_trader.portfolio.home import clean_name, first_trading_day_of_next_month
    from halal_trader.portfolio.snapshots import benchmark_moves, read_snapshot
    from halal_trader.portfolio.strict_core import BAND, BAND_FLOOR
    from halal_trader.research.forward_book import latest_weights, nav_series

    now = now or datetime.now(UTC)
    today = now.astimezone(MARKET_TZ).date()
    account = core_account(settings.core.paper)
    equity_rows = await equity_history(engine, account)
    navs = dict(await nav_series(engine, "core"))
    targets = await latest_weights(engine, "core")
    snap = await read_snapshot(engine, account)
    async with engine.connect() as conn:
        orders = (
            await conn.execute(
                text(
                    "SELECT submitted_at, symbol, side, qty, est_price, notional, reason, "
                    "screen_as_of, screen_method, status FROM core_orders WHERE account = :a "
                    "ORDER BY submitted_at DESC LIMIT 400"
                ),
                {"a": account},
            )
        ).all()
        runs = (
            await conn.execute(
                text(
                    "SELECT run_on, monthly, executed, equity, cash, orders, halted, "
                    "screen_as_of, notes FROM core_runs WHERE account = :a "
                    "ORDER BY recorded_at DESC LIMIT 12"
                ),
                {"a": account},
            )
        ).all()

    # ── equity and holdings: the minute snapshot, else the ledger at the close ──
    equity = equity_rows[-1][1] if equity_rows else None
    equity_day = equity_rows[-1][0].isoformat() if equity_rows else None
    cash: float | None = None
    change: float | None = None
    source = "ledger"
    positions: dict[str, dict[str, Any]] = {}
    if snap is not None and snap.equity > 0:
        equity, equity_day, source = snap.equity, snap.taken_at.isoformat(), "live"
        cash = snap.cash
        change = snap.equity - snap.last_equity if snap.last_equity else None
        for p in snap.positions:
            if p.get("qty"):
                positions[str(p["symbol"])] = broker_position(p)
    else:
        shares = await paper_positions(engine, today + timedelta(days=1), account)
        closes = await last_closes(engine, sorted(shares))
        for symbol, qty in shares.items():
            close = closes.get(symbol, (None, 0.0))[1]
            positions[symbol] = {
                "symbol": symbol,
                "qty": qty,
                "market_value": money(qty * close),
                "change_today": None,
                "unrealized_pl_pct": None,
            }

    symbols = sorted(set(positions) | set(targets))
    screen_as_of = await strict.newest_screen(engine)
    screen = {
        r.symbol: r
        for r in (
            await strict.screen_rows(engine, screen_as_of, symbols=symbols) if screen_as_of else []
        )
    }
    async with engine.connect() as conn:
        names = {
            r.symbol: clean_name(r.name)
            for r in await conn.execute(
                text("SELECT symbol, name FROM market_assets WHERE symbol = ANY(:s)"),
                {"s": symbols},
            )
        }

    holdings: list[dict[str, Any]] = []
    for symbol in symbols:
        pos = positions.get(symbol, {})
        value = float(pos.get("market_value") or 0.0)
        weight = value / equity if equity else None
        target = float(targets.get(symbol, 0.0))
        band = max(BAND * target, BAND_FLOOR) if target else None
        drift = weight - target if weight is not None else None
        row = screen.get(symbol)
        verdict = row.verdict if row else None
        holdings.append(
            {
                "symbol": symbol,
                "name": names.get(symbol),
                "sector": sector_of(row.sic_description if row else None),
                "shares": rounded(pos.get("qty"), 6),
                "value": money(value),
                "weight": ratio(weight),
                "target": ratio(target),
                "drift": ratio(drift),
                "band": ratio(band),
                "in_band": None if drift is None or band is None else abs(drift) <= band,
                "today": ratio(pos.get("change_today")),
                "since_buy": pos.get("unrealized_pl_pct"),
                "verdict": verdict,
                # Held but no longer halal (or no longer screened): sold at the next run.
                "to_sell": bool(pos) and verdict != "halal",
                "reason": row.reasons[0] if row and row.reasons else None,
            }
        )
    holdings.sort(key=lambda h: -float(h["target"] or 0) - float(h["weight"] or 0))
    held = [h for h in holdings if h["shares"]]
    invested = sum(float(h["value"] or 0.0) for h in held)
    by_weight = sorted(held, key=lambda h: -float(h["value"] or 0.0))
    sectors: dict[str, dict[str, Any]] = {}
    for h in by_weight:
        s = sectors.setdefault(h["sector"], {"sector": h["sector"], "value": 0.0, "names": []})
        s["value"] += float(h["value"] or 0.0)
        s["names"].append(h["symbol"])
    sector_rows = sorted(sectors.values(), key=lambda s: -s["value"])
    for s in sector_rows:
        s["weight"] = ratio(s["value"] / equity) if equity else None
        s["count"] = len(s["names"])
        s["value"] = money(s["value"])

    # ── today against the benchmarks, and since the account's first day ──
    benchmarks = await benchmark_moves(engine, today)
    change_pct = change / (equity - change) if change is not None and equity else None
    today_vs = [
        {
            "symbol": b["symbol"],
            "change_pct": b["change_pct"],
            "diff_pts": ratio(change_pct - b["change_pct"])
            if change_pct is not None and b["change_pct"] is not None
            else None,
        }
        for b in benchmarks
    ]

    series: list[dict[str, Any]] = []
    since: dict[str, Any] | None = None
    if equity_rows:
        first_day, first_equity = equity_rows[0]
        base_nav = navs.get(first_day)
        async with engine.connect() as conn:
            bench: dict[str, dict[date, float]] = {}
            for r in await conn.execute(
                text(
                    "SELECT symbol, day, close FROM daily_bars WHERE adjustment = 'all' "
                    "AND symbol = ANY(:s) AND day >= :d"
                ),
                {"s": [b["symbol"] for b in benchmarks], "d": first_day},
            ):
                bench.setdefault(r.symbol, {})[r.day] = float(r.close)
        for day, equity_then in equity_rows:
            nav = navs.get(day)
            point: dict[str, Any] = {
                "date": day.isoformat(),
                "account": rounded(100 * equity_then / first_equity, 3),
                "book": rounded(100 * nav / base_nav, 3) if nav and base_nav else None,
            }
            for symbol, closes_by_day in bench.items():
                first_close = closes_by_day.get(first_day)
                then = closes_by_day.get(day)
                point[symbol.lower()] = (
                    rounded(100 * then / first_close, 3) if then and first_close else None
                )
            series.append(point)
        now_equity = equity if equity is not None else equity_rows[-1][1]
        since = {
            "since": first_day.isoformat(),
            "start_equity": money(first_equity),
            "change": money(now_equity - first_equity),
            "change_pct": ratio(now_equity / first_equity - 1),
            "book_pct": ratio(series[-1]["book"] / 100 - 1) if series[-1]["book"] else None,
        }
        for b in benchmarks:
            key = b["symbol"].lower()
            last = series[-1].get(key)
            since[f"{key}_pct"] = ratio(last / 100 - 1) if last else None

    # ── when it trades next ──
    core_job = DAILY_JOBS[CORE_TRADE]
    trades_at = scheduled_at(core_job, today) if is_trading_day(today) else None
    ran_today = any(r.run_on == today and r.executed for r in runs)
    next_day = today if trades_at is not None and now < trades_at and not ran_today else None
    next_day = next_day or next_trading_day(today)
    next_check = scheduled_at(core_job, next_day)
    is_due = await monthly_due(engine, today, account)
    rebalance_day = next_day if is_due else first_trading_day_of_next_month(today)

    # ── the live-money gate, and the days it counts ──
    ready = await gate.check(engine, today=today, account=account)
    run_days = {r.run_on for r in runs if r.executed}
    missing = set(ready.missing_runs)
    first_day = equity_rows[0][0] if equity_rows else today
    window = []
    for d in trading_days_back(today, gate.MIN_DAYS):
        status = (
            "run"
            if d in run_days
            else "before"
            if d < first_day
            else "pending"
            if d == today
            else "missed"
            if d in missing or d < today
            else "pending"
        )
        window.append({"day": d.isoformat(), "status": status})
    earliest_days = _earliest_pass(
        today, days=ready.days, min_days=gate.MIN_DAYS, run_today_pending=next_day == today
    )
    earliest = max(earliest_days, rebalance_day) if not ready.ready else today

    # ── runs and their orders, with how each filled ──
    executed = await fill_report(engine, today - timedelta(days=60), today, account=account)
    fill_of = {(o.submitted_at, o.symbol): o for o in executed.orders}

    def order_row(o: Any) -> dict[str, Any]:
        f = fill_of.get((o.submitted_at, o.symbol))
        return {
            "at": o.submitted_at.isoformat(),
            "symbol": o.symbol,
            "side": o.side,
            "qty": rounded(o.qty, 6),
            "price": money(o.est_price),
            "notional": money(o.notional),
            "reason": o.reason,
            "screen_as_of": o.screen_as_of.isoformat() if o.screen_as_of else None,
            "screen_method": o.screen_method,
            "status": o.status,
            "fill_price": rounded(f.fill_price, 4) if f else None,
            "filled_qty": rounded(f.filled_qty, 6) if f else None,
            "fill_status": f.status if f else None,
            "close": rounded(f.close, 4) if f else None,
            "vs_arrival_bps": rounded(f.vs_arrival_bps, 1) if f else None,
            "vs_close_bps": rounded(f.vs_close_bps, 1) if f else None,
        }

    by_day: dict[date, list[Any]] = {}
    for o in orders:
        by_day.setdefault(o.submitted_at.astimezone(MARKET_TZ).date(), []).append(o)
    run_rows = []
    for r in runs:
        mine = by_day.get(r.run_on, []) if r.executed else []
        fills = [f for o in mine if (f := fill_of.get((o.submitted_at, o.symbol)))]
        run_rows.append(
            {
                "run_on": r.run_on.isoformat(),
                "monthly": r.monthly,
                "executed": r.executed,
                "equity": money(r.equity),
                "cash": money(r.cash),
                "orders": r.orders,
                "halted": r.halted,
                "screen_as_of": r.screen_as_of.isoformat() if r.screen_as_of else None,
                "notes": list(r.notes or []),
                "notional": money(sum(float(o.notional or 0.0) for o in mine)) if mine else None,
                "filled": sum(1 for f in fills if f.status == "filled"),
                "vs_arrival_bps": rounded(_weighted(fills, "vs_arrival_bps"), 1),
                "vs_close_bps": rounded(_weighted(fills, "vs_close_bps"), 1),
                "order_rows": [order_row(o) for o in mine],
            }
        )

    execution = None
    if executed.orders:
        execution = {
            **executed.summary(),
            "histogram": _fill_histogram(executed.orders),
        }

    return {
        "now": now.isoformat(),
        "enabled": await core_running(engine),
        "paper": settings.core.paper,
        "account": account,
        "equity": money(equity),
        "equity_day": equity_day,
        "equity_source": source,
        "cash": money(cash),
        "invested": money(invested),
        "change": money(change),
        "change_pct": ratio(change_pct),
        "today_vs": today_vs,
        "since": since,
        "positions": len(held),
        "top10_weight": ratio(sum(float(h["value"] or 0) for h in by_weight[:TOP]) / equity)
        if equity
        else None,
        "to_sell": [
            {"symbol": h["symbol"], "name": h["name"], "reason": h["reason"]}
            for h in holdings
            if h["to_sell"]
        ],
        "screen_as_of": screen_as_of.isoformat() if screen_as_of else None,
        "next_check": next_check.isoformat(),
        "next_rebalance": scheduled_at(core_job, rebalance_day).isoformat(),
        "monthly_due": is_due,
        "holdings": holdings,
        "sectors": sector_rows,
        "series": series,
        "readiness": {
            "ready": ready.ready,
            "days": ready.days,
            "min_days": gate.MIN_DAYS,
            "monthly_runs": ready.monthly_runs,
            "tracking_error": ratio(ready.tracking_error),
            "max_tracking_error": gate.MAX_TRACKING_ERROR,
            "gap": ratio(ready.gap),
            "max_gap": gate.MAX_GAP,
            "refused": ready.refused,
            "unfilled": ready.unfilled,
            "partial": ready.partial,
            "missing_runs": [d.isoformat() for d in ready.missing_runs],
            "halted": ready.halted,
            "failures": ready.failures,
            "window": window,
            "earliest": earliest.isoformat(),
            "earliest_days": earliest_days.isoformat(),
        },
        "execution": execution,
        "orders": [order_row(o) for o in orders[:60]],
        "runs": run_rows,
    }
