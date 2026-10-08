"""What the dashboard's home page shows, assembled from the database.

Money (both accounts, live from the bot's minute snapshots, falling back to
the ledger), the core portfolio (holdings, sectors, screen status), the
market (calendar, benchmarks) and what the automation does next. Read-only;
the web process has no broker connection.
"""

from __future__ import annotations

import re
from calendar import monthrange
from collections import Counter
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance import purification
from halal_trader.core.heartbeat import CORE_TRADE as CORE_TRADE_JOB
from halal_trader.core.heartbeat import DAILY_JOBS
from halal_trader.core.heartbeat import RESEARCH as RESEARCH_JOB
from halal_trader.core.heartbeat import WEEKLY_DIGEST as DIGEST_JOB
from halal_trader.halal import strict
from halal_trader.market_hours import (
    EARLY_CLOSE,
    EARLY_CLOSE_DATES,
    MARKET_CLOSE,
    MARKET_OPEN,
    MARKET_TZ,
    US_MARKET_HOLIDAYS,
    is_trading_day,
    next_trading_day,
)

SESSION = {
    "pre": "04:00",
    "open": f"{MARKET_OPEN:%H:%M}",
    "close": f"{MARKET_CLOSE:%H:%M}",
    "early_close": f"{EARLY_CLOSE:%H:%M}",
    "post": "20:00",
}
# The bot's own schedule (core/heartbeat.py), not a copy of it.
CORE_TRADE = DAILY_JOBS[CORE_TRADE_JOB].at
RESEARCH = DAILY_JOBS[RESEARCH_JOB].at
DIGEST = DAILY_JOBS[DIGEST_JOB].at
SCREEN_EVERY = timedelta(days=7)
TOP = 10


def holiday_name(d: date) -> str:
    """The NYSE holiday falling on (or observed on) ``d``."""
    if d.month == 1 and d.day <= 2 or d.month == 12 and d.day == 31:
        return "New Year's Day"
    if d.month == 1:
        return "Martin Luther King Jr. Day"
    if d.month == 2:
        return "Presidents' Day"
    if d.month in (3, 4):
        return "Good Friday"
    if d.month == 5:
        return "Memorial Day"
    if d.month == 6:
        return "Juneteenth"
    if d.month == 7:
        return "Independence Day"
    if d.month == 9:
        return "Labor Day"
    if d.month == 11:
        return "Thanksgiving"
    if d.month == 12:
        return "Christmas"
    return "Market holiday"


def trading_days_in_month(today: date) -> tuple[int, int]:
    """(trading days this month, those still to come after today)."""
    days = [today.replace(day=i) for i in range(1, monthrange(today.year, today.month)[1] + 1)]
    trading = [d for d in days if is_trading_day(d)]
    return len(trading), sum(1 for d in trading if d > today)


def first_trading_day_of_next_month(today: date) -> date:
    first = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
    return first if is_trading_day(first) else next_trading_day(first)


def _at(d: date, t: time) -> str:
    return datetime.combine(d, t, MARKET_TZ).isoformat()


def upcoming(
    now: datetime,
    *,
    core_enabled: bool,
    monthly_done: bool,
    last_screen: date | None,
    next_hawl: date | None,
) -> list[dict[str, str]]:
    """The automatic jobs still to come, soonest first (New York times)."""
    today = now.astimezone(MARKET_TZ).date()
    local = now.astimezone(MARKET_TZ)
    items: list[dict[str, str]] = []

    def add(when: datetime | str, label: str, detail: str) -> None:
        items.append(
            {
                "at": when if isinstance(when, str) else when.isoformat(),
                "label": label,
                "detail": detail,
            }
        )

    if core_enabled:
        day = (
            today
            if is_trading_day(today) and local.time() < CORE_TRADE
            else next_trading_day(today)
        )
        if not monthly_done or day.month != today.month:
            add(
                _at(day, CORE_TRADE),
                "Core monthly rebalance",
                "back to target weights, within bands",
            )
        else:
            add(_at(day, CORE_TRADE), "Core daily check", "sells only a holding the screen fails")
        if monthly_done:
            nxt = first_trading_day_of_next_month(today)
            add(
                _at(nxt, CORE_TRADE),
                "Next monthly rebalance",
                "back to target weights, within bands",
            )
    run_day = (
        today if today.weekday() < 5 and local.time() < RESEARCH else today + timedelta(days=1)
    )
    while run_day.weekday() >= 5:
        run_day += timedelta(days=1)
    add(_at(run_day, RESEARCH), "Evening research run", "prices, purification, forward books")
    if last_screen is not None:
        screen = max(last_screen + SCREEN_EVERY, today)
        add(
            _at(screen, RESEARCH),
            "Weekly halal screen",
            "sooner if a holding files a 10-Q/10-K; then a SPUS/HLAL check",
        )
    friday = today + timedelta(days=(4 - today.weekday()) % 7)
    if friday == today and local.time() >= DIGEST:
        friday += timedelta(days=7)
    add(_at(friday, DIGEST), "Weekly digest", "on Telegram")
    if next_hawl is not None:
        add(_at(next_hawl, time(0, 0)), "Zakat assessed", "both methods, the higher recorded")
    return sorted(items, key=lambda i: i["at"])


_NAME_NOISE = re.compile(
    r"\s+(Class [A-C] )?(Common Stock|Ordinary Shares|Capital Stock|Common Shares|"
    r"American Depositary Shares|New)\b.*$",
    re.IGNORECASE,
)


def clean_name(name: str | None) -> str | None:
    """'Alphabet Inc. Class A Common Stock' -> 'Alphabet Inc. Class A'."""
    if not name:
        return None
    share_class = re.search(r"\bClass ([A-C])\b", name)
    short = _NAME_NOISE.sub("", name).strip()
    short = re.sub(r"\s+Class [A-C]$", "", short)
    return f"{short} Class {share_class.group(1)}" if share_class else short


def total_series(
    rows: list[tuple[str, date, float]],
    accounts: list[str],
    since: date,
    current: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Combined equity per day. An account with no value yet on a day counts at
    its first known value (it was funded and in cash before its ledger began),
    and one with no daily history at all at its ``current`` value."""
    by_account: dict[str, dict[date, float]] = {a: {} for a in accounts}
    for account, day, equity in rows:
        if account in by_account and equity > 0:
            by_account[account][day] = equity
    days = sorted({d for values in by_account.values() for d in values if d >= since})
    out = []
    last: dict[str, float | None] = {
        a: (values[min(values)] if values else (current or {}).get(a))
        for a, values in by_account.items()
    }
    for day in days:
        for a, values in by_account.items():
            if day in values:
                last[a] = values[day]
        known = [v for v in last.values() if v is not None]
        out.append({"date": day.isoformat(), "equity": round(sum(known), 2)})
    return out


async def build(
    engine: AsyncEngine, settings: Any, *, now: datetime | None = None
) -> dict[str, Any]:
    from halal_trader.compliance import zakat as z
    from halal_trader.core.heartbeat import core_running
    from halal_trader.portfolio import readiness as gate
    from halal_trader.portfolio import snapshots
    from halal_trader.portfolio.core_account import DAY_TRADER, core_account
    from halal_trader.portfolio.core_executor import monthly_due
    from halal_trader.portfolio.holdings import (
        broker_position,
        failing_screen,
        ledger_positions,
        screen_view,
        sector_values,
    )

    core_name = core_account(settings.core.paper)  # "core" on paper, "core-live" live
    core_on = await core_running(engine)  # the web never sees the core's keys
    now = now or datetime.now(UTC)
    today = now.astimezone(MARKET_TZ).date()
    snaps = await snapshots.read_snapshots(engine)
    async with engine.connect() as conn:
        equity_rows = [
            (r.account, r.day, float(r.equity))
            for r in await conn.execute(
                text("SELECT account, day, equity FROM broker_equity WHERE day >= :d ORDER BY day"),
                {"d": today - timedelta(days=45)},
            )
        ]
        screen_as_of = await strict.newest_screen(engine)
        screen_counts = Counter(
            r.verdict
            for r in (await strict.screen_rows(engine, screen_as_of) if screen_as_of else [])
        )
        unpaid = await purification.unpaid(engine)
        last_run = (
            await conn.execute(
                text(
                    "SELECT equity, cash, run_on FROM core_runs WHERE executed AND account = :a "
                    "ORDER BY recorded_at DESC LIMIT 1"
                ),
                {"a": core_name},
            )
        ).first()

    benchmarks = await snapshots.benchmark_moves(engine, today)

    # ── accounts ──
    meta = {
        core_name: (
            "Core portfolio",
            "active" if core_on else "disabled",
            settings.core.paper,
        ),
        DAY_TRADER: ("Day-trader", "active", True),
    }
    ledger: dict[str, list[tuple[date, float]]] = {}
    for account, day, equity in equity_rows:
        ledger.setdefault(account, []).append((day, equity))
    accounts = []
    for account, (label, status, paper) in meta.items():
        snap = snaps.get(account)
        hist = ledger.get(account, [])
        if snap is not None:
            equity, cash = float(snap.equity), float(snap.cash)
            prev = float(snap.last_equity) if snap.last_equity else None
            positions = list(snap.positions or [])
            as_of, source = snap.taken_at.isoformat(), "live"
        elif hist:
            equity, cash = hist[-1][1], None
            prev = hist[-2][1] if len(hist) > 1 else None
            positions, as_of, source = [], hist[-1][0].isoformat(), "ledger"
        elif account == core_name and last_run is not None:
            # The run's figures predate its own orders: equity only, no cash split.
            equity, cash, prev = float(last_run.equity), None, None
            positions, as_of, source = [], last_run.run_on.isoformat(), "run"
        else:
            continue
        change = equity - prev if prev else None
        accounts.append(
            {
                "account": account,
                "label": label,
                "status": status,
                "paper": paper,
                "equity": round(equity, 2),
                "cash": None if cash is None else round(cash, 2),
                "invested": None if cash is None else round(equity - cash, 2),
                "change": None if change is None else round(change, 2),
                "change_pct": None if change is None or not prev else round(change / prev, 5),
                "positions": len(positions),
                "symbols": sorted(p["symbol"] for p in positions)[:5],
                "as_of": as_of,
                "source": source,
            }
        )
    total_equity = sum(a["equity"] for a in accounts)
    changes = [a["change"] for a in accounts if a["change"] is not None]
    total_change = sum(changes) if changes else None
    series = total_series(
        equity_rows,
        [a["account"] for a in accounts],
        today - timedelta(days=30),
        {a["account"]: a["equity"] for a in accounts},
    )
    if (
        series
        and accounts
        and series[-1]["date"] < today.isoformat()
        and any(a["source"] == "live" for a in accounts)
    ):
        series.append({"date": today.isoformat(), "equity": round(total_equity, 2)})
    base = series[0]["equity"] if series else None

    # ── the core portfolio ──
    core_snap = snaps.get(core_name)
    core_equity = next((a["equity"] for a in accounts if a["account"] == core_name), None)
    rows = (
        [broker_position(p) for p in core_snap.positions]
        if core_snap is not None and core_snap.positions
        else (await ledger_positions(engine, core_name, []))[0]
    )
    held = [
        {
            "symbol": r["symbol"],
            "value": r["market_value"] or 0.0,
            "change_today": r["change_today"],
        }
        for r in rows
    ]
    symbols = sorted(h["symbol"] for h in held)
    async with engine.connect() as conn:
        names = {
            r.symbol: clean_name(r.name)
            for r in await conn.execute(
                text("SELECT symbol, name FROM market_assets WHERE symbol = ANY(:s)"),
                {"s": symbols},
            )
        }
    verdicts = await screen_view(engine, symbols)
    invested = sum(h["value"] for h in held) or 0.0
    denominator = core_equity or invested or 1.0
    held.sort(key=lambda h: -h["value"])
    sectors = sector_values({h["symbol"]: h["value"] for h in held}, verdicts)
    failing = failing_screen((h["symbol"] for h in held), verdicts)
    top = held[:TOP]

    # ── zakat and the gate ──
    zakat: dict[str, Any] | None = None
    hawl_hijri = settings.zakat.hawl_hijri
    next_hawl = None
    if hawl_hijri:
        last_hawl, next_hawl = z.hawl_dates(z.parse_hawl(hawl_hijri), today)
        now_due = await z.assess(engine, core_name, period_start=last_hawl, hawl_date=today)
        zakat = {
            "amount": round(now_due.amount, 2),
            "chosen": now_due.chosen,
            # Valued today, before the hawl: what would be due if the year
            # ended now, not what is owed. The hawl day's assessment is.
            "estimate": today != last_hawl,
            "as_of": today.isoformat(),
            "next_hawl": next_hawl.isoformat(),
            "next_hawl_hijri": z.hijri_label(next_hawl),
        }
    ready = await gate.check(engine, today=today)
    monthly_done = not await monthly_due(engine, today, core_name)
    trading_month, trading_left = trading_days_in_month(today)
    horizon = today + timedelta(days=400)
    holidays = sorted(d for d in US_MARKET_HOLIDAYS if today <= d <= horizon)
    early = sorted(d for d in EARLY_CLOSE_DATES if today <= d <= horizon)

    return {
        "now": now.isoformat(),
        "today": today.isoformat(),
        "today_hijri": z.hijri_label(today),
        "market": {
            "session": SESSION,
            "holidays": [d.isoformat() for d in holidays],
            "early_closes": [d.isoformat() for d in early],
            "next_holiday": {"date": holidays[0].isoformat(), "name": holiday_name(holidays[0])}
            if holidays
            else None,
            "next_early_close": early[0].isoformat() if early else None,
            "trading_days_month": trading_month,
            "trading_days_left": trading_left,
            "benchmarks": benchmarks,
        },
        "accounts": accounts,
        "total": {
            "equity": round(total_equity, 2),
            "change": None if total_change is None else round(total_change, 2),
            "change_pct": None
            if total_change is None or not total_equity - total_change
            else round(total_change / (total_equity - total_change), 5),
            "series": series,
            "change_30d_pct": round(total_equity / base - 1, 5) if base else None,
        },
        "set_aside": {"purification_unpaid": round(float(unpaid or 0.0), 2), "zakat": zakat},
        "portfolio": {
            "holdings": [
                {
                    "symbol": h["symbol"],
                    "name": names.get(h["symbol"]),
                    "value": round(h["value"], 2),
                    "weight": round(h["value"] / denominator, 5),
                    "change_today": h["change_today"],
                }
                for h in top
            ],
            "count": len(held),
            "top_weight": round(sum(h["value"] for h in top) / denominator, 5) if held else None,
            "sectors": [{"sector": s, "weight": round(v / denominator, 5)} for s, v in sectors],
            "screen": {
                "as_of": screen_as_of.isoformat() if screen_as_of else None,
                "halal": screen_counts.get("halal", 0),
                "screened": sum(screen_counts.values()),
                "failing": failing,
            },
        },
        "upcoming": upcoming(
            now,
            core_enabled=core_on,
            monthly_done=monthly_done,
            last_screen=screen_as_of,
            next_hawl=next_hawl,
        ),
        "gate": {
            "ready": ready.ready,
            "days": ready.days,
            "min_days": gate.MIN_DAYS,
            "monthly_runs": ready.monthly_runs,
            "tracking_error": ready.tracking_error,
            "max_tracking_error": gate.MAX_TRACKING_ERROR,
            "gap": ready.gap,
            "max_gap": gate.MAX_GAP,
            "clean": not ready.refused and not ready.halted and not ready.unfilled,
        },
    }
