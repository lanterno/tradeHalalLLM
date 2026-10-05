"""What the dashboard's home page shows, assembled from the database.

Money (both accounts, live from the bot's minute snapshots, falling back to
the ledger), the core portfolio (holdings, sectors, screen status), the
market (calendar, benchmarks) and what the automation does next. Read-only;
the web process has no broker connection.
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.market_hours import (
    EARLY_CLOSE_DATES,
    MARKET_TZ,
    US_MARKET_HOLIDAYS,
    is_trading_day,
)

SESSION = {
    "pre": "04:00",
    "open": "09:30",
    "close": "16:00",
    "early_close": "13:00",
    "post": "20:00",
}
CORE_TRADE = time(15, 40)
RESEARCH = time(20, 30)
DIGEST = time(17, 15)
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


def next_trading_day(after: date) -> date:
    d = after + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


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
    from halal_trader.compliance.purification import paper_positions
    from halal_trader.compliance.sectors import sector_of
    from halal_trader.portfolio import readiness as gate
    from halal_trader.portfolio.core_executor import monthly_due
    from halal_trader.portfolio.snapshots import BENCHMARKS

    now = now or datetime.now(UTC)
    today = now.astimezone(MARKET_TZ).date()
    async with engine.connect() as conn:
        snaps = {r.account: r for r in await conn.execute(text("SELECT * FROM account_snapshots"))}
        equity_rows = [
            (r.account, r.day, float(r.equity))
            for r in await conn.execute(
                text("SELECT account, day, equity FROM broker_equity WHERE day >= :d ORDER BY day"),
                {"d": today - timedelta(days=45)},
            )
        ]
        quotes = {r.symbol: r for r in await conn.execute(text("SELECT * FROM quotes"))}
        closes: dict[str, list[tuple[date, float]]] = {}
        for r in await conn.execute(
            text(
                "SELECT symbol, day, close FROM ("
                " SELECT symbol, day, close, row_number() OVER "
                " (PARTITION BY symbol ORDER BY day DESC) AS n FROM daily_bars "
                " WHERE adjustment = 'raw' AND symbol = ANY(:s)) x WHERE n <= 2"
            ),
            {"s": list(BENCHMARKS)},
        ):
            closes.setdefault(r.symbol, []).append((r.day, float(r.close)))
        screen_as_of = (
            await conn.execute(text("SELECT max(as_of) FROM halal_screen_results"))
        ).scalar()
        screen_counts = {
            r.verdict: int(r.n)
            for r in await conn.execute(
                text(
                    "SELECT verdict, count(*) AS n FROM halal_screen_results "
                    "WHERE as_of = :a GROUP BY verdict"
                ),
                {"a": screen_as_of},
            )
        }
        unpaid = (
            await conn.execute(
                text(
                    "SELECT coalesce(sum(amount), 0) FROM purification_accruals "
                    "WHERE paid_at IS NULL"
                )
            )
        ).scalar()
        last_run = (
            await conn.execute(
                text(
                    "SELECT equity, cash, run_on FROM core_runs WHERE executed "
                    "ORDER BY recorded_at DESC LIMIT 1"
                )
            )
        ).first()

    # ── benchmarks: the bot's quote while it is fresh, else the last two closes ──
    benchmarks = []
    for symbol in BENCHMARKS:
        q = quotes.get(symbol)
        hist = sorted(closes.get(symbol, []))
        if q is not None and q.taken_at.astimezone(MARKET_TZ).date() == today:
            price, prev, as_of, live = float(q.price), q.prev_close, q.taken_at.isoformat(), True
        elif q is not None and hist and q.taken_at.astimezone(MARKET_TZ).date() > hist[-1][0]:
            price, prev, as_of, live = float(q.price), q.prev_close, q.taken_at.isoformat(), False
        elif hist:
            price, as_of, live = hist[-1][1], hist[-1][0].isoformat(), False
            prev = hist[-2][1] if len(hist) > 1 else None
        else:
            continue
        benchmarks.append(
            {
                "symbol": symbol,
                "price": round(price, 2),
                "change_pct": round(price / float(prev) - 1, 5) if prev else None,
                "as_of": as_of,
                "live": live,
            }
        )

    # ── accounts ──
    stocks = settings.stocks
    meta = {
        "core": (
            "Core portfolio",
            "active" if settings.core.enabled else "disabled",
            settings.core.paper,
        ),
        "paper": ("Day-trader", "active" if stocks.day_trader_enabled else "paused", True),
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
        elif account == "core" and last_run is not None:
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
    core_snap = snaps.get("core")
    core_equity = next((a["equity"] for a in accounts if a["account"] == "core"), None)
    if core_snap is not None and core_snap.positions:
        held = [
            {
                "symbol": p["symbol"],
                "value": p.get("market_value") or 0.0,
                "change_today": p.get("change_today"),
            }
            for p in core_snap.positions
        ]
    else:  # before the first snapshot: the ledger's shares at the last close
        shares = await paper_positions(engine, today + timedelta(days=1), "core")
        async with engine.connect() as conn:
            last_close = {
                r.symbol: float(r.close)
                for r in await conn.execute(
                    text(
                        "SELECT DISTINCT ON (symbol) symbol, close FROM daily_bars "
                        "WHERE adjustment = 'raw' AND symbol = ANY(:s) ORDER BY symbol, day DESC"
                    ),
                    {"s": sorted(shares)},
                )
            }
        held = [
            {"symbol": s, "value": q * last_close.get(s, 0.0), "change_today": None}
            for s, q in shares.items()
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
        verdicts = {
            r.symbol: (r.verdict, r.sic_description)
            for r in await conn.execute(
                text(
                    "SELECT symbol, verdict, sic_description FROM halal_screen_results "
                    "WHERE as_of = :a AND symbol = ANY(:s)"
                ),
                {"a": screen_as_of, "s": symbols},
            )
        }
    invested = sum(h["value"] for h in held) or 0.0
    denominator = core_equity or invested or 1.0
    held.sort(key=lambda h: -h["value"])
    sectors: dict[str, float] = {}
    for h in held:
        s = sector_of(verdicts.get(h["symbol"], (None, None))[1])
        sectors[s] = sectors.get(s, 0.0) + h["value"]
    failing = [h["symbol"] for h in held if verdicts.get(h["symbol"], ("",))[0] != "halal"]
    top = held[:TOP]

    # ── zakat and the gate ──
    zakat: dict[str, Any] | None = None
    hawl_hijri = settings.zakat.hawl_hijri
    next_hawl = None
    if hawl_hijri:
        hawl = z.parse_hawl(hawl_hijri)
        _, last_hawl = z.hawl_period(hawl, today)
        _, next_hawl = z.hawl_period(hawl, date.fromordinal(last_hawl.toordinal() + 360))
        now_due = await z.assess(engine, "core", period_start=last_hawl, hawl_date=today)
        zakat = {
            "amount": round(now_due.amount, 2),
            "chosen": now_due.chosen,
            "next_hawl": next_hawl.isoformat(),
            "next_hawl_hijri": z.hijri_label(next_hawl),
        }
    ready = await gate.check(engine, today=today)
    monthly_done = not await monthly_due(engine, today)
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
            "sectors": [
                {"sector": s, "weight": round(v / denominator, 5)}
                for s, v in sorted(sectors.items(), key=lambda kv: -kv[1])
            ],
            "screen": {
                "as_of": screen_as_of.isoformat() if screen_as_of else None,
                "halal": screen_counts.get("halal", 0),
                "screened": sum(screen_counts.values()),
                "failing": failing,
            },
        },
        "upcoming": upcoming(
            now,
            core_enabled=bool(settings.core.enabled),
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
            "clean": not ready.refused and not ready.halted,
        },
    }
