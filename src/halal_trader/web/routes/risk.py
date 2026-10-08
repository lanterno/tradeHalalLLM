"""GET /api/risk/state (the day-trader's last cycle) and /api/risk/core (the core's)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.context import DashboardContext
from halal_trader.web.dependencies import get_ctx

TOP = 10


def _r(x: float | None, digits: int = 5) -> float | None:
    return None if x is None else round(x, digits)


def drawdown(points: list[tuple[date, float]], current: float) -> dict[str, Any]:
    """The current equity against the account's own peak.

    ``points`` are (day, equity) closes; ``current`` is the newest value,
    counted as of today. Returns the drawdown (<= 0), the peak and its day,
    and the span the peak was taken over, which is short while the account
    is new: a 0% drawdown over three days says little.
    """
    series = [(d, e) for d, e in points if e > 0]
    peak_day, peak = max(series, key=lambda p: p[1]) if series else (None, current)
    if current >= peak:
        peak_day, peak = None, current  # the peak is now
    return {
        "drawdown_pct": _r(current / peak - 1) if peak else None,
        "peak_equity": round(peak, 2),
        "peak_day": peak_day.isoformat() if peak_day else None,
        "history_from": min(d for d, _ in series).isoformat() if series else None,
        "history_days": len(series),
    }


async def core_risk(engine: AsyncEngine) -> dict[str, Any]:
    """The core account's concentration and drawdown, from the bot's minute
    snapshot (falling back to its ledger at the last close) and its daily
    equity history."""
    from halal_trader.compliance.sectors import sector_of
    from halal_trader.web.routes.positions import broker_position, ledger_positions

    async with engine.connect() as conn:
        snap = (
            await conn.execute(
                text(
                    "SELECT taken_at, equity, cash, last_equity, positions "
                    "FROM account_snapshots WHERE account = 'core'"
                )
            )
        ).first()
        history = [
            (r.day, float(r.equity))
            for r in await conn.execute(
                text(
                    "SELECT day, equity FROM broker_equity WHERE account = 'core' AND equity > 0 "
                    "ORDER BY day"
                )
            )
        ]
    now = datetime.now(UTC)
    if snap is not None:
        positions = [broker_position(p) for p in snap.positions or []]
        equity: float | None = float(snap.equity)
        cash: float | None = float(snap.cash)
        as_of, source = snap.taken_at.isoformat(), "snapshot"
        age: int | None = round((now - snap.taken_at).total_seconds())
        if snap.last_equity:
            # The previous session's close, which the daily ledger may not hold yet.
            from halal_trader.market_hours import MARKET_TZ, is_trading_day

            prev_day = snap.taken_at.astimezone(MARKET_TZ).date() - timedelta(days=1)
            while not is_trading_day(prev_day):
                prev_day -= timedelta(days=1)
            if not history or history[-1][0] < prev_day:
                history.append((prev_day, float(snap.last_equity)))
    else:
        positions, as_of = await ledger_positions(engine, "core", [])
        equity = cash = age = None
        source = "ledger"
        if not positions:
            return {"available": False}

    invested = sum(p["market_value"] or 0.0 for p in positions)
    base = equity or invested
    if not base:
        return {"available": False}
    positions.sort(key=lambda p: -(p["market_value"] or 0.0))
    symbols = [p["symbol"] for p in positions]
    async with engine.connect() as conn:
        screen = {
            r.symbol: (r.verdict, r.sic_description)
            for r in await conn.execute(
                text(
                    "SELECT DISTINCT ON (symbol) symbol, verdict, sic_description "
                    "FROM halal_screen_current WHERE symbol = ANY(:s) "
                    "ORDER BY symbol, as_of DESC, screened_at DESC"
                ),
                {"s": symbols},
            )
        }
    sectors: dict[str, float] = {}
    for p in positions:
        s = sector_of(screen.get(p["symbol"], (None, None))[1])
        sectors[s] = sectors.get(s, 0.0) + (p["market_value"] or 0.0)
    ranked = sorted(sectors.items(), key=lambda kv: -kv[1])
    largest = positions[0] if positions else None
    return {
        "available": True,
        "source": source,
        "as_of": as_of,
        "age_seconds": age,
        "equity": _r(equity, 2),
        "cash": _r(cash, 2),
        "cash_pct": _r(cash / base) if cash is not None else None,
        "positions": len(positions),
        "top10_weight": _r(sum(p["market_value"] or 0.0 for p in positions[:TOP]) / base),
        "largest": {
            "symbol": largest["symbol"],
            "weight": _r((largest["market_value"] or 0.0) / base),
        }
        if largest
        else None,
        "sectors": [{"sector": s, "weight": _r(v / base)} for s, v in ranked],
        "failing_screen": [s for s in symbols if screen.get(s, ("",))[0] != "halal"],
        **drawdown(history, equity if equity is not None else invested),
    }


def register(app: FastAPI) -> None:
    @app.get("/api/risk/state")
    async def api_risk_state(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        """The risk read the day-trader's last cycle published (via its heartbeat).

        With DAY_TRADER_ENABLED=false its last read stays as it was, and
        ``day_trader_enabled`` says why.
        """
        from halal_trader.core.heartbeat import cycle_risk

        try:
            state, at = await cycle_risk(ctx.engine)
        except Exception:  # noqa: BLE001 -- an unreadable table is "not available"
            state, at = None, None
        enabled = ctx.settings.stocks.day_trader_enabled
        if state is None:
            return JSONResponse({"available": False, "day_trader_enabled": enabled})
        return JSONResponse(
            {
                "available": True,
                "market": "stocks",
                "day_trader_enabled": enabled,
                "pushed_at": at.isoformat() if at else None,
                **state,
            }
        )

    @app.get("/api/risk/core")
    async def api_risk_core(ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        """The core portfolio's risk: concentration (top ten, largest holding,
        sectors) and drawdown from its own equity peak."""
        return JSONResponse(await core_risk(ctx.engine))
