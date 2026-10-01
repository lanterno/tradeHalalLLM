"""Side by side: what the MCP adapter and the REST adapter each say Alpaca holds.

Read-only. ``halal-trader broker compare`` runs it; the REST adapter becomes
the default only after this has agreed on enough days, open and closed.
Account values and prices move between two reads, so they are compared
within a tolerance; positions and the clock must match exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from halal_trader.domain.ports import Broker

_VALUE_TOLERANCE = 0.002  # 0.2%: two reads a second apart, mid-session
_PRICE_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def _close(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance * max(abs(a), abs(b), 1.0)


def _when(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _last_trade(snapshot: Any, symbol: str) -> float | None:
    entry = snapshot.get(symbol) if isinstance(snapshot, dict) else None
    trade = entry.get("latestTrade") if isinstance(entry, dict) else None
    price = trade.get("p") if isinstance(trade, dict) else None
    return float(price) if isinstance(price, int | float) else None


def _last_bar(bars: Any, symbol: str) -> tuple[str, float] | None:
    series = bars.get("bars", {}).get(symbol) if isinstance(bars, dict) else None
    if not isinstance(series, list) or not series:
        return None
    last = series[-1]
    close = last.get("c") if isinstance(last, dict) else None
    if not isinstance(close, int | float):
        return None
    return str(last.get("t", ""))[:10], float(close)


async def compare(mcp: Broker, rest: Broker, symbols: list[str]) -> list[Check]:
    checks: list[Check] = []

    a, b = await mcp.get_account_info(), await rest.get_account_info()
    for field in ("equity", "cash", "buying_power"):
        x, y = getattr(a, field), getattr(b, field)
        checks.append(Check(f"account.{field}", _close(x, y, _VALUE_TOLERANCE), f"{x} | {y}"))

    ca, cb = await mcp.get_clock(), await rest.get_clock()
    checks.append(Check("clock.is_open", ca.is_open == cb.is_open, f"{ca.is_open} | {cb.is_open}"))
    for field in ("next_open", "next_close"):
        x, y = getattr(ca, field), getattr(cb, field)
        same = _when(x) is not None and _when(x) == _when(y)
        checks.append(Check(f"clock.{field}", same, f"{x} | {y}"))

    pa = {p.symbol: p.qty for p in await mcp.get_all_positions()}
    pb = {p.symbol: p.qty for p in await rest.get_all_positions()}
    checks.append(Check("positions", pa == pb, f"{sorted(pa.items())} | {sorted(pb.items())}"))

    if symbols:
        sa = await mcp.get_stock_snapshot(",".join(symbols))
        sb = await rest.get_stock_snapshot(",".join(symbols))
        for symbol in symbols:
            x, y = _last_trade(sa, symbol), _last_trade(sb, symbol)
            ok = x is not None and y is not None and _close(x, y, _PRICE_TOLERANCE)
            checks.append(Check(f"snapshot.{symbol}", ok, f"{x} | {y}"))
            bx = _last_bar(await mcp.get_stock_bars(symbol, days=10), symbol)
            by = _last_bar(await rest.get_stock_bars(symbol, days=10), symbol)
            ok = (
                bx is not None
                and by is not None
                and bx[0] == by[0]
                and _close(bx[1], by[1], _PRICE_TOLERANCE)
            )
            checks.append(Check(f"bars.{symbol}", ok, f"{bx} | {by}"))
    return checks
