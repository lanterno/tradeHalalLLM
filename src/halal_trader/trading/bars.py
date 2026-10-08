"""Alpaca bars → ``Kline`` adapter + indicator helper.

Shared by the cycle, the snapshot recorder and any stage that wants
indicators on stock bars. The indicator + risk infrastructure takes
``Kline`` objects; this module coerces Alpaca's ``get_stock_bars``
response into that shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from halal_trader.core.num import to_float
from halal_trader.domain.models import Kline
from halal_trader.signals.indicators import compute_all

logger = logging.getLogger(__name__)


def extract_bar_dicts(bars_payload: Any) -> list[dict[str, Any]]:
    """The raw bar dicts inside any ``get_stock_bars`` envelope, in order.

    Shared by :func:`bars_to_klines` and the prompt formatter, so the one
    parser proven against live payloads is the only one there is.
    """
    if not bars_payload:
        return []
    raw_bars: list[Any]
    if isinstance(bars_payload, dict):
        # Peel the response envelopes. The Alpaca MCP server now wraps the
        # payload under "data" (beside an "_alpaca_mcp_security" sibling), the
        # bars endpoint nests them under "bars", and again under each symbol:
        #   {"data": {"bars": {"NVDA": [...]}}}   ← current MCP format
        #   {"bars": {"NVDA": [...]}}             ← older
        #   {"bars": [...]} / [...]               ← flat
        # Unwrap "data" then "bars" until we reach a list or symbol-keyed dict.
        # (The prior code peeled only one level, so the current double-nested
        # format silently yielded ZERO klines — starving indicators/ML/the
        # daily recommendation of all bar data.)
        payload: Any = bars_payload
        for key in ("data", "bars"):
            if isinstance(payload, dict) and key in payload:
                payload = payload[key]
        if isinstance(payload, dict):
            # Symbol-keyed: {"NVDA": [...]} → flatten the underlying lists.
            flattened: list[Any] = []
            for v in payload.values():
                if isinstance(v, list):
                    flattened.extend(v)
            raw_bars = flattened
        elif isinstance(payload, list):
            raw_bars = payload
        else:
            return []
    elif isinstance(bars_payload, list):
        raw_bars = bars_payload
    else:
        return []

    return [b for b in raw_bars if isinstance(b, dict)]


def bars_to_klines(bars_for_symbol: Any) -> list[Kline]:
    """Coerce Alpaca's ``get_stock_bars`` response into ``Kline`` objects.

    Alpaca returns a list of dicts with ``t/o/h/l/c/v`` keys, OR a nested
    ``{"bars": [...]}`` envelope, OR — what ``get_stock_bars`` actually emits —
    a symbol-keyed ``{"bars": {"NVDA": [...]}}`` envelope. We tolerate all three,
    plus the ``open``/``high``/``low``/``close``/``volume`` long-key variant some
    SDK versions emit. (The symbol-keyed shape previously fell through to an empty
    list, silently starving the monitor's trend-break SMA, ML snapshots, the
    multi-timeframe analyzer, and risk indicators of data.)
    """
    raw_bars = extract_bar_dicts(bars_for_symbol)
    out: list[Kline] = []
    for i, bar in enumerate(raw_bars):
        if not isinstance(bar, dict):
            continue
        try:
            o = float(bar.get("o", bar.get("open", 0)))
            h = float(bar.get("h", bar.get("high", 0)))
            low = float(bar.get("l", bar.get("low", 0)))
            c = float(bar.get("c", bar.get("close", 0)))
            v = float(bar.get("v", bar.get("volume", 0)))
        except Exception:
            continue
        if c <= 0:
            continue
        # Synthetic monotonic times (in ms) — downstream code only uses
        # close prices for correlation, so the exact timestamp doesn't
        # matter as long as ordering is preserved.
        ts = i * 60_000
        out.append(
            Kline(
                open_time=ts,
                open=o,
                high=h,
                low=low,
                close=c,
                volume=v,
                close_time=ts + 60_000,
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One symbol's Alpaca snapshot, whatever shape it came in (each field None
    when absent or unparseable)."""

    last_trade: float | None
    bid: float | None
    ask: float | None
    minute_close: float | None
    daily_open: float | None
    daily_close: float | None
    daily_volume: float | None
    prev_close: float | None


def _field(entry: dict[str, Any], nodes: tuple[str, ...], keys: tuple[str, ...]) -> float | None:
    """The first parseable value of any ``node.key`` pair, nodes tried in order."""
    for node in nodes:
        inner = entry.get(node)
        if isinstance(inner, dict):
            for key in keys:
                value = to_float(inner.get(key))
                if value is not None:
                    return value
    return None


def snapshot_entry(payload: Any, symbol: str) -> dict[str, Any] | None:
    """The per-symbol dict inside a snapshot response: unwraps a ``"data"``
    envelope and a ``{symbol: {...}}`` map, else takes the payload as the entry."""
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("data"), dict):
        payload = payload["data"]
    entry = payload.get(symbol) or payload.get(symbol.upper()) or payload
    return entry if isinstance(entry, dict) else None


def parse_snapshot(payload: Any, symbol: str) -> Snapshot | None:
    """Alpaca's snapshot (``latestTrade.p``, ``latestQuote.bp/ap``, ``minuteBar``,
    ``dailyBar``, ``prevDailyBar``) as the live server sends it, tolerating the
    long snake_case keys some SDKs and MCP versions use."""
    entry = snapshot_entry(payload, symbol)
    if entry is None:
        return None
    daily = ("dailyBar", "daily_bar")
    return Snapshot(
        last_trade=_field(entry, ("latestTrade", "latest_trade", "trade"), ("p", "price")),
        bid=_field(entry, ("latestQuote", "latest_quote"), ("bp", "bid_price")),
        ask=_field(entry, ("latestQuote", "latest_quote"), ("ap", "ask_price")),
        minute_close=_field(entry, ("minuteBar", "minute_bar"), ("c", "close")),
        daily_open=_field(entry, daily, ("o", "open")),
        daily_close=_field(entry, daily, ("c", "close")),
        daily_volume=_field(entry, daily, ("v", "volume")),
        prev_close=_field(entry, ("prevDailyBar", "prev_daily_bar"), ("c", "close")),
    )


def extract_last_price(snap: Any, symbol: str) -> float | None:
    """The latest trade's price in a snapshot response, or None."""
    s = parse_snapshot(snap, symbol)
    return s.last_trade if s is not None else None


def compute_indicators_by_symbol(
    bars_by_symbol: dict[str, Any],
) -> tuple[dict[str, list[Kline]], dict[str, dict[str, Any]]]:
    """Run :func:`bars_to_klines` + :func:`compute_all` over a bars payload.

    Returns ``(klines_by_symbol, indicators_cache)`` so callers that need
    both (the risk engine wants klines, the regime detector wants
    indicators) only pay the parse cost once.
    """
    klines_by_symbol: dict[str, list[Kline]] = {}
    indicators_cache: dict[str, dict[str, Any]] = {}
    for symbol, raw in bars_by_symbol.items():
        klines = bars_to_klines(raw)
        if not klines:
            continue
        klines_by_symbol[symbol] = klines
        indicators_cache[symbol] = compute_all(klines)
    return klines_by_symbol, indicators_cache
