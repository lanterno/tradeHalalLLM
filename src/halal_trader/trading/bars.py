"""Alpaca bars → ``Kline`` adapter + indicator helper.

Shared by the cycle, the snapshot recorder and any stage that wants
indicators on stock bars. The indicator + risk infrastructure takes
``Kline`` objects; this module coerces Alpaca's ``get_stock_bars``
response into that shape.
"""

from __future__ import annotations

import logging
from typing import Any

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


_PRICE_PATHS: tuple[tuple[str, ...], ...] = (
    ("latestTrade", "p"),
    ("latestTrade", "price"),
    ("latest_trade", "p"),
    ("latest_trade", "price"),
    ("trade", "p"),
    ("trade", "price"),
)


def extract_last_price(snap: Any, symbol: str) -> float | None:
    """Best-effort dig through Alpaca snapshot shapes for the latest price.

    Alpaca returns either a flat dict or a nested ``{symbol: {...}}``
    depending on whether one or many symbols were requested. Inside
    each entry, the latest trade lives under ``latestTrade.p`` (or
    ``latest_trade.price`` in some SDK versions). Returns ``None`` when
    no parseable price is found.
    """
    if not isinstance(snap, dict):
        return None
    payload = snap.get(symbol) or snap.get(symbol.upper()) or snap
    if not isinstance(payload, dict):
        return None
    for path in _PRICE_PATHS:
        node: Any = payload
        ok = True
        for key in path:
            if not isinstance(node, dict) or key not in node:
                ok = False
                break
            node = node[key]
        if ok and node is not None:
            try:
                return float(node)
            except TypeError, ValueError:
                continue
    return None


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
