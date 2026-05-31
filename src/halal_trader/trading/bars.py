"""Alpaca bars → ``Kline`` adapter + indicator helper.

Promoted from a private helper in ``trading/risk.py`` so it's reusable
from the cycle, snapshot recorder, backtest harness, and any future
regime/timeframe/ML stages that want indicators on stock bars.

The crypto cycle ingests ``Kline`` objects (Binance shape); this module
coerces Alpaca's ``get_stock_bars`` response into the same shape so
the shared indicator + risk + regime infrastructure runs unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from halal_trader.crypto.indicators import compute_all
from halal_trader.domain.models import Kline

logger = logging.getLogger(__name__)


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
    if not bars_for_symbol:
        return []
    raw_bars: list[dict[str, Any]]
    if isinstance(bars_for_symbol, dict):
        raw_bars = bars_for_symbol.get("bars") or bars_for_symbol.get("data") or []
        # ``get_stock_bars`` wraps bars under the symbol: {"bars": {"NVDA": [...]}}.
        # Flatten that symbol level to the underlying bar list.
        if isinstance(raw_bars, dict):
            flattened: list[Any] = []
            for v in raw_bars.values():
                if isinstance(v, list):
                    flattened.extend(v)
            raw_bars = flattened
    elif isinstance(bars_for_symbol, list):
        raw_bars = bars_for_symbol
    else:
        return []

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
