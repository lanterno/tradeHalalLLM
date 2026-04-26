"""Stock catalyst feed — news, earnings, insider transactions for the LLM prompt.

The stock cycle today only sees price + indicators. Real day-trading
edge on stocks lives in *catalysts*:

* Breaking headlines (8-K filings, analyst actions, FDA decisions)
* Pending earnings (volatility skew + post-earnings drift)
* Insider Form 4 transactions (cluster buys / sells often precede moves)

This module is the **abstraction**: a small protocol + a default
implementation that pulls from any client exposing the right async
methods. Real wire-up to Alpaca's news endpoint (or SEC EDGAR) lives in
follow-up tasks; this PR ships the structural seam + tests so the
prompt section is ready the moment those endpoints land.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Catalyst:
    """A single time-stamped catalyst event for one symbol."""

    symbol: str
    kind: str  # "news" | "earnings" | "insider_buy" | "insider_sell" | "analyst"
    title: str
    timestamp: datetime
    sentiment: str = "neutral"  # "positive" | "negative" | "neutral"
    source: str = ""
    url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class CatalystSource(Protocol):
    """Anything that can produce a list of recent catalysts on demand."""

    async def fetch(self, symbols: Sequence[str]) -> list[Catalyst]: ...


class StockCatalystFeed:
    """Aggregates one or more :class:`CatalystSource`s into a single feed.

    Each source can fail independently — a flaky news API shouldn't take
    out the earnings calendar. Failures are logged, not raised, so the
    cycle can continue on whatever signal *is* available.
    """

    def __init__(self, sources: Sequence[CatalystSource] | None = None) -> None:
        self._sources: list[CatalystSource] = list(sources or [])

    def add_source(self, source: CatalystSource) -> None:
        self._sources.append(source)

    async def fetch_all(self, symbols: Sequence[str]) -> list[Catalyst]:
        """Pull from every source, swallow per-source errors, return combined list."""
        if not self._sources or not symbols:
            return []
        out: list[Catalyst] = []
        for source in self._sources:
            try:
                out.extend(await source.fetch(symbols))
            except Exception as e:  # noqa: BLE001 — never let a source crash the cycle
                logger.debug("Catalyst source %s failed: %s", type(source).__name__, e)
        out.sort(key=lambda c: c.timestamp, reverse=True)
        return out


_KIND_GLYPH = {
    "news": "📰",
    "earnings": "📊",
    "insider_buy": "▲",
    "insider_sell": "▼",
    "analyst": "🎯",
}
_SENTIMENT_GLYPH = {"positive": "+", "negative": "-", "neutral": "·"}


def format_catalysts_for_prompt(
    catalysts: Sequence[Catalyst],
    *,
    symbols: Sequence[str] | None = None,
    limit: int = 8,
    max_age_hours: int = 24,
) -> str:
    """Render the most recent ``limit`` catalysts as a compact bullet list.

    ``symbols`` (optional) restricts to events whose ``symbol`` is in the
    set — useful to avoid burning tokens on the full halal universe when
    we only care about today's actionable names. ``max_age_hours`` drops
    stale events so a long-running bot doesn't anchor on yesterday's
    news after a quiet morning.
    """
    if not catalysts:
        return ""

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    sym_set = {s.upper() for s in symbols} if symbols else None

    fresh: list[Catalyst] = []
    for c in catalysts:
        ts = c.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < cutoff:
            continue
        if sym_set and c.symbol.upper() not in sym_set:
            continue
        fresh.append(c)

    if not fresh:
        return ""

    chosen = fresh[:limit]
    lines: list[str] = []
    for c in chosen:
        glyph = _KIND_GLYPH.get(c.kind, "·")
        sentiment = _SENTIMENT_GLYPH.get(c.sentiment, "·")
        meta_parts = [c.kind.upper(), sentiment]
        if c.source:
            meta_parts.append(f"({c.source})")
        meta = " ".join(meta_parts)
        lines.append(f"  - {glyph} [{c.symbol}] {c.title} — {meta}")
    return "\n".join(lines)


# ── Default Alpaca-news adapter (best-effort, no extra deps) ──


class AlpacaNewsSource:
    """Adapter that pulls news from any client exposing ``get_stock_news``.

    The Alpaca MCP server may or may not expose this tool depending on
    install — when it doesn't, ``fetch`` returns an empty list and the
    feed degrades gracefully. The adapter purposely doesn't import
    ``AlpacaMCPClient`` to keep this module testable with stubs.
    """

    def __init__(self, client: Any, *, lookback_hours: int = 24) -> None:
        self._client = client
        self._lookback_hours = lookback_hours

    async def fetch(self, symbols: Sequence[str]) -> list[Catalyst]:
        if not symbols:
            return []
        if not hasattr(self._client, "get_stock_news"):
            return []
        try:
            raw = await self._client.get_stock_news(",".join(symbols))
        except Exception as e:
            logger.debug("Alpaca news fetch failed: %s", e)
            return []
        return [_parse_news_item(item) for item in (raw or []) if isinstance(item, dict)]


def _parse_news_item(item: dict) -> Catalyst:
    """Best-effort mapping from an Alpaca/IEX news payload to :class:`Catalyst`."""
    ts_raw = item.get("created_at") or item.get("published_at") or item.get("timestamp")
    if isinstance(ts_raw, str):
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.now(timezone.utc)
    elif isinstance(ts_raw, datetime):
        ts = ts_raw
    else:
        ts = datetime.now(timezone.utc)

    symbols = item.get("symbols") or item.get("tickers") or []
    sym = (symbols[0] if symbols else item.get("symbol", "")).upper()

    return Catalyst(
        symbol=sym,
        kind="news",
        title=str(item.get("headline") or item.get("title") or "")[:140],
        timestamp=ts,
        source=item.get("source") or item.get("publisher") or "alpaca",
        url=item.get("url", ""),
        sentiment=str(item.get("sentiment") or "neutral").lower(),
    )


# ── Static source (tests / offline) ──────────────────────────────


class StaticCatalystSource:
    """Returns a fixed list — useful for tests, replay, and dry-runs."""

    def __init__(self, catalysts: Sequence[Catalyst]) -> None:
        self._catalysts = list(catalysts)

    async def fetch(self, symbols: Sequence[str]) -> list[Catalyst]:
        if not symbols:
            return list(self._catalysts)
        sym_set = {s.upper() for s in symbols}
        return [c for c in self._catalysts if c.symbol.upper() in sym_set]


# ── Earnings calendar adapter ────────────────────────────────────


class EarningsCalendarSource:
    """Adapter that consumes any client exposing ``get_calendar``.

    Alpaca's calendar endpoint returns a list of ``{date, symbol,
    estimate, ...}`` rows — we map upcoming entries to
    ``kind="earnings"`` catalysts. When the underlying client doesn't
    expose ``get_calendar``, ``fetch`` quietly returns ``[]`` so the
    rest of the cycle continues.
    """

    def __init__(self, client: Any, *, look_ahead_days: int = 7) -> None:
        self._client = client
        self._look_ahead_days = look_ahead_days

    async def fetch(self, symbols: Sequence[str]) -> list[Catalyst]:
        if not symbols or not hasattr(self._client, "get_calendar"):
            return []
        try:
            rows = await self._client.get_calendar(symbols=list(symbols))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Earnings calendar fetch failed: %s", exc)
            return []
        sym_set = {s.upper() for s in symbols}
        out: list[Catalyst] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            sym = str(row.get("symbol", "")).upper()
            if sym_set and sym not in sym_set:
                continue
            ts = _parse_calendar_ts(row)
            if ts is None:
                continue
            estimate = row.get("eps_estimate") or row.get("estimate")
            out.append(
                Catalyst(
                    symbol=sym,
                    kind="earnings",
                    title=f"Earnings expected{f' (est ${estimate})' if estimate else ''}",
                    timestamp=ts,
                    source="alpaca-calendar",
                    extra={k: v for k, v in row.items() if k != "symbol"},
                )
            )
        return out


def _parse_calendar_ts(row: dict) -> datetime | None:
    raw = row.get("date") or row.get("ts") or row.get("when")
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if not isinstance(raw, str):
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


# ── Pre-event sizing policy ──────────────────────────────────────


@dataclass(frozen=True)
class CatalystRiskPolicy:
    """How aggressively to shrink positions ahead of high-impact events.

    Defaults: in the 4h window before earnings or other ``kind`` listed
    in ``high_impact_kinds``, use 50% of normal max size. Outside the
    window: full size.
    """

    high_impact_kinds: tuple[str, ...] = ("earnings", "fomc", "fda")
    pre_event_hours: float = 4.0
    pre_event_size_multiplier: float = 0.5

    def size_multiplier_for(
        self,
        symbol: str,
        catalysts: Sequence[Catalyst],
        *,
        now: datetime | None = None,
    ) -> float:
        now = now or datetime.now(timezone.utc)
        sym = symbol.upper()
        for c in catalysts:
            if c.symbol.upper() != sym:
                continue
            if c.kind not in self.high_impact_kinds:
                continue
            ts = c.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            delta = (ts - now).total_seconds() / 3600.0
            if 0 <= delta <= self.pre_event_hours:
                return self.pre_event_size_multiplier
        return 1.0


def next_catalyst_window(
    catalysts: Sequence[Catalyst],
    *,
    symbol: str | None = None,
    now: datetime | None = None,
    look_ahead_hours: float = 24.0,
) -> Catalyst | None:
    """Return the soonest upcoming catalyst within ``look_ahead_hours``."""
    now = now or datetime.now(timezone.utc)
    sym = symbol.upper() if symbol else None
    upcoming: list[Catalyst] = []
    for c in catalysts:
        ts = c.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = (ts - now).total_seconds() / 3600.0
        if delta < 0 or delta > look_ahead_hours:
            continue
        if sym is not None and c.symbol.upper() != sym:
            continue
        upcoming.append(c)
    if not upcoming:
        return None
    return min(upcoming, key=lambda c: c.timestamp)
