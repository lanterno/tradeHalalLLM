"""Options-implied-volatility surface signals.

We don't trade options (halal interpretation rules them out for now),
but the **information** in the options market is free and powerful.
The two reads we care about for stock-direction sizing:

* **ATM implied volatility** — what the options market is pricing in
  for movement over the next ~30 days. Spikes precede big moves more
  often than not.
* **Put-call skew (25Δ)** — how much more expensive out-of-the-money
  puts are than calls. Persistent positive skew = the market is
  paying up for downside protection. Useful pre-earnings signal.

Yahoo Finance exposes the full options chain at
``query2.finance.yahoo.com/v7/finance/options/<TICKER>`` — public,
no key. Rate limit isn't published but is generous; we cache 15 min
per symbol.

Output is one ``OptionsIVSnapshot`` per symbol, fed straight into the
stock-side catalyst feed alongside FRED + EDGAR. The prompt block
shows ATM IV + skew per ticker so the LLM can see what the options
market is pricing for the underlying.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


_API_BASE = "https://query2.finance.yahoo.com/v7/finance/options"
_CACHE_TTL_S = 15 * 60  # 15 min — option chains don't churn faster than that


@dataclass(frozen=True)
class OptionsIVSnapshot:
    """ATM IV + skew metrics for one underlying."""

    symbol: str
    spot: float
    atm_iv: float  # 0..1 (e.g. 0.28 = 28%)
    put_call_skew: float  # IV(put 25Δ) - IV(call 25Δ); + = puts richer
    call_volume: int
    put_volume: int
    call_open_interest: int
    put_open_interest: int

    @property
    def put_call_volume_ratio(self) -> float:
        if self.call_volume <= 0:
            return float("inf") if self.put_volume > 0 else 0.0
        return self.put_volume / self.call_volume

    @property
    def label(self) -> str:
        if self.atm_iv >= 0.6:
            return "elevated_iv"
        if self.put_call_skew >= 0.05:
            return "downside_premium"
        if self.put_call_volume_ratio >= 1.5:
            return "put_heavy_flow"
        return "neutral"


@dataclass
class _CacheEntry:
    fetched_at: float
    snapshot: OptionsIVSnapshot | None


_BREAKER_THRESHOLD = 5  # consecutive failures before opening


@dataclass
class YahooOptionsIV:
    """Pulls Yahoo Finance options chains and reduces them to one
    ``OptionsIVSnapshot`` per symbol.

    Public endpoint, no key. Always-on; the operator doesn't need to
    enable anything. ``user_agent`` is for politeness — Yahoo doesn't
    enforce it, but a generic one occasionally gets 403'd.

    Session-level circuit breaker: after ``_BREAKER_THRESHOLD``
    consecutive non-200 responses (typically 401 when Yahoo rotates
    its anti-bot tokens), the client stops calling the endpoint for
    the rest of the process lifetime. Without this, every cycle was
    burning ~10 HTTP requests + log lines on a source that wasn't
    going to recover until the process restarts.
    """

    user_agent: str = "halal-trader/0.1 (options-iv)"
    near_money_band: float = 0.10  # ±10% of spot for ATM filter
    _client: Any | None = None
    _cache: dict[str, _CacheEntry] = field(default_factory=dict)
    _consecutive_failures: int = 0
    _circuit_open: bool = False

    async def fetch(self, symbols: Sequence[str]) -> dict[str, OptionsIVSnapshot]:
        if not symbols:
            return {}
        out: dict[str, OptionsIVSnapshot] = {}
        for sym in symbols:
            sym_u = sym.upper()
            cached = self._cache.get(sym_u)
            if cached and (time.monotonic() - cached.fetched_at) < _CACHE_TTL_S:
                if cached.snapshot is not None:
                    out[sym_u] = cached.snapshot
                continue
            try:
                snap = await self._fetch_for(sym_u)
            except Exception as exc:  # noqa: BLE001
                logger.debug("options IV fetch failed for %s: %s", sym_u, exc)
                snap = None
            self._cache[sym_u] = _CacheEntry(fetched_at=time.monotonic(), snapshot=snap)
            if snap is not None:
                out[sym_u] = snap
        return out

    async def _fetch_for(self, symbol: str) -> OptionsIVSnapshot | None:
        if self._circuit_open:
            return None
        client = await self._get_client()
        resp = await client.get(f"{_API_BASE}/{symbol}")
        if resp.status_code != 200:
            logger.debug("yahoo options %s returned %d", symbol, resp.status_code)
            self._consecutive_failures += 1
            if (
                not self._circuit_open
                and self._consecutive_failures >= _BREAKER_THRESHOLD
            ):
                self._circuit_open = True
                logger.warning(
                    "yahoo options IV circuit breaker OPEN after %d consecutive "
                    "non-200 responses — silencing for the rest of this process. "
                    "Last status code: %d",
                    self._consecutive_failures,
                    resp.status_code,
                )
            return None
        # Reset on first success.
        if self._consecutive_failures or self._circuit_open:
            logger.info(
                "yahoo options IV recovered after %d failures",
                self._consecutive_failures,
            )
        self._consecutive_failures = 0
        self._circuit_open = False
        data = resp.json()
        results = (data.get("optionChain", {}) or {}).get("result", []) or []
        if not results:
            return None
        result = results[0]
        spot = float(((result.get("quote") or {}).get("regularMarketPrice") or 0) or 0)
        if spot <= 0:
            return None
        chains = result.get("options", []) or []
        if not chains:
            return None
        # Use the nearest expiry — that's what most ATM-IV reads
        # quote anyway; longer-dated expiries get noisier.
        first_expiry = chains[0]
        calls = list(first_expiry.get("calls", []) or [])
        puts = list(first_expiry.get("puts", []) or [])
        if not calls or not puts:
            return None
        return _reduce(symbol, spot, calls, puts, self.near_money_band)

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=10.0,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None


# ── Pure reducer ─────────────────────────────────────────────────


def _reduce(
    symbol: str,
    spot: float,
    calls: list[dict],
    puts: list[dict],
    band: float,
) -> OptionsIVSnapshot | None:
    """Roll one expiry's chain into a single :class:`OptionsIVSnapshot`.

    * ``atm_iv`` = mean IV of calls + puts within ``band`` of spot.
    * ``put_call_skew`` = mean IV of OTM puts (~25Δ) minus mean IV of
      OTM calls (~25Δ). 25Δ is a rough proxy via strike distance —
      the chain doesn't ship deltas, so we use ±15–25% strike offset
      as a stand-in.
    """
    lo, hi = spot * (1 - band), spot * (1 + band)

    atm_ivs: list[float] = []
    for opt in calls + puts:
        iv = _safe_float(opt.get("impliedVolatility"))
        strike = _safe_float(opt.get("strike"))
        if iv <= 0 or strike <= 0:
            continue
        if lo <= strike <= hi:
            atm_ivs.append(iv)
    if not atm_ivs:
        return None
    atm_iv = sum(atm_ivs) / len(atm_ivs)

    # 25Δ-ish proxy: OTM puts at strike ≈ spot × [0.75..0.90],
    # OTM calls at strike ≈ spot × [1.10..1.25].
    put_skew_band = (spot * 0.75, spot * 0.90)
    call_skew_band = (spot * 1.10, spot * 1.25)
    otm_put_ivs = [
        _safe_float(p.get("impliedVolatility"))
        for p in puts
        if put_skew_band[0] <= _safe_float(p.get("strike")) <= put_skew_band[1]
        and _safe_float(p.get("impliedVolatility")) > 0
    ]
    otm_call_ivs = [
        _safe_float(c.get("impliedVolatility"))
        for c in calls
        if call_skew_band[0] <= _safe_float(c.get("strike")) <= call_skew_band[1]
        and _safe_float(c.get("impliedVolatility")) > 0
    ]
    if otm_put_ivs and otm_call_ivs:
        skew = sum(otm_put_ivs) / len(otm_put_ivs) - sum(otm_call_ivs) / len(otm_call_ivs)
    else:
        skew = 0.0

    call_volume = sum(_safe_int(c.get("volume")) for c in calls)
    put_volume = sum(_safe_int(p.get("volume")) for p in puts)
    call_oi = sum(_safe_int(c.get("openInterest")) for c in calls)
    put_oi = sum(_safe_int(p.get("openInterest")) for p in puts)

    return OptionsIVSnapshot(
        symbol=symbol,
        spot=spot,
        atm_iv=atm_iv,
        put_call_skew=skew,
        call_volume=call_volume,
        put_volume=put_volume,
        call_open_interest=call_oi,
        put_open_interest=put_oi,
    )


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except TypeError, ValueError:
        return 0.0


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except TypeError, ValueError:
        return 0


# ── Prompt formatting ────────────────────────────────────────────


def format_options_iv_for_prompt(
    snapshots: dict[str, OptionsIVSnapshot],
    *,
    max_rows: int = 8,
) -> str:
    """One block summarising the options market for each watched stock.

    Empty result returns "" so the prompt builder can elide it. Sorts
    by ATM IV so the most-active names land at the top.
    """
    if not snapshots:
        return ""
    lines = ["Options market (Yahoo, nearest expiry):"]
    rows = sorted(snapshots.values(), key=lambda s: -s.atm_iv)
    for s in rows[:max_rows]:
        skew_sign = "+" if s.put_call_skew >= 0 else ""
        lines.append(
            f"  {s.symbol:<6} ATM IV {s.atm_iv:.0%} | "
            f"P-C skew {skew_sign}{s.put_call_skew:.2%} | "
            f"P/C vol {s.put_call_volume_ratio:.2f}x | "
            f"label {s.label}"
        )
    return "\n".join(lines)
