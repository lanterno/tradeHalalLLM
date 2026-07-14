"""Options-implied expected move from the ATM straddle (Phase 3).

Halal framing: this READS option-market data (public quotes) as a signal —
the bot never holds, writes, or trades a derivative. Reading the options
market to gauge an expected range is the same permissible act as reading an
index or a news feed; trading the derivative is the forbidden one.

Empirical note (verified 2026-07-14 against the live Alpaca MCP): this
account's **indicative** option feed returns per-contract quotes/trades/bars
but **NO greeks and NO implied_volatility** — so IV-based methods (IV rank,
25Δ skew, GEX) would need self-computed Black–Scholes inversion. The
straddle method needs none of that: the ATM straddle mid *is* the market's
expected move. EM ≈ 0.85·(ATM call mid + ATM put mid) — the
Brenner–Subrahmanyam approximation straddle ≈ 0.8·S·σ·√T, so the mid
divided by ~0.8 recovers the 1-σ move; practitioners quote 0.85·straddle as
the ~1-σ / ~68 % containment band to expiry.

The result is the market's own "how high / how low", to sit NEXT TO the
statistical HAR band in the recommendation — and their DISAGREEMENT is
itself signal (a much wider implied band flags an event premium, e.g.
earnings inside the horizon). Like every band input it is advisory until
its coverage is measured; the caller must not present it as validated.

Pure stdlib + the parsed chain dict; no numpy needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

# 0.85·straddle ≈ the 1-σ expected move (Brenner–Subrahmanyam); tune per the
# coverage measurement once IV history accrues.
STRADDLE_MULT = 0.85
_MIN_DTE = 2  # skip 0/1-DTE (gamma-dominated, not a "normal" range)
_MAX_SPREAD_FRAC = 0.35  # reject an ATM leg whose bid-ask spread > 35 % of mid


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """One parsed option contract with a usable mid, or a reason it's unusable."""

    symbol: str
    expiry: str  # YYYY-MM-DD
    is_call: bool
    strike: float
    mid: float | None
    spread_frac: float | None


@dataclass(frozen=True, slots=True)
class ExpectedMove:
    """Options-implied expected move to the chosen expiry."""

    expiry: str
    dte: int  # calendar days to expiry
    atm_strike: float
    spot: float
    straddle: float  # call mid + put mid
    move_abs: float  # 0.85·straddle
    move_pct: float  # move_abs / spot · 100
    low: float  # spot − move_abs
    high: float  # spot + move_abs


def parse_occ_symbol(symbol: str) -> OptionQuote | None:
    """Parse an OCC option symbol (e.g. ``AAPL260713C00215000``).

    Layout: underlying (1–6 chars, left-justified) + ``YYMMDD`` + ``C``/``P``
    + strike×1000 (8 digits). Returns a bare :class:`OptionQuote` (no mid)
    or ``None`` when the tail doesn't parse.
    """
    # The fixed tail is 15 chars: 6 date + 1 type + 8 strike.
    if len(symbol) < 16:
        return None
    tail = symbol[-15:]
    yy, mm, dd = tail[0:2], tail[2:4], tail[4:6]
    cp = tail[6]
    strike_raw = tail[7:15]
    if cp not in ("C", "P") or not (yy + mm + dd + strike_raw).isdigit():
        return None
    try:
        expiry = date(2000 + int(yy), int(mm), int(dd)).isoformat()
    except ValueError:
        return None
    return OptionQuote(
        symbol=symbol,
        expiry=expiry,
        is_call=cp == "C",
        strike=int(strike_raw) / 1000.0,
        mid=None,
        spread_frac=None,
    )


def _quote_mid(snapshot: dict[str, Any]) -> tuple[float | None, float | None]:
    """(mid, spread_fraction) from a snapshot's latestQuote, or (None, None).

    Requires a two-sided quote with positive, non-crossed bid/ask. The
    spread fraction gates illiquid ATM legs whose mid is noise.
    """
    q = snapshot.get("latestQuote") or {}
    bp, ap = q.get("bp"), q.get("ap")
    if bp is None or ap is None:
        return None, None
    try:
        bid = float(bp)
        ask = float(ap)
    except TypeError, ValueError:
        return None, None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None, None
    mid = (bid + ask) / 2.0
    return mid, (ask - bid) / mid if mid > 0 else None


def _priced_quotes(snapshots: dict[str, dict[str, Any]]) -> list[OptionQuote]:
    """Parse + price every contract in a chain snapshot dict."""
    out: list[OptionQuote] = []
    for sym, snap in snapshots.items():
        base = parse_occ_symbol(sym)
        if base is None or not isinstance(snap, dict):
            continue
        mid, spread = _quote_mid(snap)
        out.append(
            OptionQuote(
                symbol=base.symbol,
                expiry=base.expiry,
                is_call=base.is_call,
                strike=base.strike,
                mid=mid,
                spread_frac=spread,
            )
        )
    return out


def expected_move_from_chain(
    snapshots: dict[str, dict[str, Any]],
    spot: float,
    *,
    today: date,
    min_dte: int = _MIN_DTE,
) -> ExpectedMove | None:
    """Compute the ATM-straddle expected move from a chain snapshot dict.

    ``snapshots`` is the ``snapshots`` map from Alpaca's ``get_option_chain``
    (symbol → snapshot). Picks the nearest expiry at least ``min_dte``
    calendar days out (0/1-DTE is gamma-dominated, not a normal range),
    finds the strike nearest ``spot`` that has BOTH a usable call and put
    mid within the spread gate, and returns the straddle expected move.
    ``None`` when spot is invalid or no expiry has a clean ATM straddle.
    """
    if spot <= 0:
        return None
    quotes = _priced_quotes(snapshots)
    by_expiry: dict[str, list[OptionQuote]] = {}
    for q in quotes:
        if (date.fromisoformat(q.expiry) - today).days >= min_dte:
            by_expiry.setdefault(q.expiry, []).append(q)
    for expiry in sorted(by_expiry):
        legs = by_expiry[expiry]
        calls = {q.strike: q for q in legs if q.is_call and _usable(q)}
        puts = {q.strike: q for q in legs if not q.is_call and _usable(q)}
        common = sorted(set(calls) & set(puts), key=lambda k: abs(k - spot))
        if not common:
            continue
        atm = common[0]
        call_mid = calls[atm].mid
        put_mid = puts[atm].mid
        assert call_mid is not None and put_mid is not None  # _usable guaranteed
        straddle = call_mid + put_mid
        move_abs = STRADDLE_MULT * straddle
        return ExpectedMove(
            expiry=expiry,
            dte=(date.fromisoformat(expiry) - today).days,
            atm_strike=atm,
            spot=spot,
            straddle=round(straddle, 4),
            move_abs=round(move_abs, 4),
            move_pct=round(move_abs / spot * 100, 3),
            low=round(spot - move_abs, 4),
            high=round(spot + move_abs, 4),
        )
    return None


def _usable(q: OptionQuote) -> bool:
    return q.mid is not None and (q.spread_frac is None or q.spread_frac <= _MAX_SPREAD_FRAC)
