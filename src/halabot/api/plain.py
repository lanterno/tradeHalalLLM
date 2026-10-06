"""The engine's internals in plain words, for the belief board.

Pure functions, no I/O: an evidence item's name and sentence, a belief's
stance, a shadow decision's reason. The engine records what it computed
(``indicator.relstrength``, ``rel +4.26% vs SPY``); a reader wants
"Outperforming SPY by 4.3%". Every rule falls back to the raw detail, so a
new interpreter or a changed detail format reads as itself rather than as
nothing.
"""

from __future__ import annotations

import re
from typing import Any

# Evidence source -> the name a reader knows it by.
LABELS: dict[str, str] = {
    "indicator.relstrength": "Relative strength",
    "indicator.rsi": "RSI",
    "indicator.alignment": "Trend",
    "indicator.multiframe": "Moving averages",
    "indicator.structure": "Price structure",
    "indicator.momentum": "Momentum",
    "indicator.volume": "Volume",
    "news": "News",
    "forecaster": "Forecast",
    "anomaly": "Volatility",
    "drift": "Drift",
}

# Above this RSI the momentum engines still vote fully bullish; the board says
# that out loud rather than letting a stretched stock read as a clean signal.
RSI_CAUTION = 80
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

_PCT = re.compile(r"([+-]?\d+(?:\.\d+)?)%")
_NUM = r"([+-]?\d+(?:\.\d+)?)"


def evidence_label(source: str) -> str:
    if source in LABELS:
        return LABELS[source]
    tail = source.rsplit(".", 1)[-1].replace("_", " ")
    return tail[:1].upper() + tail[1:]


def _rsi(detail: str) -> int | None:
    m = re.search(r"RSI\s+(\d+(?:\.\d+)?)", detail)
    return round(float(m.group(1))) if m else None


def evidence_sentence(source: str, direction: float, detail: str) -> str:
    """One evidence item as a sentence (the raw detail when no rule fits)."""
    try:
        sentence = _sentence(source, direction, detail)
    except ValueError, IndexError:
        sentence = None
    return sentence or detail


def _sentence(source: str, direction: float, detail: str) -> str | None:
    if source == "indicator.relstrength":
        m = _PCT.search(detail)
        if not m:
            return None
        v = float(m.group(1))
        bench = re.search(r"vs\s+(\S+)", detail)
        return (
            f"{'Outperforming' if v >= 0 else 'Lagging'} "
            f"{bench.group(1) if bench else 'the market'} by {abs(v):.1f}%"
        )
    if source == "indicator.rsi":
        rsi = _rsi(detail)
        if rsi is None:
            return None
        tag = (
            " · overbought"
            if rsi >= RSI_OVERBOUGHT
            else (" · oversold" if rsi <= RSI_OVERSOLD else "")
        )
        return f"RSI {rsi}{tag}"
    if source == "indicator.alignment":
        values = [float(x) for x in _PCT.findall(detail)[:2]]
        if len(values) < 2:
            return None
        short, long_ = values
        word = (
            "Uptrend"
            if short > 0 and long_ > 0
            else ("Downtrend" if short < 0 and long_ < 0 else "Mixed trend")
        )
        return f"{word}: {short:+.1f}% short, {long_:+.1f}% long"
    if source == "indicator.multiframe":
        return "Moving averages stacked " + ("up" if direction >= 0 else "down")
    if source == "indicator.momentum":
        return "Short-term momentum " + ("up" if direction >= 0 else "down")
    if source == "news":
        headline = detail.split(": ", 1)[1] if ": " in detail else detail
        return f"News: “{headline.strip()}”"
    if source == "forecaster":
        m = re.search(r"proj\s+" + _NUM + "%", detail)
        if not m:
            return None
        v = float(m.group(1))
        if "/bar" in detail:
            return f"Forecast trend {v:+.1f}% an hour"
        bars = re.search(r"(\d+)-bar", detail)
        hours = int(bars.group(1)) if bars else 5
        return f"Forecast {v:+.1f}% over the next {hours} hours"
    if source == "indicator.structure":
        m = re.match(r"near (support|resistance)\s+" + _NUM, detail)
        if m:
            return f"Near {m.group(1)} ${float(m.group(2)):,.2f}"
        return detail[:1].upper() + detail[1:] if detail else None
    if source == "indicator.volume":
        m = re.search(r"vol\s+" + _NUM + r"x.*?" + _NUM + "%", detail)
        if not m:
            return None
        return f"Volume {float(m.group(1)):.1f}× normal behind a {float(m.group(2)):+.1f}% move"
    if source == "anomaly":
        m = re.search(r"spike\s+" + _NUM + "x", detail)
        if not m:
            return None
        return f"Volatility {float(m.group(1)):.1f}× its usual level"
    if source == "drift":
        m = re.search(r"z=" + _NUM, detail)
        if not m:
            return None
        return f"Returns shifting from their usual pattern (z {float(m.group(1)):.1f})"
    return None


def rsi_caution(evidence: list[dict[str, Any]]) -> str | None:
    """A note when a very high RSI is counted in the stock's favour."""
    for e in evidence:
        if e.get("source") != "indicator.rsi" or float(e.get("direction", 0)) <= 0:
            continue
        rsi = _rsi(str(e.get("detail", "")))
        if rsi is not None and rsi >= RSI_CAUTION:
            return (
                f"RSI {rsi} is counted as fully bullish. Momentum engines do that, "
                f"but at {rsi} the stock is stretched: a reader should know."
            )
    return None


# Stances, in the order the board filters them.
STANCES = ("long", "leaning", "none", "excluded", "benchmark")


def stance(
    *,
    asset: str,
    direction: str,
    conviction: float,
    strict: str,
    benchmark: str,
    entry_band: float,
    exit_band: float,
) -> str:
    """What the engine would do with the name now, in one word.

    ``excluded`` when the strict screen does not pass it (the engine follows
    strict-halal names only), ``benchmark`` for the index it is measured
    against, else ``long`` above the entry band, ``leaning`` above the exit
    band (it would hold, not open), and ``none`` below.
    """
    if asset == benchmark:
        return "benchmark"
    if strict != "halal":
        return "excluded"
    if direction == "long_bias" and conviction >= entry_band:
        return "long"
    if direction == "long_bias" and conviction >= exit_band:
        return "leaning"
    return "none"


_FORCED: dict[str, str] = {
    "price_break": "forced exit: price broke its invalidation level",
    "compliance_lapsed": "forced exit: its halal verdict lapsed",
    "invalidated": "forced exit: the belief was invalidated",
}

_EPS = 1e-9


def decision_reason(payload: dict[str, Any], *, entry_band: float, exit_band: float) -> str:
    """Why a shadow proposal was made, from its payload."""
    reason = str(payload.get("reason") or "")
    if payload.get("forced_exit"):
        return _FORCED.get(reason, f"forced exit: {reason.replace('_', ' ')}".rstrip(": "))
    if reason.startswith("risk halt"):
        return reason
    side = str(payload.get("side") or "")
    current = float(payload.get("current_weight") or 0.0)
    target = float(payload.get("target_weight") or 0.0)
    raw = payload.get("conviction_raw")
    level = f"{float(raw) * 100:.0f}%" if isinstance(raw, int | float) else None
    if side == "buy":
        text = (
            f"conviction crossed {entry_band * 100:.0f}%"
            if current <= _EPS
            else (f"conviction rose to {level}" if level else "conviction rose")
        )
    elif side == "sell" and target <= _EPS:
        # Sold out with conviction still above the exit band: something else
        # zeroed the target (the direction turned, a gate), so don't claim a
        # fall the payload contradicts.
        if isinstance(raw, int | float) and float(raw) >= exit_band:
            text = f"target fell to zero at conviction {level}"
        else:
            text = f"conviction fell below {exit_band * 100:.0f}%"
    elif side == "sell":
        text = f"conviction fell to {level}" if level else "conviction fell"
    else:
        text = reason or "—"
    if "gross-normalized" in reason:
        text += " · book scaled to its exposure cap"
    return text


MACRO_PLAIN: dict[str, str] = {
    "CPI": "inflation print",
    "FOMC": "Fed rate decision",
    "NFP": "jobs report",
    "GDP": "growth estimate",
}
