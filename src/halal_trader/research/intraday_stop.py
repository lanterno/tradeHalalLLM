"""Should a day-trader buy carry a default stop? A stop below the fill, tested on 15-minute bars.
(pre-registered 2026-10-22, HAL-2)

The day-trader's prompt asks every buy for a stop_loss, and
StockPositionMonitor sells at market once the price reaches it. 59 of the
140 buys of 2026-10-08..16 had none, so the question was whether to give
them one. The five stop exits so far were wrong-way (the names beat SPY
after the sale), and intraday moves tend to revert, so a stop may sell the
lows. This tests the stop itself, before one is added.

Pre-registered (written before any result was seen):

* **entries:** every observation of the entry test (research/intraday_entry.py):
  its universe, bars, decision times and volume baseline, bought at T (the
  open of the bar starting at T);
* **stop:** ``DEPTHS`` below that price, 1% and 2%. The first bar from T on
  whose low reaches it sells at the stop, or at the bar's open when it
  opened below (a gap), less ``SLIPPAGE`` (the monitor polls every 30 s and
  sells at market). Without the stop the position is sold at the close, as
  the end-of-day flatten does;
* **outcome:** the stop's value, (stop sale - close) / entry: what selling at
  the stop instead of at the close made per unit bought, zero when the stop
  is never reached. Either way the position is sold once, so the round trip
  costs the same and the value is net. Not against SPY: after the stop the
  money sits in cash;
* **statistic:** mean of session means, t across sessions;
* **rule, ALL entries, each depth:** trained on 2016-10..2021, then held out
  on 2022-2024. *Supported* (a default stop pays) if the value is positive
  with t >= 2 in both windows; *refuted* (it costs) if negative with t <= -2
  in both; otherwise inconclusive. CHASE and DIP entries, the share of
  entries stopped and the value over stopped entries alone are reported,
  not judged. 2025-2026 stays untouched.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

from halal_trader.research import intraday_entry as ie
from halal_trader.trading.decision_quality import Stat, clustered

DEPTHS = (0.01, 0.02)
SLIPPAGE = 0.0005


@dataclass(frozen=True, slots=True)
class Summary:
    group: str
    depth: float
    value: Stat  # over every entry
    stopped: float  # share of entries the stop sold
    when_stopped: Stat  # over the entries it sold


def stop_sale(s: ie.Session, k: int, depth: float) -> float | None:
    """The price a stop ``depth`` below a buy at bar ``k``'s open sells at, if it is reached."""
    stop = s.open[k] * (1 - depth)
    for j in range(k, ie.BARS):
        if s.low[j] <= stop:
            return min(s.open[j], stop) * (1 - SLIPPAGE)
    return None


def summarise(
    obs: Iterable[ie.Observation],
    bars: Mapping[str, Mapping[date, ie.Session]],
    group: str,
    depth: float,
) -> Summary:
    every: list[tuple[date, float]] = []
    sold: list[tuple[date, float]] = []
    for o in obs:
        if not ie.in_group(o, group):
            continue
        s = bars[o.symbol][o.day]
        sale = stop_sale(s, o.bar, depth)
        value = 0.0 if sale is None else (sale - s.close[-1]) / s.open[o.bar]
        every.append((o.day, value))
        if sale is not None:
            sold.append((o.day, value))
    share = len(sold) / len(every) if every else 0.0
    return Summary(group, depth, clustered(every), share, clustered(sold))


def verdict(train: Stat, holdout: Stat) -> str:
    """The pre-registered rule on one depth's ALL entries (t has the mean's sign)."""
    ts = (train.t, holdout.t)
    if all(t is not None and t <= -ie.T_BAR for t in ts):
        return "refuted"
    if all(t is not None and t >= ie.T_BAR for t in ts):
        return "supported"
    return "inconclusive"
