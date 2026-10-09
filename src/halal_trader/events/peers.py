"""Peer read-through: does an earnings surprise move the company's industry peers?
(the reactor's second rebuilt test, pre-registered 2026-10-09)

The first rebuilt test (events/earnings_signal.py) found nothing in reacting to
a company's own earnings at its first tradable price: the auction prices it.
News about one company reaches its industry more slowly (slow diffusion along
economic links is a documented effect), which is the industry-aware reactor
the operator asked for. So each earnings release (the leader) is read for its
peers: the other companies in its SEC industry.

Pre-registered (written before any result was seen):

* **leaders:** every earnings release with a revenue surprise
  (earnings_signal.releases); the signal is that surprise;
* **peers:** the ``PEERS`` largest other companies in the leader's SEC
  industry by the market cap of the screen in force at the release, leaving
  out any peer that reported its own earnings within ``OWN_REPORT`` days;
* **outcome:** each peer entered at its first tradable price after the release
  (events/study.py: timing, costs, abnormal to SPY); **one observation per
  release**, the peers' equal-weighted mean, since hundreds of peer returns
  from one event are not independent; fewer than ``MIN_PEERS`` peers: skipped;
* **primary:** leaders in Technology (halal/sector_limits.py); secondary: all;
* **rule:** trained on 2016-2021: the top decile of leader surprise must give
  its peers a positive 5-day mean net abnormal return with t >= 2; then
  2022-2024 must agree in sign with t >= 2; 2025-2026 stays untouched.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.earnings_signal import Release
from halal_trader.events.study import Observation, evaluate
from halal_trader.market_hours import MARKET_TZ

PEERS = 20
MIN_PEERS = 3
OWN_REPORT = timedelta(days=3)
HORIZONS = (1, 5, 20)


async def _industries(engine: AsyncEngine) -> tuple[dict[str, str], dict[date, dict[str, float]]]:
    """(symbol -> SEC industry, screen date -> symbol -> market cap)."""
    sic: dict[str, str] = {}
    caps: dict[date, dict[str, float]] = defaultdict(dict)
    async with engine.connect() as conn:
        for r in await conn.execute(
            text(
                "SELECT as_of, symbol, sic_description AS sic, "
                "(metrics->>'price')::float * (metrics->>'shares_outstanding')::float AS cap "
                "FROM halal_screen_results"
            )
        ):
            if r.sic:
                sic[r.symbol] = r.sic
            if r.cap:
                caps[r.as_of][r.symbol] = float(r.cap)
    return sic, dict(caps)


def peers_of(
    leader: Release,
    *,
    sic: dict[str, str],
    caps_by_screen: dict[date, dict[str, float]],
    screen_dates: list[date],
    reported: dict[str, list[date]],
) -> list[str]:
    """The ``PEERS`` largest same-industry companies not reporting near the leader."""
    industry = sic.get(leader.symbol)
    day = leader.published_at.astimezone(MARKET_TZ).date()
    i = bisect_right(screen_dates, day) - 1
    if industry is None or i < 0:
        return []
    caps = caps_by_screen[screen_dates[i]]

    def reports_near(symbol: str) -> bool:
        return any(abs(d - day) <= OWN_REPORT for d in reported.get(symbol, []))

    candidates = [
        s
        for s, c in caps.items()
        if s != leader.symbol and sic.get(s) == industry and not reports_near(s)
    ]
    return sorted(candidates, key=lambda s: -caps[s])[:PEERS]


async def peer_outcomes(
    engine: AsyncEngine, leaders: list[Release], *, sector: str | None
) -> list[tuple[Observation, int, float]]:
    """One (observation, horizon, mean peer net abnormal return) per leader release."""
    from halal_trader.events.earnings_signal import releases
    from halal_trader.halal.sector_limits import cap_sector

    sic, caps_by_screen = await _industries(engine)
    screen_dates = sorted(caps_by_screen)
    reported: dict[str, list[date]] = defaultdict(list)
    for r in await releases(engine):
        reported[r.symbol].append(r.published_at.astimezone(MARKET_TZ).date())

    peer_obs: list[Observation] = []
    leader_of: list[Release] = []
    for leader in leaders:
        if leader.sales_surprise is None:
            continue
        if sector is not None and cap_sector(leader.symbol, sic.get(leader.symbol)) != sector:
            continue
        peers = peers_of(
            leader,
            sic=sic,
            caps_by_screen=caps_by_screen,
            screen_dates=screen_dates,
            reported=reported,
        )
        if len(peers) < MIN_PEERS:
            continue
        k = len(leader_of)
        leader_of.append(leader)
        peer_obs += [
            Observation(p, leader.published_at, leader.sales_surprise, {"leader": k}) for p in peers
        ]

    by_leader: dict[tuple[int, int], list[float]] = defaultdict(list)
    tags: dict[int, dict[str, Any]] = {}
    for obs, h, ret in await evaluate(engine, peer_obs, HORIZONS):
        k = int(obs.tags["leader"])  # type: ignore[call-overload]
        by_leader[(k, h)].append(ret)
        tags[k] = {"year": obs.tags.get("year")}
    out = []
    for (k, h), rets in by_leader.items():
        if len(rets) >= MIN_PEERS:
            leader = leader_of[k]
            obs = Observation(
                leader.symbol, leader.published_at, leader.sales_surprise or 0.0, tags[k]
            )
            out.append((obs, h, sum(rets) / len(rets)))
    return out
