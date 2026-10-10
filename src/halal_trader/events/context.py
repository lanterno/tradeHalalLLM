"""Point-in-time context for news trials: what was known about a name before its story.

A story's reaction session S is judged on information dated strictly before
it, every comparison strict:

* **Screen.** The newest ``halal_screen_current`` screen with ``as_of < S``;
  PRIMARY needs ``verdict == 'halal'`` and a CIK. ``strict.verdict`` is not
  used: its freshness rule is for live orders, and history's screens are
  quarterly. BROAD (a sensitivity) also admits rows whose one reason is a
  Shariah index's exclusion, and unmapped rows (no CIK) of names that are not
  funds (``ticker_ciks.status = 'fund'``).
* **Liquidity.** The 0-based index in ``universe_at(S, top_n=3000)``, which
  ranks on the months before S's month; eligible below 1000.
* **Share class.** Among the names admitted at S (screen and rank) that share
  the screen row's CIK, only the most liquid is kept.
* **Price.** ``prev_close_s >= 5``: the previous session's raw close in
  session-S units, ``close_raw(S−1) · A(S−1)/A(S)`` with
  ``A(d) = close_all(d)/close_raw(d)``. The ratio holds exactly the corporate
  actions effective at S's open, which are announced beforehand.
* **Volatility.** σ and β come from the 60 sessions whose close is strictly
  before the news (``at_news``, always required): pre-open news uses S−1
  back, news after the close of N uses N back, news late in N's session N−1
  back. At least 40 daily abnormal returns ``C_all(x)/C_all(x−1) − 1 − (SPY
  the same)`` must be valid, else ``no_sigma``. News at or after S's close,
  or no later than S−2's close, cannot belong to a story reacting in S
  (every story item comes after S−1's close less 90 minutes): refused.
* **Descriptives** (ATR, dollar volume, momentum, levels) use sessions up to
  S−1, and are NaN where their bars are missing; they gate nothing.
  **Facts** are the earnings facts published strictly before ``at``.

Loading reads each source once per run: the screens in one query, the
universe once per month, daily bars (raw and all-adjusted) streamed in
batches of 100 symbols, SPY always. Not point in time, and stated: the SIC
description behind the sector, and the restated SEC facts behind a screen.
"""

from __future__ import annotations

import logging
import math
import re
from bisect import bisect_left
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.data.minutes import session_bounds
from halal_trader.events.earnings_parse import EarningsFacts
from halal_trader.events.study import BENCHMARK, cost_bps
from halal_trader.halal.sector_limits import TECHNOLOGY, cap_sector
from halal_trader.halal.strict import ScreenRow, all_screens
from halal_trader.signals.indicators import atr

logger = logging.getLogger(__name__)

Reason = Literal[
    "ok",
    "no_screen",
    "not_halal",
    "unmapped",
    "rank",
    "price",
    "share_class",
    "no_daily",
    "no_sigma",
]
Universe = Literal["primary", "broad"]

TOP_N: Final = 3000  # the universe the rank is read in
MAX_RANK: Final = 1000  # eligible below this 0-based rank
MIN_PREV_CLOSE: Final = 5.0
SIGMA_SESSIONS: Final = 60
SIGMA_MIN_OBS: Final = 40
BETA_CLIP: Final = (0.5, 2.0)
ATR_PERIOD: Final = 14
ATR_SESSIONS: Final = 60  # bars Wilder's ATR runs over: fixed, so loaded history cannot move it
ADV_SESSIONS: Final = 20
LEVEL_SESSIONS: Final = (20, 252)
# Daily bars are read from LOOKBACK_DAYS before the first session (252
# sessions for the 52-week levels, with holidays to spare) to LOOKAHEAD_DAYS
# after the last, which covers a 3-session path's closes and A-ratios.
LOOKBACK_DAYS: Final = 380
LOOKAHEAD_DAYS: Final = 7
FACTS_LOOKBACK_DAYS: Final = 200
BATCH_SYMBOLS: Final = 100
# compliance/index_veto.apply_veto's reason, as SQL's LIKE 'excluded by %Shariah index%'.
INDEX_VETO: Final = re.compile(r"excluded by .*Shariah index", re.DOTALL)

# Rows of a symbol's (8, sessions) array: raw open/high/low/close/volume, then
# the all-adjusted high/low/close. NaN where the bar is missing.
_RO, _RH, _RL, _RC, _RV, _AH, _AL, _AC = range(8)


@dataclass(frozen=True, slots=True)
class Eligibility:
    """Whether a name is in the trial's universe at a session, and why not."""

    eligible: bool
    reason: Reason
    universe: Universe  # broad adds index-veto-only and unmapped-only rows (sensitivity)
    screen_as_of: date | None
    verdict: str | None
    cik: int | None
    sector: str | None
    tech: bool
    liquidity_rank: int | None  # 0-based index in universe_at(S, top_n=3000)
    cost_bps: float  # study.cost_bps(rank), one way


@dataclass(frozen=True, slots=True)
class PreEvent:
    """A name's state before its story's news, levels in session-S raw units."""

    session: date
    prev_session: date
    prev_close_s: float  # close_raw(S-1) * A(S-1) / A(S)
    spy_prev_close_s: float
    sigma: float  # sd(ddof=1) of daily abnormal returns over the sigma window
    sigma_n: int  # valid returns in that window
    beta: float  # OLS slope on SPY over the same returns, clipped to [0.5, 2.0]
    atr_pct: float  # Wilder ATR(14) on all-adjusted bars / their last close (descriptive)
    adv20_usd: float  # mean raw close * raw volume, 20 sessions
    ret5_vs_spy: float
    ret20_vs_spy: float
    hi20_s: float
    lo20_s: float
    hi252_s: float
    lo252_s: float


@dataclass(frozen=True, slots=True)
class DailyPoint:
    """One session's raw daily bar and its adjustment factor A."""

    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj: float  # A(day) = close_all / close_raw


class _Series:
    """One symbol's daily bars aligned to the context's sessions."""

    __slots__ = ("data",)

    def __init__(self, sessions: int) -> None:
        self.data: NDArray[np.float64] = np.full((8, sessions), np.nan)

    def adj(self, j: int) -> float | None:
        raw, adjusted = float(self.data[_RC, j]), float(self.data[_AC, j])
        if not (raw > 0.0 and adjusted > 0.0):  # NaN fails too
            return None
        return adjusted / raw


def index_veto_only(reasons: Sequence[str]) -> bool:
    """Whether a screen row failed only because a Shariah index excludes it."""
    return len(reasons) == 1 and INDEX_VETO.match(reasons[0]) is not None


class PitContext:
    """Screens, liquidity ranks, daily bars and earnings facts, read once, asked
    point in time. Build it with :meth:`load`."""

    def __init__(
        self,
        *,
        start: date,
        end: date,
        sessions: Sequence[date],
        screens: Mapping[date, Mapping[str, ScreenRow]],
        funds: Collection[str],
        ranks: Mapping[date, Mapping[str, int]],
        series: Mapping[str, _Series | None],
        facts: Mapping[str, Sequence[tuple[datetime, EarningsFacts]]],
        facts_from: datetime,
        facts_to: datetime,
    ) -> None:
        self.sessions: list[date] = list(sessions)
        self._start, self._end = start, end
        self._bars_from = start - timedelta(days=LOOKBACK_DAYS)
        self._bars_to = end + timedelta(days=LOOKAHEAD_DAYS)
        self._index = {d: j for j, d in enumerate(self.sessions)}
        self._closes = [session_bounds(d)[1] for d in self.sessions]
        self._screens = screens
        self._screen_dates = sorted(screens)
        self._funds = frozenset(funds)
        self._ranks = ranks
        self._series = dict(series)
        spy = self._series.get(BENCHMARK)
        if spy is None:
            raise ValueError(f"{BENCHMARK} has no daily bars in the loaded range")
        self._spy = spy
        self._fact_times = {s: [t for t, _ in rows] for s, rows in facts.items()}
        self._facts = {s: [f for _, f in rows] for s, rows in facts.items()}
        self._facts_from, self._facts_to = facts_from, facts_to
        self._winners: dict[tuple[date, date, Universe], dict[int, str]] = {}

    @classmethod
    async def load(
        cls, engine: AsyncEngine, *, symbols: Collection[str], start: date, end: date
    ) -> PitContext:
        """Everything the questions about ``symbols`` at sessions in [start, end] need."""
        from halal_trader.compliance.delisted import fund_symbols
        from halal_trader.data.universe import month_starts, universe_at

        if end < start:
            raise ValueError(f"end {end} is before start {start}")
        wanted = sorted(set(symbols) - {BENCHMARK})
        lo = start - timedelta(days=LOOKBACK_DAYS)
        hi = end + timedelta(days=LOOKAHEAD_DAYS)
        screens = await _screens(engine)
        funds = await fund_symbols(engine)
        ranks: dict[date, dict[str, int]] = {}
        for month in month_starts(start, end):
            names = await universe_at(engine, month, top_n=TOP_N)
            ranks[month] = {s: i for i, s in enumerate(names)}
        sessions = await _sessions(engine, lo, hi)
        spy = await _bars(engine, [BENCHMARK], sessions, lo, hi)
        series: dict[str, _Series | None] = {BENCHMARK: spy.series.get(BENCHMARK)}
        off_calendar = 0
        for i in range(0, len(wanted), BATCH_SYMBOLS):
            batch = wanted[i : i + BATCH_SYMBOLS]
            loaded = await _bars(engine, batch, sessions, lo, hi)
            off_calendar += loaded.off_calendar
            for symbol in batch:
                series[symbol] = loaded.series.get(symbol)
        if off_calendar:
            logger.warning("pit context: %d daily bars on days SPY did not trade", off_calendar)
        facts_from = session_bounds(start)[0] - timedelta(days=FACTS_LOOKBACK_DAYS + 7)
        facts_to = session_bounds(end)[1] + timedelta(days=1)
        facts = await _facts(engine, wanted, facts_from, facts_to)
        logger.info(
            "pit context: %d symbols, %d sessions, %d screens, %d months, %d facts",
            len(wanted),
            len(sessions),
            len(screens),
            len(ranks),
            sum(len(v) for v in facts.values()),
        )
        return cls(
            start=start,
            end=end,
            sessions=sessions,
            screens=screens,
            funds=funds,
            ranks=ranks,
            series=series,
            facts=facts,
            facts_from=facts_from,
            facts_to=facts_to,
        )

    # ── questions ────────────────────────────────────────────────

    def eligibility(
        self,
        symbol: str,
        session: date,
        *,
        at_news: datetime,
        universe: Universe = "primary",
    ) -> Eligibility:
        """Whether ``symbol`` is in ``universe`` at reaction session ``session``
        for a story whose news came at ``at_news``.

        The first failing rule names the reason, in the order screen, CIK,
        rank, share class, daily bars, price, σ. σ is judged at ``at_news``
        (C.1), the window :meth:`pre_event` uses, so a story is never counted
        eligible on one σ window and traded on another.
        """
        j = self._session(session)
        series = self._loaded(symbol)
        k = self._news_index(j, at_news)
        month = session.replace(day=1)
        rank = self._ranks[month].get(symbol)
        as_of = self._screen_as_of(session)
        row = self._screens[as_of].get(symbol) if as_of is not None else None
        sector = cap_sector(symbol, row.sic_description) if row is not None else None

        def result(reason: Reason) -> Eligibility:
            return Eligibility(
                eligible=reason == "ok",
                reason=reason,
                universe=universe,
                screen_as_of=as_of,
                verdict=row.verdict if row is not None else None,
                cik=row.cik if row is not None else None,
                sector=sector,
                tech=sector == TECHNOLOGY,
                liquidity_rank=rank,
                cost_bps=cost_bps(rank),
            )

        if as_of is None:
            return result("no_screen")
        refused = self._refused(symbol, row, universe)
        if refused is not None:
            return result(refused)
        if rank is None or rank >= MAX_RANK:
            return result("rank")
        if row is not None and row.cik is not None:
            if self._class_winners(as_of, month, universe).get(row.cik) != symbol:
                return result("share_class")
        prev_close = _prev_close_s(series, j) if series is not None else None
        if prev_close is None:
            return result("no_daily")
        if prev_close < MIN_PREV_CLOSE:
            return result("price")
        if series is None or self._sigma_beta(series, k) is None:
            return result("no_sigma")
        return result("ok")

    def pre_event(self, symbol: str, session: date, at_news: datetime) -> PreEvent | None:
        """``symbol``'s pre-news state for a story at ``session`` whose news came at
        ``at_news``; None without the previous close and A-ratios, or without σ."""
        j = self._session(session)
        series = self._loaded(symbol)
        k = self._news_index(j, at_news)
        if series is None:
            return None
        prev_close = _prev_close_s(series, j)
        spy_prev_close = _prev_close_s(self._spy, j)
        if prev_close is None or spy_prev_close is None:
            return None
        fit = self._sigma_beta(series, k)
        if fit is None:
            return None
        sigma, sigma_n, beta = fit
        p = j - 1
        a_s = series.adj(j)
        if a_s is None:  # _prev_close_s already needed it
            return None
        d = series.data
        hi20, lo20 = _levels(series, p, LEVEL_SESSIONS[0], a_s)
        hi252, lo252 = _levels(series, p, LEVEL_SESSIONS[1], a_s)
        return PreEvent(
            session=session,
            prev_session=self.sessions[p],
            prev_close_s=prev_close,
            spy_prev_close_s=spy_prev_close,
            sigma=sigma,
            sigma_n=sigma_n,
            beta=beta,
            atr_pct=_atr_pct(series, p),
            adv20_usd=_mean_finite(
                d[_RC, _through(p, ADV_SESSIONS)] * d[_RV, _through(p, ADV_SESSIONS)]
            ),
            ret5_vs_spy=self._ret_vs_spy(series, p, 5),
            ret20_vs_spy=self._ret_vs_spy(series, p, 20),
            hi20_s=hi20,
            lo20_s=lo20,
            hi252_s=hi252,
            lo252_s=lo252,
        )

    def adj(self, symbol: str, day: date) -> float | None:
        """A(day) = close_all / close_raw; None without both bars on ``day``."""
        j = self._day(day)
        series = self._loaded(symbol)
        return series.adj(j) if series is not None and j is not None else None

    def daily(self, symbol: str, day: date) -> DailyPoint | None:
        """``symbol``'s raw daily bar on ``day`` with its A; None without both bars."""
        j = self._day(day)
        series = self._loaded(symbol)
        if series is None or j is None:
            return None
        a = series.adj(j)
        if a is None:
            return None
        o, h, lo, c, v = (float(x) for x in series.data[_RO : _RV + 1, j])
        if not all(math.isfinite(x) for x in (o, h, lo, v)):
            return None
        return DailyPoint(day, o, h, lo, c, v, a)

    def screen_verdict(self, symbol: str, day: date) -> str:
        """The verdict of the newest screen dated before ``day``: 'no_screen'
        when there is none, 'not_halal' when that screen does not hold ``symbol``."""
        as_of = self._screen_as_of(day)
        if as_of is None:
            return "no_screen"
        row = self._screens[as_of].get(symbol)
        return row.verdict if row is not None else "not_halal"

    def facts_before(
        self, symbol: str, at: datetime, *, lookback_days: int = FACTS_LOOKBACK_DAYS
    ) -> list[EarningsFacts]:
        """Earnings facts on ``symbol`` published in [at − lookback, at), oldest first."""
        if at.tzinfo is None:
            raise ValueError("at must be timezone-aware")
        since = at - timedelta(days=lookback_days)
        if since < self._facts_from or at > self._facts_to:
            raise ValueError(
                f"facts were loaded for [{self._facts_from}, {self._facts_to}), not [{since}, {at})"
            )
        self._loaded(symbol)
        times = self._fact_times.get(symbol, [])
        return self._facts.get(symbol, [])[bisect_left(times, since) : bisect_left(times, at)]

    # ── internals ────────────────────────────────────────────────

    def _session(self, day: date) -> int:
        if not self._start <= day <= self._end:
            raise ValueError(f"{day} is outside the loaded sessions {self._start}..{self._end}")
        j = self._index.get(day)
        if j is None:
            raise ValueError(f"{day} is not a session")
        return j

    def _day(self, day: date) -> int | None:
        if not self._bars_from <= day <= self._bars_to:
            raise ValueError(
                f"{day} is outside the loaded daily bars {self._bars_from}..{self._bars_to}"
            )
        return self._index.get(day)

    def _loaded(self, symbol: str) -> _Series | None:
        if symbol not in self._series:
            raise ValueError(f"{symbol} was not loaded")
        return self._series[symbol]

    def _screen_as_of(self, day: date) -> date | None:
        i = bisect_left(self._screen_dates, day)
        return self._screen_dates[i - 1] if i else None

    def _refused(self, symbol: str, row: ScreenRow | None, universe: Universe) -> Reason | None:
        """Why the screen keeps ``symbol`` out of ``universe``, or None when it admits it."""
        if row is None:
            return "not_halal"
        if row.verdict == "halal" and row.cik is not None:
            return None
        unmapped = row.cik is None and symbol not in self._funds
        if universe == "broad" and (unmapped or index_veto_only(row.reasons)):
            return None
        return "unmapped" if unmapped else "not_halal"

    def _class_winners(self, as_of: date, month: date, universe: Universe) -> dict[int, str]:
        """CIK -> its most liquid admitted symbol, for one screen and month."""
        key = (as_of, month, universe)
        if key not in self._winners:
            ranks = self._ranks[month]
            best: dict[int, tuple[int, str]] = {}
            for symbol, row in self._screens[as_of].items():
                rank = ranks.get(symbol)
                if row.cik is None or rank is None or rank >= MAX_RANK:
                    continue
                if self._refused(symbol, row, universe) is not None:
                    continue
                if row.cik not in best or rank < best[row.cik][0]:
                    best[row.cik] = (rank, symbol)
            self._winners[key] = {cik: symbol for cik, (_, symbol) in best.items()}
        return self._winners[key]

    def _news_index(self, j: int, at_news: datetime) -> int:
        """k, the number of sessions closed strictly before ``at_news``, for a
        story reacting in session j: always j−1 or j.

        News at or after session j's close cannot belong to a story reacting
        in j, and news no later than j−2's close belongs to an earlier
        session (a story's items come after j−1's close less 90 minutes).
        Either is a caller's mistake (say, a parent story's time), refused
        rather than let σ see session j or quietly measure an older window.
        """
        if at_news.tzinfo is None:
            raise ValueError("at_news must be timezone-aware")
        if at_news >= self._closes[j]:
            raise ValueError(
                f"news at {at_news} is after the close of its story's session {self.sessions[j]}"
            )
        k = bisect_left(self._closes, at_news)  # sessions[:k] closed before the news
        if k < j - 1:
            raise ValueError(
                f"news at {at_news} is no later than the close of {self.sessions[j - 2]}: "
                f"too early for a story reacting in {self.sessions[j]}"
            )
        return k

    def _sigma_beta(self, series: _Series, k: int) -> tuple[float, int, float] | None:
        """(σ, valid returns, clipped β) over the 60 sessions before index ``k``
        (those closed strictly before the news); None under 40 valid returns."""
        lo = max(k - SIGMA_SESSIONS, 1)  # a return on x needs the close of x-1
        if k - lo < SIGMA_MIN_OBS:
            return None
        c, m = series.data[_AC], self._spy.data[_AC]
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = c[lo:k] / c[lo - 1 : k - 1] - 1.0
            rm = m[lo:k] / m[lo - 1 : k - 1] - 1.0
        ok = np.isfinite(rs) & np.isfinite(rm)
        n = int(ok.sum())
        if n < SIGMA_MIN_OBS:
            return None
        rs, rm = rs[ok], rm[ok]
        sigma = float(np.std(rs - rm, ddof=1))
        var_m = float(np.var(rm, ddof=1))
        beta = float(np.cov(rs, rm, ddof=1)[0, 1]) / var_m if var_m > 0.0 else math.nan
        return sigma, n, min(max(beta, BETA_CLIP[0]), BETA_CLIP[1])

    def _ret_vs_spy(self, series: _Series, p: int, n: int) -> float:
        if p - n < 0:
            return math.nan
        c, m = series.data[_AC], self._spy.data[_AC]
        return float((c[p] / c[p - n] - 1.0) - (m[p] / m[p - n] - 1.0))


def _through(p: int, n: int) -> slice:
    """The ``n`` sessions ending at index ``p``, inclusive."""
    return slice(max(0, p - n + 1), p + 1)


def _prev_close_s(series: _Series, j: int) -> float | None:
    """close_raw(S−1) · A(S−1)/A(S), or None without those bars."""
    if j < 1:
        return None
    a_s, a_p = series.adj(j), series.adj(j - 1)
    if a_s is None or a_p is None:
        return None
    return float(series.data[_RC, j - 1]) * a_p / a_s


def _mean_finite(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else math.nan


def _atr_pct(series: _Series, p: int) -> float:
    window = _through(p, ATR_SESSIONS)
    h, lo, c = (series.data[row, window] for row in (_AH, _AL, _AC))
    ok = np.isfinite(h) & np.isfinite(lo) & np.isfinite(c)
    if int(ok.sum()) <= ATR_PERIOD:
        return math.nan
    return atr(h[ok], lo[ok], c[ok], ATR_PERIOD) / float(series.data[_AC, p])


def _levels(series: _Series, p: int, n: int, a_s: float) -> tuple[float, float]:
    """Highest high and lowest low of the ``n`` sessions through ``p``, in S units."""
    window = _through(p, n)
    d = series.data
    with np.errstate(divide="ignore", invalid="ignore"):
        a = d[_AC, window] / d[_RC, window]
        highs = d[_RH, window] * a / a_s
        lows = d[_RL, window] * a / a_s
    highs, lows = highs[np.isfinite(highs)], lows[np.isfinite(lows)]
    if not highs.size or not lows.size:
        return math.nan, math.nan
    return float(highs.max()), float(lows.min())


async def _screens(engine: AsyncEngine) -> dict[date, dict[str, ScreenRow]]:
    """Every screen's rows (the newest method's), by date then symbol."""
    screens = await all_screens(engine)
    return {as_of: {r.symbol: r for r in rows} for as_of, rows in screens.items()}


@dataclass(slots=True)
class _Loaded:
    series: dict[str, _Series]
    off_calendar: int = 0  # bars on a day that is not one of the sessions


async def _sessions(engine: AsyncEngine, lo: date, hi: date) -> list[date]:
    """The benchmark's raw daily-bar days in [lo, hi]: the calendar."""
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT day FROM daily_bars WHERE symbol = :s AND adjustment = 'raw' "
                "AND day >= :lo AND day <= :hi ORDER BY day"
            ),
            {"s": BENCHMARK, "lo": lo, "hi": hi},
        )
        return [r.day for r in rows]


async def _bars(
    engine: AsyncEngine, symbols: Sequence[str], sessions: Sequence[date], lo: date, hi: date
) -> _Loaded:
    """Raw and all-adjusted daily bars of ``symbols`` in [lo, hi], aligned to ``sessions``."""
    index = {d: j for j, d in enumerate(sessions)}
    out = _Loaded({})
    async with engine.connect() as conn:
        result = await conn.stream(
            text(
                "SELECT symbol, day, adjustment, open, high, low, close, volume FROM daily_bars "
                "WHERE symbol = ANY(:s) AND day >= :lo AND day <= :hi "
                "AND adjustment IN ('raw', 'all')"
            ),
            {"s": list(symbols), "lo": lo, "hi": hi},
        )
        async for r in result:
            j = index.get(r.day)
            if j is None:
                out.off_calendar += 1
                continue
            series = out.series.get(r.symbol)
            if series is None:
                series = out.series[r.symbol] = _Series(len(sessions))
            if r.adjustment == "raw":
                series.data[_RO : _RV + 1, j] = (r.open, r.high, r.low, r.close, r.volume)
            else:
                series.data[_AH : _AC + 1, j] = (r.high, r.low, r.close)
    return out


async def _facts(
    engine: AsyncEngine, symbols: Sequence[str], lo: datetime, hi: datetime
) -> dict[str, list[tuple[datetime, EarningsFacts]]]:
    """The current extractor's result and guidance facts on ``symbols``
    published in [lo, hi), each symbol's oldest first."""
    from halal_trader.events import earnings_parse

    out: dict[str, list[tuple[datetime, EarningsFacts]]] = {}
    if not symbols:
        return out
    async with engine.connect() as conn:
        result = await conn.stream(
            text(
                "SELECT e.symbol, e.published_at, f.kind, f.fields FROM event_facts f "
                "JOIN events e ON e.id = f.event_id WHERE f.extractor = :x "
                "AND f.kind IN ('result', 'guidance') AND e.symbol = ANY(:s) "
                "AND e.published_at >= :lo AND e.published_at < :hi "
                "ORDER BY e.symbol, e.published_at, e.id, f.id"
            ),
            {"x": earnings_parse.EXTRACTOR, "s": list(symbols), "lo": lo, "hi": hi},
        )
        async for r in result:
            out.setdefault(r.symbol, []).append(
                (r.published_at, EarningsFacts(r.kind, dict(r.fields)))
            )
    return out
