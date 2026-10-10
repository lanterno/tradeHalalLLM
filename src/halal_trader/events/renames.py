"""Renamed tickers, and the news Benzinga filed under the old ones (news engine, §A.8).

The screen's history is keyed by today's tickers (SEC's ticker file maps each
filer to its current symbol), but Benzinga tagged each article with the
ticker of its day. The news backfill kept articles for the symbols the screen
covers, so a company that changed ticker has no news before the change:
Facebook's articles say FB, the screen says META.

``TICKER_RENAMES`` maps each old ticker to the current symbol and the last
session the company traded under the old one. It was seeded with
``seed_candidates`` -- symbols whose first news comes more than a year after
their first halal screen -- plus the symbols whose news has halal quarters
with none at all (a ticker taken over from another company hides the gap:
Gardner Denver took IR from Ingersoll-Rand, Axon took AXON from Axovant),
and every date was checked against the company's own filing or release.

``backfill_renamed_news`` fetches each old ticker's articles over the days it
named this company (``news_window``) and stores them under the current
symbol. ``payload.symbols`` stays as Benzinga sent it. Resumable: one unit per
old ticker and month (task ``news-renamed``, unit ``OLD:YYYY-MM``).

Prices follow the company, news follows the ticker: a news row under a ticker
another company held on its day is that company's. ``owner`` says, for a news
row, which of today's symbols it belongs to (see there for the rule); the
alias learner and the story builder keep a row only where it is its own
symbol's.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.http import Pacer
from halal_trader.events.history import NEWS_FROM, _done, mark_units, news_records
from halal_trader.events.store import EventRecord, EventRecorder
from halal_trader.market_hours import (
    MARKET_TZ,
    next_trading_day,
    today_eastern,
    trading_day_end_utc,
    trading_day_start_utc,
)

logger = logging.getLogger(__name__)

TASK: Final = "news-renamed"
SEED_GAP_DAYS: Final = 365  # first news this long after the first halal screen: a candidate
# Benzinga keeps tagging an old ticker for a few days after the switch (DWDP
# has rows after 2019-05-31), so an old ticker nobody took over at once names
# its company for this many sessions more.
GRACE_SESSIONS: Final = 5
_MAX_PAGES = 500  # 25,000 articles: more than any old ticker's busiest month

# old ticker -> (current symbol, last session under the old ticker).
# The date is the session before the new ticker's first, from the company's
# 8-K, press release or exchange notice.
TICKER_RENAMES: Final[dict[str, tuple[str, date]]] = {
    "AAXN": ("AXON", date(2021, 1, 25)),  # Axon Enterprise; AXON from 2021-01-26
    "AOBC": ("SWBI", date(2020, 5, 29)),  # American Outdoor Brands Corp -> Smith & Wesson
    "BBBY": ("NXH", date(2026, 8, 14)),  # Bed Bath & Beyond, Inc. -> Neighborhood Intelligence
    "BIGC": ("CMRC", date(2025, 7, 31)),  # BigCommerce -> Commerce.com
    "BTX": ("ERNA", date(2022, 10, 14)),  # Brooklyn ImmunoTherapeutics -> Eterna (Ernexa)
    "BYON": ("NXH", date(2025, 8, 28)),  # Beyond -> Bed Bath & Beyond, Inc.
    "CDEV": ("PR", date(2022, 9, 1)),  # Centennial Resource Development -> Permian Resources
    "CHK": ("EXE", date(2024, 10, 1)),  # Chesapeake Energy -> Expand Energy
    "COH": ("TPR", date(2017, 10, 30)),  # Coach -> Tapestry
    "CREE": ("WOLF", date(2021, 10, 1)),  # Cree -> Wolfspeed
    "DPS": ("KDP", date(2018, 7, 9)),  # Dr Pepper Snapple -> Keurig Dr Pepper
    "DRQ": ("INVX", date(2024, 9, 6)),  # Dril-Quip -> Innovex International
    "DSW": ("DBI", date(2019, 4, 1)),  # DSW -> Designer Brands
    "DWDP": ("DD", date(2019, 5, 31)),  # DowDuPont -> DuPont de Nemours
    "ELY": ("CALY", date(2022, 9, 6)),  # Callaway Golf -> Topgolf Callaway Brands (MODG)
    "EXPI": ("AGNT", date(2026, 5, 7)),  # eXp World Holdings -> AGNT
    "FB": ("META", date(2022, 6, 8)),  # Facebook -> Meta Platforms
    "FBHS": ("FBIN", date(2022, 12, 14)),  # Fortune Brands Home & Security -> Innovations
    "FLT": ("CPAY", date(2024, 3, 22)),  # FLEETCOR -> Corpay
    "GDI": ("IR", date(2020, 2, 28)),  # Gardner Denver -> Ingersoll Rand Inc.
    "GPS": ("GAP", date(2024, 8, 21)),  # Gap Inc. kept its name, changed its ticker
    "HRS": ("LHX", date(2019, 6, 28)),  # Harris -> L3Harris Technologies
    "IAC": ("PPLI", date(2026, 6, 3)),  # IAC (the 2020 spin-off) -> People Inc.
    "IR": ("TT", date(2020, 2, 28)),  # Ingersoll-Rand plc -> Trane Technologies
    "JCOM": ("ZD", date(2021, 10, 7)),  # J2 Global -> Ziff Davis
    "JEC": ("J", date(2019, 12, 9)),  # Jacobs Engineering -> Jacobs
    "KORS": ("CPRI", date(2018, 12, 31)),  # Michael Kors Holdings -> Capri Holdings
    "MODG": ("CALY", date(2026, 1, 15)),  # Topgolf Callaway Brands -> Callaway Golf
    "OSTK": ("NXH", date(2023, 11, 3)),  # Overstock.com -> Beyond
    "PCLN": ("BKNG", date(2018, 2, 26)),  # Priceline Group -> Booking Holdings
    "PKI": ("RVTY", date(2023, 5, 15)),  # PerkinElmer -> Revvity
    "PSTG": ("P", date(2026, 4, 16)),  # Pure Storage -> Everpure
    "Q": ("IQV", date(2017, 11, 14)),  # Quintiles IMS -> IQVIA
    "SQ": ("XYZ", date(2025, 1, 17)),  # Square / Block
    "SWHC": ("SWBI", date(2016, 12, 30)),  # Smith & Wesson Holding -> AOBC
    "TASR": ("AXON", date(2017, 4, 5)),  # TASER International -> Axon Enterprise (AAXN)
    "UBNT": ("UI", date(2019, 8, 19)),  # Ubiquiti Networks -> Ubiquiti Inc.
    "UTX": ("RTX", date(2020, 4, 2)),  # United Technologies -> Raytheon Technologies
    "VSCO": ("VSXY", date(2026, 6, 1)),  # Victoria's Secret & Co.
    "ZI": ("GTM", date(2025, 5, 12)),  # ZoomInfo Technologies
}

# (ticker, current symbol of the company) -> the first session the ticker
# named that company, where the rule below does not already give it.
#   * An old ticker names its company from the session after the previous
#     old ticker of the same company, else from NEWS_FROM.
#   * A current ticker with old tickers names its company from the day after
#     the last of them; one without names it throughout, unless listed.
# Rows under a ticker before the company held it are another company's
# (or nobody's): the old IAC, BioTime, Quintiles' Q before Qnity took it.
HELD_SINCE: Final[dict[tuple[str, str], date]] = {
    ("BTX", "ERNA"): date(2021, 3, 26),  # BioTime (now LCTX) traded as BTX until 2019
    ("DWDP", "DD"): date(2017, 9, 1),  # DowDuPont's first session (the merger closed 08-31)
    ("IAC", "PPLI"): date(2020, 7, 1),  # the old IAC, now Match Group, was IAC to 2020-06-30
    ("Q", "Q"): date(2025, 11, 3),  # Qnity Electronics, regular-way after DuPont's spin-off
}


def old_tickers(symbol: str) -> tuple[str, ...]:
    """The tickers ``symbol``'s company traded under before, oldest first."""
    olds = [(last, old) for old, (current, last) in TICKER_RENAMES.items() if current == symbol]
    return tuple(old for _, old in sorted(olds))


def window(old: str) -> tuple[date, date]:
    """The sessions [first, last] the old ticker named its company, from NEWS_FROM."""
    current, last = TICKER_RENAMES[old]
    first = HELD_SINCE.get((old, current))
    if first is None:
        earlier = [
            prior for o, (c, prior) in TICKER_RENAMES.items() if c == current and prior < last
        ]
        first = next_trading_day(max(earlier)) if earlier else NEWS_FROM
    return max(first, NEWS_FROM), last


def held_since(symbol: str) -> date | None:
    """The first day ``symbol`` named the company trading under it today, when
    another company (or none) held it inside the news history; None otherwise.

    A ``HELD_SINCE`` entry for (symbol, symbol), else the day after the last
    session of its company's last old ticker (the switch happens at that
    close, so the weekend after it is already the new ticker's).
    """
    explicit = HELD_SINCE.get((symbol, symbol))
    if explicit is not None:
        return explicit
    lasts = [last for current, last in TICKER_RENAMES.values() if current == symbol]
    return max(lasts) + timedelta(days=1) if lasts else None


def news_window(old: str) -> tuple[date, date]:
    """The days [first, last] news under the old ticker is its company's.

    ``window`` plus ``GRACE_SESSIONS`` sessions, but never into the days
    another company held the ticker (IR passed straight to Gardner Denver).
    """
    first, last = window(old)
    end = last
    for _ in range(GRACE_SESSIONS):
        end = next_trading_day(end)
    taken = held_since(old)
    if taken is not None and taken > last:
        end = min(end, taken - timedelta(days=1))
    return first, end


def months(first: date, last: date) -> list[tuple[date, date]]:
    """[first, last] cut into calendar months: (first day, last day) of each, clipped."""
    out: list[tuple[date, date]] = []
    lo = first
    while lo <= last:
        nxt = date(lo.year + lo.month // 12, lo.month % 12 + 1, 1)
        out.append((lo, min(nxt - timedelta(days=1), last)))
        lo = nxt
    return out


def unit(old: str, month: date) -> str:
    return f"{old}:{month:%Y-%m}"


# ── who a news row belongs to ─────────────────────────────────


@dataclass(frozen=True, slots=True)
class TickerHistory:
    """``TICKER_RENAMES`` and ``HELD_SINCE`` as lookups (``owner`` per row)."""

    windows: dict[str, tuple[str, date, date]]  # old -> (current, news_window)
    olds: dict[str, tuple[str, ...]]  # current -> its old tickers
    since: dict[str, date]  # ticker -> held_since
    involved: frozenset[str]

    @classmethod
    def build(cls) -> TickerHistory:
        windows = {old: (now, *news_window(old)) for old, (now, _) in TICKER_RENAMES.items()}
        olds = {current: old_tickers(current) for current, _ in TICKER_RENAMES.values()}
        tickers = {t for pair in HELD_SINCE for t in pair} | set(windows) | set(olds)
        since = {t: s for t in tickers if (s := held_since(t)) is not None}
        return cls(windows, olds, since, frozenset(windows.keys() | olds.keys() | since.keys()))

    def owner(self, symbol: str, day: date, tagged: Collection[str] | None) -> str | None:
        if symbol not in self.involved:
            return symbol
        if tagged:
            for old in self.olds.get(symbol, ()):
                _, first, last = self.windows[old]
                if old in tagged and first <= day <= last:
                    return symbol
        named = self.windows.get(symbol)
        if named is not None and named[1] <= day <= named[2]:
            return named[0]
        since = self.since.get(symbol)
        if since is not None:
            return symbol if day >= since else None
        return None  # an old ticker outside its window: another company's, or nobody's


_history: dict[str, tuple[object, object, TickerHistory]] = {}


def ticker_history() -> TickerHistory:
    """The lookups for the current tables (rebuilt when either table is replaced)."""
    cached = _history.get("now")
    if cached is None or cached[0] is not TICKER_RENAMES or cached[1] is not HELD_SINCE:
        cached = (TICKER_RENAMES, HELD_SINCE, TickerHistory.build())
        _history["now"] = cached
    return cached[2]


def owner(symbol: str, day: date, tagged: Collection[str] | None) -> str | None:
    """Which of today's symbols a news row stored under ``symbol`` belongs to.

    ``day`` is the New York date the row was published; ``tagged`` is its
    ``payload.symbols`` as Benzinga sent it (None for a live row, which has
    none). The first rule that applies decides:

    1. ``tagged`` names an old ticker of ``symbol`` on a day of that ticker's
       ``news_window``: ``symbol``. These are the rows ``backfill_renamed_news``
       copied here (FB's articles under META), and articles tagging both
       tickers (Gardner Denver's GDI next to the old IR).
    2. ``symbol`` is an old ticker and ``day`` falls in its ``news_window``:
       its current symbol. The row is a duplicate: the backfill stored the
       same article under the current symbol, and that copy is the one kept
       (IAC's rows count for PPLI once, DWDP's for DD).
    3. ``symbol`` has a ``held_since`` day: ``symbol`` from that day on, None
       before (Pandora's P, Axovant's AXON, E.I. du Pont's DD, the metaverse
       ETF's META, Grupo Aeroportuario's GAP, the old IAC).
    4. ``symbol`` is an old ticker outside its window: None (another company's,
       or Benzinga still tagging a retired ticker).
    5. Otherwise ``symbol``.

    A row belongs to its own symbol only when this returns that symbol; the
    alias learner and the story builder drop every other row. Rule 2 relies
    on the backfill being complete.
    """
    return ticker_history().owner(symbol, day, tagged)


def renamed_records(articles: Iterable[Any], old: str, current: str) -> list[EventRecord]:
    """The old ticker's articles as events of the current symbol (``news_records``' rows)."""
    return [dataclasses.replace(r, symbol=current) for r in news_records(articles, {old})]


# ── seeding ───────────────────────────────────────────────────


async def seed_candidates(
    engine: AsyncEngine, *, today: date | None = None
) -> list[tuple[str, date, date | None]]:
    """(symbol, first halal screen, first news) of every symbol whose first news
    is more than ``SEED_GAP_DAYS`` after its first halal screen.

    A symbol with no news at all is a candidate once its first halal screen
    is that old (its first news is then None).
    """
    today = today or today_eastern()
    async with engine.connect() as conn:
        halal = {
            r.symbol: r.first
            for r in await conn.execute(
                text(
                    "SELECT symbol, min(as_of) AS first FROM halal_screen_current "
                    "WHERE verdict = 'halal' GROUP BY symbol"
                )
            )
        }
        news = {
            r.symbol: r.first.astimezone(MARKET_TZ).date()
            for r in await conn.execute(
                text(
                    "SELECT symbol, min(published_at) AS first FROM events "
                    "WHERE kind = 'news' AND symbol = ANY(:s) GROUP BY symbol"
                ),
                {"s": sorted(halal)},
            )
        }
    out: list[tuple[str, date, date | None]] = []
    for symbol, first_halal in sorted(halal.items()):
        first_news = news.get(symbol)
        reference = first_news if first_news is not None else today
        if (reference - first_halal).days > SEED_GAP_DAYS:
            out.append((symbol, first_halal, first_news))
    return out


# ── the backfill ──────────────────────────────────────────────


async def backfill_renamed_news(
    engine: AsyncEngine,
    market: Any,
    *,
    rate_per_min: int = 100,
    now: datetime | None = None,
) -> int:
    """Store every old ticker's articles under its current symbol; returns events written.

    ``market.news`` is called at most ``rate_per_min`` times a minute (the
    client paces its own page requests). A month is marked done once its
    last day is over.
    """
    now = now or datetime.now(UTC)
    done = await _done(engine, TASK)
    recorder = EventRecorder(engine, raise_errors=True)
    pacer = Pacer(60.0 / max(rate_per_min, 1))
    stored = 0
    for old in sorted(TICKER_RENAMES):
        current = TICKER_RENAMES[old][0]
        first, last = news_window(old)
        written_old = 0
        for lo, hi in months(first, last):
            u = unit(old, lo)
            if u in done:
                continue
            end = trading_day_end_utc(hi)
            await pacer.wait()
            articles = await market.news(
                [old],
                start=trading_day_start_utc(lo),
                end=end - timedelta(seconds=1),
                max_pages=_MAX_PAGES,
            )
            written = await recorder.record(renamed_records(articles, old, current))
            if end <= now:
                await mark_units(engine, TASK, {u: written})
            written_old += written
        if written_old:
            logger.info("renamed news %s -> %s: %d events", old, current, written_old)
        stored += written_old
    return stored
