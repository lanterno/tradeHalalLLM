"""Zakat on a portfolio, by both of Dar al-Ifta's methods, choosing the higher.

Egypt's Dar al-Ifta (fatwa 8767, 2 October 2025, Grand Mufti Nazir Ayad)
distinguishes shares by why they are held:

* **trade goods** (held to profit from the price): 2.5% of the market value
  at the end of the lunar year;
* **income** (held for dividends): no zakat on the shares themselves, 2.5%
  of the dividends received over the year.

The operator's decision (2026-10-04): compute both every year, show both,
and pay the higher. Dividends are taken net of their purified share, which
was never the holder's to keep.

The hawl is a Hijri day and month (``ZAKAT_HAWL_HIJRI``, "MM-DD"),
converted with the Umm al-Qura calendar; Egypt's moon-sighting can differ
from it by a day. Not done here, because they concern the whole of the
holder's wealth rather than one account: the nisab test (85 g of 21-carat
gold in Dar al-Ifta's fatwa) and the cash held alongside the shares.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from hijridate import Gregorian, Hijri
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

RATE = 0.025
SOURCE = "Dar al-Ifta al-Misriyya, fatwa 8767 (2025-10-02): both methods, the higher paid"


def parse_hawl(hijri: str) -> tuple[int, int]:
    """'09-01' -> (9, 1): month and day of the Hijri hawl."""
    month, day = (int(x) for x in hijri.strip().split("-"))
    if not (1 <= month <= 12 and 1 <= day <= 30):
        raise ValueError(f"ZAKAT_HAWL_HIJRI must be MM-DD, got {hijri!r}")
    return month, day


def _gregorian(year: int, month: int, day: int) -> date:
    """The Gregorian date of a Hijri day, a 30th falling back to a 29-day month's end."""
    length = Hijri(year, month, 1).month_length()
    g = Hijri(year, month, min(day, length)).to_gregorian()
    return date(g.year, g.month, g.day)


def hawl_period(hawl: tuple[int, int], on_or_before: date) -> tuple[date, date]:
    """(previous hawl, latest hawl on or before ``on_or_before``): one lunar year."""
    month, day = hawl
    year = Gregorian(on_or_before.year, on_or_before.month, on_or_before.day).to_hijri().year
    end = _gregorian(year, month, day)
    if end > on_or_before:
        year -= 1
        end = _gregorian(year, month, day)
    return _gregorian(year - 1, month, day), end


def hawl_dates(hawl: tuple[int, int], today: date) -> tuple[date, date]:
    """(the latest hawl on or before ``today``, the next one after it)."""
    _, last = hawl_period(hawl, today)
    # A lunar year is about 354 days: 360 days on is past the next hawl.
    _, following = hawl_period(hawl, date.fromordinal(last.toordinal() + 360))
    return last, following


def hijri_label(day: date) -> str:
    h = Gregorian(day.year, day.month, day.day).to_hijri()
    return f"{h.year}-{h.month:02d}-{h.day:02d} AH"


@dataclass(frozen=True, slots=True)
class Assessment:
    account: str
    period_start: date
    hawl_date: date
    market_value: float
    dividends: float
    purified: float
    holdings: dict[str, float] = field(default_factory=dict)  # symbol -> market value

    @property
    def trade_goods(self) -> float:
        return RATE * self.market_value

    @property
    def income(self) -> float:
        return RATE * max(self.dividends - self.purified, 0.0)

    @property
    def chosen(self) -> str:
        return "trade goods" if self.trade_goods >= self.income else "income"

    @property
    def amount(self) -> float:
        return max(self.trade_goods, self.income)


async def _close_on_or_before(engine: AsyncEngine, symbol: str, day: date) -> float | None:
    async with engine.connect() as conn:
        value = (
            await conn.execute(
                text(
                    "SELECT close FROM daily_bars WHERE symbol = :s AND adjustment = 'raw' "
                    "AND day <= :d ORDER BY day DESC LIMIT 1"
                ),
                {"s": symbol, "d": day},
            )
        ).scalar()
    return float(value) if value else None


async def _holdings(engine: AsyncEngine, account: str, day: date) -> dict[str, float]:
    """Market value per symbol at the close of ``day``."""
    from datetime import timedelta

    from halal_trader.compliance.purification import BOOK_NOTIONAL, held_shares

    if not account.startswith("book:"):  # a broker account: "paper", "core", "core-live"
        out = {}
        for symbol, shares in (await held_shares(engine, day + timedelta(days=1), account)).items():
            close = await _close_on_or_before(engine, symbol, day)
            if close:
                out[symbol] = shares * close
        return out
    book = account.removeprefix("book:")
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT nav, weights FROM forward_book_days WHERE book = :b AND day <= :d "
                    "ORDER BY day DESC LIMIT 1"
                ),
                {"b": book, "d": day},
            )
        ).first()
    if row is None:
        return {}
    return {s: w * row.nav * BOOK_NOTIONAL for s, w in dict(row.weights).items()}


async def _dividends(
    engine: AsyncEngine, account: str, start: date, end: date
) -> tuple[float, float]:
    """(dividends, purified share) paid in (start, end]."""
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT coalesce(sum(dividend), 0) AS d, coalesce(sum(amount), 0) AS p "
                    "FROM purification_accruals WHERE account = :a "
                    "AND coalesce(payable_date, ex_date) > :s "
                    "AND coalesce(payable_date, ex_date) <= :e"
                ),
                {"a": account, "s": start, "e": end},
            )
        ).one()
    return float(row.d), float(row.p)


async def assess(
    engine: AsyncEngine, account: str, *, period_start: date, hawl_date: date
) -> Assessment:
    holdings = await _holdings(engine, account, hawl_date)
    dividends, purified = await _dividends(engine, account, period_start, hawl_date)
    return Assessment(
        account, period_start, hawl_date, sum(holdings.values()), dividends, purified, holdings
    )


async def record(engine: AsyncEngine, a: Assessment) -> None:
    """Store a year's assessment (one per account and hawl; a re-run replaces it)."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO zakat_assessments (account, hawl_date, hawl_hijri, period_start, "
                "market_value, trade_goods_zakat, dividends, purified, income_zakat, chosen, "
                "amount, holdings, source, computed_at) VALUES (:acc, :h, :hh, :ps, :mv, :tg, "
                ":d, :p, :inc, :ch, :amt, CAST(:hold AS JSONB), :src, :at) "
                "ON CONFLICT (account, hawl_date) DO UPDATE SET "
                "market_value = EXCLUDED.market_value, "
                "trade_goods_zakat = EXCLUDED.trade_goods_zakat, dividends = EXCLUDED.dividends, "
                "purified = EXCLUDED.purified, income_zakat = EXCLUDED.income_zakat, "
                "chosen = EXCLUDED.chosen, amount = EXCLUDED.amount, holdings = EXCLUDED.holdings, "
                "computed_at = EXCLUDED.computed_at"
            ),
            {
                "acc": a.account,
                "h": a.hawl_date,
                "hh": hijri_label(a.hawl_date),
                "ps": a.period_start,
                "mv": a.market_value,
                "tg": a.trade_goods,
                "d": a.dividends,
                "p": a.purified,
                "inc": a.income,
                "ch": a.chosen,
                "amt": a.amount,
                "hold": json.dumps(a.holdings),
                "src": SOURCE,
                "at": datetime.now(UTC),
            },
        )
