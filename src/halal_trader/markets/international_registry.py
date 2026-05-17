"""International equity exchange registry.

The roadmap pins Wave 1.J: "Once Saxo is in (1.E), enable
cross-listed halal stocks: BABA on HKSE, TCS on NSE, Aramco on
TADAWUL. The strategy code is asset-class-agnostic; mainly a
screener + exchange-rules update." This module is the
**pure-Python catalogue + cross-listing resolver** that the
broker plugin (deferred until Saxo / IBKR / Alpaca International
land) consults to map symbols to exchanges, surface trading
hours, and route halal traders to their home-market listings.

Picked a focused registry over per-broker hardcoded exchange
lookups because (a) cross-listed stocks (Aramco trades on TADAWUL
*and* US OTC; BABA on HKSE *and* NYSE) need a single source of
truth that the executor consults — duplicating exchange metadata
across broker adapters means a holiday update has to land in 4
places, (b) halal traders outside the US prefer their home-market
listings (a Saudi operator buying Aramco wants TADAWUL, not OTC)
— pinning the home-market preference here lets the cycle's symbol
selector pick the right venue without re-deriving the policy, (c)
halal-jurisdiction tagging (which exchanges have shariah-screening
infrastructure built in, vs. need our own screener layer) is a
factual catalogue rather than runtime state.

Pinned semantics:
- **Closed-set Exchange enum.** Adding an exchange is a code
  review change so the registry doesn't drift silently when a
  contributor adds a free-form ISO 10383 MIC string.
- **Trading hours are exchange-local UTC offsets.** Renamed to
  trading_hours_local to make this obvious; computed open-now
  is via the standard library tzinfo + day-of-week check.
- **Cross-listings prefer home-market for the issuer's
  jurisdiction.** A Saudi-issued symbol resolves to TADAWUL
  before US-OTC; a HK-issued resolves to HKSE before NYSE.
  Pinned via test.
- **Halal-friendly tag is informational.** TADAWUL has built-in
  shariah index; HKSE doesn't. The bot's screener applies our
  Wave 1.G / 1.H / 1.I logic regardless; the tag drives operator
  UI hints rather than executor behavior.
- **Render output never includes broker API keys / customer IDs.**
  The registry is pure catalogue data; the no-secret pin is
  structural.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from enum import Enum


class Exchange(str, Enum):
    """ISO 10383 MIC values for the in-scope exchanges.

    Pinned string values for JSON / DB / broker-routing stability.
    Adding an exchange is a code review change.
    """

    # US (existing — included for cross-listing resolution)
    NYSE = "XNYS"
    NASDAQ = "XNAS"
    OTC = "OTCM"
    # International (Wave 1.J target)
    LSE = "XLON"  # London
    TSE = "XTKS"  # Tokyo
    HKSE = "XHKG"  # Hong Kong
    NSE = "XNSE"  # India National
    BSE = "XBOM"  # India Bombay
    TADAWUL = "XSAU"  # Saudi
    DIFC = "DIFX"  # Dubai International Financial Centre / Nasdaq Dubai
    EGX = "XCAI"  # Egypt
    KLSE = "XKLS"  # Malaysia (Bursa Malaysia)
    IDX = "XIDX"  # Indonesia
    PSX = "XKAR"  # Pakistan (Karachi Stock Exchange)


class Jurisdiction(str, Enum):
    """Country / region of issuer or exchange."""

    US = "us"
    UK = "uk"
    JAPAN = "japan"
    HONG_KONG = "hong_kong"
    INDIA = "india"
    SAUDI_ARABIA = "saudi_arabia"
    UAE = "uae"
    EGYPT = "egypt"
    MALAYSIA = "malaysia"
    INDONESIA = "indonesia"
    PAKISTAN = "pakistan"


class HalalInfrastructure(str, Enum):
    """Whether the exchange offers shariah-screening infrastructure.

    `BUILT_IN` = exchange operates a shariah index (TADAWUL, KLSE).
    `THIRD_PARTY` = exchange has third-party shariah screening
    (Zoya for US/UK, etc.). `OUR_SCREENER_ONLY` = no public
    shariah infrastructure; we apply our screener directly.
    """

    BUILT_IN = "built_in"
    THIRD_PARTY = "third_party"
    OUR_SCREENER_ONLY = "our_screener_only"


@dataclass(frozen=True)
class TradingHours:
    """One exchange's regular trading-session hours.

    All times are exchange-local; `tz_offset_minutes` is the
    exchange's UTC offset (positive east of UTC). Pre-market and
    after-hours sessions are deferred to follow-ups (most
    international exchanges don't have them; the regular session
    is the meaningful trading window).
    """

    open_time: time
    close_time: time
    tz_offset_minutes: int
    trading_days: frozenset[int]  # 0=Mon, 6=Sun

    def __post_init__(self) -> None:
        if not self.trading_days:
            raise ValueError("trading_days must be non-empty")
        for day in self.trading_days:
            if not 0 <= day <= 6:
                raise ValueError(f"trading_day {day} must be in [0, 6]")
        if self.close_time <= self.open_time:
            raise ValueError("close_time must be after open_time")
        if not -720 <= self.tz_offset_minutes <= 840:
            raise ValueError(f"tz_offset_minutes {self.tz_offset_minutes} out of [-720, 840]")


@dataclass(frozen=True)
class ExchangeProfile:
    """One exchange's metadata."""

    exchange: Exchange
    display_name: str
    jurisdiction: Jurisdiction
    trading_hours: TradingHours
    halal_infrastructure: HalalInfrastructure
    settlement_days: int  # T+N

    def __post_init__(self) -> None:
        if not self.display_name or not self.display_name.strip():
            raise ValueError("display_name must be non-empty")
        if self.settlement_days < 0:
            raise ValueError("settlement_days must be non-negative")
        if self.settlement_days > 5:
            raise ValueError(f"settlement_days {self.settlement_days} above sane upper bound 5")


# Canonical exchange registry. Module-level immutable.
_EXCHANGE_REGISTRY: dict[Exchange, ExchangeProfile] = {
    Exchange.NYSE: ExchangeProfile(
        exchange=Exchange.NYSE,
        display_name="New York Stock Exchange",
        jurisdiction=Jurisdiction.US,
        trading_hours=TradingHours(
            open_time=time(9, 30),
            close_time=time(16, 0),
            tz_offset_minutes=-300,  # EST; DST handled by broker
            trading_days=frozenset({0, 1, 2, 3, 4}),
        ),
        halal_infrastructure=HalalInfrastructure.THIRD_PARTY,
        settlement_days=2,
    ),
    Exchange.NASDAQ: ExchangeProfile(
        exchange=Exchange.NASDAQ,
        display_name="NASDAQ",
        jurisdiction=Jurisdiction.US,
        trading_hours=TradingHours(
            open_time=time(9, 30),
            close_time=time(16, 0),
            tz_offset_minutes=-300,
            trading_days=frozenset({0, 1, 2, 3, 4}),
        ),
        halal_infrastructure=HalalInfrastructure.THIRD_PARTY,
        settlement_days=2,
    ),
    Exchange.OTC: ExchangeProfile(
        exchange=Exchange.OTC,
        display_name="OTC Markets",
        jurisdiction=Jurisdiction.US,
        trading_hours=TradingHours(
            open_time=time(9, 30),
            close_time=time(16, 0),
            tz_offset_minutes=-300,
            trading_days=frozenset({0, 1, 2, 3, 4}),
        ),
        halal_infrastructure=HalalInfrastructure.THIRD_PARTY,
        settlement_days=2,
    ),
    Exchange.LSE: ExchangeProfile(
        exchange=Exchange.LSE,
        display_name="London Stock Exchange",
        jurisdiction=Jurisdiction.UK,
        trading_hours=TradingHours(
            open_time=time(8, 0),
            close_time=time(16, 30),
            tz_offset_minutes=0,  # GMT; broker handles BST
            trading_days=frozenset({0, 1, 2, 3, 4}),
        ),
        halal_infrastructure=HalalInfrastructure.THIRD_PARTY,
        settlement_days=2,
    ),
    Exchange.TSE: ExchangeProfile(
        exchange=Exchange.TSE,
        display_name="Tokyo Stock Exchange",
        jurisdiction=Jurisdiction.JAPAN,
        trading_hours=TradingHours(
            open_time=time(9, 0),
            close_time=time(15, 0),
            tz_offset_minutes=540,
            trading_days=frozenset({0, 1, 2, 3, 4}),
        ),
        halal_infrastructure=HalalInfrastructure.OUR_SCREENER_ONLY,
        settlement_days=2,
    ),
    Exchange.HKSE: ExchangeProfile(
        exchange=Exchange.HKSE,
        display_name="Hong Kong Stock Exchange",
        jurisdiction=Jurisdiction.HONG_KONG,
        trading_hours=TradingHours(
            open_time=time(9, 30),
            close_time=time(16, 0),
            tz_offset_minutes=480,
            trading_days=frozenset({0, 1, 2, 3, 4}),
        ),
        halal_infrastructure=HalalInfrastructure.OUR_SCREENER_ONLY,
        settlement_days=2,
    ),
    Exchange.NSE: ExchangeProfile(
        exchange=Exchange.NSE,
        display_name="National Stock Exchange of India",
        jurisdiction=Jurisdiction.INDIA,
        trading_hours=TradingHours(
            open_time=time(9, 15),
            close_time=time(15, 30),
            tz_offset_minutes=330,
            trading_days=frozenset({0, 1, 2, 3, 4}),
        ),
        halal_infrastructure=HalalInfrastructure.THIRD_PARTY,
        settlement_days=1,
    ),
    Exchange.BSE: ExchangeProfile(
        exchange=Exchange.BSE,
        display_name="Bombay Stock Exchange",
        jurisdiction=Jurisdiction.INDIA,
        trading_hours=TradingHours(
            open_time=time(9, 15),
            close_time=time(15, 30),
            tz_offset_minutes=330,
            trading_days=frozenset({0, 1, 2, 3, 4}),
        ),
        halal_infrastructure=HalalInfrastructure.THIRD_PARTY,
        settlement_days=1,
    ),
    Exchange.TADAWUL: ExchangeProfile(
        exchange=Exchange.TADAWUL,
        display_name="Saudi Stock Exchange (Tadawul)",
        jurisdiction=Jurisdiction.SAUDI_ARABIA,
        trading_hours=TradingHours(
            open_time=time(10, 0),
            close_time=time(15, 0),
            tz_offset_minutes=180,
            trading_days=frozenset({6, 0, 1, 2, 3}),  # Sun-Thu
        ),
        halal_infrastructure=HalalInfrastructure.BUILT_IN,
        settlement_days=2,
    ),
    Exchange.DIFC: ExchangeProfile(
        exchange=Exchange.DIFC,
        display_name="Nasdaq Dubai (DIFX)",
        jurisdiction=Jurisdiction.UAE,
        trading_hours=TradingHours(
            open_time=time(10, 0),
            close_time=time(14, 0),
            tz_offset_minutes=240,
            trading_days=frozenset({0, 1, 2, 3, 4}),
        ),
        halal_infrastructure=HalalInfrastructure.BUILT_IN,
        settlement_days=2,
    ),
    Exchange.EGX: ExchangeProfile(
        exchange=Exchange.EGX,
        display_name="Egyptian Exchange",
        jurisdiction=Jurisdiction.EGYPT,
        trading_hours=TradingHours(
            open_time=time(10, 0),
            close_time=time(14, 30),
            tz_offset_minutes=120,
            trading_days=frozenset({6, 0, 1, 2, 3}),  # Sun-Thu
        ),
        halal_infrastructure=HalalInfrastructure.OUR_SCREENER_ONLY,
        settlement_days=2,
    ),
    Exchange.KLSE: ExchangeProfile(
        exchange=Exchange.KLSE,
        display_name="Bursa Malaysia",
        jurisdiction=Jurisdiction.MALAYSIA,
        trading_hours=TradingHours(
            open_time=time(9, 0),
            close_time=time(17, 0),
            tz_offset_minutes=480,
            trading_days=frozenset({0, 1, 2, 3, 4}),
        ),
        halal_infrastructure=HalalInfrastructure.BUILT_IN,
        settlement_days=2,
    ),
    Exchange.IDX: ExchangeProfile(
        exchange=Exchange.IDX,
        display_name="Indonesia Stock Exchange",
        jurisdiction=Jurisdiction.INDONESIA,
        trading_hours=TradingHours(
            open_time=time(9, 0),
            close_time=time(15, 50),
            tz_offset_minutes=420,
            trading_days=frozenset({0, 1, 2, 3, 4}),
        ),
        halal_infrastructure=HalalInfrastructure.BUILT_IN,
        settlement_days=2,
    ),
    Exchange.PSX: ExchangeProfile(
        exchange=Exchange.PSX,
        display_name="Pakistan Stock Exchange",
        jurisdiction=Jurisdiction.PAKISTAN,
        trading_hours=TradingHours(
            open_time=time(9, 30),
            close_time=time(15, 30),
            tz_offset_minutes=300,
            trading_days=frozenset({0, 1, 2, 3, 4}),
        ),
        halal_infrastructure=HalalInfrastructure.BUILT_IN,
        settlement_days=2,
    ),
}


def exchange_profile(exchange: Exchange) -> ExchangeProfile:
    return _EXCHANGE_REGISTRY[exchange]


def all_exchanges() -> tuple[ExchangeProfile, ...]:
    return tuple(_EXCHANGE_REGISTRY[e] for e in Exchange)


def exchanges_for_jurisdiction(
    jurisdiction: Jurisdiction,
) -> tuple[ExchangeProfile, ...]:
    """Return all exchanges in the given jurisdiction (canonical order)."""

    return tuple(
        _EXCHANGE_REGISTRY[e]
        for e in Exchange
        if _EXCHANGE_REGISTRY[e].jurisdiction is jurisdiction
    )


def is_market_open(exchange: Exchange, *, now: datetime) -> bool:
    """Check whether the given exchange's regular session is currently open.

    Holiday calendar is operator-side (broker reports holiday-closed
    via API); this only checks day-of-week + time-of-day in the
    exchange's local timezone.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    profile = exchange_profile(exchange)
    hours = profile.trading_hours
    local_tz = timezone(timedelta(minutes=hours.tz_offset_minutes))
    local = now.astimezone(local_tz)
    if local.weekday() not in hours.trading_days:
        return False
    local_time = local.time()
    return hours.open_time <= local_time < hours.close_time


@dataclass(frozen=True)
class CrossListing:
    """One symbol's listing on a specific exchange.

    The same issuer (e.g. Alibaba / BABA / 9988.HK) may have several
    `CrossListing` rows in the registry — one per exchange where it
    trades.
    """

    issuer_id: str  # canonical issuer identifier (e.g. "alibaba")
    symbol: str  # exchange-specific ticker (e.g. "BABA", "9988")
    exchange: Exchange
    is_primary_listing: bool  # True only for the issuer's home venue

    def __post_init__(self) -> None:
        if not self.issuer_id or not self.issuer_id.strip():
            raise ValueError("issuer_id must be non-empty")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")


def resolve_home_market(
    listings: Iterable[CrossListing],
    *,
    issuer_id: str,
    operator_jurisdiction: Jurisdiction | None = None,
) -> CrossListing | None:
    """Return the preferred listing for an issuer.

    Preference order:
    1. If `operator_jurisdiction` is provided AND the issuer has a
       listing in that jurisdiction's exchanges, return that listing
       (the home-market preference for halal traders).
    2. Otherwise return the issuer's `is_primary_listing=True` row.
    3. If neither matches, return None.
    """

    issuer_listings = [lst for lst in listings if lst.issuer_id == issuer_id]
    if not issuer_listings:
        return None

    if operator_jurisdiction is not None:
        local = [
            lst
            for lst in issuer_listings
            if exchange_profile(lst.exchange).jurisdiction is operator_jurisdiction
        ]
        if local:
            # Prefer primary if available within the local set
            primaries = [lst for lst in local if lst.is_primary_listing]
            return primaries[0] if primaries else local[0]

    primaries = [lst for lst in issuer_listings if lst.is_primary_listing]
    if primaries:
        return primaries[0]
    return None


_HALAL_INFRA_EMOJI: dict[HalalInfrastructure, str] = {
    HalalInfrastructure.BUILT_IN: "🕌",
    HalalInfrastructure.THIRD_PARTY: "🔍",
    HalalInfrastructure.OUR_SCREENER_ONLY: "🛠️",
}


def render_exchange_profile(profile: ExchangeProfile) -> str:
    """Format an exchange profile for ops display.

    No-secret-leak: registry is pure catalogue data; structural.
    """

    emoji = _HALAL_INFRA_EMOJI[profile.halal_infrastructure]
    hours = profile.trading_hours
    days_str = ",".join(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d] for d in sorted(hours.trading_days)
    )
    offset_h = hours.tz_offset_minutes / 60.0
    return (
        f"{emoji} {profile.display_name} ({profile.exchange.value}) — "
        f"{profile.jurisdiction.value}\n"
        f"  hours: {hours.open_time}-{hours.close_time} (UTC{offset_h:+.1f})\n"
        f"  days: {days_str}\n"
        f"  settlement: T+{profile.settlement_days}\n"
        f"  halal: {profile.halal_infrastructure.value}"
    )


__all__ = [
    "CrossListing",
    "Exchange",
    "ExchangeProfile",
    "HalalInfrastructure",
    "Jurisdiction",
    "TradingHours",
    "all_exchanges",
    "exchange_profile",
    "exchanges_for_jurisdiction",
    "is_market_open",
    "render_exchange_profile",
    "resolve_home_market",
]
