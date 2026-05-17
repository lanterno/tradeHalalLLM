"""Tests for `halal_trader.markets.international_registry` (Wave 1.J).

Covers: exchange enum + profile registry, trading-hours validation,
is_market_open across timezones, cross-listing home-market resolver,
no-secret render contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, time, timezone

import pytest

from halal_trader.markets.international_registry import (
    CrossListing,
    Exchange,
    ExchangeProfile,
    HalalInfrastructure,
    Jurisdiction,
    TradingHours,
    all_exchanges,
    exchange_profile,
    exchanges_for_jurisdiction,
    is_market_open,
    render_exchange_profile,
    resolve_home_market,
)

UTC = timezone.utc


# --------------------------- Enum string pins --------------------------------


def test_exchange_string_values_pinned() -> None:
    """Pin: ISO 10383 MIC values used as the canonical identifier."""

    assert Exchange.NYSE.value == "XNYS"
    assert Exchange.NASDAQ.value == "XNAS"
    assert Exchange.LSE.value == "XLON"
    assert Exchange.TSE.value == "XTKS"
    assert Exchange.HKSE.value == "XHKG"
    assert Exchange.NSE.value == "XNSE"
    assert Exchange.BSE.value == "XBOM"
    assert Exchange.TADAWUL.value == "XSAU"
    assert Exchange.DIFC.value == "DIFX"
    assert Exchange.EGX.value == "XCAI"
    assert Exchange.KLSE.value == "XKLS"
    assert Exchange.IDX.value == "XIDX"
    assert Exchange.PSX.value == "XKAR"


def test_jurisdiction_string_values_pinned() -> None:
    assert Jurisdiction.US.value == "us"
    assert Jurisdiction.UK.value == "uk"
    assert Jurisdiction.JAPAN.value == "japan"
    assert Jurisdiction.HONG_KONG.value == "hong_kong"
    assert Jurisdiction.INDIA.value == "india"
    assert Jurisdiction.SAUDI_ARABIA.value == "saudi_arabia"
    assert Jurisdiction.UAE.value == "uae"
    assert Jurisdiction.EGYPT.value == "egypt"
    assert Jurisdiction.MALAYSIA.value == "malaysia"
    assert Jurisdiction.INDONESIA.value == "indonesia"
    assert Jurisdiction.PAKISTAN.value == "pakistan"


def test_halal_infrastructure_string_values_pinned() -> None:
    assert HalalInfrastructure.BUILT_IN.value == "built_in"
    assert HalalInfrastructure.THIRD_PARTY.value == "third_party"
    assert HalalInfrastructure.OUR_SCREENER_ONLY.value == "our_screener_only"


# --------------------------- TradingHours ------------------------------------


def test_trading_hours_basic() -> None:
    h = TradingHours(
        open_time=time(9, 30),
        close_time=time(16, 0),
        tz_offset_minutes=-300,
        trading_days=frozenset({0, 1, 2, 3, 4}),
    )
    assert h.open_time == time(9, 30)


def test_trading_hours_rejects_close_before_open() -> None:
    with pytest.raises(ValueError, match="close_time"):
        TradingHours(
            open_time=time(16, 0),
            close_time=time(9, 30),
            tz_offset_minutes=0,
            trading_days=frozenset({0}),
        )


def test_trading_hours_rejects_close_equals_open() -> None:
    with pytest.raises(ValueError, match="close_time"):
        TradingHours(
            open_time=time(9, 0),
            close_time=time(9, 0),
            tz_offset_minutes=0,
            trading_days=frozenset({0}),
        )


def test_trading_hours_rejects_empty_days() -> None:
    with pytest.raises(ValueError, match="trading_days"):
        TradingHours(
            open_time=time(9, 0),
            close_time=time(16, 0),
            tz_offset_minutes=0,
            trading_days=frozenset(),
        )


def test_trading_hours_rejects_invalid_day() -> None:
    with pytest.raises(ValueError, match="trading_day"):
        TradingHours(
            open_time=time(9, 0),
            close_time=time(16, 0),
            tz_offset_minutes=0,
            trading_days=frozenset({7}),
        )


def test_trading_hours_rejects_extreme_tz_offset() -> None:
    with pytest.raises(ValueError, match="tz_offset"):
        TradingHours(
            open_time=time(9, 0),
            close_time=time(16, 0),
            tz_offset_minutes=900,
            trading_days=frozenset({0}),
        )


def test_trading_hours_is_frozen() -> None:
    h = TradingHours(
        open_time=time(9, 0),
        close_time=time(16, 0),
        tz_offset_minutes=0,
        trading_days=frozenset({0}),
    )
    with pytest.raises(FrozenInstanceError):
        h.tz_offset_minutes = 60  # type: ignore[misc]


# --------------------------- ExchangeProfile ---------------------------------


def test_profile_rejects_empty_display_name() -> None:
    h = TradingHours(
        open_time=time(9, 0),
        close_time=time(16, 0),
        tz_offset_minutes=0,
        trading_days=frozenset({0}),
    )
    with pytest.raises(ValueError, match="display_name"):
        ExchangeProfile(
            exchange=Exchange.NYSE,
            display_name="",
            jurisdiction=Jurisdiction.US,
            trading_hours=h,
            halal_infrastructure=HalalInfrastructure.THIRD_PARTY,
            settlement_days=2,
        )


def test_profile_rejects_negative_settlement() -> None:
    h = TradingHours(
        open_time=time(9, 0),
        close_time=time(16, 0),
        tz_offset_minutes=0,
        trading_days=frozenset({0}),
    )
    with pytest.raises(ValueError, match="settlement_days"):
        ExchangeProfile(
            exchange=Exchange.NYSE,
            display_name="X",
            jurisdiction=Jurisdiction.US,
            trading_hours=h,
            halal_infrastructure=HalalInfrastructure.THIRD_PARTY,
            settlement_days=-1,
        )


def test_profile_rejects_settlement_above_5() -> None:
    h = TradingHours(
        open_time=time(9, 0),
        close_time=time(16, 0),
        tz_offset_minutes=0,
        trading_days=frozenset({0}),
    )
    with pytest.raises(ValueError, match="settlement_days"):
        ExchangeProfile(
            exchange=Exchange.NYSE,
            display_name="X",
            jurisdiction=Jurisdiction.US,
            trading_hours=h,
            halal_infrastructure=HalalInfrastructure.THIRD_PARTY,
            settlement_days=10,
        )


def test_profile_is_frozen() -> None:
    p = exchange_profile(Exchange.NYSE)
    with pytest.raises(FrozenInstanceError):
        p.settlement_days = 99  # type: ignore[misc]


# --------------------------- exchange_profile lookup -------------------------


def test_every_exchange_has_profile() -> None:
    """Pin: registry coverage."""

    profiles = all_exchanges()
    assert len(profiles) == len(Exchange)


def test_every_profile_in_canonical_order() -> None:
    profiles = all_exchanges()
    spec_order = [p.exchange for p in profiles]
    assert spec_order == list(Exchange)


def test_us_jurisdiction_pin() -> None:
    """Pin: NYSE / NASDAQ / OTC are US."""

    for ex in (Exchange.NYSE, Exchange.NASDAQ, Exchange.OTC):
        assert exchange_profile(ex).jurisdiction is Jurisdiction.US


def test_saudi_jurisdiction_pin() -> None:
    """Pin: TADAWUL is SAUDI_ARABIA with BUILT_IN halal infra."""

    profile = exchange_profile(Exchange.TADAWUL)
    assert profile.jurisdiction is Jurisdiction.SAUDI_ARABIA
    assert profile.halal_infrastructure is HalalInfrastructure.BUILT_IN


def test_uae_jurisdiction_pin() -> None:
    profile = exchange_profile(Exchange.DIFC)
    assert profile.jurisdiction is Jurisdiction.UAE
    assert profile.halal_infrastructure is HalalInfrastructure.BUILT_IN


def test_pakistan_pin() -> None:
    """Pin: PSX has BUILT_IN halal (KMI-30 index per Wave 2.G)."""

    profile = exchange_profile(Exchange.PSX)
    assert profile.jurisdiction is Jurisdiction.PAKISTAN
    assert profile.halal_infrastructure is HalalInfrastructure.BUILT_IN


def test_malaysia_indonesia_built_in_halal() -> None:
    """Pin: KLSE + IDX have BUILT_IN shariah infrastructure."""

    assert exchange_profile(Exchange.KLSE).halal_infrastructure is HalalInfrastructure.BUILT_IN
    assert exchange_profile(Exchange.IDX).halal_infrastructure is HalalInfrastructure.BUILT_IN


# --------------------------- exchanges_for_jurisdiction ----------------------


def test_exchanges_for_us() -> None:
    us_exchanges = exchanges_for_jurisdiction(Jurisdiction.US)
    us_set = {p.exchange for p in us_exchanges}
    assert us_set == {Exchange.NYSE, Exchange.NASDAQ, Exchange.OTC}


def test_exchanges_for_india() -> None:
    in_exchanges = exchanges_for_jurisdiction(Jurisdiction.INDIA)
    in_set = {p.exchange for p in in_exchanges}
    assert in_set == {Exchange.NSE, Exchange.BSE}


def test_exchanges_for_saudi() -> None:
    saudi = exchanges_for_jurisdiction(Jurisdiction.SAUDI_ARABIA)
    assert {p.exchange for p in saudi} == {Exchange.TADAWUL}


# --------------------------- is_market_open ----------------------------------


def test_is_market_open_nyse_during_session() -> None:
    """NYSE 14:00 UTC = 9:00 EST (open) on Tuesday."""

    now = datetime(2026, 5, 5, 14, 30, 0, tzinfo=UTC)  # Tuesday
    assert is_market_open(Exchange.NYSE, now=now) is True


def test_is_market_open_nyse_before_open() -> None:
    """NYSE 13:00 UTC = 8:00 EST (before open)."""

    now = datetime(2026, 5, 5, 13, 0, 0, tzinfo=UTC)
    assert is_market_open(Exchange.NYSE, now=now) is False


def test_is_market_open_nyse_after_close() -> None:
    """NYSE 22:00 UTC = 17:00 EST (after close)."""

    now = datetime(2026, 5, 5, 22, 0, 0, tzinfo=UTC)
    assert is_market_open(Exchange.NYSE, now=now) is False


def test_is_market_open_nyse_weekend() -> None:
    """NYSE Saturday is closed."""

    now = datetime(2026, 5, 9, 14, 30, 0, tzinfo=UTC)  # Saturday
    assert is_market_open(Exchange.NYSE, now=now) is False


def test_is_market_open_tadawul_sunday() -> None:
    """Pin: TADAWUL trades Sun-Thu — Sunday during session is OPEN."""

    # Sunday 2026-05-03 11:00 Saudi (UTC+3) = 08:00 UTC
    now = datetime(2026, 5, 3, 8, 0, 0, tzinfo=UTC)
    assert is_market_open(Exchange.TADAWUL, now=now) is True


def test_is_market_open_tadawul_friday_closed() -> None:
    """Pin: TADAWUL closed Fri (4) and Sat (5)."""

    # Friday 2026-05-08 11:00 Saudi (UTC+3) = 08:00 UTC
    now = datetime(2026, 5, 8, 8, 0, 0, tzinfo=UTC)
    assert is_market_open(Exchange.TADAWUL, now=now) is False


def test_is_market_open_tse_during_session() -> None:
    """TSE 09:30 JST (UTC+9) = 00:30 UTC on Monday."""

    now = datetime(2026, 5, 4, 0, 30, 0, tzinfo=UTC)  # Monday
    assert is_market_open(Exchange.TSE, now=now) is True


def test_is_market_open_egx_friday_closed() -> None:
    """Pin: EGX trades Sun-Thu like TADAWUL."""

    # Friday during local session
    now = datetime(2026, 5, 8, 10, 0, 0, tzinfo=UTC)
    assert is_market_open(Exchange.EGX, now=now) is False


def test_is_market_open_at_exact_open_boundary() -> None:
    """Pin: open_time inclusive (>=)."""

    # NYSE 9:30 EST = 14:30 UTC on Tuesday
    now = datetime(2026, 5, 5, 14, 30, 0, tzinfo=UTC)
    assert is_market_open(Exchange.NYSE, now=now) is True


def test_is_market_open_at_exact_close_boundary() -> None:
    """Pin: close_time exclusive (<)."""

    # NYSE 16:00 EST = 21:00 UTC on Tuesday — exactly at close
    now = datetime(2026, 5, 5, 21, 0, 0, tzinfo=UTC)
    assert is_market_open(Exchange.NYSE, now=now) is False


def test_is_market_open_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        is_market_open(Exchange.NYSE, now=datetime(2026, 5, 5, 14, 0))


# --------------------------- CrossListing ------------------------------------


def test_cross_listing_rejects_empty_issuer() -> None:
    with pytest.raises(ValueError, match="issuer_id"):
        CrossListing(
            issuer_id="",
            symbol="X",
            exchange=Exchange.NYSE,
            is_primary_listing=True,
        )


def test_cross_listing_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        CrossListing(
            issuer_id="x",
            symbol="",
            exchange=Exchange.NYSE,
            is_primary_listing=True,
        )


def test_cross_listing_is_frozen() -> None:
    lst = CrossListing(
        issuer_id="x",
        symbol="X",
        exchange=Exchange.NYSE,
        is_primary_listing=True,
    )
    with pytest.raises(FrozenInstanceError):
        lst.symbol = "Y"  # type: ignore[misc]


# --------------------------- resolve_home_market -----------------------------


def _aramco_listings() -> list[CrossListing]:
    """Aramco trades primarily on TADAWUL, also OTC in US."""

    return [
        CrossListing(
            issuer_id="aramco",
            symbol="2222",
            exchange=Exchange.TADAWUL,
            is_primary_listing=True,
        ),
        CrossListing(
            issuer_id="aramco",
            symbol="ARMCO",
            exchange=Exchange.OTC,
            is_primary_listing=False,
        ),
    ]


def _baba_listings() -> list[CrossListing]:
    """BABA primary on HKSE, secondary on NYSE."""

    return [
        CrossListing(
            issuer_id="alibaba",
            symbol="9988",
            exchange=Exchange.HKSE,
            is_primary_listing=True,
        ),
        CrossListing(
            issuer_id="alibaba",
            symbol="BABA",
            exchange=Exchange.NYSE,
            is_primary_listing=False,
        ),
    ]


def test_resolve_returns_primary_when_no_jurisdiction() -> None:
    listings = _aramco_listings()
    home = resolve_home_market(listings, issuer_id="aramco")
    assert home is not None
    assert home.exchange is Exchange.TADAWUL


def test_resolve_returns_local_when_jurisdiction_matches() -> None:
    """Pin: a Saudi operator gets TADAWUL for Aramco."""

    listings = _aramco_listings()
    home = resolve_home_market(
        listings, issuer_id="aramco", operator_jurisdiction=Jurisdiction.SAUDI_ARABIA
    )
    assert home is not None
    assert home.exchange is Exchange.TADAWUL


def test_resolve_returns_us_otc_for_us_operator() -> None:
    """Pin: a US operator without TADAWUL access gets OTC for Aramco."""

    listings = _aramco_listings()
    home = resolve_home_market(listings, issuer_id="aramco", operator_jurisdiction=Jurisdiction.US)
    assert home is not None
    assert home.exchange is Exchange.OTC


def test_resolve_baba_for_hk_operator() -> None:
    listings = _baba_listings()
    home = resolve_home_market(
        listings, issuer_id="alibaba", operator_jurisdiction=Jurisdiction.HONG_KONG
    )
    assert home is not None
    assert home.exchange is Exchange.HKSE


def test_resolve_baba_for_us_operator_returns_us_listing() -> None:
    """Pin: US operator without HKSE access gets NYSE BABA."""

    listings = _baba_listings()
    home = resolve_home_market(listings, issuer_id="alibaba", operator_jurisdiction=Jurisdiction.US)
    assert home is not None
    assert home.exchange is Exchange.NYSE


def test_resolve_unknown_issuer_returns_none() -> None:
    listings = _aramco_listings()
    home = resolve_home_market(listings, issuer_id="unknown_issuer")
    assert home is None


def test_resolve_no_jurisdiction_match_falls_back_to_primary() -> None:
    """Pin: if operator's jurisdiction has no listing, fall back to primary."""

    listings = _aramco_listings()
    # Aramco doesn't trade in India; should fall back to TADAWUL primary
    home = resolve_home_market(
        listings, issuer_id="aramco", operator_jurisdiction=Jurisdiction.INDIA
    )
    assert home is not None
    assert home.exchange is Exchange.TADAWUL


def test_resolve_with_no_primary_returns_none() -> None:
    """Pin: if no listing is marked primary AND jurisdiction doesn't match,
    return None rather than guessing.
    """

    listings = [
        CrossListing(
            issuer_id="x",
            symbol="X",
            exchange=Exchange.OTC,
            is_primary_listing=False,
        ),
    ]
    home = resolve_home_market(listings, issuer_id="x", operator_jurisdiction=Jurisdiction.UK)
    assert home is None


# --------------------------- render_exchange_profile -------------------------


def test_render_includes_display_name_and_mic() -> None:
    out = render_exchange_profile(exchange_profile(Exchange.TADAWUL))
    assert "Tadawul" in out
    assert "XSAU" in out


def test_render_includes_halal_emoji_for_built_in() -> None:
    out = render_exchange_profile(exchange_profile(Exchange.TADAWUL))
    assert "🕌" in out


def test_render_includes_halal_emoji_for_third_party() -> None:
    out = render_exchange_profile(exchange_profile(Exchange.NYSE))
    assert "🔍" in out


def test_render_includes_halal_emoji_for_our_screener_only() -> None:
    out = render_exchange_profile(exchange_profile(Exchange.TSE))
    assert "🛠️" in out


def test_render_includes_settlement_days() -> None:
    out = render_exchange_profile(exchange_profile(Exchange.NSE))
    assert "T+1" in out


def test_render_no_secret_leak() -> None:
    """Pin: registry contains no secrets; structural."""

    for ex in Exchange:
        out = render_exchange_profile(exchange_profile(ex))
        assert "api_key" not in out.lower()
        assert "secret" not in out.lower()
        assert "cus_" not in out.lower()


# --------------------------- e2e flows ---------------------------------------


def test_e2e_saudi_operator_trading_aramco() -> None:
    """Real: Saudi operator wants Aramco; should route to TADAWUL with
    BUILT_IN halal infra."""

    listings = _aramco_listings()
    home = resolve_home_market(
        listings, issuer_id="aramco", operator_jurisdiction=Jurisdiction.SAUDI_ARABIA
    )
    assert home is not None
    profile = exchange_profile(home.exchange)
    assert profile.halal_infrastructure is HalalInfrastructure.BUILT_IN
    assert profile.jurisdiction is Jurisdiction.SAUDI_ARABIA


def test_e2e_us_operator_baba_routes_to_nyse() -> None:
    listings = _baba_listings()
    home = resolve_home_market(listings, issuer_id="alibaba", operator_jurisdiction=Jurisdiction.US)
    assert home is not None
    assert home.exchange is Exchange.NYSE


def test_e2e_market_open_during_overlap() -> None:
    """Pin: at 09:00 UTC Tuesday, LSE (UTC+0) is open; NYSE (UTC-5) is not yet."""

    now = datetime(2026, 5, 5, 9, 0, 0, tzinfo=UTC)  # Tuesday 09:00 UTC
    assert is_market_open(Exchange.LSE, now=now) is True
    assert is_market_open(Exchange.NYSE, now=now) is False
