"""Tests for the regional-regulator halal-index module."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from halal_trader.halal.regulator_index import (
    DEFAULT_THRESHOLDS,
    IndexListing,
    Market,
    RegulatorIndex,
    RegulatorScreenResult,
    RegulatorSource,
    RegulatorThresholds,
    RegulatorVerdict,
    listing_age_days,
    newest_index,
    regulator_market,
    render_screen_result,
    screen_with_regulator,
)

_NOW = datetime(2026, 5, 1, tzinfo=UTC)


def _listing(
    *,
    symbol: str = "1010",
    verdict: RegulatorVerdict = RegulatorVerdict.HALAL,
    listed_at: datetime | None = None,
    notes: str = "",
) -> IndexListing:
    return IndexListing(
        symbol=symbol,
        verdict=verdict,
        listed_at=listed_at or (_NOW - timedelta(days=30)),
        notes=notes,
    )


def _index(
    *,
    source: RegulatorSource = RegulatorSource.TADAWUL,
    fetched_at: datetime | None = None,
    listings: tuple[IndexListing, ...] = (),
) -> RegulatorIndex:
    return RegulatorIndex(
        source=source,
        fetched_at=fetched_at or _NOW,
        listings=listings,
    )


# ---------------------------------------------------------------------------
# Authority registry
# ---------------------------------------------------------------------------


def test_tadawul_covers_saudi() -> None:
    assert regulator_market(RegulatorSource.TADAWUL) is Market.SAUDI


def test_cma_covers_saudi() -> None:
    assert regulator_market(RegulatorSource.CMA_HALAL) is Market.SAUDI


def test_kmi30_covers_pakistan() -> None:
    assert regulator_market(RegulatorSource.KMI30) is Market.PAKISTAN


def test_secp_covers_pakistan() -> None:
    assert regulator_market(RegulatorSource.SECP_HALAL) is Market.PAKISTAN


def test_index_market_property() -> None:
    idx = _index(source=RegulatorSource.KMI30)
    assert idx.market is Market.PAKISTAN


# ---------------------------------------------------------------------------
# IndexListing validation
# ---------------------------------------------------------------------------


def test_listing_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        IndexListing(
            symbol="",
            verdict=RegulatorVerdict.HALAL,
            listed_at=_NOW,
        )


def test_listing_rejects_whitespace_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        IndexListing(
            symbol="   ",
            verdict=RegulatorVerdict.HALAL,
            listed_at=_NOW,
        )


def test_listing_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        IndexListing(
            symbol="1010",
            verdict=RegulatorVerdict.HALAL,
            listed_at=datetime(2026, 4, 1),
        )


# ---------------------------------------------------------------------------
# RegulatorIndex validation
# ---------------------------------------------------------------------------


def test_index_rejects_naive_fetched_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        RegulatorIndex(
            source=RegulatorSource.TADAWUL,
            fetched_at=datetime(2026, 5, 1),
            listings=(),
        )


def test_index_rejects_duplicate_symbols() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _index(
            listings=(
                _listing(symbol="1010"),
                _listing(symbol="1010"),
            )
        )


def test_index_duplicate_check_is_case_insensitive() -> None:
    # "1010" and "  1010 " are duplicates after normalisation
    with pytest.raises(ValueError, match="duplicate"):
        _index(
            listings=(
                _listing(symbol="1010"),
                _listing(symbol="  1010  "),
            )
        )


def test_index_lookup_case_insensitive() -> None:
    idx = _index(listings=(_listing(symbol="hbl"),))
    assert idx.lookup("HBL") is not None
    assert idx.lookup("Hbl") is not None
    assert idx.lookup("  hbl  ") is not None


def test_index_lookup_returns_none_on_miss() -> None:
    idx = _index(listings=(_listing(symbol="1010"),))
    assert idx.lookup("9999") is None


def test_index_lookup_empty_string_returns_none() -> None:
    idx = _index(listings=(_listing(symbol="1010"),))
    assert idx.lookup("") is None
    assert idx.lookup("   ") is None


# ---------------------------------------------------------------------------
# RegulatorThresholds validation
# ---------------------------------------------------------------------------


def test_thresholds_default_ladder() -> None:
    assert DEFAULT_THRESHOLDS.stale_days == 90
    assert DEFAULT_THRESHOLDS.expired_days == 365


def test_thresholds_reject_zero_stale() -> None:
    with pytest.raises(ValueError, match="stale_days"):
        RegulatorThresholds(stale_days=0)


def test_thresholds_reject_negative_expired() -> None:
    with pytest.raises(ValueError, match="expired_days"):
        RegulatorThresholds(expired_days=-1)


def test_thresholds_reject_misordered_ladder() -> None:
    with pytest.raises(ValueError, match=">= stale_days"):
        RegulatorThresholds(stale_days=180, expired_days=90)


def test_thresholds_accept_equal_ladder() -> None:
    # operators who want stale = expired (no warning band) can do this
    t = RegulatorThresholds(stale_days=90, expired_days=90)
    assert t.stale_days == 90


# ---------------------------------------------------------------------------
# Screen — basic outcomes
# ---------------------------------------------------------------------------


def test_screen_saudi_listing_in_tadawul_is_halal() -> None:
    indices = (
        _index(
            source=RegulatorSource.TADAWUL,
            listings=(_listing(symbol="1010", verdict=RegulatorVerdict.HALAL),),
        ),
    )
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=indices, now=_NOW)
    assert result.verdict is RegulatorVerdict.HALAL
    assert RegulatorSource.TADAWUL in result.sources
    assert result.is_stale is False
    assert result.is_expired is False


def test_screen_pakistani_listing_in_kmi30_is_halal() -> None:
    indices = (
        _index(
            source=RegulatorSource.KMI30,
            listings=(_listing(symbol="HBL"),),
        ),
    )
    result = screen_with_regulator(symbol="HBL", market=Market.PAKISTAN, indices=indices, now=_NOW)
    assert result.verdict is RegulatorVerdict.HALAL


def test_screen_not_halal_listing_returns_not_halal() -> None:
    indices = (
        _index(
            source=RegulatorSource.TADAWUL,
            listings=(_listing(symbol="1180", verdict=RegulatorVerdict.NOT_HALAL),),
        ),
    )
    result = screen_with_regulator(symbol="1180", market=Market.SAUDI, indices=indices, now=_NOW)
    assert result.verdict is RegulatorVerdict.NOT_HALAL


# ---------------------------------------------------------------------------
# Authority guards (cross-market)
# ---------------------------------------------------------------------------


def test_us_symbol_with_tadawul_index_returns_unknown() -> None:
    """Cross-market authority pin: TADAWUL governs Saudi only.

    The screener should NOT silently flag AAPL as NOT_HALAL just
    because it's not in the Tadawul index — that index has no
    authority over US-listed equities.
    """

    indices = (
        _index(
            source=RegulatorSource.TADAWUL,
            listings=(_listing(symbol="1010"),),
        ),
    )
    result = screen_with_regulator(symbol="AAPL", market=Market.OTHER, indices=indices, now=_NOW)
    assert result.verdict is RegulatorVerdict.UNKNOWN
    assert result.sources == ()
    assert any("Market.OTHER" in w or "no regional" in w for w in result.warnings)


def test_pakistani_symbol_with_only_saudi_index_returns_unknown() -> None:
    indices = (
        _index(
            source=RegulatorSource.TADAWUL,
            listings=(_listing(symbol="HBL"),),  # accidentally listed
        ),
    )
    result = screen_with_regulator(symbol="HBL", market=Market.PAKISTAN, indices=indices, now=_NOW)
    # The TADAWUL row exists but has no authority over the Pakistani market.
    assert result.verdict is RegulatorVerdict.UNKNOWN
    assert result.sources == ()


# ---------------------------------------------------------------------------
# Absence ≠ NOT_HALAL pin
# ---------------------------------------------------------------------------


def test_symbol_absent_from_covering_index_is_unknown() -> None:
    """The pin: absence in a regulator's index is UNKNOWN, not NOT_HALAL."""

    indices = (
        _index(
            source=RegulatorSource.TADAWUL,
            listings=(_listing(symbol="1010"),),
        ),
    )
    result = screen_with_regulator(symbol="9999", market=Market.SAUDI, indices=indices, now=_NOW)
    assert result.verdict is RegulatorVerdict.UNKNOWN
    assert result.sources == ()
    assert any("not present" in w for w in result.warnings)


def test_screen_with_no_indices_at_all_returns_unknown() -> None:
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=(), now=_NOW)
    assert result.verdict is RegulatorVerdict.UNKNOWN
    assert result.sources == ()


def test_screen_market_other_returns_unknown() -> None:
    result = screen_with_regulator(symbol="AAPL", market=Market.OTHER, indices=(), now=_NOW)
    assert result.verdict is RegulatorVerdict.UNKNOWN
    assert any("Market.OTHER" in w or "no regional" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Multiple-index combination + conservative tiebreak
# ---------------------------------------------------------------------------


def test_two_indices_agree_halal() -> None:
    indices = (
        _index(
            source=RegulatorSource.TADAWUL,
            listings=(_listing(symbol="1010"),),
        ),
        _index(
            source=RegulatorSource.CMA_HALAL,
            listings=(_listing(symbol="1010"),),
        ),
    )
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=indices, now=_NOW)
    assert result.verdict is RegulatorVerdict.HALAL
    assert RegulatorSource.TADAWUL in result.sources
    assert RegulatorSource.CMA_HALAL in result.sources


def test_two_indices_disagree_resolve_to_not_halal() -> None:
    """Conservative-tiebreak pin: any NOT_HALAL → NOT_HALAL."""

    indices = (
        _index(
            source=RegulatorSource.TADAWUL,
            listings=(_listing(symbol="1010", verdict=RegulatorVerdict.HALAL),),
        ),
        _index(
            source=RegulatorSource.CMA_HALAL,
            listings=(_listing(symbol="1010", verdict=RegulatorVerdict.NOT_HALAL),),
        ),
    )
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=indices, now=_NOW)
    assert result.verdict is RegulatorVerdict.NOT_HALAL
    assert len(result.sources) == 2
    assert len(result.matched_listings) == 2


def test_one_halal_one_unknown_resolves_halal() -> None:
    """An UNKNOWN listing doesn't override a HALAL one."""

    indices = (
        _index(
            source=RegulatorSource.TADAWUL,
            listings=(_listing(symbol="1010", verdict=RegulatorVerdict.HALAL),),
        ),
        _index(
            source=RegulatorSource.CMA_HALAL,
            listings=(_listing(symbol="1010", verdict=RegulatorVerdict.UNKNOWN),),
        ),
    )
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=indices, now=_NOW)
    assert result.verdict is RegulatorVerdict.HALAL


# ---------------------------------------------------------------------------
# Staleness ladder
# ---------------------------------------------------------------------------


def test_fresh_listing_no_warning() -> None:
    indices = (_index(listings=(_listing(symbol="1010", listed_at=_NOW - timedelta(days=10)),)),)
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=indices, now=_NOW)
    assert result.is_stale is False
    assert result.is_expired is False
    assert result.warnings == ()
    assert result.oldest_listing_age_days == 10


def test_stale_listing_warns_but_keeps_verdict() -> None:
    indices = (_index(listings=(_listing(symbol="1010", listed_at=_NOW - timedelta(days=120)),)),)
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=indices, now=_NOW)
    assert result.is_stale is True
    assert result.is_expired is False
    assert result.verdict is RegulatorVerdict.HALAL
    assert any("stale" in w for w in result.warnings)


def test_expired_listing_demotes_to_unknown() -> None:
    indices = (_index(listings=(_listing(symbol="1010", listed_at=_NOW - timedelta(days=400)),)),)
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=indices, now=_NOW)
    assert result.is_expired is True
    assert result.verdict is RegulatorVerdict.UNKNOWN
    assert any("expir" in w.lower() or "demoted" in w for w in result.warnings)


def test_stale_threshold_is_inclusive() -> None:
    """Pin: at exactly stale_days, the listing is stale."""

    indices = (_index(listings=(_listing(symbol="1010", listed_at=_NOW - timedelta(days=90)),)),)
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=indices, now=_NOW)
    assert result.is_stale is True


def test_expired_threshold_is_inclusive() -> None:
    """Pin: at exactly expired_days, the listing is expired."""

    indices = (_index(listings=(_listing(symbol="1010", listed_at=_NOW - timedelta(days=365)),)),)
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=indices, now=_NOW)
    assert result.is_expired is True
    assert result.verdict is RegulatorVerdict.UNKNOWN


def test_custom_thresholds_flow_through() -> None:
    indices = (_index(listings=(_listing(symbol="1010", listed_at=_NOW - timedelta(days=200)),)),)
    strict = RegulatorThresholds(stale_days=30, expired_days=180)
    result = screen_with_regulator(
        symbol="1010",
        market=Market.SAUDI,
        indices=indices,
        now=_NOW,
        thresholds=strict,
    )
    assert result.is_expired is True
    assert result.verdict is RegulatorVerdict.UNKNOWN


# ---------------------------------------------------------------------------
# Mixed staleness across sources
# ---------------------------------------------------------------------------


def test_one_fresh_one_expired_uses_fresh_for_verdict() -> None:
    """A fresh HALAL listing should still pass even when an
    expired one for the same symbol exists in another regulator's
    index. The expired source contributes UNKNOWN; the fresh source
    contributes HALAL; combined → HALAL.
    """

    indices = (
        _index(
            source=RegulatorSource.TADAWUL,
            listings=(_listing(symbol="1010", listed_at=_NOW - timedelta(days=400)),),
        ),
        _index(
            source=RegulatorSource.CMA_HALAL,
            listings=(_listing(symbol="1010", listed_at=_NOW - timedelta(days=20)),),
        ),
    )
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=indices, now=_NOW)
    assert result.verdict is RegulatorVerdict.HALAL
    assert result.is_expired is True  # oldest_age >= expired_days
    assert result.oldest_listing_age_days == 400


# ---------------------------------------------------------------------------
# Symbol normalisation in lookup
# ---------------------------------------------------------------------------


def test_symbol_lookup_strips_whitespace() -> None:
    indices = (
        _index(
            listings=(_listing(symbol="HBL"),),
        ),
    )
    indices_pkr = (
        RegulatorIndex(
            source=RegulatorSource.KMI30,
            fetched_at=_NOW,
            listings=(_listing(symbol="HBL"),),
        ),
    )
    result = screen_with_regulator(
        symbol="  HBL  ",
        market=Market.PAKISTAN,
        indices=indices_pkr,
        now=_NOW,
    )
    assert result.verdict is RegulatorVerdict.HALAL
    # the saudi indices are unused in this assertion but we keep them
    # to confirm the helper's compose works
    assert indices is not None


def test_symbol_lookup_is_case_insensitive_in_screen() -> None:
    indices = (
        RegulatorIndex(
            source=RegulatorSource.TADAWUL,
            fetched_at=_NOW,
            listings=(_listing(symbol="aramco"),),
        ),
    )
    result = screen_with_regulator(symbol="ARAMCO", market=Market.SAUDI, indices=indices, now=_NOW)
    assert result.verdict is RegulatorVerdict.HALAL


# ---------------------------------------------------------------------------
# Screener input validation
# ---------------------------------------------------------------------------


def test_screen_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        screen_with_regulator(
            symbol="1010",
            market=Market.SAUDI,
            indices=(),
            now=datetime(2026, 5, 1),
        )


def test_screen_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        screen_with_regulator(symbol="", market=Market.SAUDI, indices=(), now=_NOW)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def test_listing_age_days() -> None:
    listing = _listing(listed_at=_NOW - timedelta(days=42))
    assert listing_age_days(listing, _NOW) == 42


def test_listing_age_days_rejects_naive_now() -> None:
    listing = _listing()
    with pytest.raises(ValueError, match="timezone-aware"):
        listing_age_days(listing, datetime(2026, 5, 1))


def test_newest_index_returns_latest_fetched() -> None:
    a = _index(fetched_at=_NOW - timedelta(days=10))
    b = _index(fetched_at=_NOW - timedelta(days=2))
    c = _index(fetched_at=_NOW - timedelta(days=20))
    assert newest_index((a, b, c)) is b


def test_newest_index_returns_none_on_empty() -> None:
    assert newest_index(()) is None


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_listing_is_frozen() -> None:
    listing = _listing()
    with pytest.raises(dataclasses.FrozenInstanceError):
        listing.symbol = "OTHER"  # type: ignore[misc]


def test_index_is_frozen() -> None:
    idx = _index()
    with pytest.raises(dataclasses.FrozenInstanceError):
        idx.source = RegulatorSource.KMI30  # type: ignore[misc]


def test_thresholds_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_THRESHOLDS.stale_days = 30  # type: ignore[misc]


def test_result_is_frozen() -> None:
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=(), now=_NOW)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = RegulatorVerdict.HALAL  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB serialisation
# ---------------------------------------------------------------------------


def test_source_string_values_pinned() -> None:
    assert RegulatorSource.TADAWUL.value == "tadawul"
    assert RegulatorSource.CMA_HALAL.value == "cma_halal"
    assert RegulatorSource.KMI30.value == "kmi_30"
    assert RegulatorSource.SECP_HALAL.value == "secp_halal"


def test_market_string_values_pinned() -> None:
    assert Market.SAUDI.value == "saudi"
    assert Market.PAKISTAN.value == "pakistan"
    assert Market.OTHER.value == "other"


def test_verdict_string_values_pinned() -> None:
    assert RegulatorVerdict.HALAL.value == "halal"
    assert RegulatorVerdict.NOT_HALAL.value == "not_halal"
    assert RegulatorVerdict.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_halal_result() -> None:
    result = screen_with_regulator(
        symbol="1010",
        market=Market.SAUDI,
        indices=(_index(listings=(_listing(symbol="1010"),)),),
        now=_NOW,
    )
    text = render_screen_result(result)
    assert "✅" in text
    assert "1010" in text
    assert "saudi" in text
    assert "tadawul" in text


def test_render_not_halal_result() -> None:
    result = screen_with_regulator(
        symbol="1180",
        market=Market.SAUDI,
        indices=(
            _index(
                listings=(_listing(symbol="1180", verdict=RegulatorVerdict.NOT_HALAL),),
            ),
        ),
        now=_NOW,
    )
    text = render_screen_result(result)
    assert "❌" in text
    assert "NOT_HALAL" in text


def test_render_unknown_result() -> None:
    result = screen_with_regulator(symbol="AAPL", market=Market.OTHER, indices=(), now=_NOW)
    text = render_screen_result(result)
    assert "❓" in text
    assert "UNKNOWN" in text


def test_render_includes_stale_marker() -> None:
    indices = (
        _index(
            listings=(_listing(symbol="1010", listed_at=_NOW - timedelta(days=120)),),
        ),
    )
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=indices, now=_NOW)
    text = render_screen_result(result)
    assert "(stale)" in text


def test_render_includes_expired_marker() -> None:
    indices = (
        _index(
            listings=(_listing(symbol="1010", listed_at=_NOW - timedelta(days=400)),),
        ),
    )
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=indices, now=_NOW)
    text = render_screen_result(result)
    assert "(expired)" in text


# ---------------------------------------------------------------------------
# RegulatorScreenResult shape sanity
# ---------------------------------------------------------------------------


def test_result_shape() -> None:
    indices = (
        _index(
            listings=(_listing(symbol="1010"),),
        ),
    )
    result = screen_with_regulator(symbol="1010", market=Market.SAUDI, indices=indices, now=_NOW)
    assert isinstance(result, RegulatorScreenResult)
    assert result.symbol == "1010"
    assert result.market is Market.SAUDI
    assert len(result.matched_listings) == 1
    assert result.matched_listings[0].symbol == "1010"
