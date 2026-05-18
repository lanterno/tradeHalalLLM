"""Tests for the commodity-ETF Shariah screener."""

from __future__ import annotations

import dataclasses

import pytest

from halal_trader.halal.commodities_screener import (
    DEFAULT_THRESHOLDS,
    BackingMode,
    CommodityETFFinancials,
    CommodityScreenResult,
    CommodityThresholds,
    CommodityType,
    CommodityVerdict,
    StorageLocation,
    render_screen_result,
    screen_commodity_etf,
)


def _financials(
    *,
    symbol: str = "PHYS",
    name: str = "Sprott Physical Gold Trust",
    commodity: CommodityType = CommodityType.GOLD,
    backing_mode: BackingMode = BackingMode.ALLOCATED_PHYSICAL,
    storage_location: StorageLocation = StorageLocation.SEGREGATED,
    leverage_factor: float = 1.0,
    physical_holdings_pct: float = 99.0,
    has_audited_holdings: bool = True,
) -> CommodityETFFinancials:
    return CommodityETFFinancials(
        symbol=symbol,
        name=name,
        commodity=commodity,
        backing_mode=backing_mode,
        storage_location=storage_location,
        leverage_factor=leverage_factor,
        physical_holdings_pct=physical_holdings_pct,
        has_audited_holdings=has_audited_holdings,
    )


# ---------------------------------------------------------------------------
# Threshold validation
# ---------------------------------------------------------------------------


def test_default_thresholds_are_aaoifi_aligned() -> None:
    assert DEFAULT_THRESHOLDS.min_physical_holdings_pct == 95.0
    assert DEFAULT_THRESHOLDS.max_leverage_factor == 1.0


def test_thresholds_reject_zero_physical_pct() -> None:
    with pytest.raises(ValueError, match="min_physical_holdings_pct"):
        CommodityThresholds(min_physical_holdings_pct=0.0)


def test_thresholds_reject_above_100_physical_pct() -> None:
    with pytest.raises(ValueError):
        CommodityThresholds(min_physical_holdings_pct=101.0)


def test_thresholds_reject_zero_leverage() -> None:
    with pytest.raises(ValueError, match="max_leverage_factor"):
        CommodityThresholds(max_leverage_factor=0.0)


def test_thresholds_reject_negative_leverage_max() -> None:
    with pytest.raises(ValueError):
        CommodityThresholds(max_leverage_factor=-1.0)


# ---------------------------------------------------------------------------
# CommodityETFFinancials validation
# ---------------------------------------------------------------------------


def test_financials_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        _financials(symbol="")


def test_financials_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        _financials(name="")


def test_financials_rejects_negative_physical_pct() -> None:
    with pytest.raises(ValueError, match="physical_holdings_pct"):
        _financials(physical_holdings_pct=-1.0)


def test_financials_rejects_above_100_physical_pct() -> None:
    with pytest.raises(ValueError, match="physical_holdings_pct"):
        _financials(physical_holdings_pct=101.0)


# ---------------------------------------------------------------------------
# Hard rejections — unconditional NOT_HALAL gates
# ---------------------------------------------------------------------------


def test_swap_backed_is_not_halal() -> None:
    """SWAP_BACKED is categorically NOT_HALAL regardless of other flags."""

    result = screen_commodity_etf(_financials(backing_mode=BackingMode.SWAP_BACKED))
    assert result.verdict is CommodityVerdict.NOT_HALAL
    assert any("swap-backed" in f for f in result.failures)


def test_paper_storage_is_not_halal() -> None:
    result = screen_commodity_etf(_financials(storage_location=StorageLocation.PAPER))
    assert result.verdict is CommodityVerdict.NOT_HALAL
    assert any("paper" in f for f in result.failures)


def test_leveraged_etf_is_not_halal() -> None:
    result = screen_commodity_etf(_financials(leverage_factor=2.0))
    assert result.verdict is CommodityVerdict.NOT_HALAL
    assert any("leverage" in f for f in result.failures)


def test_inverse_etf_is_not_halal() -> None:
    """Pin: inverse / short ETFs (negative leverage) are NOT_HALAL."""

    result = screen_commodity_etf(_financials(leverage_factor=-1.0))
    assert result.verdict is CommodityVerdict.NOT_HALAL
    assert any("inverse" in f or "short" in f for f in result.failures)


def test_low_physical_holdings_is_not_halal() -> None:
    result = screen_commodity_etf(_financials(physical_holdings_pct=80.0))
    assert result.verdict is CommodityVerdict.NOT_HALAL
    assert any("physical holdings" in f for f in result.failures)


def test_physical_holdings_at_threshold_is_inclusive() -> None:
    """Pin: 95% physical → passes (boundary inclusive)."""

    result = screen_commodity_etf(_financials(physical_holdings_pct=95.0))
    assert result.verdict is CommodityVerdict.HALAL


def test_physical_holdings_just_below_threshold_fails() -> None:
    result = screen_commodity_etf(_financials(physical_holdings_pct=94.99))
    assert result.verdict is CommodityVerdict.NOT_HALAL


def test_leverage_at_threshold_is_inclusive() -> None:
    """Pin: 1.0x leverage → passes (boundary inclusive)."""

    result = screen_commodity_etf(_financials(leverage_factor=1.0))
    assert result.verdict is CommodityVerdict.HALAL


def test_leverage_just_above_threshold_fails() -> None:
    result = screen_commodity_etf(_financials(leverage_factor=1.01))
    assert result.verdict is CommodityVerdict.NOT_HALAL


# ---------------------------------------------------------------------------
# INSUFFICIENT_DATA — backing UNKNOWN
# ---------------------------------------------------------------------------


def test_unknown_backing_returns_insufficient_data() -> None:
    result = screen_commodity_etf(_financials(backing_mode=BackingMode.UNKNOWN))
    assert result.verdict is CommodityVerdict.INSUFFICIENT_DATA
    assert any("UNKNOWN" in w for w in result.warnings)


def test_unknown_backing_returns_insufficient_data_even_with_other_flags_clean() -> None:
    result = screen_commodity_etf(
        _financials(
            backing_mode=BackingMode.UNKNOWN,
            storage_location=StorageLocation.SEGREGATED,
            physical_holdings_pct=99.0,
            has_audited_holdings=True,
        )
    )
    assert result.verdict is CommodityVerdict.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# HALAL — every check passes
# ---------------------------------------------------------------------------


def test_phys_like_clean_gold_etf_is_halal() -> None:
    """Real-world: Sprott Physical Gold Trust (PHYS) — allocated physical
    bullion in segregated LBMA-vaulted storage, no leverage, audited."""

    result = screen_commodity_etf(_financials())
    assert result.verdict is CommodityVerdict.HALAL
    assert result.failures == ()
    assert result.warnings == ()


def test_clean_silver_etf_is_halal() -> None:
    result = screen_commodity_etf(
        _financials(
            symbol="PSLV",
            name="Sprott Physical Silver Trust",
            commodity=CommodityType.SILVER,
        )
    )
    assert result.verdict is CommodityVerdict.HALAL


def test_clean_platinum_etf_is_halal() -> None:
    result = screen_commodity_etf(
        _financials(
            symbol="PPLT",
            commodity=CommodityType.PLATINUM,
        )
    )
    assert result.verdict is CommodityVerdict.HALAL


# ---------------------------------------------------------------------------
# DOUBTFUL — soft warnings drive doubtful
# ---------------------------------------------------------------------------


def test_unallocated_physical_gold_is_doubtful() -> None:
    """Pin: gold is ribawi; unallocated → DOUBTFUL even when audited."""

    result = screen_commodity_etf(_financials(backing_mode=BackingMode.UNALLOCATED_PHYSICAL))
    assert result.verdict is CommodityVerdict.DOUBTFUL
    assert any("unallocated" in w for w in result.warnings)
    assert any("ribawi" in w for w in result.warnings)


def test_unallocated_physical_copper_is_doubtful() -> None:
    """Copper is non-ribawi but unallocated still triggers DOUBTFUL warning."""

    result = screen_commodity_etf(
        _financials(
            commodity=CommodityType.COPPER,
            backing_mode=BackingMode.UNALLOCATED_PHYSICAL,
        )
    )
    assert result.verdict is CommodityVerdict.DOUBTFUL


def test_futures_backed_is_doubtful() -> None:
    """Futures-backed ETFs are DOUBTFUL — scholar disagreement on roll yield."""

    result = screen_commodity_etf(
        _financials(
            symbol="USO",
            name="United States Oil Fund",
            commodity=CommodityType.OIL,
            backing_mode=BackingMode.FUTURES_BACKED,
            storage_location=StorageLocation.PAPER,
            physical_holdings_pct=0.0,
        )
    )
    # PAPER storage is NOT_HALAL — overrides the futures-backed warning
    assert result.verdict is CommodityVerdict.NOT_HALAL


def test_futures_backed_gold_with_clean_storage_still_doubtful() -> None:
    """Even if storage isn't paper, futures-backed ribawi → DOUBTFUL."""

    result = screen_commodity_etf(
        _financials(
            backing_mode=BackingMode.FUTURES_BACKED,
            storage_location=StorageLocation.SEGREGATED,
        )
    )
    assert result.verdict is CommodityVerdict.DOUBTFUL
    assert any("futures-backed" in w for w in result.warnings)


def test_commingled_storage_is_doubtful() -> None:
    result = screen_commodity_etf(_financials(storage_location=StorageLocation.COMMINGLED))
    # gold + commingled triggers ribawi-segregated warning → DOUBTFUL
    assert result.verdict is CommodityVerdict.DOUBTFUL


def test_no_audited_holdings_is_doubtful() -> None:
    result = screen_commodity_etf(_financials(has_audited_holdings=False))
    assert result.verdict is CommodityVerdict.DOUBTFUL
    assert any("audited" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Ribawi commodities (gold / silver) — stricter rules
# ---------------------------------------------------------------------------


def test_ribawi_gold_unallocated_segregated_is_doubtful() -> None:
    """Gold (ribawi) requires allocated-physical even with segregated storage."""

    result = screen_commodity_etf(
        _financials(
            commodity=CommodityType.GOLD,
            backing_mode=BackingMode.UNALLOCATED_PHYSICAL,
            storage_location=StorageLocation.SEGREGATED,
        )
    )
    assert result.verdict is CommodityVerdict.DOUBTFUL
    assert any("ribawi" in w and "allocated" in w for w in result.warnings)


def test_ribawi_silver_allocated_commingled_is_doubtful() -> None:
    """Silver (ribawi) requires segregated storage even with allocated backing."""

    result = screen_commodity_etf(
        _financials(
            commodity=CommodityType.SILVER,
            backing_mode=BackingMode.ALLOCATED_PHYSICAL,
            storage_location=StorageLocation.COMMINGLED,
        )
    )
    assert result.verdict is CommodityVerdict.DOUBTFUL
    assert any("ribawi" in w and "segregated" in w for w in result.warnings)


def test_non_ribawi_copper_with_allocated_commingled_is_doubtful() -> None:
    """Non-ribawi copper with commingled storage triggers commingled warning,
    but no ribawi-specific warning."""

    result = screen_commodity_etf(
        _financials(
            commodity=CommodityType.COPPER,
            backing_mode=BackingMode.ALLOCATED_PHYSICAL,
            storage_location=StorageLocation.COMMINGLED,
        )
    )
    assert result.verdict is CommodityVerdict.DOUBTFUL
    # warnings include commingled but not ribawi-specific
    assert any("commingled" in w for w in result.warnings)
    assert not any("ribawi" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Debated commodities — sector-scholar disagreement
# ---------------------------------------------------------------------------


def test_oil_with_clean_backing_is_doubtful() -> None:
    """Oil has scholar disagreement; even clean backing → DOUBTFUL."""

    result = screen_commodity_etf(
        _financials(
            commodity=CommodityType.OIL,
            backing_mode=BackingMode.ALLOCATED_PHYSICAL,
            storage_location=StorageLocation.SEGREGATED,
        )
    )
    assert result.verdict is CommodityVerdict.DOUBTFUL
    assert any("scholar disagreement" in w for w in result.warnings)


def test_natural_gas_with_clean_backing_is_doubtful() -> None:
    result = screen_commodity_etf(
        _financials(
            commodity=CommodityType.NATURAL_GAS,
            backing_mode=BackingMode.ALLOCATED_PHYSICAL,
            storage_location=StorageLocation.SEGREGATED,
        )
    )
    assert result.verdict is CommodityVerdict.DOUBTFUL


def test_agricultural_with_clean_backing_is_doubtful() -> None:
    result = screen_commodity_etf(
        _financials(
            commodity=CommodityType.AGRICULTURAL,
            backing_mode=BackingMode.ALLOCATED_PHYSICAL,
            storage_location=StorageLocation.SEGREGATED,
        )
    )
    assert result.verdict is CommodityVerdict.DOUBTFUL


def test_platinum_palladium_with_clean_backing_is_halal() -> None:
    """Platinum / palladium are non-ribawi industrial metals; clean backing → HALAL."""

    for c in (CommodityType.PLATINUM, CommodityType.PALLADIUM):
        result = screen_commodity_etf(_financials(commodity=c))
        assert result.verdict is CommodityVerdict.HALAL, c


# ---------------------------------------------------------------------------
# Multiple-failure aggregation
# ---------------------------------------------------------------------------


def test_multiple_failures_all_listed() -> None:
    result = screen_commodity_etf(
        _financials(
            backing_mode=BackingMode.SWAP_BACKED,
            storage_location=StorageLocation.PAPER,
            leverage_factor=2.0,
            physical_holdings_pct=0.0,
        )
    )
    assert result.verdict is CommodityVerdict.NOT_HALAL
    assert len(result.failures) >= 3


# ---------------------------------------------------------------------------
# Threshold customisation
# ---------------------------------------------------------------------------


def test_stricter_physical_threshold_flips_verdict() -> None:
    f = _financials(physical_holdings_pct=96.0)
    assert screen_commodity_etf(f).verdict is CommodityVerdict.HALAL
    strict = CommodityThresholds(min_physical_holdings_pct=98.0)
    assert screen_commodity_etf(f, thresholds=strict).verdict is CommodityVerdict.NOT_HALAL


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_screen_result_is_frozen() -> None:
    result = screen_commodity_etf(_financials())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = CommodityVerdict.NOT_HALAL  # type: ignore[misc]


def test_financials_is_frozen() -> None:
    f = _financials()
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.leverage_factor = 99.0  # type: ignore[misc]


def test_thresholds_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_THRESHOLDS.min_physical_holdings_pct = 50.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned
# ---------------------------------------------------------------------------


def test_commodity_type_string_values() -> None:
    assert CommodityType.GOLD.value == "gold"
    assert CommodityType.SILVER.value == "silver"
    assert CommodityType.PLATINUM.value == "platinum"
    assert CommodityType.PALLADIUM.value == "palladium"
    assert CommodityType.COPPER.value == "copper"
    assert CommodityType.OIL.value == "oil"
    assert CommodityType.NATURAL_GAS.value == "natural_gas"
    assert CommodityType.AGRICULTURAL.value == "agricultural"


def test_backing_mode_string_values() -> None:
    assert BackingMode.ALLOCATED_PHYSICAL.value == "allocated_physical"
    assert BackingMode.UNALLOCATED_PHYSICAL.value == "unallocated_physical"
    assert BackingMode.FUTURES_BACKED.value == "futures_backed"
    assert BackingMode.SWAP_BACKED.value == "swap_backed"
    assert BackingMode.UNKNOWN.value == "unknown"


def test_storage_location_string_values() -> None:
    assert StorageLocation.SEGREGATED.value == "segregated"
    assert StorageLocation.COMMINGLED.value == "commingled"
    assert StorageLocation.PAPER.value == "paper"


def test_verdict_string_values() -> None:
    assert CommodityVerdict.HALAL.value == "halal"
    assert CommodityVerdict.NOT_HALAL.value == "not_halal"
    assert CommodityVerdict.DOUBTFUL.value == "doubtful"
    assert CommodityVerdict.INSUFFICIENT_DATA.value == "insufficient_data"


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_halal_result() -> None:
    result = screen_commodity_etf(_financials())
    text = render_screen_result(result)
    assert "✅" in text
    assert "PHYS" in text
    assert "HALAL" in text
    assert "allocated_physical" in text


def test_render_not_halal_result() -> None:
    result = screen_commodity_etf(_financials(backing_mode=BackingMode.SWAP_BACKED))
    text = render_screen_result(result)
    assert "❌" in text
    assert "NOT_HALAL" in text
    assert "failures:" in text


def test_render_doubtful_result() -> None:
    result = screen_commodity_etf(_financials(backing_mode=BackingMode.UNALLOCATED_PHYSICAL))
    text = render_screen_result(result)
    assert "⚠️" in text
    assert "DOUBTFUL" in text
    assert "warnings:" in text


def test_render_insufficient_data_result() -> None:
    result = screen_commodity_etf(_financials(backing_mode=BackingMode.UNKNOWN))
    text = render_screen_result(result)
    assert "❓" in text
    assert "INSUFFICIENT_DATA" in text


def test_render_includes_physical_pct_and_leverage() -> None:
    result = screen_commodity_etf(_financials(physical_holdings_pct=99.5))
    text = render_screen_result(result)
    assert "99.50%" in text
    assert "1.0x" in text


# ---------------------------------------------------------------------------
# Result shape sanity
# ---------------------------------------------------------------------------


def test_result_carries_all_fields() -> None:
    result = screen_commodity_etf(_financials())
    assert isinstance(result, CommodityScreenResult)
    assert result.symbol == "PHYS"
    assert result.commodity is CommodityType.GOLD
    assert result.backing_mode is BackingMode.ALLOCATED_PHYSICAL
    assert result.storage_location is StorageLocation.SEGREGATED
    assert result.physical_holdings_pct == 99.0
    assert result.leverage_factor == 1.0


# ---------------------------------------------------------------------------
# End-to-end: real-world cases
# ---------------------------------------------------------------------------


def test_real_world_gld_unallocated_pooled_is_doubtful() -> None:
    """SPDR Gold Trust (GLD) — historically described as unallocated /
    commingled with HSBC. Even audited, ribawi rules push to DOUBTFUL."""

    result = screen_commodity_etf(
        CommodityETFFinancials(
            symbol="GLD",
            name="SPDR Gold Trust",
            commodity=CommodityType.GOLD,
            backing_mode=BackingMode.UNALLOCATED_PHYSICAL,
            storage_location=StorageLocation.COMMINGLED,
            leverage_factor=1.0,
            physical_holdings_pct=99.5,
            has_audited_holdings=True,
        )
    )
    assert result.verdict is CommodityVerdict.DOUBTFUL


def test_real_world_uco_2x_oil_is_not_halal() -> None:
    """ProShares Ultra Crude Oil (UCO) — 2x daily oil; swap-financed."""

    result = screen_commodity_etf(
        CommodityETFFinancials(
            symbol="UCO",
            name="ProShares Ultra Crude Oil",
            commodity=CommodityType.OIL,
            backing_mode=BackingMode.SWAP_BACKED,
            storage_location=StorageLocation.PAPER,
            leverage_factor=2.0,
            physical_holdings_pct=0.0,
            has_audited_holdings=True,
        )
    )
    assert result.verdict is CommodityVerdict.NOT_HALAL
    assert len(result.failures) >= 3  # swap + paper + leverage + physical%
