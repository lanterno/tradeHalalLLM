"""Kelly-fractional sizer tests — confidence, vol scaling, and clamps."""

from decimal import Decimal

from halal_trader.core.sizing import (
    KELLY_FRACTION,
    VOL_SCALE_MAX,
    VOL_SCALE_MIN,
    SizingInputs,
    size_position,
)


def _inputs(**overrides) -> SizingInputs:
    base = dict(
        equity_usd=Decimal("10000"),
        confidence=0.7,
        atr_pct=0.02,
        atr_baseline=0.02,
        base_max_position_pct=0.25,
        available_usd=None,
    )
    base.update(overrides)
    return SizingInputs(**base)


def test_low_confidence_returns_zero():
    """We never size into a trade where p ≤ 0.5 — those are coin flips."""
    r = size_position(_inputs(confidence=0.5))
    assert r.notional_usd == Decimal("0")
    assert r.capped_by == "kelly"


def test_quarter_kelly_at_baseline_vol():
    # confidence 0.7 → edge 0.4 → quarter-Kelly = 0.10 of equity.
    # Baseline vol means vol_scale = 1.0. 10% of $10k = $1000.
    r = size_position(_inputs(confidence=0.7, atr_pct=0.02, atr_baseline=0.02))
    assert r.kelly_fraction == Decimal("0.4") * KELLY_FRACTION
    assert r.vol_scale == Decimal("1")
    assert r.notional_usd == Decimal("1000.00")


def test_high_vol_shrinks_size():
    # ATR 4% vs baseline 2% → vol_scale = 0.5; same edge gives half.
    r = size_position(_inputs(confidence=0.7, atr_pct=0.04, atr_baseline=0.02))
    assert r.vol_scale == Decimal("0.5")
    assert r.notional_usd == Decimal("500.00")


def test_low_vol_boosts_size_clamped():
    # ATR 0.001 vs baseline 0.02 → raw scale 20×, clamped to VOL_SCALE_MAX.
    r = size_position(_inputs(confidence=0.7, atr_pct=0.001, atr_baseline=0.02))
    assert r.vol_scale == VOL_SCALE_MAX


def test_high_vol_clamped_at_floor():
    # ATR 1.0 vs baseline 0.02 → raw scale 0.02, clamped to VOL_SCALE_MIN.
    r = size_position(_inputs(confidence=0.7, atr_pct=1.0, atr_baseline=0.02))
    assert r.vol_scale == VOL_SCALE_MIN


def test_base_max_position_pct_caps_extreme_confidence():
    # confidence 1.0 → edge 1.0 → quarter-Kelly = 0.25; vol boost lifts
    # it to 0.50 — the base ceiling pulls it back down to 0.25.
    r = size_position(
        _inputs(confidence=1.0, atr_pct=0.01, atr_baseline=0.02, base_max_position_pct=0.25)
    )
    assert r.fraction_used == Decimal("0.25")
    assert r.capped_by == "base_max"
    assert r.notional_usd == Decimal("2500.00")


def test_available_usd_caps_below_kelly_target():
    r = size_position(_inputs(confidence=0.9, available_usd=Decimal("400")))
    assert r.notional_usd <= Decimal("400")
    assert r.capped_by == "available"


def test_dust_returns_zero():
    """Sub-dollar notional is zeroed out — no point spending fees on it."""
    r = size_position(_inputs(equity_usd=Decimal("5"), confidence=0.6))
    assert r.notional_usd == Decimal("0")
    assert r.capped_by == "min_dust"


def test_no_atr_falls_back_to_unscaled():
    """Missing ATR (e.g. cold start on a new pair) shouldn't NaN — vol_scale = 1."""
    r = size_position(_inputs(confidence=0.7, atr_pct=0.0))
    assert r.vol_scale == Decimal("1")
    assert r.notional_usd > Decimal("0")


def test_confidence_above_one_is_clamped():
    """Defensive: an LLM that hallucinates confidence=1.5 shouldn't bypass caps."""
    r = size_position(_inputs(confidence=1.5))
    # Edge maxes at 1.0, then quarter-Kelly = 0.25 (= base_max). Still safe.
    assert r.fraction_used <= Decimal("0.25")
