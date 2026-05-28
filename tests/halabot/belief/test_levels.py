"""Level engine — all-None guard, ratchet-up invalidation, support/resistance."""

from __future__ import annotations

from halabot.belief.levels import update_levels
from halabot.belief.schema import Levels


def test_cold_start_all_none_returns_none_invalidation_not_crash():
    """No swings, no ATR, no prior → None invalidation (fix R, all-None max)."""
    out = update_levels(
        last_price=None, swing_lows=[], swing_highs=[], atr=None, prev=Levels()
    )
    assert out.invalidation is None
    assert out.stop is None


def test_invalidation_from_atr_floor_when_no_swings():
    out = update_levels(
        last_price=100.0, swing_lows=[], swing_highs=[], atr=2.0, prev=Levels(),
        atr_stop_mult=2.0,
    )
    assert out.invalidation == 96.0  # 100 - 2*2


def test_invalidation_ratchets_up_never_down():
    # Prior invalidation 95; new structural/atr lower → keep the higher prior.
    out = update_levels(
        last_price=100.0, swing_lows=[90.0], swing_highs=[], atr=2.0,
        prev=Levels(invalidation=95.0), atr_stop_mult=2.0,
    )
    assert out.invalidation == 96.0  # max(90, 96, 95) — never loosens below 95


def test_invalidation_rises_with_price():
    out = update_levels(
        last_price=120.0, swing_lows=[110.0], swing_highs=[], atr=2.0,
        prev=Levels(invalidation=96.0), atr_stop_mult=2.0,
    )
    assert out.invalidation == 116.0  # max(110, 116, 96)


def test_support_and_resistance_nearest_to_price():
    out = update_levels(
        last_price=100.0,
        swing_lows=[80.0, 95.0, 70.0],
        swing_highs=[105.0, 130.0],
        atr=1.0,
        prev=Levels(),
    )
    assert out.support == 95.0       # nearest low below
    assert out.resistance == 105.0   # nearest high above


def test_stop_mirrors_invalidation():
    out = update_levels(
        last_price=100.0, swing_lows=[], swing_highs=[], atr=3.0, prev=Levels(),
        atr_stop_mult=2.0,
    )
    assert out.stop == out.invalidation == 94.0
