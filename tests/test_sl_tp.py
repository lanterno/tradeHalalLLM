"""Setup-typed SL/TP profile tests."""

import pytest

from halal_trader.core.sl_tp import (
    SetupType,
    coerce_setup_type,
    derive_sl_tp,
    profile_for,
)


def test_setup_type_enum_values():
    # Pin the wire format so prompt templates referencing these strings
    # don't silently break if the enum is renamed.
    assert SetupType.BREAKOUT.value == "breakout"
    assert SetupType.MEAN_REVERSION.value == "mean_reversion"
    assert SetupType.MOMENTUM.value == "momentum"
    assert SetupType.RANGE.value == "range"
    assert SetupType.UNKNOWN.value == "unknown"


def test_coerce_normalises_case_and_separators():
    assert coerce_setup_type("Breakout") is SetupType.BREAKOUT
    assert coerce_setup_type("MEAN-REVERSION") is SetupType.MEAN_REVERSION
    assert coerce_setup_type("mean reversion") is SetupType.MEAN_REVERSION
    assert coerce_setup_type(None) is SetupType.UNKNOWN
    assert coerce_setup_type("") is SetupType.UNKNOWN


def test_unknown_raw_falls_back_to_unknown_setup():
    """LLM typos shouldn't crash the cycle; they should fall back."""
    assert coerce_setup_type("scalp_overnight") is SetupType.UNKNOWN


def test_profiles_distinct_per_setup():
    profiles = {st: profile_for(st) for st in SetupType}
    # Each setup must produce a different (sl, tp) pair so the table
    # actually does work — guarding against an accidental copy-paste.
    seen = set()
    for st, p in profiles.items():
        if st is SetupType.UNKNOWN:
            continue
        seen.add((p.stop_loss_pct, p.take_profit_pct))
    assert len(seen) == 4


def test_mean_reversion_has_tighter_sl_than_breakout():
    """Mean-reversion thesis is fragile — a tighter stop is the whole point."""
    mr = profile_for(SetupType.MEAN_REVERSION)
    bo = profile_for(SetupType.BREAKOUT)
    assert mr.stop_loss_pct < bo.stop_loss_pct
    assert bo.take_profit_pct > mr.take_profit_pct


def test_reward_risk_ratios_positive():
    for st in SetupType:
        p = profile_for(st)
        assert p.reward_risk > 1.0  # always at least 1:1


def test_derive_sl_tp_for_breakout_long():
    sl, tp = derive_sl_tp(entry_price=100.0, setup_type="breakout")
    assert sl == pytest.approx(100.0 * (1 - 0.012))
    assert tp == pytest.approx(100.0 * (1 + 0.030))


def test_derive_sl_tp_unknown_falls_back():
    sl, tp = derive_sl_tp(entry_price=100.0, setup_type=None)
    # Falls through to legacy 1% / 2% defaults.
    assert sl == pytest.approx(99.0)
    assert tp == pytest.approx(102.0)


def test_derive_sl_tp_rejects_short_side():
    """Shorts violate halal — keep the surface area visibly absent."""
    with pytest.raises(NotImplementedError, match="halal"):
        derive_sl_tp(entry_price=100.0, setup_type="momentum", side="sell")
